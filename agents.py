from keras.models import Model, load_model
from keras.layers import Dense, Conv2D, Flatten, Input, Lambda, add
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
# needs to stay as K because keras otherwise doesn't know how to load the model
from keras import backend as k
import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, state_size):
        state_shape = (capacity, state_size[0], state_size[1], state_size[2])
        self.state = np.zeros(state_shape, dtype=np.float32)
        self.next_state = np.zeros(state_shape, dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        self.state[self.index, :, :, :] = state
        self.action[self.index] = action
        if not done:
            self.next_state[self.index, :, :, :] = next_state
        self.done[self.index] = done
        self.reward[self.index] = reward

        # Mod of capacity is going to overwrite from beginning
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, sample_size):
        i = random.sample(range(0, self.size), sample_size)
        return (self.state[i], self.action[i], self.reward[i],
                self.next_state[i], self.done[i])


class DuelingDoom:
    def __init__(self, state_size, action_size, initialise_model=True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.update_target_frequency = 3000
        self.episode = 0

        # Exploration parameters
        self.epsilon_initial = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.exploration_steps = 250000

        self.memory = ReplayMemory(capacity=100000, state_size=self.state_size)

        self.weight_backup = 'models/doom_defend_the_center.hd5'
        self.checkpointer = ModelCheckpoint(filepath=self.weight_backup,
                                            verbose=0, save_best_only=False,
					    period=1000)
        if initialise_model:
            self.model = self.build_model()
            self.target_model = self.build_model()
        else:
            self.model = None
            self.target_model = None

    def build_model(self):
        state_input = Input(shape=self.state_size)
        x = Conv2D(16, kernel_size=8, strides=4, activation='relu')(
            state_input)
        x = Conv2D(32, kernel_size=4, strides=2, activation='relu')(x)
        x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, activation='relu')(x)
        state_value = Dense(1, kernel_initializer='uniform')(state_value)
        state_value = Lambda(lambda s: k.expand_dims(s[:, 0], -1),
                             output_shape=(self.action_size,))(state_value)

        # action advantage tower - A
        action_advantage = Dense(256, activation='relu')(x)
        action_advantage = Dense(self.action_size)(action_advantage)
        action_advantage = Lambda(
            lambda a: a[:, :] - k.mean(a[:, :], keepdims=True),
            output_shape=(self.action_size,))(action_advantage)

        # merge to state-action value function Q
        state_action_value = add([state_value, action_advantage])

        model = Model(inputs=state_input, outputs=state_action_value)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return model

    def load_model(self):
        self.model = load_model(self.weight_backup)
        self.target_model = load_model(self.weight_backup)
        self.epsilon = 0.0

    def act(self, state):
        # Explore randomly
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            img = state.reshape((-1, *state.shape))
            action = self.model.predict(img)
            return np.argmax(action[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, sample_batch_size):
        batch_size = min(self.memory.size, sample_batch_size)

        # Get a sample replay from memory
        state, action, reward, next_state, done = self.memory.sample(
            batch_size)

        # DDQN -    selection of action is from model
        #           update is from target model
        target_value = reward + self.gamma * np.amax(
            self.target_model.predict(next_state), axis=1) * (1 - done)

        target = self.model.predict(state)
        target_action = np.argmax(self.model.predict(next_state), axis=1)
        target[np.arange(target_value.shape[0]), target_action] = target_value

        # Every so often update target model
        if self.episode % self.update_target_frequency:
            self.target_model.set_weights(self.model.get_weights())

        self.model.fit(state.reshape((-1, *self.state_size)),
                       target.reshape((-1, self.action_size)),
                       batch_size=sample_batch_size, epochs=1, verbose=0,
                       callbacks=[self.checkpointer])

        # Make sure that model still experiments after long time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_initial - self.epsilon_min) \
                            / self.exploration_steps
