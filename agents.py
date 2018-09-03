from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, Lambda, add
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, RMSprop
# needs to stay as K because keras otherwise doesn't know how to load the model
from keras import backend as K
import random
import numpy as np


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        clip = 0.2

        prob = K.sum(y_true * y_pred)
        old_prob = K.sum(y_true * old_prediction)
        # Add small number to avoid division by 0
        ratio = prob / (old_prob + 1e-10)

        # Conservative surrogate
        surr1 = ratio * advantage
        # Clipping surrogate
        surr2 = K.clip(ratio, 1.0 - clip, 1.0 + clip) * advantage

        # PPO pessimistic surrogate
        return - K.minimum(surr1, surr2)

    return loss


def stochastic_policy_gradient_loss(discounted_reward):
    def loss(y_true, y_pred):
        negative_log_probability = K.categorical_crossentropy(
            target=y_true, output=y_pred, from_logits=False)
        return K.mean(negative_log_probability * discounted_reward)

    return loss


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

    def bulk(self):
        return (self.state[:self.size, :, :, :],
                self.action[:self.size],
                self.reward[:self.size],
                self.next_state[:self.size, :, :, :],
                self.done[:self.size])


class BaseQDoom:
    def __init__(self, state_size, action_size, initialise_model=True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.update_target_frequency = 3000
        self.episode = 0

        # Exploration parameters
        self.epsilon_initial = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.exploration_steps = 25000

        self.memory = ReplayMemory(capacity=25000, state_size=self.state_size)

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
        x = Conv2D(8, kernel_size=6, strides=3, activation='relu')(
            state_input)
        x = Conv2D(8, kernel_size=3, strides=2, activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        x = Dense(128, activation='relu')(x)
        x = Dense(self.action_size)(x)

        model = Model(inputs=state_input, outputs=x)
        rms = RMSprop(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=rms)

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

        q2 = np.max(self.model.predict(next_state), axis=1)
        target_q = self.model.predict(state)
        target_q[np.arange(target_q.shape[0]), action] = reward + (
                self.gamma * (1 - done) * q2)

        history = self.model.fit(state.reshape((-1, *self.state_size)),
                                 target_q.reshape((-1, self.action_size)),
                                 batch_size=sample_batch_size, epochs=1,
                                 verbose=0, callbacks=[self.checkpointer])

        mean_loss = np.mean(history.history['loss'])

        # Make sure that model still experiments after long time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_initial - self.epsilon_min) \
                            / self.exploration_steps

        return mean_loss


class DuelingDoom:
    def __init__(self, state_size, action_size, initialise_model=True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.update_target_frequency = 3000
        self.episode = 0

        # Exploration parameters
        self.epsilon_initial = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.exploration_steps = 25000

        self.memory = ReplayMemory(capacity=25000, state_size=self.state_size)

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
        x = Conv2D(32, kernel_size=8, strides=4, activation='relu')(
            state_input)
        x = Conv2D(64, kernel_size=4, strides=2, activation='relu')(x)
        x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(512, activation='relu')(x)
        state_value = Dense(1, kernel_initializer='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1),
                             output_shape=(self.action_size,))(state_value)

        # action advantage tower - A
        action_advantage = Dense(512, activation='relu')(x)
        action_advantage = Dense(self.action_size)(action_advantage)
        action_advantage = Lambda(
            lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
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

        history = self.model.fit(state.reshape((-1, *self.state_size)),
                                 target.reshape((-1, self.action_size)),
                                 batch_size=sample_batch_size, epochs=1,
                                 verbose=0, callbacks=[self.checkpointer])

        mean_loss = np.mean(history.history['loss'])

        # Make sure that model still experiments after long time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_initial - self.epsilon_min) \
                            / self.exploration_steps

        return mean_loss


class PPODoom:
    def __init__(self, state_size, action_size, actor_path, critic_path,
                 initialise_model=True):
        self.learning_rate = 0.00025
        self.epochs_per_replay = 10

        # Exploration parameters
        self.epsilon_initial = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.exploration_steps = 25000 / self.epochs_per_replay

        self.state_size = state_size
        self.action_size = action_size
        self.actor_path = actor_path
        self.critic_path = critic_path

        self.memory = ReplayMemory(capacity=25000, state_size=self.state_size)
        self.save_period = 1000
        self.actor_checkpointer = ModelCheckpoint(filepath=self.actor_path,
                                                  verbose=0,
                                                  save_best_only=False,
                                                  period=self.save_period)
        self.critic_checkpointer = ModelCheckpoint(filepath=self.critic_path,
                                                   verbose=0,
                                                   save_best_only=False,
                                                   period=self.save_period)

        if initialise_model:
            self.actor = self.build_actor()
            self.critic = self.build_critic()
        else:
            self.actor = None
            self.critic = None

    def build_actor(self):
        state_input = Input(shape=self.state_size)
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.action_size,))

        x = Conv2D(8, kernel_size=6, strides=3, activation='relu')(
            state_input)
        x = Conv2D(8, kernel_size=3, strides=2, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output_actions = Dense(self.action_size)(x)

        model = Model(inputs=[state_input, advantage, old_prediction],
                      outputs=[output_actions])
        model.compile(optimizer=RMSprop(lr=self.learning_rate),
                      loss=[proximal_policy_optimization_loss(advantage,
                                                              old_prediction)])

        return model

    def build_critic(self):
        state_input = Input(shape=self.state_size)

        x = Conv2D(8, kernel_size=6, strides=3, activation='relu')(
            state_input)
        x = Conv2D(8, kernel_size=3, strides=2, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output_actions = Dense(1)(x)

        model = Model(inputs=[state_input], outputs=[output_actions])
        model.compile(optimizer=RMSprop(lr=self.learning_rate), loss='mse')

        return model

    def load_model(self):
        self.actor = self.build_actor()
        self.actor.load_weights(self.actor_path)

        self.critic = self.build_critic()
        self.critic.load_weights(self.critic_path)

        self.epsilon = 0.0

    def act(self, state):
        # Explore randomly
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            img = state.reshape((-1, *state.shape))
            dummy_advantage = np.zeros((1, 1))
            dummy_action = np.zeros((1, self.action_size))
            action = self.actor.predict([img, dummy_advantage, dummy_action])
            return np.argmax(action[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self, sample_batch_size):
        batch_size = min(self.memory.size, sample_batch_size)

        # Get a sample replay from memory
        state, action_index, reward, next_state, done = self.memory.sample(
            batch_size)
        action = np.eye(self.action_size)[action_index]
        old_prediction = action.copy()

        # Calculate advantage, refer to https://arxiv.org/pdf/1602.01783.pdf
        critic_pred = self.critic.predict(state).reshape((-1))
        advantage = reward - critic_pred

        state = state.reshape((-1, *self.state_size))

        history = self.actor.fit([state, advantage, old_prediction], [action],
                                 batch_size=sample_batch_size,
                                 epochs=self.epochs_per_replay, verbose=0,
                                 callbacks=[self.actor_checkpointer])

        mean_loss = np.mean(history.history['loss'])

        self.critic.fit([state], [reward], batch_size=sample_batch_size,
                        epochs=self.epochs_per_replay, verbose=0,
                        callbacks=[self.critic_checkpointer])

        # Make sure that model still experiments after long time
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_initial - self.epsilon_min) \
                            / self.exploration_steps

        return mean_loss


class PolicyGradientAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma,
                 save_path, initialise_model=True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.weight_backup = save_path
        self.memory = ReplayMemory(50000, self.state_size)
        self.checkpointer = ModelCheckpoint(filepath=self.weight_backup,
                                            verbose=0, save_best_only=False,
                                            period=10)

        if initialise_model:
            self.model = self.build_model()
        else:
            self.model = None

    def build_model(self):
        state = Input(shape=self.state_size)
        discounted_reward = Input(shape=(1,))
        x = Conv2D(32, kernel_size=8, strides=4, activation='relu',
                   input_shape=self.state_size)(state)
        x = BatchNormalization()(x)

        x = Conv2D(64, kernel_size=4, strides=2, activation='relu')(x)
        x = BatchNormalization()(x)

        x = Conv2D(64, kernel_size=3, strides=1, activation='relu')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.action_size, activation='softmax')(x)

        model = Model(inputs=[state, discounted_reward], outputs=[output])

        adam = Adam(lr=self.learning_rate)
        model.compile(loss=stochastic_policy_gradient_loss(discounted_reward),
                      optimizer=adam)

        return model

    def load_model(self):
        self.model = self.build_model()
        self.model.load_weights(self.weight_backup)

    def reset_memory(self):
        self.memory = ReplayMemory(50000, self.state_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        state = state.reshape((-1, *state.shape))
        dummy_reward = np.zeros((1, 1))

        policy = self.model.predict([state, dummy_reward])
        action_index = np.random.choice(self.action_size, p=policy.flatten())
        return action_index

    def discount_and_normalize_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0.0

        # Reward for an action is also all discounted rewards that it resulted
        # in the future
        for i in reversed(range(len(episode_rewards))):
            cumulative = cumulative * self.gamma + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative

        # Normalize rewards by subtracting mean and dividing by std
        # This avoids big variation in rewards for different episodes
        mean = np.mean(discounted_episode_rewards)
        std = np.std(discounted_episode_rewards)
        discounted_episode_rewards = (discounted_episode_rewards - mean) / std

        return discounted_episode_rewards

    def train(self):
        state, action_index, reward, next_state, done = self.memory.bulk()

        targets = np.zeros((len(action_index), self.action_size))
        for i in range(len(action_index)):
            targets[i, action_index] = 1

        discounted_reward = self.discount_and_normalize_rewards(reward)

        loss = self.model.fit(x=[state, discounted_reward], y=targets,
                              epochs=1, verbose=0,
                              callbacks=[self.checkpointer])

        return loss.history['loss']




