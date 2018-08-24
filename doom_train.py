from agents import DuelingDoom, PPODoom, BaseQDoom
from vizdoom_wrapper import VizdoomWrapper

import tqdm
import numpy as np
from collections import OrderedDict


training_episodes_per_epoch = 2000
testing_episodes_per_epoch = 50
epochs = 20
replay_batch_size = 32

load_pretrained_network = False

print('Initialising VizDoom...')
config_path = 'scenarios/basic.cfg'
actor_path = 'models/basic_actor.hd5'
critic_path = 'models/basic_critic.hd5'
reward_table = OrderedDict({'FRAGCOUNT': 1})
resolution = (90, 60)
doom = VizdoomWrapper(config_path=config_path, reward_table=reward_table,
                      frame_resolution=resolution, show_mode=False,
                      frame_stack=1)

print('Initialising Doomguy...')
doomguy = BaseQDoom(doom.get_state_size(), doom.get_action_size())
# doomguy = DuelingDoom(doom.get_state_size(), doom.get_action_size())
if load_pretrained_network:
    doomguy.load_model()


for epoch in range(epochs):
    print('\nEpoch {}\n-------'.format(epoch + 1))
    print('\nTraining...')
    doom.new_game()
    train_scores = []
    prev_variables = []

    for episode in tqdm.trange(training_episodes_per_epoch, leave=False):
        # Get state, action, reward, done and next state
        state = doom.get_current_state()
        best_action_index = doomguy.act(state)
        next_state, reward, done = doom.step(best_action_index)

        # Save to memory
        doomguy.remember(state, best_action_index, reward, next_state, done)
        # Replay from memory
        doomguy.replay(replay_batch_size)

        # Store results on game end
        if done:
            score = doom.get_total_reward()
            train_scores.append(score)
            doom.new_game()

    if len(train_scores) > 0:
        train_scores = np.array(train_scores)

        print('Results: mean: {}±{}, min: {}, max: {}'
              .format(train_scores.mean(), train_scores.std(),
                      train_scores.min(), train_scores.max()))
        print('Current exploration rate: {}'.format(doomguy.epsilon))

    print('\nTesting...')
    test_scores = []

    for episode in tqdm.trange(testing_episodes_per_epoch, leave=False):
        done = False
        doom.new_game()
        while not done:
            state = doom.get_current_state()
            best_action_index = doomguy.act(state)
            next_state, reward, done = doom.step(best_action_index)

        total_reward = doom.get_total_reward()
        test_scores.append(total_reward)

    test_scores = np.array(test_scores)

    print('Results: mean: {}±{}, min: {}, max: {}'
          .format(test_scores.mean(), test_scores.std(), test_scores.min(),
                  test_scores.max()))
