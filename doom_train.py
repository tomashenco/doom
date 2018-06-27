from agent import DuelingDoom
import game_utils

from itertools import product
import tqdm
import numpy as np

training_episodes_per_epoch = 2000
testing_episodes_per_epoch = 100
epochs = 20
replay_batch_size = 64

# Parameters for enhancing reward
enhance = False
health_loss_penalty = 0
killcount_reward = 1.0

load_pretrained_network = False

print('Initialising VizDoom...')
game = game_utils.initialise_game(show_mode=False)

# Action = which buttons can be pressed
action_size = game.get_available_buttons_size()
# Get all combinations of possible actions
actions = [list(a) for a in product([0, 1], repeat=action_size)]

print('Initialising Doomguy...')
doomguy = DuelingDoom(game_utils.screen_resolution, len(actions))
if load_pretrained_network:
    doomguy.load_model()


def enhance_reward(current_variables, game_reward, previous_variables):
    if len(previous_variables) != len(current_variables):
        return game_reward
    else:
        current_ammo, current_health, current_killcount = current_variables
        prev_ammo, prev_health, prev_killcount = previous_variables

        game_reward += (current_health - prev_health) * health_loss_penalty
        game_reward += (current_killcount - prev_killcount) * killcount_reward

        return game_reward


for epoch in range(epochs):
    print('\nEpoch {}\n-------'.format(epoch + 1))
    print('\nTraining...')
    game.new_episode()
    train_scores = []
    prev_variables = []

    for episode in tqdm.trange(training_episodes_per_epoch, leave=False):
        # Get state, action, reward, done and next state
        state = game_utils.preprocess_image(game.get_state().screen_buffer)
        best_action = doomguy.act(state)
        reward = game.make_action(actions[best_action],
                                  game_utils.frame_repeat)
        done = game.is_episode_finished()
        next_state = game_utils.preprocess_image(
            game.get_state().screen_buffer) if not done else None

        # Enhance reward
        if enhance and not done:
            variables = game.get_state().game_variables
            reward = enhance_reward(variables, reward, prev_variables)
            prev_variables = variables

        # Save to memory
        doomguy.remember(state, best_action, reward, next_state, done)
        # Replay from memory
        doomguy.replay(replay_batch_size)

        # Store results on game end
        if game.is_episode_finished():
            score = game.get_total_reward()
            train_scores.append(score)
            game.new_episode()

    if len(train_scores) > 0:
        train_scores = np.array(train_scores)

        print('Results: mean: {}±{}, min: {}, max: {}'
              .format(train_scores.mean(), train_scores.std(),
                      train_scores.min(), train_scores.max()))

    print('\nTesting...')
    test_scores = []

    for episode in tqdm.trange(testing_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state = game_utils.preprocess_image(game.get_state().screen_buffer)
            best_action = doomguy.act(state)
            reward = game.make_action(actions[best_action],
                                      game_utils.frame_repeat)

        total_reward = game.get_total_reward()
        test_scores.append(total_reward)

    test_scores = np.array(test_scores)

    print('Results: mean: {}±{}, min: {}, max: {}'
          .format(test_scores.mean(), test_scores.std(), test_scores.min(),
                  test_scores.max()))

game.close()
