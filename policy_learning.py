from vizdoom_wrapper import VizdoomWrapper
from agents import PolicyGradientAgent

from collections import OrderedDict
from time import sleep

load_pretrained_network = True
train_network = False
show_results = True

config_path = 'scenarios/health_gathering.cfg'
model_path = 'models/health.hd5'
reward_table = OrderedDict({})
resolution = (84, 84)

episodes = 500
gamma = 0.99
learning_rate = 0.0002

print('Initialising Doom...')
doom = VizdoomWrapper(config_path=config_path, reward_table=reward_table,
                      frame_resolution=resolution, show_mode=False,
                      frame_stack=4)

doomguy = PolicyGradientAgent(doom.get_state_size(), doom.get_action_size(),
                              learning_rate, gamma, save_path=model_path)

if load_pretrained_network:
    doomguy.load_model()

if train_network:
    for episode in range(episodes):
        print('Episode', episode)
        doom.new_game()
        done = False
        step = 0

        while not done:
            state = doom.get_current_state()
            action_index = doomguy.act(state)
            next_state, reward, done = doom.step(action_index)
            doomguy.remember(state, action_index, reward, next_state, done)

            step += 1

        loss = doomguy.train()
        doomguy.reset_memory()
        print('Total steps: {}, loss was: {}'.format(step, loss))

if show_results:
    doom = VizdoomWrapper(config_path=config_path, reward_table=reward_table,
                          frame_resolution=resolution, show_mode=True,
                          frame_stack=4)
    for episode in range(3):
        doom.new_game()
        done = False

        while not done:
            state = doom.get_current_state()
            action_index = doomguy.act(state)
            done = doom.set_action(action_index)

        # Sleep between episodes
        sleep(1.0)
        score = doom.get_total_reward()
        print('Total score: {}'.format(score))
