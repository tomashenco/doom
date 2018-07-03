from vizdoom_wrapper import VizdoomWrapper
from agents import DuelingDoom

from time import sleep
from collections import OrderedDict


episodes_to_watch = 10

print('Initialising VizDoom...')
config_path = '/home/tomasz.dobrzycki@UK.CF247.NET/dev/ViZDoom/scenarios/' \
              'defend_the_line.cfg'
reward_table = OrderedDict({'KILLCOUNT': 10})
resolution = (84, 84)
doom = VizdoomWrapper(config_path=config_path, reward_table=reward_table,
                      state_resolution=resolution, show_mode=True)

print('Initialising Doomguy...')
doomguy = DuelingDoom(doom.get_state_size(), doom.get_action_size())
doomguy.load_model()

for _ in range(episodes_to_watch):
    done = False
    doom.new_game()
    while not done:
        state = doom.get_current_state()
        best_action = doomguy.act(state)
        doom.set_action(best_action)

    # Sleep between episodes
    sleep(1.0)
    score = doom.get_total_reward()
    print('Total score: {}'.format(score))
