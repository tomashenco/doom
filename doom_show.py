from vizdoom_wrapper import VizdoomWrapper
from agents import DuelingDoom, PPODoom, BaseQDoom

from time import sleep
from collections import OrderedDict


episodes_to_watch = 2

print('Initialising VizDoom...')
config_path = 'scenarios/basic.cfg'
actor_path = 'models/defend_the_center_actor.hd5'
critic_path = 'models/defend_the_center_critic.hd5'
reward_table = OrderedDict({'FRAGCOUNT': 1})
resolution = (90, 60)
doom = VizdoomWrapper(config_path=config_path, reward_table=reward_table,
                      frame_resolution=resolution, show_mode=True,
                      frame_stack=1)

print('Initialising Doomguy...')
doomguy = BaseQDoom(doom.get_state_size(), doom.get_action_size())
doomguy.load_model()

for _ in range(episodes_to_watch):
    done = False
    doom.new_game()
    while not done:
        state = doom.get_current_state()
        best_action = doomguy.act(state)
        done = doom.set_action(best_action)

    # Sleep between episodes
    sleep(1.0)
    score = doom.get_total_reward()
    print('Total score: {}'.format(score))
