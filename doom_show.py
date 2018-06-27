import game_utils
from agent import DuelingDoom

from itertools import product
from time import sleep


episodes_to_watch = 10

print('Initialising VizDoom...')
game = game_utils.initialise_game(show_mode=True)

# Action = which buttons can be pressed
action_size = game.get_available_buttons_size()
# Get all combinations of possible actions
actions = [list(a) for a in product([0, 1], repeat=action_size)]

print('Initialising Doomguy...')
doomguy = DuelingDoom(game_utils.screen_resolution, action_size)
doomguy.load_model()

for _ in range(episodes_to_watch):
    game.new_episode()
    while not game.is_episode_finished():
        state = game_utils.preprocess_image(game.get_state().screen_buffer)
        best_action = doomguy.act(state)

        game.set_action(actions[best_action])
        for _ in range(game_utils.frame_repeat):
            game.advance_action()

        # if game.is_episode_finished():
        #     sleep(1.0)

    # Sleep between episodes
    sleep(1.0)
    score = game.get_total_reward()
    print("Total score: ", score)
