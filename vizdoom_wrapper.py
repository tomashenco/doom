import vizdoom
from skimage.transform import resize
import numpy as np
from collections import OrderedDict


class VizdoomWrapper:
    def __init__(self,
                 config_path: str,
                 reward_table: OrderedDict,
                 state_resolution: tuple=(30, 45, 1),
                 frame_repeat: int=4,
                 frame_stack: int=4,
                 show_mode: bool=False,
                 ):
        self.config_path = config_path
        self.state_resolution = state_resolution
        self.frame_repeat = frame_repeat
        self.frame_stack = frame_stack
        self.show_mode = show_mode
        self.reward_table = reward_table
        self.total_reward = 0

        self.game = self.__initialise_game()

        self.possible_actions = np.eye(self.game.get_available_buttons_size())

        self.stacked_frames = np.zeros((self.state_resolution[0],
                                        self.state_resolution[1],
                                        self.frame_stack), dtype=np.float32)
        self.variables = ()

    def __initialise_game(self):
        game = vizdoom. DoomGame()
        game.load_config(self.config_path)
        # Screen settings
        game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
        game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
        # Display settings
        game.set_window_visible(self.show_mode)
        game.set_sound_enabled(self.show_mode)
        if self.show_mode:
            game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
        # Game variables used for reward enhancing
        for name, _ in self.reward_table:
            try:
                game.add_available_game_variable(eval(
                    'vizdoom.GameVariable.'+name))
            except NameError:
                print('{} is not defined as possible game variable'
                      .format(name))
        game.init()

        return game

    def __preprocess_image(self, raw_image):
        img = resize(raw_image, (self.state_resolution[0],
                                 self.state_resolution[1]))
        img = img.reshape((-1, *self.state_resolution))
        img = img.astype(np.float32)
        return img

    def __enhance_reward(self, reward):
        # Will enhance reward by custom defined table
        previous_variables = self.variables
        current_variables = self.game.get_state().game_variables
        if len(previous_variables) != len(current_variables):
            return reward
        else:

            for prev_var, cur_var, reward_modifier in zip(
                    previous_variables, current_variables,
                    self.reward_table.values()):
                reward += (cur_var - prev_var) * reward_modifier

            return reward

    def __update(self):
        self.variables = self.game.get_state().game_variables

        current_frame = self.__preprocess_image(
            self.game.get_state().screen_buffer)
        self.stacked_frames = np.append(self.stacked_frames[1:], current_frame,
                                        axis=0)

    def __is_done(self):
        return self.game.is_episode_finished()

    def get_current_state(self):
        return self.stacked_frames

    def get_action_size(self):
        return len(self.possible_actions)

    def get_state_size(self):
        return self.stacked_frames.shape

    def get_total_reward(self):
        return self.total_reward

    def step(self, action_index):
        action = list(self.possible_actions[action_index])
        base_reward = self.game.make_action(action, self.frame_repeat)
        reward = self.__enhance_reward(base_reward)
        self.total_reward += reward
        done = self.__is_done()
        if not done:
            self.__update()
            next_state = self.get_current_state()
        else:
            next_state = None

        return next_state, reward, done

    def set_action(self, action_index):
        # Method used when showing the game as it smoothes FPS
        action = list(self.possible_actions[action_index])
        self.game.set_action(action)
        for _ in range(self.frame_repeat):
            self.game.advance_action()

    def new_game(self):
        self.game.new_episode()
        self.stacked_frames = np.zeros_like(self.stacked_frames)
        self.variables = ()
        self.total_reward = 0
