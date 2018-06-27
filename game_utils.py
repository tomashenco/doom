import vizdoom
from skimage.transform import resize
import numpy as np


screen_resolution = (60, 80, 1)
frame_repeat = 4


def initialise_game(show_mode=False):
    game = vizdoom. DoomGame()
    # Scenarios trained:
    # simpler_basic (30, 45, 1)
    # simpler_basic_compare (30, 45, 1)
    game.load_config('/home/tomasz.dobrzycki@UK.CF247.NET/dev/ViZDoom/'
                     'scenarios/simpler_basic.cfg')
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.set_window_visible(show_mode)
    game.set_sound_enabled(show_mode)
    if show_mode:
        game.set_mode(vizdoom.Mode.ASYNC_PLAYER)
    game.init()

    return game


def preprocess_image(raw_image):
    img = resize(raw_image, (screen_resolution[0], screen_resolution[1]))
    img = img.reshape((-1, *screen_resolution))
    img = img.astype(np.float32)
    return img
