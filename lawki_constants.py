
# STATION_CONFIG:
#   - x, y: location of player on screen
#   - width, height: size of player
#   - addr: location of player relative to main server
#   - port: listening port
#   - tile_params: list off tuples (x, y, scale) of additional video quads



STATION_ADDRESSES = {
#    0: {'addr': '127.0.0.1',        'port': 1880},
    1: {'addr': '127.0.0.1',        'port': 1881},
#    1: {'addr': '192.168.178.158',        'port': 1881},
#    2: {'addr': '127.0.0.1',        'port': 1882},
#    3: {'addr': '192.168.0.147',    'port': 1883},
#    4: {'addr': '192.168.0.147',    'port': 1884},
#    5: {'addr': '192.168.0.147',    'port': 1885},
}


STATION_VIDEO_CONFIG = {
#   1: {'x': 0,     'y': 0, 'width': 500,   'height': 500, 'tile_params': [(0, 0, None, None)]},
    1: {'x': 0,     'y': 0,   'width': 2432,   'height': 1000, 'tile_params': [(10, 10, 640, 480), (1920, 0, 512, 480),
    		(660, 10, 768, 480), (10, 500, 640, 480), (660, 500, 512, 384), (1182, 500, 512, 384),
    		]},
#   2: {'x': 1920,     'y': 0, 'width': 512,   'height': 480, 'tile_params': [(0, 0, None, None)]},
#    2: {'x': 1920,  'y': 0,   'width': 640,   'height': 384, 'tile_params': [(0, 0, 1)]},
#    3: {'x': 0,     'y': 0,   'width': 640,   'height': 384, 'tile_params': [(0, 0, 1)]},
#    5: {'x': 1920,  'y': 0,   'width': 1152,  'height': 384, 'tile_params': [(0, 0, 1)]},
#    4: {'x': 3840,  'y': 0,   'width': 768,   'height': 384, 'tile_params': [(0, 0, 1)]},
}



STATION_AUDIO_CONFIG = {
    0: {'gen_kwargs': {'allow_repeat': 1}},
    1: {'gen_kwargs': {'allow_repeat': 16}},
    2: {'gen_kwargs': {'allow_repeat': 3}},
    3: {'gen_kwargs': {'allow_repeat': 4}},
    4: {'gen_kwargs': {'allow_repeat': 8}},
    5: {'gen_kwargs': {'allow_repeat': 2}},
}


RW_PARAMS = {
    1: {'cluster_change_prob': 0.2, 'cluster_poisson_lambda': 4, 'memory_len': 100, 'nudge_var': 1.0},
    2: {'cluster_change_prob': 0.2, 'cluster_poisson_lambda': 4, 'memory_len': 100, 'nudge_var': 1.0},
    3: {'cluster_change_prob': 0.2, 'cluster_poisson_lambda': 4, 'memory_len': 100, 'nudge_var': 1.0},
    4: {'cluster_change_prob': 0.2, 'cluster_poisson_lambda': 4, 'memory_len': 100, 'nudge_var': 1.0},
    5: {'cluster_change_prob': 0.8, 'cluster_poisson_lambda': 9, 'memory_len': 100, 'nudge_var': 0.2},
}


# subset of all the sound samples. Bass track as separate station
COMPOSER_SUBSETS = {
    0: [],
    1: ['vocals', 'bass', 'cloud', 'cloud', 'cloud'],
    2: ['bass', 'long', 'longer'],
    3: ['samples', 'samples', 'samples'],
    4: ['bass', 'samples', 'samples'],
    5: ['samples', 'samples', 'samples', 'samples'],
}


SENSOR_DESTINATION = {
    'addr': '192.168.0.142', 'port': 4040
}


FILTER_DEFAULT_UNIFORMS = {
    'threshold': 0.02,
    'decay': 0.7,
    'seed': 0.5,
    'shade': 1.0,
    'reaction': 0.0,
    'scaleY': 1.0,
    'green': True,
    'highlight': 0,
}

# log_views selecting filter proportions
FILTER_LEVEL_BOUNDS = [0.0, 2.7, 5.0, 7.3, 9.5, 11.8, 100]


#VIDEO_BASE_FP = '/home/lawki/raw'
#VIDEO_BASE_FP = '/mnt/new/raw/'
VIDEO_BASE_FP = '/mnt/usb/usb1/LAWKI_Passages/videos/'
#VIDEO_BASE_FP = '../vids/'
#VIDEO_BASE_FP = "/mnt/usb/LAWKI_ALIVE/videos/"
AUDIO_BASE_FP = 'samples/'

AUDIO_TEMP_DIR = '/tmp/'

SAMPLE_RATE = 44100
