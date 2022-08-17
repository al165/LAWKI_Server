import os
import sys
import random
import threading
from multiprocessing import Process, Event, Queue
from time import time, sleep
from queue import Empty
from collections import defaultdict

import numpy as np
import pandas as pd

from pythonosc.udp_client import SimpleUDPClient

from rw import Line2D, NeighbourRW, DecoyRW, NoRW
from lawki_constants import *
from logprint import logprint
from eventScheduler import EventScheduler, sendSample


def testFunc(*args, **kwargs):
    print(args, kwargs)


def main():
    clients = dict()
    for station_number, conf in STATION_ADDRESSES.items():
        clients[station_number] = SimpleUDPClient(conf['addr'], conf['port'])
    print(f'Created {len(clients)} clients')

    playback_queue = Queue()

    event_scheduler = EventScheduler(playback_queue, send_function=sendSample,
                                        send_function_kwargs={'clients': clients})

    video_curator = RandomVideoSelector(playback_queue=playback_queue)
    #video_curator = SimpleVideoSelector(playback_queue=playback_queue)
    #video_curator = RandomWalker(playback_queue=playback_queue)
    #video_curator = RandomWalker2(playback_queue=playback_queue)

    #video_curator.daemon = True
    video_curator.start()
    event_scheduler.start()

    mainLoop()

    video_curator.join()
    event_scheduler.join()

    logprint('DONE')





class RandomWalker2(Process):

    def __init__(self, playback_queue=None):
        super(RandomWalker2, self).__init__()

        os.nice(10)

        self.playback_queue = playback_queue

        self.rws = dict()
        self.end_times = dict()

        for sn in range(1, 7):
            self.rws[sn] = Line2D(directory='/home/lawki/development/data/', n_neighbours=200, restart_percentage=0.5, hist_len=5)
            self.end_times[sn] = 0.0

        self.stop = threading.Event()


    def run(self):
        '''Start thread'''
        while not self.stop.is_set():
            self.update()
            sleep(0.5)


    def join(self, timeout=1):
        self.stop.set()
        super(RandomWalker2, self).join(timeout)


    def update(self):
        now = time()
        for sn, rw in self.rws.items():
            if now < self.end_times[sn]:
                continue

            _, row, (start_time, duration) = rw.step(0)
            #logprint(f'step done in {time() - now:.2f}s')

            fn = row.link + '.mp4'
            views = np.log(row.views + 1)

            self.end_times[sn] = now + duration - 0.2
            
            addr = f'/station_1/video/{sn}/'
            msg = [(now + 1.0, (addr, (str(fn), float(views), float(start_time), True)))]
            
            #if self.playback_queue is not None:
            self.playback_queue.put({'type': 'events', 'events': msg})
            
            logprint('sending', msg)
            #logprint(f'update done in {time() - now:.2f}s')
        


class SimpleVideoSelector(Process):

    def __init__(self, playback_queue=None):

        super(SimpleVideoSelector, self).__init__()

        import glob

        self.playback_queue = playback_queue

        logprint(VIDEO_BASE_FP)
        self.video_path = os.path.join(VIDEO_BASE_FP, '*.mp4')

        #clusters_df = pd.read_csv('./cluster_centres_big.csv').set_index('vid_id')
        #self.fps = set()
        #for i in clusters_df.index:
        #    self.fps.add(os.path.join(VIDEO_BASE_FP, i + '.mp4'))
        #self.fps = list(self.fps)

        self.fps = list(glob.glob(self.video_path))

        if len(self.fps) == 0:
            raise ValueError(f'no video files found in {self.video_path}')

        # curr_videos = {station_number: {track: []}}
        self.curr_videos = defaultdict(list)

        self.stop = threading.Event()
        logprint('initiated')
        logprint(f'number of videos: {len(self.fps)}')

        self.rws = [NeighbourRW(directory='/home/lawki/development/data/') for _ in range(6)]
        logprint('made random walkers')
        self.asdf = 0

    def run(self):
        '''Start thread'''
        while not self.stop.is_set():
            self.update()
            sleep(0.1)


    def join(self, timeout=1):
        self.stop.set()
        super(SimpleVideoSelector, self).join(timeout)


    def update(self):
        '''Send out a random video at a random time''' 
        if self.playback_queue is None: # or random.random() < 0.9:
            return

        idx = random.randint(0, len(self.fps)-1)
        fn = os.path.split(self.fps[idx])[-1]
        snippet = random.randint(0, 10)

        #station_number = random.randint(len(STATION_VIDEO_CONFIG) - 1) + 1
        station_number = np.random.choice(list(STATION_VIDEO_CONFIG.keys()))

        num_tiles = len(STATION_VIDEO_CONFIG[station_number]['tile_params'])
        if num_tiles == 1:
            tile_number = 0
        else:
            tile_number = random.randint(0, num_tiles - 1)

        for rw in self.rws:
            _, row, (start_time, duration) = rw.step(0)


        addr = f'/station_{station_number}/video/{tile_number}/'
        msg = [(time() + 1, (addr, (fn, 0, 10, True)))]
        logprint(addr, msg)

        self.playback_queue.put({'type': 'events', 'events': msg})

        sleep(random.randint(0, 2000)/1000 + 1)


class RandomVideoSelector(Process):

    def __init__(self, playback_queue=None):
        super(RandomVideoSelector, self).__init__()

        import glob

        self.playback_queue = playback_queue

        logprint(VIDEO_BASE_FP)
        self.video_path = os.path.join(VIDEO_BASE_FP, '*.mp4')
        self.fps = list(glob.glob(self.video_path))

        if len(self.fps) == 0:
            raise ValueError(f'no video files found in {self.video_path}')

        # curr_videos = {station_number: {track: []}}
        self.curr_videos = defaultdict(list)

        self.stop = threading.Event()
        logprint('initiated')
        logprint(f'number of videos: {len(self.fps)}')
    
    def run(self):
        '''Start thread'''
        while not self.stop.is_set():
            self.update()
            sleep(0.1)

    def join(self, timeout=1):
        self.stop.set()
        super(RandomVideoSelector, self).join(timeout)

    def update(self):
        '''Send out a random video at a random time''' 
        if self.playback_queue is None: # or random.random() < 0.9:
            return

        idx = random.randint(0, len(self.fps)-1)
        fn = os.path.split(self.fps[idx])[-1]
        station_number = 1 #np.random.choice(list(STATION_VIDEO_CONFIG.keys()))

        num_tiles = 6 #len(STATION_VIDEO_CONFIG[station_number]['tile_params'])
        if num_tiles == 1:
            tile_number = 0
        else:
            tile_number = random.randint(0, num_tiles - 1)

        addr = '/video'
        msg = [(time() + 1, (addr, (int(tile_number), fn)))]
        logprint(addr, msg)

        self.playback_queue.put({'type': 'events', 'events': msg})

        sleep(random.randint(0, 2000)/1000 + 1)


def mainLoop():
    while True:
        try:
            sleep(10)
        except KeyboardInterrupt:
            return



if __name__ == '__main__':
    main()

