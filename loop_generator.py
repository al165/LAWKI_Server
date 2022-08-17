import os
import time
import json

import numpy as np
from scipy.spatial import KDTree

import librosa
import soundfile
from perlin_noise import PerlinNoise

from lawki_constants import *
from logprint import logprint


def main():
    lg = LoopGenerator(points_fp='/tmp/reduced_points.json')

    while True:
        try:
            #lg.update('/tmp/reduced_points.json')
            lg.update()
            time.sleep(30)
        except KeyboardInterrupt:
            break


class LoopGenerator:

    def __init__(self, points_fp:str=None, octaves:int=2, dt:float=0.05) -> None:
        '''
        Class to generate audio loops by exploring embedding space of audio units

        Parameters
        ----------
        points_fp : str
            Location of the JSON file containg point data and file paths. Default None.
        octaves : int, optional
            Number of octaves in Perlin Noise. Default 2.
        dt : float, optional
            Time step when updating the loop walk. Default 0.05.
        '''

        self.points_fp = points_fp
        self.noise = PerlinNoise(octaves=octaves)

        self.points = None
        self.fps = None
        self.kd_tree = None

        self.t = 0.0
        self.dt = dt

        self.audio_cache = dict()

        if points_fp is not None:
            self.load_points(points_fp)


    def update(self, loop_fp:str=None) -> None:
        '''
        Update the loop, generate new audio and save result.

        Parameters
        ----------
        loop_fp : str, optional
            Save path of loop audio. If `None`, defaults to the `/tmp/` directory. Defualt None.
        '''

        # 1. update points
        self.load_points()
        
        # 2. update path
        path = self.make_path(offset=self.t, n=8)
        self.t += self.dt

        # 3. make audio from path
        audio = self.audio_from_path(path, bpm=80, extend=2)

        if audio is None:
            return

        # 4. save audio loop
        if loop_fp is None:
            loop_fp = '/tmp/loop.wav'

        soundfile.write(loop_fp, audio, SAMPLE_RATE)
        logprint(f'saved audio loop at {loop_fp}')


    def make_path(self, n:int=8, offset:float=0.0, scale:float=10) -> np.array:
        '''
        Generates a path (loop).

        Parameters
        ----------
        n : int, optional
            Number of points in loop. Defualt 8.
        offset : float, optional
            Time step of paths evlolution. Default 0.0.
        scale : float, optional
            scale of the Perlin Noise function. Defualt 10.

        Returns
        -------
        path : np.array
            Coordinate of path. Shape (n, 2).
        '''
        ts = np.linspace(0, 2*np.pi, n)
        
        st = np.sin(ts)
        ct = np.cos(ts)
        
        xx = st + offset
        xy = ct + offset
        
        yx = 1.618 * st + 10 + offset
        yy = 1.618 * ct + 10 + offset
        
        X = np.array([self.noise([x/scale, y/scale]) for x, y in zip(xx, xy)])
        Y = np.array([self.noise([x/scale, y/scale]) for x, y in zip(yx, yy)])
        
        # scale to [0, 1]
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())
        
        return np.array([X, Y]).T


    def audio_from_path(self, path:np.array, bpm:int=80, extend:int=2, sf:int=None) -> np.array:
        '''
        Make audio from a given path (loop).

        Parameters
        ----------
        path : np.array
            Path to generate audio from. Must be of shape (n, 2).
        bpm : int, optional
            Beats per minute. Defualt 80.
        extend : int, optional
            [Not Implemented] Determins how long sample should be extended by so that is continurs to
            play while the next sample starts. Defualt 2.
        sf : int, optional
            Sample frequency of output audio. If None, use the defualt specified by `SAMPLE_RATE`.
            Defualt None.


        Returns
        -------
        audio : np.array
            Resulting raw audio data.

        '''

        if not self.kd_tree:
            logprint('kd_tree not initialised (still need to load points?)')
            return None


        if sf is None:
            sf = SAMPLE_RATE

        _, idxs = self.kd_tree.query(path)
        
        grouped = []
        for i in idxs:
            if len(grouped) > 0 and grouped[-1][1] == i:
                grouped[-1][0] += 1
            else:
                grouped.append([1, i])
        
        length = 60 / (bpm / 4)
        grain_length = length / len(idxs)
        grain_samples = int(grain_length * sf)
        
        audio = np.zeros(int(length * sf))
        
        start = 0
        for i in grouped:
            # load audio sample
            if i[1] in self.audio_cache:
                y = self.audio_cache[i[1]]
            else:
                fp = self.fps[i[1]]
                y, _ = librosa.load(fp, sf)
                self.audio_cache[i[1]] = y
            
            # set length (in grains) of sample
            l1 = (i[0]) * grain_samples
            l2 = min(len(y), l1)
            audio[start:start+l2] += y[:l2]
            start += l1
            
        return audio


    def load_points(self, fp:str=None) -> None:
        '''
        JSON file has format:
            {
                'points': [...],
                'fps': [...]
            }

        Parameters
        ----------
        fp : str, None
            path to JSON file containing point data. If `None`, reuse last filepath.
            Default None.
        '''

        if fp is None and self.points_fp is not None:
            fp = self.points_fp
        elif fp is not None:
            self.points_fp = fp
        else:
            return None

        try:
            with open(fp, 'r') as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            logprint(f'could not find {fp}')
            return

        self.points = np.array(data['points'])
        self.fps = data['fps']

        self.kd_tree = KDTree(self.points)



if __name__ == '__main__':
    main()
