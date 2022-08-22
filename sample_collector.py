import os
import time
import json
import hashlib
import pickle as pkl
from threading import Thread

import numpy as np

import librosa
import soundfile

import torch

from lawki_constants import *
from logprint import logprint
from music_generator.models import LinearAE
import music_generator.feature_extractor as fe


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('computing on', DEVICE)


AUDIO_SAMPLE_FOLDER = '/tmp/samples'
EMBEDDING_DATA = '/tmp/reduced_points.json'
VIDEO_TITLE_FILE = '/tmp/curr_video_title.txt'

MAX_POINTS = 200


def main():
    sc = SampleCollector()

    while True:
        try:
            sc.update(max_len=20.0)

            time.sleep(60)
        except KeyboardInterrupt:
            break

        #x = input('make new unit?')

        #if x == 'q':
        #    break



def load_audio(fp: str) -> np.array:
    y, _ = librosa.load(fp, sr=SAMPLE_RATE)
    return y


class SampleCollector(Thread):

    def __init__(self, min_length:float=0.5):
        super(SampleCollector, self).__init__()

        self.min_length_samples = int(min_length * SAMPLE_RATE)

        # make sure the sample folder exists
        if not os.path.exists(AUDIO_SAMPLE_FOLDER):
            logprint(f'making {AUDIO_SAMPLE_FOLDER}')
            os.mkdir(AUDIO_SAMPLE_FOLDER)

        # load embedding model
        logprint('loading embedding_model...', end= ' ', flush=True)
        embedding_parameters = torch.load('/home/lawki/LAWKI_Server/music_generator/models/AE_embedding_model.pt', map_location=torch.device('cpu'))
        self.embedding_model = LinearAE(**embedding_parameters['model_params']).to(DEVICE)
        self.embedding_model.load_state_dict(embedding_parameters['model_state_dict'])
        self.embedding_model.eval()
        print('done')

        logprint('loading reducer model...', end=' ', flush=True)
        # load reducer model
        with open('/home/lawki/development/music_generator/models/reducer.pkl', 'rb') as f:
            self.reducer = pkl.load(f)
        print('done')
        logprint('loading scaler', end=' ', flush=True)
        with open('/home/lawki/development/music_generator/models/reducer_scaler.pkl', 'rb') as f:
            self.reducer_scaler = pkl.load(f)
        print('done')


    #def run(self):
    #    pass;


    #def join(self, timeout=1):
    #    super(SampleCollector, self).join(timeout)


    def update(self, audio_fp:str=None, max_len:float=None):
        start_time = time.time()

        # 1. load audio stream
        if audio_fp:
            y = load_audio(fp)
        else:
            fallback_audio_path = os.path.join(AUDIO_TEMP_DIR, 'audio.wav')
            try:
                y = load_audio(fallback_audio_path)
            except FileNotFoundError:
                logprint(f'no audio found (`audio_fp` is None and {fallback_audio_path} does not exist.)')
                return

        if max_len:
            y = y[:int(max_len*SAMPLE_RATE)]


        # 2. splice into units
        units = self.split_audio_into_units(y)
        logprint(f'made {len(units)} units')
        if len(units) == 0:
            logprint('did not make any new units')
            return


        # 3. embed and reduce units
        unit_features = [fe.feature_extractor(u, SAMPLE_RATE)['features'] for u in units]        
        unit_embedding = self.embedding_model.encode(
            torch.tensor(unit_features, device=DEVICE, dtype=torch.float)
        )
        unit_reduced = self.reducer.transform(unit_embedding.cpu().detach().numpy())
        unit_reduced = self.reducer_scaler.transform(unit_reduced)
        logprint(unit_reduced.shape)


        # TODO: 4. select subset of units (uniformly)
        subset = list(range(len(units)))
        logprint(f'subset length {len(subset)}')


        # 5. add to sample pool and update list
        if audio_fp:
            fn = hashlib.md5(audio_fp.encode('utf-8')).hexdigest()[:5]
        elif os.path.exists(VIDEO_TITLE_FILE):
            with open(VIDEO_TITLE_FILE, 'r') as f:
                fn = hashlib.md5(f.readlines()[0].encode('utf-8')).hexdigest()[:5]
        else:
            fn = hashlib.md5(f'{time.time()}'.encode('utf-8')).hexdigest()[:5]

        logprint(f'saving samples as {fn}_XX.wav')

        try:
            with open(EMBEDDING_DATA, 'r') as f:
                data = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            logprint(f'embedding file {EMBEDDING_DATA} cannot be read')
            data = {'points': [], 'fps': []}
        except FileNotFoundError:
            logprint(f'embedding file {EMBEDDING_DATA} not found, starting new')
            data = {'points': [], 'fps': []}


        for i in subset:
            fp = os.path.join(AUDIO_SAMPLE_FOLDER, f'{fn}_{i:02}.wav')

            if fp in data['fps']:
                continue

            soundfile.write(fp, units[i], SAMPLE_RATE)

            data['points'].append(unit_reduced[i].tolist())
            data['fps'].append(fp)



        # 6. remove old points
        if len(data['points']) > MAX_POINTS:
            logprint(f'removing {MAX_POINTS - len(data["points"])} points')
            points_to_remove = data['fps'][:-MAX_POINTS]
            data['points'] = data['points'][-MAX_POINTS:]
            data['fps'] = data['fps'][-MAX_POINTS:]

            assert len(data['points']) <= MAX_POINTS

            for p in points_to_remove:
                print('DELETING', p)
                os.remove(p)

        with open(EMBEDDING_DATA, 'w') as f:
            f.write(json.dumps(data))

        logprint(f'wrote data file (length {len(data["points"])})')

        print()
        logprint(f'completed in {time.time() - start_time:.2f}s')


    def split_audio_into_units(self, y:np.array, maximum:int=50) -> list:

        units = []
        unit_starts = librosa.onset.onset_detect(y=y, sr=SAMPLE_RATE, backtrack=True, units='samples')
        
        for i in range(min(len(unit_starts) - 1, maximum)):
            us = unit_starts[i]
            ue = us + max(self.min_length_samples, unit_starts[i+1] - us)
            unit = y[us:ue]

            if len(unit) < self.min_length_samples:
                continue

            units.append(unit)

        return units


if __name__ == '__main__':
    main()
