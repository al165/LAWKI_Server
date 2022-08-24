import os
import warnings

import numpy as np

import librosa


def feature_extractor(signal, sf):

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        mfccs = librosa.feature.mfcc(y=signal, sr=sf)
        mfccs -= mfccs.mean()
        mfccs /= mfccs.std()
    
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # mean first order difference
        mfccs_mfod = np.mean(np.diff(mfccs, axis=1), axis=1)  

        chromo = librosa.feature.chroma_stft(y=signal, sr=sf)
        chromo_mean = np.mean(chromo, axis=1)
        chromo_std = np.std(chromo, axis=1)

    onset = librosa.onset.onset_strength(y=signal, sr=sf)
    onenv = np.zeros(8)
    hop = max(1, int(len(onset) / 8))
    for i in range(8):
        sliced = onset[hop*i:hop*(i+1)]
        if len(sliced) == 0:
            x = 0
        else:
            x = np.nanmean(sliced)
        if np.isfinite(x):
            onenv[i] = x
    
    zcr = librosa.feature.zero_crossing_rate(signal)
    zcr_mean = np.array([zcr.mean()])
    zcr_std = np.array([zcr.std()])
    zcr_mfod = np.array([np.mean(np.diff(zcr))])
    
    features = np.concatenate([mfccs_mean, mfccs_std, mfccs_mfod, chromo_mean, chromo_std, zcr_mean, zcr_std, zcr_mfod, onenv],)
    
    length = len(signal) / sf
    
    data = {
        'features': features,
        'length': length
    }
    
    return data


def feature_extractor_from_file(fp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signal, sf = librosa.load(fp, None)
    
    data = feature_extractor(signal, sr=sf)
    data['fp'] = fp
    data['fn'] = os.path.split(fp)[-1]
    return data
