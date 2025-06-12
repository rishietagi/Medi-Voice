import numpy as np
import pandas as pd
import librosa

def extract_age_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Basic spectral features
    meanfreq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    sd = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
    median = np.median(librosa.feature.spectral_centroid(y=y, sr=sr))
    Q25 = np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 25)
    Q75 = np.percentile(librosa.feature.spectral_centroid(y=y, sr=sr), 75)
    IQR = Q75 - Q25

    centroid = meanfreq 
    mode = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  
    skew = (meanfreq - median) / (sd + 1e-6)  
    kurt = np.mean(((librosa.feature.spectral_centroid(y=y, sr=sr) - meanfreq) ** 4)) / (sd ** 4 + 1e-6)

    sp_ent = -np.sum(librosa.feature.spectral_flatness(y=y) * np.log2(librosa.feature.spectral_flatness(y=y) + 1e-6)) / len(y)
    sfm = np.mean(librosa.feature.spectral_flatness(y=y))

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) == 0:
        meanfun = minfun = maxfun = 0.0
    else:
        meanfun = np.mean(pitches)
        minfun = np.min(pitches)
        maxfun = np.max(pitches)

    stft = np.abs(librosa.stft(y))
    dom_freqs = librosa.fft_frequencies(sr=sr)
    mean_dom = np.mean(dom_freqs)
    min_dom = np.min(dom_freqs)
    max_dom = np.max(dom_freqs)
    df_range = max_dom - min_dom

    modindx = np.std(y) / (np.mean(y) + 1e-6)

    features = {
        'meanfreq': meanfreq,
        'sd': sd,
        'median': median,
        'Q25': Q25,
        'Q75': Q75,
        'IQR': IQR,
        'skew': skew,
        'kurt': kurt,
        'sp.ent': sp_ent,
        'sfm': sfm,
        'mode': mode,
        'centroid': centroid,
        'meanfun': meanfun,
        'minfun': minfun,
        'maxfun': maxfun,
        'meandom': mean_dom,
        'mindom': min_dom,
        'maxdom': max_dom,
        'dfrange': df_range,
        'modindx': modindx
    }

    return pd.DataFrame([features])
