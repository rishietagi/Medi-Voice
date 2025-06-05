import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def predict_gender(filepath):
    y, sr = librosa.load(filepath, sr=16000)
    
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    pitch = pitch[pitch > 0]

    if len(pitch) == 0:
        return "unknown"
    
    pitch_mean = np.mean(pitch)
    pitch_min = np.min(pitch)
    pitch_max = np.max(pitch)
    pitch_depth = pitch_max - pitch_min

    energy = np.sum(librosa.feature.rms(y=y))

    if pitch_mean < 160 and pitch_depth < 40:
        return 'male'
    elif pitch_mean >= 160 and energy > 0.1:
        return 'female'
    else:
        return 'female' if pitch_mean >= 150 else 'male'

def extract_features_for_file(filepath):
    y, sr = librosa.load(filepath, sr=16000)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes > np.median(magnitudes)]
    pitch = pitch[pitch > 0]  

    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0
    pitch_min = np.min(pitch) if len(pitch) > 0 else 0
    pitch_max = np.max(pitch) if len(pitch) > 0 else 0
    pitch_depth = pitch_max - pitch_min

    energy = np.sum(librosa.feature.rms(y=y))

    envelope = np.abs(y)
    peaks, _ = find_peaks(envelope, height=0.02, distance=1000)
    duration = len(y) / sr
    speaking_rate = len(peaks) / duration if duration > 0 else 0

    gender = predict_gender(filepath)

    feature_dict = {
        "filename": os.path.basename(filepath),
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_depth": pitch_depth,
        "energy": energy,
        "speaking_rate": speaking_rate,
        "gender": gender,
        **{f"mfcc_{i+1}": mfcc_mean[i] for i in range(len(mfcc_mean))}
    }

    return pd.DataFrame([feature_dict])
