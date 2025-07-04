import os
import librosa
import numpy as np
import pandas as pd
import joblib
from scipy.signal import find_peaks
from feat_gender import extract_features_from_audio
from feat_age import extract_age_features

gender_model = joblib.load("gender_classifier.pkl")
gender_scaler = joblib.load("gender_scaler.pkl")
age_model = joblib.load("age_classifier.pkl")

def predict_age(filepath):
    try:
        age_features = extract_age_features(filepath)

        age = age_model.predict(age_features)[0]

        return age
    except Exception as e:
        print(f"Error predicting age for {filepath}: {e}")
        return "unknown"

def predict_gender(filepath):
    try:
        gender_features = extract_features_from_audio(filepath)  

        features_scaled = gender_scaler.transform(gender_features)

        gender = gender_model.predict(features_scaled)[0]

        return gender
    except Exception as e:
        print(f"Error predicting gender for {filepath}: {e}")
        return "unknown"

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

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_mean = np.mean(mel_spec_db, axis=1)[:10]  

    envelope = np.abs(y)
    peaks, _ = find_peaks(envelope, height=0.02, distance=1000)
    duration = len(y) / sr
    speaking_rate = len(peaks) / duration if duration > 0 else 0

    gender = predict_gender(filepath)
    age = predict_age(filepath)


    mfcc_dict = {f"mfcc_{i+1}": mfcc_mean[i] for i in range(len(mfcc_mean))}
    mel_dict = {f"mel_{i+1}": mel_mean[i] for i in range(len(mel_mean))}

    feature_dict = {
        "filename": os.path.basename(filepath),
        "pitch_mean": pitch_mean,
        "pitch_min": pitch_min,
        "pitch_max": pitch_max,
        "pitch_depth": pitch_depth,
        "energy": energy,
        "speaking_rate": speaking_rate,
        "gender": gender,
        "age": age,
        **mfcc_dict,
        **mel_dict
    }

    return pd.DataFrame([feature_dict])
