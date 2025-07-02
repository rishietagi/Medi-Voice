import librosa
import numpy as np

def extract_health_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel, axis=1)[:13]  # Only take first 13

        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0

        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y))

        features = np.concatenate([
            mfcc_mean,         # 13
            mel_mean,          # 13
            [pitch_mean, zcr, spec_centroid, spec_bw, rolloff, rms]  # 6
        ])  # Total = 32

        return features

    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return None

