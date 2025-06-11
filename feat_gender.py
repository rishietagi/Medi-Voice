import numpy as np
import librosa
import scipy.stats
import warnings

warnings.filterwarnings("ignore")

def extract_features_from_audio(filepath):
    y, sr = librosa.load(filepath, sr=16000)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    pitches = pitches[pitches > 0]

    if len(pitches) == 0:
        pitches = np.array([0])

    mean_freq = np.mean(pitches)
    sd_freq = np.std(pitches)
    median_freq = np.median(pitches)
    q25_freq = np.percentile(pitches, 25)
    q75_freq = np.percentile(pitches, 75)
    iqr_freq = q75_freq - q25_freq
    skewness = scipy.stats.skew(pitches)
    kurtosis_val = scipy.stats.kurtosis(pitches)

    flatness = np.mean(librosa.feature.spectral_flatness(y=y).flatten())
    centroid_freq = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).flatten())
    peak_freq = np.max(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max))

    tempogram = np.mean(librosa.feature.tempogram(y=y, sr=sr).flatten())

    counts = np.bincount(np.round(pitches).astype(int))
    mode_freq = np.argmax(counts) if len(counts) > 0 else 0

    rms = librosa.feature.rms(y=y).flatten()
    mean_fun = np.mean(rms)
    min_fun = np.min(rms)
    max_fun = np.max(rms)

    dom = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max).flatten()
    mean_dom = np.mean(dom)
    min_dom = np.min(dom)
    max_dom = np.max(dom)

    fund_freq = np.mean(pitches)

    mod_index = sd_freq / mean_freq if mean_freq != 0 else 0

    features = [
        mean_freq, sd_freq, median_freq, q25_freq, q75_freq, iqr_freq,
        skewness, kurtosis_val, flatness, tempogram, mode_freq,
        centroid_freq, peak_freq,
        mean_fun, min_fun, max_fun,
        mean_dom, min_dom, max_dom,
        fund_freq, mod_index
    ]

    return np.array(features).reshape(1, -1)
