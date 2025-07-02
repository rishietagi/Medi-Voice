import librosa
import soundfile as sf
from collections import defaultdict
import numpy as np

def split_audio_by_words(audio_path, words, durations, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    word_segments = {}
    start = 0
    for word, dur in zip(words, durations):
        end = start + int(dur * sr)
        word_audio = y[start:end]
        word_segments[word] = word_audio
        start = end
    return word_segments

def get_pronunciation_score(audio_segment, sr=16000):
    rms = librosa.feature.rms(y=audio_segment).mean()
    flatness = librosa.feature.spectral_flatness(y=audio_segment).mean()
    mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean()
    
    score = rms * 10 - flatness * 5 + mfcc_mean / 5
    return round(score, 3)



word_to_letters = {
    "Bhaskar": ["B"],
    "tried": ["T", "R"],
    "jumping": ["J"],
    "quickly": ["Q", "K"],
    "on": ["O"],
    "ten": ["T"],
    "big": ["B"],
    "chaat": ["C"],
    "carts": ["K"],
    "while": ["W"],
    "singing": ["S"],
    "Hindi": ["H"],
    "songs": ["S"],
    "near": ["N"],
    "Kochi": ["K"]
}





def score_letters_from_words(word_scores, word_to_letters):
    letter_scores = defaultdict(list)
    
    for word, score in word_scores.items():
        letters = word_to_letters.get(word, [])
        for letter in letters:
            letter_scores[letter].append(score)

    averaged_scores = {letter: np.mean(scores) for letter, scores in letter_scores.items()}
    return averaged_scores



def predict_first_letter_from_pronunciation(word_scores):
    letter_scores = score_letters_from_words(word_scores, word_to_letters)
    predicted_letter = max(letter_scores, key=letter_scores.get)
    return predicted_letter, letter_scores
