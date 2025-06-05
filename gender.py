import torchaudio
import torch
from speechbrain.pretrained import SpeakerRecognition # type: ignore
import numpy as np
import joblib
import os

# Load speaker recognition model from your local folder
verification = SpeakerRecognition.from_hparams(
    source="spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
)

# Path to your trained gender classifier
GENDER_CLASSIFIER_PATH = "gender_classifier.joblib"

# Load gender classifier if exists, else None
if os.path.exists(GENDER_CLASSIFIER_PATH):
    gender_clf = joblib.load(GENDER_CLASSIFIER_PATH)
else:
    gender_clf = None
    print(f"Warning: Gender classifier model not found at {GENDER_CLASSIFIER_PATH}. Please train and save it.")

def extract_embedding(filepath):
    """Extract speaker embedding from audio file."""
    signal, fs = torchaudio.load(filepath)
    with torch.no_grad():
        embedding = verification.encode_batch(signal)
    return embedding.squeeze().cpu().numpy()

def predict_gender(filepath):
    """Predict gender ('male' or 'female') from audio file."""
    if gender_clf is None:
        return "unknown"

    embedding = extract_embedding(filepath).reshape(1, -1)
    gender_pred = gender_clf.predict(embedding)[0]
    return gender_pred
