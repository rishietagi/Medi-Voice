# Your full existing imports (unchanged)
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import torch
import torchaudio
import tempfile
import joblib
import os
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier 
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from stable_baselines3 import DDPG
from feat import extract_features_for_file
from feat_diabetes import extract_health_features  

model = DDPG.load("ddpg_hr_model")
norm_stats = joblib.load("normalization_stats.pkl")
mean = norm_stats["mean"]
std = norm_stats["std"]

health_model = tf.keras.models.load_model("health_condition_multilabel_model.h5")
health_scaler = joblib.load("input_scaler.pkl")
health_flags = ['diabetes', 'ht', 'asthma', 'fever', 'smoker', 'cld', 'ihd']

@st.cache_resource
def load_name_predictor():
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    clf = joblib.load("first_letter_predictor.pkl")
    encoder = joblib.load("first_letter_label_encoder.pkl")
    return ecapa, clf, encoder

st.title("Voice-Based Health Analyzer")
st.write("Upload or record your voice to estimate heart rate and predict first name letter.")

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None
if "features_df" not in st.session_state:
    st.session_state.features_df = None
if "recorded_audio_path" not in st.session_state:
    st.session_state.recorded_audio_path = None

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame.to_ndarray())
        return frame

ctx = webrtc_streamer(
    key="live-audio",
    mode=WebRtcMode.SENDONLY,
    media_stream_constraints={"audio": True},
    audio_processor_factory=AudioProcessor,
)

if ctx.audio_processor and len(ctx.audio_processor.frames) > 0:
    if st.button("Stop & Use Recorded Audio"):
        audio = np.concatenate(ctx.audio_processor.frames, axis=0).astype(np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio, 48000)
            st.session_state.audio_path = tmp_file.name
            st.success("Audio recorded and saved!")
            st.audio(tmp_file.name, format="audio/wav")

st.subheader(" Upload an Audio File")
uploaded_file = st.file_uploader("Upload audio (.wav, .mp3, .flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1]
    temp_path = f"temp_audio.{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.audio_path = temp_path
    st.success("File uploaded and saved!")
    st.audio(temp_path, format="audio/wav")

if st.session_state.audio_path and st.session_state.features_df is None:
    try:
        features_df = extract_features_for_file(st.session_state.audio_path)
        st.session_state.features_df = features_df
        st.success("Features extracted from audio.")
    except Exception as e:
        st.error(f"Failed to extract features: {e}")

features_df = st.session_state.features_df
if features_df is not None:
    gender = features_df["gender"].values[0]
    age = features_df["age"].values[0]
    st.markdown(f"**Detected Gender:** `{gender.capitalize()}`")
    st.markdown(f"**Detected Age Group:** `{age.capitalize()}`")

    if st.button("Predict First Letter of Name"):
        ecapa, clf, encoder = load_name_predictor()
        try:
            signal, fs = torchaudio.load(st.session_state.audio_path)
            emb = ecapa.encode_batch(signal).squeeze().detach().cpu().numpy()
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            pred = clf.predict(emb)
            letter = encoder.inverse_transform(pred)[0]
            st.success(f"Predicted First Letter: **{letter}**")
        except Exception as e:
            st.error(f"Failed: {e}")

    if st.button("Predict Heart Rate"):
        age_categories = ['young', 'mature', 'old']
        age_encoded = np.array([[1.0 if age == cat else 0.0 for cat in age_categories]], dtype=np.float32)
        input_feats = features_df.drop(columns=["filename", "gender", "age"]).values.astype(np.float32)
        input_feats = (input_feats - mean) / (std + 1e-8)
        full_input = np.hstack([input_feats, age_encoded])

        if full_input.shape[1] != 32:
            st.error(f"Feature shape mismatch: got {full_input.shape[1]}, expected 32.")
        else:
            pred_action, _ = model.predict(full_input, deterministic=True)
            pred_hr = 0.5 * (pred_action[0] + 1) * (160 - 50) + 50
            st.success(f"Estimated Heart Rate: **{round(float(pred_hr), 2)} BPM**")

    if st.button("Predict Health Conditions"):
        health_input = extract_health_features(st.session_state.audio_path)
        health_input = health_input.reshape(1, -1)
        health_scaled = health_scaler.transform(health_input)
        preds = health_model.predict(health_scaled)
        preds_binary = (preds > 0.5).astype(int)[0]

        if np.sum(preds_binary) == 0:
            st.success("✅ Predicted Health Status: **Healthy**")
        else:
            flags = [flag for flag, val in zip(health_flags, preds_binary) if val == 1]
            st.warning(f"🚨 Detected Potential Conditions: **{', '.join(flags)}**")

st.subheader("Speak to Predict First Letter, Gender, Age & Heart Rate")

def record_voice_input(duration=10, fs=16000):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        wav.write(temp_wav.name, fs, audio)
        audio_path = temp_wav.name
    st.session_state.recorded_audio_path = audio_path
    st.audio(audio_path, format="audio/wav")
    return audio_path

st.markdown("Bhaskar tried jumping quickly on ten big chaat carts while singing Hindi songs near Kochi.")

if st.button(" Record and speak this"):
    try:
        audio_path = record_voice_input()
        ecapa, clf, encoder = load_name_predictor()
        signal, fs = torchaudio.load(audio_path)
        emb = ecapa.encode_batch(signal).squeeze().detach().cpu().numpy()
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        letter_pred = clf.predict(emb)
        first_letter = encoder.inverse_transform(letter_pred)[0]
        st.success(f" Predicted First Letter: **{first_letter}**")

        features_df_live = extract_features_for_file(audio_path)
        gender_live = features_df_live["gender"].values[0]
        age_live = features_df_live["age"].values[0]
        st.markdown(f"** Detected Gender:** `{gender_live.capitalize()}`")
        st.markdown(f"**Detected Age Group:** `{age_live.capitalize()}`")

        age_categories = ['young', 'mature', 'old']
        age_encoded = np.array([[1.0 if age_live == cat else 0.0 for cat in age_categories]], dtype=np.float32)
        input_feats = features_df_live.drop(columns=["filename", "gender", "age"]).values.astype(np.float32)
        input_feats = (input_feats - mean) / (std + 1e-8)
        full_input = np.hstack([input_feats, age_encoded])

        if full_input.shape[1] != 32:
            st.error(f"Feature shape mismatch: got {full_input.shape[1]}, expected 32.")
        else:
            pred_action, _ = model.predict(full_input, deterministic=True)
            pred_hr = 0.5 * (pred_action[0] + 1) * (160 - 50) + 50
            st.success(f" Estimated Heart Rate: **{round(float(pred_hr), 2)} BPM**")

    except Exception as e:
        st.error(f"Prediction from recorded speech failed: {e}")

if st.button("Predict Health Conditions from Spoken Sentence"):
    try:
        if st.session_state.recorded_audio_path:
            features = extract_health_features(st.session_state.recorded_audio_path)

            if features is None:
                st.error("❌ Feature extraction failed. Please try again with a clearer recording.")
            else:
                features = features.reshape(1, -1)
                features_scaled = health_scaler.transform(features)
                preds = health_model.predict(features_scaled)
                preds_binary = (preds > 0.5).astype(int)[0]

                if np.sum(preds_binary) == 0:
                    st.success("✅ Predicted Health Status: **Healthy**")
                else:
                    flags = [flag for flag, val in zip(health_flags, preds_binary) if val == 1]
                    st.warning(f"🚨 Detected Potential Conditions: **{', '.join(flags)}**")
        else:
            st.warning("Please speak the sentence first.")
    except Exception as e:
        st.error(f"Health prediction failed: {e}")


st.subheader("First Name Predictor 2 (Pronunciation Comfort Based)")

# Add pronunciation predictor code here (unchanged)
