import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os

from stable_baselines3 import DDPG
from feat import extract_features_for_file

# Load trained model and normalization stats
model = DDPG.load("ddpg_hr_model")
norm_stats = joblib.load("normalization_stats.pkl")
mean = norm_stats["mean"]
std = norm_stats["std"]

st.title("Voice-Based Heart Rate Estimator")
st.write("Upload an audio sample to get heart rate estimate.")
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Extract original extension
    ext = uploaded_file.name.split('.')[-1].lower()
    temp_path = f"temp_audio.{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    features_df = extract_features_for_file(temp_path)
    st.success("Audio processed and features extracted.")

    gender = features_df["gender"].values[0]
    st.markdown(f"**Detected Gender:** `{gender.capitalize()}`")

    if st.button("Detect Heart Rate"):
        input_feats = features_df.drop(columns=["filename", "gender"]).values.astype(np.float32)

        input_feats = (input_feats - mean) / (std + 1e-8)

        if input_feats.shape[1] == 19:
            input_feats = np.hstack([input_feats, np.array([[0.0]])])  # Add dummy HR

        pred_action, _ = model.predict(input_feats, deterministic=True)

        # Scale action to HR range (50 to 160 BPM)
        pred_hr = 0.5 * (pred_action[0] + 1) * (160 - 50) + 50
        pred_hr = round(float(pred_hr), 2)

        st.success(f"Predicted Heart Rate: **{pred_hr} BPM**")
