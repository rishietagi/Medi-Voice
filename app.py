import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os

from stable_baselines3 import DDPG
from feat import extract_features_for_file

model = DDPG.load("ddpg_hr_model")
norm_stats = joblib.load("normalization_stats.pkl")
mean = norm_stats["mean"]
std = norm_stats["std"]

st.title("Voice-Based Heart Rate Estimator")
st.write("Upload an audio sample to get heart rate estimate.")
uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    ext = uploaded_file.name.split('.')[-1].lower()
    temp_path = f"temp_audio.{ext}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    features_df = extract_features_for_file(temp_path)
    st.success("Audio processed and features extracted.")

    gender = features_df["gender"].values[0]
    st.markdown(f"**Detected Gender:** `{gender.capitalize()}`")
    age = features_df["age"].values[0]
    st.markdown(f"**Detected Age Group:** `{age.capitalize()}`")

if st.button("Detect Heart Rate"):

    age_val = features_df["age"].values[0]
    age_categories = ['young', 'mature', 'old']
    age_encoded = np.array([[1.0 if age_val == age_cat else 0.0 for age_cat in age_categories]], dtype=np.float32)

    input_feats = features_df.drop(columns=["filename", "gender", "age"]).values.astype(np.float32)

    input_feats = (input_feats - mean) / (std + 1e-8)

    full_input = np.hstack([input_feats, age_encoded])

    # if input_feats.shape[1] == 30:
    #     input_feats = np.hstack([input_feats, np.array([[0.0]])]) 

    if full_input.shape[1] != 32:
        st.error(f"Input feature shape mismatch: got {full_input.shape[1]}, expected 32.")
    else:
        pred_action, _ = model.predict(full_input, deterministic=True)
        pred_hr = 0.5 * (pred_action[0] + 1) * (160 - 50) + 50
        pred_hr = round(float(pred_hr), 2)
        st.success(f"Predicted Heart Rate: **{pred_hr} BPM**")

