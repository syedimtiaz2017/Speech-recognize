import streamlit as st
import pickle
import numpy as np
import os
from utils.feature import extract_features

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))

st.set_page_config(page_title="Emotion Detector", layout="centered")

st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV file and detect emotion")

audio_file = st.file_uploader("Upload Audio", type=["wav"])

if audio_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    st.audio(audio_file)

    features = extract_features("temp.wav")

    if features is not None:
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        st.success(f"🎯 Emotion: {prediction}")

        st.subheader("Confidence Scores")
        for label, prob in zip(model.classes_, proba):
            st.write(f"{label}: {prob:.2f}")

    else:
        st.error("Could not process audio")

    os.remove("temp.wav")