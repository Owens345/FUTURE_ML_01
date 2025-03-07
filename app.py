import streamlit as st
import numpy as np
import joblib

# Load trained model, encoder, and scaler
model = joblib.load("spotify_mood_classifier.pkl")
le = joblib.load("mood_label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("🎵 Spotify Song Mood Classifier")

st.write("Enter the song's features to predict its mood:")

features = ["Danceability", "Energy", "Valence", "Tempo", "Loudness", "Speechiness", "Acousticness"]
user_inputs = []

for feature in features:
    value = st.slider(feature, min_value=0.0, max_value=1.0, step=0.01, key=feature)
    user_inputs.append(value)

if st.button("Predict Mood 🎶"):
    # Scale input
    user_features = np.array(user_inputs).reshape(1, -1)
    user_features = scaler.transform(user_features)

    # Predict mood
    pred = model.predict(user_features)
    mood = le.inverse_transform(pred)

    st.success(f"Predicted Mood: **{mood[0]}** 🎧")
 
