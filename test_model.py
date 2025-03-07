import numpy as np
import joblib

# Load trained model, encoder, and scaler
model = joblib.load("spotify_mood_classifier.pkl")
le = joblib.load("mood_label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Feature names
features = ["danceability", "energy", "valence", "tempo", "loudness", "speechiness", "acousticness"]

print("🎵 Enter song features to predict mood:")
user_features = [float(input(f"{f}: ")) for f in features]

# Scale input
user_features = np.array(user_features).reshape(1, -1)
user_features = scaler.transform(user_features)

# Predict mood
pred = model.predict(user_features)
mood = le.inverse_transform(pred)

print(f"\n🎶 Predicted Mood: {mood[0]}")
 
