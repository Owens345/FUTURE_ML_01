import pandas as pd
import numpy as np
import librosa
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("spotify_songs.csv")

# Check available columns
print("Columns in dataset:", df.columns)

# Check if mood column exists, else classify using valence & energy
if 'mood' not in df.columns:
    print("⚠️ 'mood' column missing. Generating moods...")

    def classify_mood(row):
        if row["valence"] > 0.6:
            return "Happy"
        elif row["valence"] < 0.4:
            return "Sad"
        elif row["energy"] > 0.7:
            return "Energetic"
        elif row["energy"] < 0.4 and row["acousticness"] > 0.6:
            return "Calm"
        else:
            return "Neutral"

    df["mood"] = df.apply(classify_mood, axis=1)
    df.to_csv("spotify_songs_with_mood.csv", index=False)
    print("✅ Mood column added!")

# Select features
features = ["danceability", "energy", "valence", "tempo", "loudness", "speechiness", "acousticness"]
target_column = "mood"

# Encode mood labels
le = LabelEncoder()
df["mood_encoded"] = le.fit_transform(df[target_column])

# Save encoder
joblib.dump(le, "mood_label_encoder.pkl")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
y = df["mood_encoded"]

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "spotify_mood_classifier.pkl")

print("🎵✅ Mood classification model trained & saved!")
 
