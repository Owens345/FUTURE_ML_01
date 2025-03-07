import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("spotify_songs_with_mood.csv")

# Plot mood distribution
plt.figure(figsize=(10, 6))
sns.countplot(x="mood", data=df, palette="viridis")
plt.title("Mood Distribution in Spotify Songs")
plt.xlabel("Mood")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
 
