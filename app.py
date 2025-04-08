import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model and scaler
with open("kmeans_model.pkl", "rb") as f:
    data = pickle.load(f)
    kmeans = data["model"]
    scaler = data["scaler"]

# Load clustered dataset
df = pd.read_csv("clustered_df.csv")

# Select numerical features used for clustering
numerical_features = [
    "valence", "danceability", "energy", "tempo", 
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def recommend_songs(song_name, num_recommendations=5):
    if song_name not in df["name"].values:
        return pd.DataFrame([["Song not found", "", ""]], columns=["name", "year", "artists"])
    
    song_cluster = df[df["name"] == song_name]["Cluster"].values[0]
    same_cluster_songs = df[df["Cluster"] == song_cluster]
    
    song_index = same_cluster_songs[same_cluster_songs["name"] == song_name].index[0]
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)
    
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]
    
    return recommendations

# Streamlit UI
st.title("🎵 Music Recommendation System")
song_input = st.selectbox("Choose a song:", sorted(df["name"].unique()))

if st.button("Recommend"):
    st.subheader(f"Songs similar to '{song_input}':")
    recs = recommend_songs(song_input)
    st.dataframe(recs)
