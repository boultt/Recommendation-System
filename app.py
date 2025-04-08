import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# --- Spotify API Setup ---
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# --- Load model and scaler ---
with open("kmeans_model.pkl", "rb") as f:
    data = pickle.load(f)
    kmeans = data["model"]
    scaler = data["scaler"]

# --- Load clustered dataset ---
df = pd.read_csv("clustered_df.csv")

# --- Numerical features ---
numerical_features = [
    "valence", "danceability", "energy", "tempo", 
    "acousticness", "liveness", "speechiness", "instrumentalness"
]

def get_spotify_metadata(song_name, artist):
    query = f"track:{song_name} artist:{artist}"
    result = sp.search(q=query, type='track', limit=1)
    if result["tracks"]["items"]:
        track = result["tracks"]["items"][0]
        return {
            "spotify_url": track["external_urls"]["spotify"],
            "album_art": track["album"]["images"][0]["url"],
            "preview_url": track["preview_url"]
        }
    return None

def recommend_songs(song_name, df, num_recommendations=5):
    # Get the cluster of the input song
    song_cluster = df[df["name"] == song_name]["Cluster"].values[0]

    # Filter songs from the same cluster
    same_cluster_songs = df[df["Cluster"] == song_cluster].reset_index(drop=True)

    # Find song within the cluster
    song_row = same_cluster_songs[same_cluster_songs["name"] == song_name]
    if song_row.empty:
        return f"'{song_name}' not found in the cluster."

    song_index = song_row.index[0]

    # Compute similarity
    cluster_features = same_cluster_songs[numerical_features]
    similarity = cosine_similarity(cluster_features, cluster_features)

    # Get top recommendations
    similar_songs = np.argsort(similarity[song_index])[-(num_recommendations + 1):-1][::-1]
    recommendations = same_cluster_songs.iloc[similar_songs][["name", "year", "artists"]]

    return recommendations

# --- Streamlit UI ---
st.set_page_config(page_title="VibeVerse", layout="wide")

st.markdown("<h1 style='font-size: 3rem; color:#1DB954;'>VibeVerse</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:-20px;'>Music Recommender System</h3>", unsafe_allow_html=True)

song_input = st.selectbox("Choose a song:", sorted(df["name"].unique()))

if st.button("Recommend"):
    st.subheader(f"Songs similar to '{song_input}':")
    recs = recommend_songs(song_input, df)

    for _, row in recs.iterrows():
        st.markdown(f"**{row['name']}** ({row['year']}) by {row['artists']}")
        artist = eval(row['artists'])[0] if isinstance(row['artists'], str) else row['artists'][0]
        metadata = get_spotify_metadata(row['name'], artist)
        if metadata:
            st.image(metadata["album_art"], width=100)
            st.markdown(f"[Listen on Spotify]({metadata['spotify_url']})")
            if metadata["preview_url"]:
                st.audio(metadata["preview_url"])
        st.markdown("---")
