import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    spotify_df = pd.read_csv('dataset.csv')
    spotify_df = spotify_df[spotify_df['duration_ms'] != 0.0]
    spotify_df = spotify_df.drop_duplicates(subset='track_name', keep='first')
    return spotify_df

spotify_df = load_data()

# Load scaler
scaler = joblib.load("scaler.save")

# Audio columns
colonnes_audio = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

audio_df = spotify_df[colonnes_audio]
scaled_audio_df = pd.DataFrame(scaler.transform(audio_df), columns=colonnes_audio)
scaled_audio_df['track_name'] = spotify_df['track_name'].values

# Sort and take first 10000
X = scaled_audio_df.sort_values('track_name').head(10000)
X.drop(columns=['track_name'], inplace=True)
short_spotify_df = spotify_df.sort_values('track_name').head(10000)

# Compute similarity
@st.cache_data
def compute_similarity(_X):
    matrice_similarite = cosine_similarity(_X)
    df_similarite = pd.DataFrame(matrice_similarite, index=short_spotify_df['track_name'], columns=short_spotify_df['track_name'])
    return df_similarite

df_similarite = compute_similarity(X)

# App UI
st.title("🎵 Spotify Music Recommendation App")
st.markdown("Get personalized music recommendations based on your favorite songs!")

# Song input
selected_song = st.text_input("Enter the title of a song you've listened to:")

if st.button("Get Recommendations", type="primary") and selected_song:
    if selected_song not in short_spotify_df['track_name'].values:
        st.error("Song not found in our database. Please try another song title.")
    else:
        # Get recommendations
        similaires = df_similarite[selected_song].sort_values(ascending=False)[1:6]

        st.subheader("Top 5 Recommended Songs:")

        # Display recommendations
        for i, song in enumerate(similaires.index, 1):
            # Get song details
            song_info = short_spotify_df[short_spotify_df['track_name'] == song].iloc[0]
            artists = song_info['artists']

            st.markdown(f"""
            **{i}. {song}**  
            *Artists:* {artists}
            ---
            """)

st.markdown("---")
st.markdown("Built with Streamlit and machine learning similarity algorithms.")
