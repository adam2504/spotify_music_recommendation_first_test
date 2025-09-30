import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.process import extractOne

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

# Number of recommendations
num_recs = st.slider("Number of recommendations:", 1, 10, 5)

# Song input
selected_song = st.text_input("Enter the title of a song you've listened to:")

song_list = sorted(short_spotify_df['track_name'].unique())

if st.button("Get Recommendations", type="primary") and selected_song:
    selected_song = selected_song.strip()
    song = None
    if selected_song in short_spotify_df['track_name'].values:
        song = selected_song
    else:
        best_match = extractOne(selected_song, song_list)
        if best_match and best_match[1] > 70:
            song = best_match[0]
            st.info(f"Couldn't find exact match. Using closest match: '{song}' (similarity: {best_match[1]:.1f}%)")
        else:
            st.error("Song not found and no close matches found. Please try another song title.")
    if song:
        similaires = df_similarite[song].sort_values(ascending=False)[1:num_recs+1]
        st.subheader(f"Top {num_recs} Recommended Songs:")
        for i, rec_song in enumerate(similaires.index, 1):
            song_info = short_spotify_df[short_spotify_df['track_name'] == rec_song].iloc[0]
            artists = song_info['artists']
            st.markdown(f"**{i}. {rec_song}**  \n*Artists:* {artists}  \n---")

st.markdown("---")
st.markdown("Built with Streamlit and machine learning similarity algorithms.")
