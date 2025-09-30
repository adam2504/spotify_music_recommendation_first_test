# 🎵 Spotify Music Recommendation App

A sophisticated music recommendation system built with Python and Streamlit that suggests similar songs based on audio features using machine learning techniques.

## 🌟 Features

- **Intelligent Song Matching**: Enter any song title with fuzzy search that handles typos, case variations, and minor spelling differences
- **Customizable Recommendations**: Choose how many songs you want recommended (1-10)
- **Audio-Based Similarity**: Recommendations powered by cosine similarity on 9 key audio features
- **User-Friendly Interface**: Clean Streamlit UI with real-time feedback
- **Artist Information**: Displays artists for both input and recommended songs

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/spotify_music_recommendation_first_test.git
   cd spotify_music_recommendation_first_test
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501` and start discovering music!

## 📊 How It Works

The recommendation engine uses **cosine similarity** on standardized audio features to find songs with similar musical characteristics:

- **Audio Features Used**: Danceability, Energy, Loudness, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo
- **Algorithm**: Cosine similarity matrix computed on the first 10,000 unique tracks
- **Matching**: Fuzzy string matching (rapidfuzz) for flexible song title input

## 🎯 Usage

1. **Select Number of Recommendations**: Use the slider to choose 1-10 suggestions
2. **Enter Song Title**: Type any song name you've enjoyed
3. **Get Recommendations**: Click the button and discover similar music!

The app intelligently handles:
- Exact title matches
- Close matches with confirmation (for typos)
- Case-insensitive search
- Partial title recognition

## 📊 Dataset

The application uses the [**Spotify Tracks Dataset**](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download) available on Kaggle. This comprehensive dataset contains over 100,000 Spotify tracks with detailed audio features and metadata, providing the foundation for our music similarity analysis.

## 📁 Project Structure

```
spotify_music_recommendation_first_test/
├── app.py                 # Main Streamlit application
├── scaler.save           # Trained StandardScaler model
├── dataset.csv          # Spotify tracks dataset
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── spotify_music_recommendation.ipynb  # Original Jupyter notebook
```

## 🛠 Technical Details

- **Machine Learning**: Scikit-learn for feature scaling and similarity computation
- **Similarity Algorithm**: Cosine similarity for music feature matching
- **Text Matching**: RapidFuzz for fuzzy string comparison
- **Web Framework**: Streamlit for interactive UI
- **Data Processing**: Pandas for data manipulation

## 📋 Requirements

The dependencies are listed in `requirements.txt`:
- streamlit
- pandas
- numpy
- scikit-learn
- rapidfuzz
- joblib
- matplotlib
- seaborn

## 🍀 About

This project demonstrates how machine learning can enhance music discovery by analyzing objective audio characteristics rather than relying solely on collaborative filtering or user preferences.

This was my first project in data science and machine learning – a modest but meaningful step into the fascinating world of AI-powered recommendation systems, transforming raw Spotify data into personalized music suggestions.

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Star the repository!

---
