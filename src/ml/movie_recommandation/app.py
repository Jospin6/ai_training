import pickle
import streamlit as st
import requests
import sys
from pathlib import Path


st.header("Movies Recommendation System Using Machine Learning")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_DIR = PROJECT_ROOT / "notebooks" / "artificats"

movies = pickle.load(open(ARTIFACTS_DIR/"movie_list.pkl", 'rb'))
similarity = pickle.load(open(ARTIFACTS_DIR/"similarity.pkl", 'rb'))


movie_list = movies['title'].values
selected_movie = st.selectbox(
    'Type or select a movie to get recommendation',
    movie_list
)

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=<<api_key>>&language=en-US"

    data = requests.get(url)

    data = data.json()

    poster_path = data['poster_path']
    full_path = "http://image.tmdb.org/t/p/w500/" + poster_path

    return full_path

def recommend(movie: str):
    index = movies[movies['title'] == movie].index[0]
    
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key= lambda x: x[1])

    recommended_movies_name = []
    recommended_movies_poster = []

    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies_poster.append(fetch_poster(movie_id))
        recommended_movies_name.append(movies.iloc[i[0]].title)

    return (recommended_movies_name, recommended_movies_poster)

if st.button("Show recommendation"):
    recommended_movies_name, recommended_movies_poster = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(recommended_movies_name[0])
        st.image(recommended_movies_poster[0])

    with col2:
            st.text(recommended_movies_name[1])
            st.image(recommended_movies_poster[1])

    with col3:
            st.text(recommended_movies_name[2])
            st.image(recommended_movies_poster[2])

    with col4:
            st.text(recommended_movies_name[3])
            st.image(recommended_movies_poster[3])

    with col5:
            st.text(recommended_movies_name[4])
            st.image(recommended_movies_poster[4])






