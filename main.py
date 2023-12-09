import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

df = pd.read_excel('movies.xlsx')

data_selected = ['genres','keywords','cast','director','overview']

for data in data_selected:
  df[data] = df[data].fillna('')

combined_data = df['genres']+' '+df['keywords']+' '+df['cast']+' '+df['director']+' '+df['overview']
print(combined_data)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
dataset = tfidf_vectorizer.fit_transform(combined_data)

k_neighbors = 20
knn_model = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
knn_model.fit(dataset)

def recommend_movie(movie_title, dataset, knn_model, df):
  movie_index = df.index[df['title'] == movie_title].tolist()[0]
  _, neighbor = knn_model.kneighbors(dataset[movie_index])
  recommended_movies = df['title'].iloc[neighbor[0][1:]].tolist()
  return recommended_movies

movie = st.input("Masukkan judul film:")
recomendation = recommend_movie(movie, dataset, knn_model, df)

f"Rekomendasi film dari {movie}"
st.write(recomendation)
