import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the titles dataset
titles = pd.read_csv('titles.csv')

# Preprocess genres and descriptions
titles['genres'] = titles['genres'].apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').replace(' ', ''))
titles['description'] = titles['description'].fillna('')  # Ensure descriptions are not NaN
titles['features'] = titles['genres'].apply(lambda x: ' '.join(x.split(','))) + ' ' + titles['description']

# Create a TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(titles['features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=titles['title'], columns=titles['title'])

# Function to recommend movies
def recommend_movies(movie_name, num_recommendations=5):
    sim_scores = cosine_sim_df[movie_name].sort_values(ascending=False)
    recommended_movies = sim_scores.head(num_recommendations + 1).index.tolist()
    recommended_movies.remove(movie_name)
    return recommended_movies

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Enter a movie name to get recommendations.")

movie_name = st.text_input("Movie Name", "")
num_recommendations = st.slider("Number of Recommendations", 1, 10, 5)

if st.button("Recommend"):
    if movie_name in cosine_sim_df.columns:
        recommendations = recommend_movies(movie_name, num_recommendations)
        st.write(f"Movies similar to '{movie_name}':")
        for movie in recommendations:
            st.write(movie)
    else:
        st.write("Movie not found. Please check the spelling or try a different movie.")


