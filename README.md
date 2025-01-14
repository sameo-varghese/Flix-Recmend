# Flix-Recmend
A content-based movie recommendation system 🎬 that suggests movies similar to a given movie based on genres and descriptions 📚, utilizing TF-IDF vectorization and cosine similarity. This project uses a dataset of 5,000 Netflix films and shows, making it perfect for small-scale applications. 📊
## Features 
- Load and preprocess movie data
- Compute TF-IDF matrix for movie features
- Calculate cosine similarity between movies
- Provide movie recommendations based on similarity scores
- Simple and interactive UI using Streamlit
## Prerequisites 
- Python 3.x
- Required Python libraries:
  - pandas
  - scikit-learn
  - streamlit
## Installation 
Clone the repository: git clone https://github.com/your-username/netflixnook.git cd netflixnook
## Install the required libraries:
pip install pandas scikit-learn streamlit
## Run the Streamlit app:
streamlit run app.py
## Usage
- Enter a movie name in the input field.
- Select the number of recommendations you want.
- Click the "Recommend" button to get a list of similar movies.
