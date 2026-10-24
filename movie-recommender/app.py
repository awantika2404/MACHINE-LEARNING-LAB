
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Small sample dataset
movies = pd.DataFrame({
    "title": [
        "Toy Story",
        "Finding Nemo",
        "The Incredibles",
        "Shrek",
        "Jumanji",
        "The Lion King",
        "Frozen",
        "Moana",
        "Spider-Man",
        "Batman"
    ],
    "genres": [
        "animation family comedy",
        "animation family adventure",
        "animation action family",
        "animation family comedy fantasy",
        "adventure family fantasy",
        "animation family drama adventure",
        "animation family fantasy musical",
        "animation family adventure musical",
        "action adventure superhero",
        "action crime superhero"
    ]
})

# Convert genres into numerical features
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genres"])

# Calculate movie-to-movie similarity
similarity = cosine_similarity(genre_matrix)


def recommend_movies(movie_title, number=5):
    movie_index = movies[
        movies["title"] == movie_title
    ].index[0]

    scores = list(enumerate(similarity[movie_index]))

    # Sort by similarity, highest first
    scores = sorted(
        scores,
        key=lambda item: item[1],
        reverse=True
    )

    # Exclude the selected movie itself
    scores = [
        item for item in scores
        if item[0] != movie_index
    ]

    recommendations = []

    for index, score in scores[:number]:
        recommendations.append({
            "Movie": movies.iloc[index]["title"],
            "Genres": movies.iloc[index]["genres"],
            "Similarity": round(float(score), 2)
        })

    return pd.DataFrame(recommendations)


# Streamlit interface
st.title("🎬 Movie Recommendation System")
st.write("Select a movie to discover similar movies.")

selected_movie = st.selectbox(
    "Choose a movie",
    movies["title"].tolist()
)

if st.button("Recommend Movies"):
    results = recommend_movies(selected_movie)

    st.subheader("Recommended for you")
    st.dataframe(results, hide_index=True)

st.caption("Powered by content-based filtering and cosine similarity.")
