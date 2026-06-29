"""
Movie Recommendation Engine

This project recommends similar movies based on
their overview using a content-based recommendation system.

Algorithm : Content-Based Recommendation
Technique : TF-IDF + Cosine Similarity
Domain : Artificial Intelligence
"""
# Import required libraries

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# Load the movie dataset

print("Loading movie dataset...\n")

movies = pd.read_csv("../data/movies.csv")

print("Dataset loaded successfully.\n")

# Display dataset information

print("First five movies:\n")

print(movies.head())

print("\nTotal Movies :", len(movies))

print("\nDataset Information:\n")

movies.info()

print("\nMissing Values:\n")

print(movies.isnull().sum())

# Keep only the required columns

movies = movies[["title", "overview"]]

# Remove rows with missing overview

movies = movies.dropna()

print("\nDataset after cleaning:")

print(movies.head())

# Convert movie overviews into numerical vectors

print("\nCreating movie vectors...")

vectorizer = TfidfVectorizer(stop_words="english")

movie_vectors = vectorizer.fit_transform(movies["overview"])

print("Movie vectors created successfully.")

# Calculate similarity between movies

print("\nCalculating similarity matrix...")

similarity = cosine_similarity(movie_vectors)

print("Similarity matrix created successfully.")

# Recommend similar movies

def recommend_movies(movie_name):

    movie_name = movie_name.lower()

    movie_titles = movies["title"].str.lower()

    if movie_name not in movie_titles.values:
        print("\nMovie not found in dataset.")
        return

    index = movie_titles[movie_titles == movie_name].index[0]

    similarity_scores = list(enumerate(similarity[index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 10 Recommended Movies\n")

    count = 0

    recommendations = []

    for movie in similarity_scores[1:11]:

        movie_index = movie[0]

        similarity_score = round(movie[1] * 100, 2)

        movie_title = movies.iloc[movie_index]["title"]

        print(f"{count + 1}. {movie_title} (Similarity: {similarity_score}%)")

        recommendations.append({
            "Searched Movie": movies.iloc[index]["title"],
            "Recommended Movie": movie_title,
            "Similarity (%)": similarity_score
        })

        count += 1

    pd.DataFrame(recommendations).to_csv(
        "../outputs/recommendations.csv",
        index=False
    )

    print("\nRecommendations saved successfully.")
    
# Ask the user for a movie name

print("\nMovie Recommendation System")

movie = input("\nEnter a movie name: ")

recommend_movies(movie)

print("\nProgram completed successfully.")
