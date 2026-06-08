# movie-recommender-system
Hybrid movie recommender system using CF, content-based filtering, and NLP semantic search.

---
**Notice:** 
This application is a Final Year Project (FYP) Prototype. Public user registration is currently closed. External evaluators may use 'Guest Mode' or enter a pre-assigned Test User ID to evaluate the recommendation engine.

## 1. Overview
With thousands of movies released across multiple streaming platforms every year, users often struggle to find content that matches their interests. Traditional searching methods require users to browse through large catalogs manually, which can be time-consuming and overwhelming.

This project presents a **Hybrid Movie Recommender System** that combines multiple recommendation techniques to deliver personalized and relevant movie suggestions efficiently.

The system utilizes a **Switch Hybrid Recommendation Strategy**, dynamically selecting the most suitable recommendation method based on the user's input and available information.

### Features
- Personalized recommendations
- Similar movie recommendations
- Natural language movie search
- Movie poster display
- Cold-start user support
- Interactive web interface

## 2. Problem Statement

The rapid growth of digital entertainment platforms has created a movie discovery problem:

- Thousands of movies are available online.
- Users have limited time to search manually.
- Traditional keyword searches may fail to capture user preferences.
- New users may have insufficient rating history for collaborative filtering.

This project aims to help users discover movies more efficiently through intelligent recommendation techniques.

## 3. Objectives
- Develop a personalized movie recommendation system.
- Improve movie discovery efficiency.
- Reduce the time users spend searching for movies.
- Handle both existing users and new users effectively

## 4. System Architecture

**Switch Hybrid Strategy** : Instead of combining recommendation scores directly, this project uses a Switch Hybrid Recommender which user can select one of the recommendation engine at a time.

The application consists of three recommendation modules and one filtering feature :

### **A) Recommendation Modules**

#### i. Collaborative Filtering (CF)

Recommends movies based on user behavior and ratings.
Implemented approaches:
- Item-Based Collaborative Filtering

The system identifies movies with similar rating patterns using the MovieLens dataset and recommends movies that are similar to those previously liked by the user.

Suitable for:
- Users with sufficient rating history.
- Personalized recommendations based on community preferences.

#### ii. Content-Based Filtering (CBF)

Recommends movies similar to a selected movie based on movie attributes.
Features used include:
- Genres
- Movie Overview
- Keywords
- Production Companies
- Cast
- Directors

Movie metadata is collected from TMDB and transformed into feature vectors for similarity computation.

Suitable for:
- Users who already know a movie they like.
- Similar movie discovery.

#### iii. NLP Semantic Search

Allows users to search using natural language descriptions.

Examples:
- "I want a funny space adventure."
- "Recommend emotional movies about friendship."
- "Movies similar to Harry Potter but darker."

The system converts movie descriptions into semantic embeddings and retrieves the most relevant movies using vector similarity search.

Suitable for:
- Users who cannot remember movie titles.
- Intent-based movie discovery.

###  **B) Filtering Feature**

These do not generate recommendations. They help users narrow down results. Including : 
- Genre filter
- Actor/Actress filter
- Director filter


## Dataset

We have two datasets using in this project 

### i) MovieLens Dataset (dataset/rating_clean.csv)
Used for:
- User ratings
- User IDs
- Collaborative Filtering

Dataset Source:
- Kaggle(https://www.kaggle.com/datasets/abhikjha/movielens-100k)

### ii) TMDB Dataset (dataset/tmdb_clean.csv)
Used for:
- Content-Based Filtering
- NLP Semantic Search

Dataset Source:
- TMDB API key
- Kaggle (https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## 5. Technologies Used

**The application is built using:**
- Python 3.14
- NLP
  - TFIDF
  - SBERT(Sentence Transformers)
- Consine Similarity
- Streamlit

## 6. Evaluation Metrics

The recommendation models are evaluated using ranking-based metrics:

**Precision@K**
- Measures how many recommended movies are relevant.

**Recall@K**
- Measures how many relevant movies are successfully recommended.

**Hit Rate@K**
- Measures whether at least one relevant movie appears in recommendations.

**NDCG@K**
- Measures ranking quality by rewarding relevant movies appearing higher in the recommendation list.

## 7. Notebook
The project includes Jupyter notebooks used for data preprocessing, exploratory analysis, and model development. These notebooks document the end-to-end machine learning pipeline from raw data to final recommendation models. Can refer inside this repository :
- notebook/Movie_Recommender_System_(PART_A)_.ipynb
- notebook/Movie_Recommender_System_(PART_B)_.ipynb

## 8. Recommendation Workflow

## 9. How to use the app

## 10. Output
