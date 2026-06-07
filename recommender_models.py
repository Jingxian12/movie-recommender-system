import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# ========================================================================================================================================================================================================
#                                                                                 DATA & MODEL LOADERS
# ========================================================================================================================================================================================================
# Cache data/models so they don't reload heavy pandas filtering on every button click/refresh.
# How : Save a duplicate copy inside the local cache storage so that streamlit will not re-reading files.

@st.cache_data
def load_data():
    """Loads and prepares the DataFrames."""
    movies = pd.read_csv("dataset/tmdb_clean.csv",encoding="utf-8-sig")
    ratings = pd.read_csv("dataset/ratings_clean.csv",encoding="utf-8-sig")
    
    # User-Item Matrix
    user_item_matrix = ratings.pivot_table(
        index="userId",
        columns="tmdbId",
        values="rating"
    ).fillna(0)
    
    return movies,ratings, user_item_matrix

# Load AI model
@st.cache_resource
def load_models():
    """Loads the precomputed recommendation files."""
    # Item Similarity (CF)
    with gzip.open("models/cf/item_topk.pkl.gz", "rb") as f:
        item_topk = pickle.load(f)

    # TF-IDF Content Top-K
    with gzip.open("models/content/tfidf_topk.pkl.gz", "rb") as f:
        content_topk = pickle.load(f)

    # SBERT Embeddings
    with gzip.open("models/nlp/sbert_embeddings.pkl.gz", "rb") as f:
        embeddings = pickle.load(f)
        
    return item_topk, content_topk, embeddings
    
# ========================================================================================================================================================================================================
#                                                                                RECOMMENDATION ENGINES
# ========================================================================================================================================================================================================

# ============================================================================================================= 
# A) Item-Based (Collaborative Filtering)
# ============================================================================================================= 
def recommend_cf(user_id, user_item_matrix, item_topk, movies):
    """Collaborative Filtering recommendation strategy with correct ranking and rating weighting."""
    
    # 1. Fallback Mechanism (Cold Start)
    if user_id not in user_item_matrix.index:
        # Return 5 random movies and reset index to ensure consistent rendering
        return movies.sample(10).reset_index(drop=True)

    # 2. Get user's profile and identify historical movies
    user_ratings = user_item_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0].index
    scores = {}
    
    # 3. Traverse watched movies to aggregate similarity scores
    for movie in watched:
        if movie in item_topk:
            rating_weight = user_ratings[movie] # 🌟 Fix: Get the user's specific rating score
            for neighbor, sim in item_topk[movie]:
                if neighbor in watched:
                    continue
                # 🌟 Fix: Multiply similarity by user's actual rating for accurate personalization
                scores[neighbor] = scores.get(neighbor, 0) + (sim * rating_weight)

    if not scores:
        return movies.sample(5).reset_index(drop=True)

    # 4. Extract Top-5 ranked items
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rec_tmdb = [i[0] for i in ranked[:10]]

    # 5. 🌟 Fix: Filter the DataFrame while STRICTLY PRESERVING the recommended rank order
    recommendations = movies[movies["tmdbId"].isin(rec_tmdb)].copy()
    recommendations['tmdbId'] = recommendations['tmdbId'].astype(recommendations['tmdbId'].dtype)
    
    # Force pandas to sort rows exactly by the order of rec_tmdb list
    recommendations = recommendations.set_index('tmdbId').loc[rec_tmdb].reset_index()

    return recommendations
    
# =============================================================================================================   
#  B) TF-IDF PART (Content-Based Filtering) 
# ============================================================================================================= 
"""Generates content-based recommendations sorted strictly by similarity score."""
def recommend_content(movie_title, content_topk, movies):
    # 1. Finding the Target Movie's ID
    movie_row = movies[movies['title'] == movie_title] # complete row
    if movie_row.empty:
        return None
    target_id = movie_row.iloc[0]['tmdbId'] #[0] : to prevent movies have same title, so we take first one

    # 2. Extract similar neighbors from the precomputed topk dictionary
    if target_id not in content_topk:
        return None
    top_matches = content_topk[target_id][:10] # Slice to retrieve the Top 10 matches
    rec_tmdb = [item[0] for item in top_matches]
    scores = [round(item[1], 4) for item in top_matches]

    # 3. Core Protection: Prevent the pandas .isin() filter from breaking the ranked order
    # Extract the matching subset of movies from the global dataframe (order break here because pandas scan from top to bottom, return based on original database order)
    recommendations = movies[movies["tmdbId"].isin(rec_tmdb)].copy()
    
    # Create a temporary score mapping dictionary {tmdbId: score}
    score_map = dict(zip(rec_tmdb, scores))
    
    # Map and inject the calculated similarity scores as a new column
    recommendations['score'] = recommendations['tmdbId'].map(score_map)
    
    # Force pandas to explicitly re-sort the rows based on the 'score' column in descending order
    recommendations = recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)

    return recommendations
    
# ============================================================================================================= 
#   C) SBERT (NLP PART) 
# ============================================================================================================= 
# 🌟 Cache the Bi-Encoder inside the model file
@st.cache_resource
def load_bi_encoder():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 🌟 Cache the Cross-Encoder inside the model file
@st.cache_resource
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize instances globally within this module's scope
model = load_bi_encoder()
cross_encoder = load_cross_encoder()

def recommend_semantic(query, embeddings, movies, top_k=10):
    """Hybrid Semantic Search that intercepts short keywords and falls back to AI ranking."""
    clean_query = query.strip().lower()
    if not clean_query:
        return pd.DataFrame()  # return empty dataframe
        
    # Handle NaN safety check for string comparisons
    movies_clean = movies.copy()
    for col in ['director', 'title', 'cast','production_companies']:
        if col in movies_clean.columns:
            movies_clean[col] = movies_clean[col].fillna("")
            
    # STEP 1: KEYWORD INTERCEPTION (For short queries like "nolan")
    keyword_matches = movies_clean[
        (movies_clean['director'].str.lower().str.contains(clean_query)) |
        (movies_clean['title'].str.lower().str.contains(clean_query)) |
        (movies_clean['cast'].str.lower().str.contains(clean_query)) |
        (movies_clean['production_companies'].str.lower().str.contains(clean_query))    
    ]
    
    if len(clean_query.split()) <= 2 and not keyword_matches.empty:
        sort_col = 'popularity' if 'popularity' in keyword_matches.columns else keyword_matches.index.name
        result = keyword_matches.sort_values(by=sort_col, ascending=False).head(top_k).copy()
        result['rerank_score'] = 99.0  
        return result

    # STEP 2: DEEP SEMANTIC AI RE-RANKING (Using local cached models) 
    # BI encoder
    query_vec = model.encode([query])
    scores = cosine_similarity(query_vec, embeddings)[0]
    candidate_indices = np.argsort(scores)[::-1][:30] # get top 30 candidate movies ([::-1]reverse the order)
    
    candidates = movies_clean.iloc[candidate_indices].copy()
    
    if 'tags' not in candidates.columns:
        return candidates.head(top_k) 
        
    pairs = [[query, row['tags']] for _, row in candidates.iterrows()]
    # Cross-Encoder (re-rank the top 30 candidates by feeding the query and movie tags together for deep semantic matching.)
    cross_scores = cross_encoder.predict(pairs)
    candidates['rerank_score'] = cross_scores
    
    final_sorted = candidates.sort_values(by='rerank_score', ascending=False).head(top_k)
    return final_sorted
