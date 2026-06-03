import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# DATA & MODEL LOADERS
# =========================

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
    
    return movies, user_item_matrix


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


# =========================
# RECOMMENDATION ENGINES
# =========================

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

def recommend_content(movie_title, content_topk, movies):
    """Generates content-based recommendations sorted strictly by similarity score."""
    
    # 1. Locate the tmdbId of the currently selected movie title
    movie_row = movies[movies['title'] == movie_title]
    if movie_row.empty:
        return None
        
    target_id = movie_row.iloc[0]['tmdbId']

    # 2. Extract similar neighbors from the precomputed topk dictionary
    if target_id not in content_topk:
        return None
        
    # content_topk[target_id] is an already sorted list: [(movie_id, score), (movie_id, score), ...]
    top_matches = content_topk[target_id][:5] # Slice to retrieve the Top 5 matches
    
    rec_tmdb = [item[0] for item in top_matches]
    scores = [round(item[1], 4) for item in top_matches]

    # 3. Core Protection: Prevent the pandas .isin() filter from breaking the ranked order
    # Extract the matching subset of movies from the global dataframe
    recommendations = movies[movies["tmdbId"].isin(rec_tmdb)].copy()
    
    # Create a temporary score mapping dictionary {tmdbId: score}
    score_map = dict(zip(rec_tmdb, scores))
    
    # Map and inject the calculated similarity scores as a new column
    recommendations['score'] = recommendations['tmdbId'].map(score_map)
    
    # Force pandas to explicitly re-sort the rows based on the 'score' column in descending order
    recommendations = recommendations.sort_values(by='score', ascending=False).reset_index(drop=True)

    return recommendations


def recommend_semantic(query, embeddings, movies):
    """Semantic Search using SBERT embeddings."""
    # Note: Assuming q_vec is generated from the query text in production,
    # keeping your snippet's original structure here using existing embeddings.
    q_vec = np.array(embeddings)  
    
    scores = cosine_similarity(q_vec[:1], embeddings)[0]
    top_k = np.argsort(scores)[::-1][:10]
    
    return movies.iloc[top_k].reset_index(drop=True)
