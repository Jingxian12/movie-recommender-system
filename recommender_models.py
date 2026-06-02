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
    movies = pd.read_csv("dataset/tmdb_clean.csv")
    ratings = pd.read_csv("dataset/ratings_clean.csv")
    links = pd.read_csv("dataset/movieLens.csv")
    
    # Build Mapping
    ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId")
    ratings = ratings.merge(movies, on="tmdbId")
    
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
    """Collaborative Filtering recommendation strategy."""
    # Fallback Mechanism
    if user_id not in user_item_matrix.index:
        return movies.sample(5)

    user_ratings = user_item_matrix.loc[user_id]
    watched = user_ratings[user_ratings > 0].index
    scores = {}
    
    for movie in watched:
        if movie in item_topk:
            for neighbor, sim in item_topk[movie]:
                scores[neighbor] = scores.get(neighbor, 0) + sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rec_tmdb = [i[0] for i in ranked[:5]]

    return movies[movies["tmdbId"].isin(rec_tmdb)]


def recommend_content(movie_title, content_topk, movies):
    """Content-Based recommendation strategy using precomputed TF-IDF."""
    # 1. Extract raw tmdbId
    matching_movies = movies[movies["title"] == movie_title]
    if matching_movies.empty:
        return None
        
    raw_id = matching_movies["tmdbId"].values[0]
    
    # 2. Force conversion to integer to fix potential float mismatch keys
    tmdb_id_int = int(float(raw_id))
    
    # Try looking up both int and string variants just in case guest mode used strings
    if tmdb_id_int in content_topk:
        rec_list = content_topk[tmdb_id_int]
    elif str(tmdb_id_int) in content_topk:
        rec_list = content_topk[str(tmdb_id_int)]
    else:
        return None

    # 3. Get recommended movie IDs
    rec_tmdb = [int(float(i[0])) for i in rec_list]
    
    # 4. Filter out recommended data safely
    return movies[movies["tmdbId"].astype(float).astype(int).isin(rec_tmdb)].reset_index(drop=True)


def recommend_semantic(query, embeddings, movies):
    """Semantic Search using SBERT embeddings."""
    # Note: Assuming q_vec is generated from the query text in production,
    # keeping your snippet's original structure here using existing embeddings.
    q_vec = np.array(embeddings)  
    
    scores = cosine_similarity(q_vec[:1], embeddings)[0]
    top_k = np.argsort(scores)[::-1][:10]
    
    return movies.iloc[top_k].reset_index(drop=True)
