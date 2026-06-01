import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender System")

# =========================
# LOAD DATA
# =========================
movies = pd.read_csv("data/tmdb_clean.csv")
ratings = pd.read_csv("data/ratings.csv") 
links = pd.read_csv("data/movieLens.csv")  # bridge

# =========================
# MAP MOVIELENS → TMDB
# =========================
ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId")
ratings = ratings.merge(movies, on="tmdbId")

# =========================
# LOAD MODELS
# =========================

# CF item-based top-k
with gzip.open("models/cf/item_topk.pkl.gz", "rb") as f:
    item_topk = pickle.load(f)

# TF-IDF content-based top-k
with gzip.open("models/content/content_topk.pkl.gz", "rb") as f:
    content_topk = pickle.load(f)

# SBERT embeddings
with gzip.open("models/nlp/sbert_embeddings.pkl.gz", "rb") as f:
    embeddings = pickle.load(f)

# SBERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# USER-ITEM MATRIX (CF)
# =========================
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="tmdbId",
    values="rating"
).fillna(0)

# =========================
# SESSION STATE (LOGIN)
# =========================
if "mode" not in st.session_state:
    st.session_state.mode = "home"

# =========================
# HOME
# =========================
if st.session_state.mode == "home":

    st.title("🎬 Hybrid Movie Recommender System")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔐 Login"):
            st.session_state.mode = "login"

    with col2:
        if st.button("🚀 Guest Mode"):
            st.session_state.mode = "guest"

# =========================
# LOGIN
# =========================
elif st.session_state.mode == "login":

    st.subheader("Enter User ID")

    uid = st.text_input("User ID")

    if st.button("Submit"):

        if uid.isdigit() and int(uid) in user_item_matrix.index:
            st.session_state.user_id = int(uid)
            st.session_state.mode = "user"
        else:
            st.error("User ID not found")

# =========================
# CF FUNCTION
# =========================
def recommend_cf(user_id):

    if user_id not in user_item_matrix.index:
        return movies.sample(10)

    user_ratings = user_item_matrix.loc[user_id]

    watched = user_ratings[user_ratings > 0].index

    scores = {}

    for movie in watched:
        if movie in item_topk:

            for neighbor, sim in item_topk[movie]:
                scores[neighbor] = scores.get(neighbor, 0) + sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    rec_tmdb = [i[0] for i in ranked[:10]]

    return movies[movies["tmdbId"].isin(rec_tmdb)]

# =========================
# USER MODE
# =========================
elif st.session_state.mode == "user":

    st.success(f"Welcome User {st.session_state.user_id}")

    tab1, tab2, tab3 = st.tabs([
        "🏠 For You (CF)",
        "🎥 Content-Based",
        "🧠 Semantic Search"
    ])

    # -------------------------
    # CF
    # -------------------------
    with tab1:

        st.header("Personalized Recommendations")

        recs = recommend_cf(st.session_state.user_id)

        st.dataframe(recs[["title", "genres", "vote_average"]])

    # -------------------------
    # TF-IDF
    # -------------------------
    with tab2:

        st.header("Similar Movies (TF-IDF)")

        movie = st.selectbox("Select Movie", movies["title"])

        if st.button("Recommend TF-IDF"):

            idx = movies[movies["title"] == movie].index[0]

            rec_list = content_topk[str(idx)]

            rec_movies = movies.iloc[[i[0] for i in rec_list]]

            st.dataframe(rec_movies[["title", "genres", "overview"]])

    # -------------------------
    # SBERT
    # -------------------------
    with tab3:

        st.header("Semantic Search")

        query = st.text_area("Describe a movie")

        if st.button("Search"):

            q_vec = model.encode([query])

            scores = cosine_similarity(q_vec, embeddings)[0]

            top_k = np.argsort(scores)[::-1][:10]

            st.dataframe(movies.iloc[top_k][["title", "genres", "overview"]])

# =========================
# GUEST MODE
# =========================
elif st.session_state.mode == "guest":

    st.title("🎥 Guest Mode")

    tab1, tab2 = st.tabs([
        "🎥 Content-Based",
        "🧠 Semantic Search"
    ])

    with tab1:

        movie = st.selectbox("Select Movie", movies["title"])

        if st.button("Recommend"):

            idx = movies[movies["title"] == movie].index[0]

            rec_list = content_topk[str(idx)]

            rec_movies = movies.iloc[[i[0] for i in rec_list]]

            st.dataframe(rec_movies[["title", "genres"]])

    with tab2:

        query = st.text_area("Describe a movie")

        if st.button("Search"):

            q_vec = model.encode([query])

            scores = cosine_similarity(q_vec, embeddings)[0]

            top_k = np.argsort(scores)[::-1][:10]

            st.dataframe(movies.iloc[top_k][["title", "genres", "overview"]])
