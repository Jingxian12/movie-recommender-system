import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip

from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommender System")
st.caption("Welcome to MovieMatch! Log in with your User ID to unlock personalized recommendations based on your unique rating history, or jump straight into Guest Mode to explore movies by matching your favorite vibes and descriptions")
st.divider()

# =========================
# LOAD DATA
# =========================
movies = pd.read_csv("dataset/tmdb_clean.csv")
ratings = pd.read_csv("dataset/ratings_clean.csv")
links = pd.read_csv("dataset/movieLens.csv")

# =========================
# BUILD MAPPING (CRITICAL FIX)
# =========================
ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId")
ratings = ratings.merge(movies, on="tmdbId")

# =========================
# USER-ITEM MATRIX
# =========================
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="tmdbId",
    values="rating"
).fillna(0)

# =========================
# LOAD MODELS
# =========================
# CF
with gzip.open("models/cf/item_topk.pkl.gz", "rb") as f:
    item_topk = pickle.load(f)

# TF-IDF (KEY FIX: should be tmdbId-based)
with gzip.open("models/content/tfidf_topk.pkl.gz", "rb") as f:
    content_topk = pickle.load(f)

# SBERT embeddings
with gzip.open("models/nlp/sbert_embeddings.pkl.gz", "rb") as f:
    embeddings = pickle.load(f)

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
# MOVIE DETAILS UI
# =========================
@st.dialog("🎬 Movie Details",width="large")
def show_movie(movie):

    col1, col2 = st.columns([1,2])

    with col1:
        st.image(movie["poster_url"], use_container_width=True)

    with col2:
        st.subheader(movie["title"])
        st.write("**Genres:**", movie["genres"])
        st.write("**Overview:**", movie["overview"])
        st.write("**Cast:**", movie["cast"])
        st.write("**Director:**", movie["director"])
        st.write("**Rating:**", movie["vote_average"])
        st.write("**Runtime:**", movie["runtime"])
        st.write("**Release Date:**", movie["release_date"])

# =========================
# LOGOUT FUNCTION
# =========================
def logout():
    st.session_state.mode = "home"

    if "user_id" in st.session_state:
        del st.session_state.user_id


# =========================
# SESSION STATE
# =========================
if "mode" not in st.session_state:
    st.session_state.mode = "home"

# =========================
# HOME
# =========================
if st.session_state.mode == "home":

    st.subheader("Choose Mode")

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
            st.error("User not found")

# =========================
# USER MODE
# =========================
elif st.session_state.mode == "user":

    col1, col2 = st.columns([8, 1])

    with col2:
        if st.button("🚪 Sign Out"):
            logout()
            st.rerun()
    st.success(f"Welcome User {st.session_state.user_id}")

    tab1, tab2, tab3 = st.tabs([
        "🏠 CF",
        "🎥 Content-Based",
        "🧠 SBERT"
    ])

    # =========================
    # CF TAB
    # =========================
    with tab1:

        st.header("Personalized Recommendations")

        recs = recommend_cf(st.session_state.user_id).reset_index(drop=True)

        selected = None
        cols = st.columns(5)

        for i, row in recs.iterrows():

            with cols[i % 5]:
                st.image(row["poster_url"], use_container_width=True)

                if st.button(row["title"], key=f"cf_{i}"):
                    selected = row

        if selected is not None:
            st.divider()
            show_movie(selected)

    # =========================
    # TF-IDF TAB (FIXED KEY ERROR)
    # =========================
    with tab2:

        st.header("Similar Movies (Content-Based)")

        movie = st.selectbox("Select Movie", movies["title"])

        if st.button("Recommend"):

            tmdb_id = movies[movies["title"] == movie]["tmdbId"].values[0]

            # ✅ FIX: USE tmdbId (NOT index)
            if str(tmdb_id) not in content_topk:
                st.error("No recommendations found")
            else:
                rec_list = content_topk[str(tmdb_id)]

                rec_tmdb = [i[0] for i in rec_list]
                rec_movies = movies[movies["tmdbId"].isin(rec_tmdb)].reset_index(drop=True)

                selected = None
                cols = st.columns(5)

                for i, row in rec_movies.iterrows():

                    with cols[i % 5]:
                        st.image(row["poster_url"], use_container_width=True)

                        if st.button(row["title"], key=f"tfidf_{i}"):
                              selected = row

                if selected is not None:
                    st.divider()
                    show_movie(selected)

    # =========================
    # SBERT TAB (FIXED ID USAGE)
    # =========================
    with tab3:

        st.header("Semantic Search")

        query = st.text_area("Describe a movie")

        if st.button("Search"):

            q_vec = np.array(embeddings)  # assuming precomputed similarity-ready embeddings

            scores = cosine_similarity(q_vec[:1], embeddings)[0]

            top_k = np.argsort(scores)[::-1][:10]

            results = movies.iloc[top_k].reset_index(drop=True)

            selected = None
            cols = st.columns(5)

            for i, row in results.iterrows():

                with cols[i % 5]:
                    st.image(row["poster_url"], use_container_width=True)

                    if st.button(row["title"], key=f"sbert_{i}"):
                        selected = row

            if selected is not None:
                st.divider()
                show_movie(selected)

# =========================
# GUEST MODE
# =========================
elif st.session_state.mode == "guest":
    col1, col2 = st.columns([8, 1])

    with col2:
        if st.button("🏠 Home"):
            logout()
            st.rerun()

    st.title("🎥 Guest Mode")

    tab1, tab2 = st.tabs([
        "🎥 Content-Based",
        "🧠 SBERT"
    ])

    with tab1:

        movie = st.selectbox("Select Movie", movies["title"])

        if st.button("Recommend"):

            tmdb_id = movies[movies["title"] == movie]["tmdbId"].values[0]

            if str(tmdb_id) not in content_topk:
                st.error("No recommendations found")
            else:
                rec_list = content_topk[str(tmdb_id)]

                rec_tmdb = [i[0] for i in rec_list]
                rec_movies = movies[movies["tmdbId"].isin(rec_tmdb)].reset_index(drop=True)

                selected = None
                cols = st.columns(5)

                for i, row in rec_movies.iterrows():

                    with cols[i % 5]:
                        st.image(row["poster_url"], use_container_width=True)

                        if st.button(row["title"], key=f"g_tf_{i}"):
                            selected = row

                if selected is not None:
                    st.divider()
                    show_movie(selected)

    with tab2:

        query = st.text_area("Describe a movie")

        if st.button("Search"):

            q_vec = np.array(embeddings)

            scores = cosine_similarity(q_vec[:1], embeddings)[0]

            top_k = np.argsort(scores)[::-1][:10]

            results = movies.iloc[top_k].reset_index(drop=True)

            selected = None
            cols = st.columns(5)

            for i, row in results.iterrows():

                with cols[i % 5]:
                    st.image(row["poster_url"], use_container_width=True)

                    if st.button(row["title"], key=f"g_sb_{i}"):
                        selected = row

            if selected is not None:
                st.divider()
                show_movie(selected)
