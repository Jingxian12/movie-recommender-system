import streamlit as st
import pandas as pd
import numpy as np

# Import functions from our newly created module
from recommender_models import (
    load_data, 
    load_models, 
    recommend_cf, 
    recommend_content, 
    recommend_semantic
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="MovieMatch", layout="wide")

st.title("🎬 Hybrid Movie Recommender System")
st.write("Welcome to MovieMatch! Log in with your User ID to unlock personalized recommendations or use Guest Mode.")
st.divider()

# =========================
# INITIALIZE DATA & MODELS
# =========================
# Cache data/models so they don't reload on every button click/refresh
@st.cache_data
def get_cached_data():
    return load_data()

@st.cache_resource
def get_cached_models():
    return load_models()

movies, user_item_matrix = get_cached_data()
item_topk, content_topk, embeddings = get_cached_models()

# =========================
# MOVIE DETAILS UI DIALOG
# =========================
@st.dialog("🎬 Movie Details", width="large")
def show_movie(movie):
    col1, col2 = st.columns([1, 2])
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
# HELPER ACTIONS
# =========================
def logout():
    st.session_state.mode = "home"
    if "user_id" in st.session_state:
        del st.session_state.user_id

def render_movie_grid(movie_df, key_prefix):
    """Helper method to clean up repetitive grid generation code"""
    selected = None
    cols = st.columns(5)
    for i, row in movie_df.iterrows():
        with cols[i % 5]:
            st.image(row["poster_url"], use_container_width=True)
            if st.button(row["title"], key=f"{key_prefix}_{i}"):
                selected = row
    return selected


# =========================
# SESSION STATE CONTROL
# =========================
if "mode" not in st.session_state:
    st.session_state.mode = "home"

# =========================
# 1. HOME MODE
# =========================
if st.session_state.mode == "home":
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("### 🔐 Personalized Experience")
            st.markdown("- 📈 **Collaborative Filtering**: History analysis.\n- 🎯 **Predictive Scoring**: Guessing your favorites.")
            if st.button("Log In to Your Profile", use_container_width=True, type="primary",icon="🚨"):
                st.session_state.mode = "login"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 🚀 Quick Guest Mode")
            st.markdown("- 📄 **Content-Based Discovery**: Keywords & vibes.\n- 🍿 **Instant Results**: No account needed.")
            if st.button("Continue as Guest", use_container_width=True):
                st.session_state.mode = "guest"
                st.rerun()

    st.divider()
    # Trending Movie Poster 
    st.subheader("🔥 Explore Trending Movies Right Now")
    
    try:
        # 1. Filter out movies with very few votes to maintain quality control
        min_votes = 2000
        qualified_movies = movies[movies["vote_count"] >= min_votes]
        
        # 2. Sort primarily by popularity, secondarily by vote_count
        hot_movies = qualified_movies.sort_values(by=["popularity", "vote_count"], ascending=[False, False]).head(5)
    
        poster_cols = st.columns(5)
        for idx, (_, row) in enumerate(hot_movies.iterrows()):
            with poster_cols[idx]:
                poster = row["poster_url"] if str(row["poster_url"]) != "nan" else "https://placeholder.com"
                st.image(poster, use_container_width=True)
                
                # --- KEY IMPLEMENTATION HERE ---
                # Use the title as a button. When clicked, it calls your show_movie function.
                # We add key=f"trend_{idx}" so Streamlit can track each unique button.
                if st.button(row['title'], key=f"trend_{idx}", use_container_width=True):
                    show_movie(row)
                    
    except Exception as e:
        # Optional: print the error to your terminal during development to debug issues
        print(f"Error on trending section: {e}")
        pass


# =========================
# 2. LOGIN MODE
# =========================
elif st.session_state.mode == "login":
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("⬅️ Home"):
            st.session_state.mode = "home"
            st.rerun()

    st.subheader("Enter User ID")
    uid = st.text_input("User ID")
    if st.button("Submit"):
        if uid.isdigit() and int(uid) in user_item_matrix.index:
            st.session_state.user_id = int(uid)
            st.session_state.mode = "user"
            st.rerun()
        else:
            st.error("❌ User not found. Please enter a valid ID!")

# =========================
# 3. USER MODE
# =========================
elif st.session_state.mode == "user":
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🚪 Sign Out"):
            logout()
            st.rerun()
    st.success(f"Welcome User {st.session_state.user_id}")

    tab1, tab2, tab3 = st.tabs(["🏠 CF", "🎥 Content-Based", "🧠 SBERT"])

    with tab1:
        st.header("Personalized Recommendations")
        recs = recommend_cf(st.session_state.user_id, user_item_matrix, item_topk, movies).reset_index(drop=True)
        selected = render_movie_grid(recs, "cf")
        if selected is not None:
            show_movie(selected)

    with tab2:
        st.header("Similar Movies (Content-Based)")
        movie = st.selectbox("Select Movie", movies["title"], key="user_cb_select")
        if st.button("Recommend", key="user_cb_btn"):
            rec_movies = recommend_content(movie, content_topk, movies)
            if rec_movies is None or rec_movies.empty:
                st.error("No recommendations found")
            else:
                selected = render_movie_grid(rec_movies, "tfidf")
                if selected is not None:
                    show_movie(selected)

    with tab3:
        st.header("Semantic Search")
        query = st.text_area("Describe a movie", key="user_sb_query")
        if st.button("Search", key="user_sb_btn"):
            results = recommend_semantic(query, embeddings, movies)
            selected = render_movie_grid(results, "sbert")
            if selected is not None:
                show_movie(selected)

# =========================
# 4. GUEST MODE
# =========================
elif st.session_state.mode == "guest":
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🏠 Home"):
            logout()
            st.rerun()

    st.title("🎥 Guest Mode")
    tab1, tab2 = st.tabs(["🎥 Content-Based", "🧠 SBERT"])

    with tab1:
        movie = st.selectbox("Select Movie", movies["title"], key="guest_cb_select")
        if st.button("Recommend", key="guest_cb_btn"):
            rec_movies = recommend_content(movie, content_topk, movies)
            if rec_movies is None or rec_movies.empty:
                st.error("No recommendations found")
            else:
                selected = render_movie_grid(rec_movies, "g_tf")
                if selected is not None:
                    show_movie(selected)

    with tab2:
        query = st.text_area("Describe a movie", key="guest_sb_query")
        if st.button("Search", key="guest_sb_btn"):
            results = recommend_semantic(query, embeddings, movies)
            selected = render_movie_grid(results, "g_sb")
            if selected is not None:
                show_movie(selected)
