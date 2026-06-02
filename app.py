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
    col1, col2 = st.columns([1, 2]) # col1 : image / col2 : information
    with col1:
        st.image(movie["poster_url"], use_container_width=True)
    with col2:
        # Safe formatting helper function
        def get_clean_val(val, suffix=""):
            # Checks if value is NaN, None, or an empty string
            if pd.isna(val) or str(val).strip().lower() in ["nan", "none", ""]:
                return "Not Available"
            return f"{val}{suffix}"

        st.write("**Genres:**", get_clean_val(movie["genres"]))
        st.write("**Overview:**", get_clean_val(movie["overview"]))
        st.write("**Cast:**", get_clean_val(movie["cast"]))
        st.write("**Director:**", get_clean_val(movie["director"]))
        st.write("**Rating:**", get_clean_val(movie["vote_average"], " / 10"))
        # Automatically adds " mins" if the runtime exists, otherwise says "Not Available"
        st.write("**Runtime:**", get_clean_val(movie["runtime"], " mins"))
        st.write("**Release Date:**", get_clean_val(movie["release_date"]))

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
            st.markdown("### 🔐 For Returning Members")
            st.markdown(
                """
                * 📈 **Made For You**: Get smart picks based on movies you've already rated.
                * 🎯 **Smart Predictions**: The more you rate, the better our guesses get.
                * 📂 **Saved History**: Keep track of your personal viewing history.
                """
            )
            if st.button("Log In to Your Profile", use_container_width=True, type="primary", icon="🔑"):
                st.session_state.mode = "login"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 🚀 Quick Guest Mode")
            st.markdown(
                """
                * 📄 **Search by Vibe**: Find movies by matching descriptions, moods.
                * 🎭 **Instant Mix**: Pick a favorite film and instantly see choices just like it.
                * 🍿 **Zero Setup**: No account or password needed—just jump straight in!
                """
            )
            if st.button("Continue as Guest", use_container_width=True,type="primary", icon="👋"):
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
    
        # Create the 5-column grid layout
        poster_cols = st.columns(5)
        
        for idx, (_, row) in enumerate(hot_movies.iterrows()):
            with poster_cols[idx]:
                # This creates a beautiful distinct frame around the content
                with st.container(border=True):
                    
                    # A. Show the movie poster
                    poster = row["poster_url"] if str(row["poster_url"]) != "nan" else "https://placeholder.com"
                    st.image(poster, use_container_width=True)
                    
                    # B. Clean, bold title
                    st.markdown(f"**{row['title']}**")
                    
                    # C. Tiny, useful metadata subtext (Rating badge)
                    # Safeguard in case vote_average is missing
                    rating = row.get("vote_average", "N/A")
                    st.caption(f"⭐ {rating} / 10")
                    
                    # D. THE FIX: Invisible structural filler that forces the button 
                    # to stay glued to the bottom of the card regardless of title length
                    st.html("<div style='min-height: 10px;'></div>")
                    
                    # E. A distinct, clearly visible action button
                    if st.button("🎬 View Info", key=f"trend_{idx}", use_container_width=True, type="secondary"):
                        show_movie(row)
                    
    except Exception as e:
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
# 3. USER MODE (Valid User Page)
# =========================
elif st.session_state.mode == "user":
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🚪 Sign Out"):
            logout()
            st.rerun()
    st.success(f"Welcome User {st.session_state.user_id}")

    tab1, tab2, tab3 = st.tabs(["🏠 CF", "🎥 Content-Based", "🧠 SBERT"])

# =========================
    # CF TAB (PERSONALIZED)
    # =========================
    with tab1:
        # 1. Fetch the user's historical rating metrics for a personalized greeting
        user_ratings_all = user_item_matrix.loc[st.session_state.user_id]
        watched_movies = user_ratings_all[user_ratings_all > 0]
        total_watched = len(watched_movies)
        avg_user_rating = round(watched_movies.mean(), 1) if total_watched > 0 else 0

        # Welcome Summary Banner
        st.markdown(f"### 👋 Welcome back, Explorer!")
        st.markdown(
            f"You have rated **{total_watched} movies** with an average score of **{avg_user_rating} ⭐**. "
            "Based on your unique taste footprint, we have curated these special match categories for you:"
        )

        # 2. Helpful Context Box
        st.info(
            "💡 **How these are chosen:** We found other film lovers who share your exact movie ratings, "
            "and brought forward the titles they rated highly that you haven't discovered yet!"
        )
        
        st.divider()

        # 3. Fetch recommendations from your model
        # We fetch a larger pool (e.g., 20 movies) so we can filter and shuffle them!
        raw_recs = recommend_cf(st.session_state.user_id, user_item_matrix, item_topk, movies)
        
        if raw_recs.empty:
            st.warning("Complete a few more ratings in our system to unlock custom picks!")
        else:
            # Action controls row: Let them mix things up
            col_header, col_btn = st.columns([4, 1])
            with col_btn:
                # A shuffle button that triggers a simple UI state rerun
                shuffle_clicked = st.button("🔄 Shuffle My Picks", use_container_width=True)

            # Row 1: Top Picks (Sorted by highest popularity within your recommended pool)
            st.markdown("#### 🏆 Your Absolute Top Matches")
            st.caption("Highly-rated blockbusters that fit your profile perfectly.")
            
            top_picks = raw_recs.sort_values(by="popularity", ascending=False).head(5)
            if shuffle_clicked:
                top_picks = raw_recs.sample(min(5, len(raw_recs)))
                
            selected_top = render_movie_grid(top_picks.reset_index(drop=True), "cf_top")
            if selected_top is not None:
                show_movie(selected_top)

            st.write("") # Spacer

            # Row 2: Hidden Gems (Low popularity scores in the dataset, but highly recommended by CF)
            st.markdown("#### 💎 Underrated Hidden Gems")
            st.caption("Less mainstream movies that people with your exact taste absolutely loved.")
            
            # Grabbing movies with lower popularity metrics to serve as lesser-known gems
            hidden_gems = raw_recs.sort_values(by="popularity", ascending=True).head(5)
            if shuffle_clicked:
                hidden_gems = raw_recs.sample(min(5, len(raw_recs)))
                
            selected_gem = render_movie_grid(hidden_gems.reset_index(drop=True), "cf_gem")
            if selected_gem is not None:
                show_movie(selected_gem)

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
