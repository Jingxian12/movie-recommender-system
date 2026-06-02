import streamlit as st
import pandas as pd
import numpy as np

from recommender_models import (load_data, load_models, recommend_cf, recommend_content, recommend_semantic)

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
            
        st.write("**Title:**", get_clean_val(movie["title"]))
        st.write("**Genres:**", get_clean_val(movie["genres"]))
        st.write("**Overview:**", get_clean_val(movie["overview"]))
        st.write("**Cast:**", get_clean_val(movie["cast"]))
        st.write("**Director:**", get_clean_val(movie["director"]))
        st.write("**Production Companies:**", get_clean_val(movie["production_companies"]))
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
    """
    Renders a standard 5-column grid of movie cards inside bordered containers.
    Returns the movie row if its 'Info' button is clicked, otherwise returns None.
    """
    selected_movie = None
    
    # Ensure we don't try to render more columns than we have data for
    num_movies = len(movie_df)
    if num_movies == 0:
        return None
        
    # Create exactly 5 columns for layout uniformity
    poster_cols = st.columns(5)
    
    for idx, (_, row) in enumerate(movie_df.iterrows()):
        # Use modulo to wrap elements if the dataframe has more than 5 movies
        col_idx = idx % 5
        
        with poster_cols[col_idx]:
            # Distinct structural frame around each movie card
            with st.container(border=True):
                
                # A. Show the movie poster
                poster = row["poster_url"] if str(row["poster_url"]) != "nan" else "https://placeholder.com"
                st.image(poster, use_container_width=True)
                
                # B. Clean, bold title
                st.markdown(f"**{row['title']}**")
                
                # C. Metadata subtext (Rating badge)
                rating = row.get("vote_average", "N/A")
                st.caption(f"⭐ {rating} / 10")
                
                # D. Structural spacing filler to keep layout lengths balanced
                st.html("<div style='min-height: 10px;'></div>")
                
                # E. Distinct action button
                # We combine key_prefix and idx to ensure every button across the app has a unique identifier
                if st.button("🎬 Info", key=f"{key_prefix}_{idx}", use_container_width=True, type="secondary"):
                    selected_movie = row

    return selected_movie

def render_similar_mix_tab(prefix):
    """Reusable Component for Tab 2: Content-Based Matching"""
    st.header("🎥 Similar Mix")
    st.caption("Select a movie you love to find others built with a similar recipe.")
    
    movie = st.selectbox("Select Movie", movies["title"], key=f"{prefix}_cb_select")
    if st.button("Find Matches", key=f"{prefix}_cb_btn"):
        rec_movies = recommend_content(movie, content_topk, movies)
        if rec_movies is None or rec_movies.empty:
            st.error("No matches found in our database.")
        else:
            selected = render_movie_grid(rec_movies, f"{prefix}_tfidf")
            if selected is not None:
                show_movie(selected)


def render_search_vibe_tab(prefix):
    """Reusable Component for Tab 3: NLP Semantic Search"""
    st.header("🧠 Search by Vibe")
    st.caption("Describe your ideal movie vibe, mood, or plot elements in everyday words.")
    
    query = st.text_area(
        "What are you in the mood for?", 
        key=f"{prefix}_sb_query", 
        placeholder="e.g., A suspenseful spacesuit thriller with an unexpected twist ending."
    )
    if st.button("Search Mood", key=f"{prefix}_sb_btn"):
        results = recommend_semantic(query, embeddings, movies)
        selected = render_movie_grid(results, f"{prefix}_sbert")
        if selected is not None:
            show_movie(selected)


def render_browse_categories_tab(prefix):
    """Reusable Component for Tab 4: Genre Filtering"""
    st.header("🎭 Browse Categories")
    st.caption("Filter our collection down by your favorite genres.")
    
    # Safely extract unique categories out of the dataframe string data
    all_genres = sorted(list(set([g.strip() for sublist in movies['genres'].dropna().str.split(',') for g in sublist])))
    selected_genre = st.selectbox("Pick a Category", all_genres, key=f"{prefix}_genre_select")
    
    genre_filtered = movies[movies['genres'].str.contains(selected_genre, na=False, case=False)].head(5).reset_index(drop=True)
    selected = render_movie_grid(genre_filtered, f"{prefix}_genre")
    if selected is not None:
        show_movie(selected)

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
        new_movies = movies.sort_values(by=["release_date","popularity"], ascending=[False, False]).head(5)
    
       # 3. Call the function
        clicked_trending = render_movie_grid(new_movies, "trend")
        if clicked_trending is not None:
            show_movie(clicked_trending)
                    
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
# 3. USER MODE
# =========================
elif st.session_state.mode == "user":
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🚪 Sign Out", key="user_logout"):
            logout()
            st.rerun()
            
    st.success(f"Welcome back, User {st.session_state.user_id}!")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏠 Made For You", 
        "🎥 Similar Mix", 
        "🧠 Search by Vibe", 
        "🎭 Browse Categories"
    ])

    with tab1:
        st.header("Personalized Picks")
        st.caption("Custom tailored choices calculated from your historical ratings.")
        recs = recommend_cf(st.session_state.user_id, user_item_matrix, item_topk, movies).reset_index(drop=True)
        selected = render_movie_grid(recs.head(5), "user_cf")
        if selected is not None:
            show_movie(selected)

    with tab2:
        render_similar_mix_tab(prefix="user")

    with tab3:
        render_search_vibe_tab(prefix="user")

    with tab4:
        render_browse_categories_tab(prefix="user")

# =========================
# 4. GUEST MODE
# =========================
elif st.session_state.mode == "guest":
    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🏠 Home", key="guest_home"):
            logout()
            st.rerun()

    st.title("🎥 Guest Dashboard")
    st.info("💡 Logging in with a User ID unlocks premium custom tracking and history calculations!")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 What's Hot", 
        "🎥 Similar Mix", 
        "🧠 Search by Vibe", 
        "🎭 Browse Categories"
    ])

    with tab1:
        st.header("What's Hot")
        st.caption("The most famous blockbuster movies actively trending globally right now.")
        min_votes = 2000
        qualified = movies[movies["vote_count"] >= min_votes]
        popular_recs = qualified.sort_values(by=["popularity", "vote_count"], ascending=[False, False]).head(5).reset_index(drop=True)
        
        selected = render_movie_grid(popular_recs, "guest_hot")
        if selected is not None:
            show_movie(selected)

    with tab2:
        render_similar_mix_tab(prefix="guest")

    with tab3:
        render_search_vibe_tab(prefix="guest")

    with tab4:
        render_browse_categories_tab(prefix="guest")
