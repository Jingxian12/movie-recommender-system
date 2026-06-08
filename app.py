import streamlit as st
import pandas as pd
import numpy as np
import math
from recommender_models import (load_data, load_models,build_cf_matrix, recommend_cf, recommend_content, recommend_semantic)

# Call the function to get data
movies,ratings, user_item_matrix = load_data()
user_item_matrix, valid_users = build_cf_matrix(ratings)
item_topk, content_topk, embeddings = load_models()


# ========================================================================================================================================================================================================    
#                                                                                FUNCTION / FORMAT PART
# ======================================================================================================================================================================================================== 

# ============================================================================================
#                                    TAB DETAILS 
# ============================================================================================
# Tab 1
def get_popular_movies(movies,min_votes=10000,top_n=5):
    qualified = movies[movies["vote_count"] >= min_votes]
    return (qualified.sort_values(by=["popularity", "vote_count"],ascending=[False, False]).head(top_n).reset_index(drop=True))

# Tab 2
def render_similar_mix_tab(prefix):
    """Reusable Component for Tab 2: Content-Based Matching"""
    st.header("🍿 Find Movies Similar to Your Favorites")
    st.caption("Select a movie you love to find others built with a similar recipe.")
    
    # 1. Initialize session state variables for this tab if they don't exist
    if f"{prefix}_rec_data" not in st.session_state:
        st.session_state[f"{prefix}_rec_data"] = None

    movie_selected = st.selectbox("Select Movie", movies["title"], key=f"{prefix}_cb_select")
    
    # 2. When the button is clicked, fetch the data and save it in Session State
    if st.button("Find Matches", key=f"{prefix}_cb_btn", type="primary"):
        with st.spinner("Finding similar movies..."):
            rec_movies = recommend_content(movie_selected, content_topk, movies)
            if rec_movies is None or rec_movies.empty:
                st.session_state[f"{prefix}_rec_data"] = None
                st.error("No matches found in our database.")
            else:
                # Save the DataFrame to state so it survives clicks/reruns
                st.session_state[f"{prefix}_rec_data"] = rec_movies

    # 3. Render the grid from Session State (NOT from inside the button condition)
    if st.session_state[f"{prefix}_rec_data"] is not None:
        st.divider()
        st.header("Best Match for You:")
        st.caption("Below are the top 10 movies calculated by our engine:")
        selected = render_movie_grid(st.session_state[f"{prefix}_rec_data"], f"{prefix}_tfidf")
        
        # 4. Show info pop-up modal safely
        if selected is not None:
            show_movie(selected)

# Tab 3
def render_search_vibe_tab(prefix):
    """Reusable Component for Tab 3: NLP Semantic Search"""
    st.header("💬 Search by Movie Vibe or Plot")
    st.caption("Describe your ideal movie vibe, mood, or plot elements in everyday words.")
    
    # Track results via session_state to prevent posters from disappearing on click
    if f"{prefix}_search_results" not in st.session_state:
        st.session_state[f"{prefix}_search_results"] = None

    query = st.text_area(
        "What are you in the mood for?", 
        key=f"{prefix}_sb_query", 
        placeholder="e.g., A suspenseful spacesuit thriller with an unexpected twist ending."
    )
    
    if st.button("Search", key=f"{prefix}_sb_btn", type="primary" ):
        if query.strip() == "":
            st.error("Please write something before searching!")
        else:
            with st.spinner("Analyzing your mood..."):
                # Call the imported function cleanly
                results = recommend_semantic(query, embeddings, movies, top_k=10)
                if results.empty:
                    st.error("No movies matched your vibe. Try adjusting your description!")
                    st.session_state[f"{prefix}_search_results"] = None
                else:
                    st.session_state[f"{prefix}_search_results"] = results

    # Render layout safely outside the click branch logic
    if st.session_state[f"{prefix}_search_results"] is not None:
        selected = render_movie_grid(st.session_state[f"{prefix}_search_results"], f"{prefix}_sbert")
        if selected is not None:
            show_movie(selected)

# Tab 4
def render_advanced_search_tab(prefix):
    """Renders the advanced search tab with dynamic filters, sorting by date/title, and integrated pagination. """
    st.header("🔍 Global Studio Search")
    st.caption("Choose a category method below to find your next favorite movie.")

    # 1. Initialize tracking variables in session state if they don't exist 
    if f"{prefix}_current_page" not in st.session_state:
        st.session_state[f"{prefix}_current_page"] = 0
    if f"{prefix}_previous_search" not in st.session_state:
        st.session_state[f"{prefix}_previous_search"] = ""
    if f"{prefix}_previous_sort" not in st.session_state:
        st.session_state[f"{prefix}_previous_sort"] = ""

    # 2. Main navigation radio selection
    search_method = st.radio("How would you like to browse?",["🎥 Browse by Genre", "🎭 Find by Actor/Actress", "🎬 Find by Director"], horizontal=True, key=f"{prefix}_search_method_radio")
    st.write("") 

    filtered_df = pd.DataFrame()
    search_triggered = False
    current_search_key = ""

    # 3. Render dropdown menus dynamically based on selection
    # =========================
    #           Genre
    # =========================
    if search_method == "🎥 Browse by Genre":
        all_genres = ["Select a Genre..."] + sorted(list(set([g.strip() for sublist in movies['genres'].dropna().str.split(',') for g in sublist])))
        selected_genre = st.selectbox("Pick a Category", all_genres, key=f"{prefix}_adv_genre")

        # User have select a genre, so system can start search 
        if selected_genre != "Select a Genre...":
            filtered_df = movies[movies['genres'].str.contains(selected_genre, na=False, case=False)]
            search_triggered = True
            current_search_key = f"genre_{selected_genre}"

    # =========================
    #           People
    # =========================
    elif search_method == "🎭 Find by Actor/Actress":
        all_actors = ["Select an Actor/Actress..."] + sorted(list(set([actor.strip() for sublist in movies['cast'].dropna().str.split(',') for actor in sublist])))
        selected_actor = st.selectbox("Pick an Actor/Actress", all_actors, key=f"{prefix}_adv_cast")

         # User have select a person, so system can start search 
        if selected_actor != "Select an Actor/Actress...":
            filtered_df = movies[movies['cast'].str.contains(selected_actor, na=False, case=False)]
            search_triggered = True
            current_search_key = f"actor_{selected_actor}"

    # =========================
    #          Director
    # =========================
    elif search_method == "🎬 Find by Director":
        all_directors = ["Select a Director..."] + sorted(list(set([dir_name.strip() for sublist in movies['director'].dropna().str.split(',') for dir_name in sublist])))
        selected_director = st.selectbox("Pick a Director", all_directors, key=f"{prefix}_adv_director")

        # User have select a director, so system can start search 
        if selected_director != "Select a Director...":
            filtered_df = movies[movies['director'].str.contains(selected_director, na=False, case=False)]
            search_triggered = True
            current_search_key = f"director_{selected_director}"

    # Reset page back to 0 if the user switches to a completely different search value
    if current_search_key != st.session_state[f"{prefix}_previous_search"]:
        st.session_state[f"{prefix}_current_page"] = 0
        st.session_state[f"{prefix}_previous_search"] = current_search_key

    st.divider()
    
    # 4. Sorting & Pagination Logic Engine
    if not search_triggered:
        st.info("💡 Select an option from the dropdown menu above to display matching movies instantly!")
    else:
        total_results = len(filtered_df)
        
        if total_results > 0:
            # Layout splitting into Results Info text and the Sorting Dropdown widget
            col_info, col_sort = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"### Total Results Found: **{total_results}**")
                
            with col_sort:
                sort_option = st.selectbox(
                    "Sort results by:",
                    ["⭐ Rating (Highest to Lowest)", "⭐ Rating (Lowest to Highest)","📅 Release Date (Newest)", "📅 Release Date (Oldest)", "🔤 Title (A - Z)", "🔤 Title (Z - A)"],
                    key=f"{prefix}_sort_by"
                )
            
            # Reset page back to 0 if the user changes the sorting strategy
            if sort_option != st.session_state[f"{prefix}_previous_sort"]:
                st.session_state[f"{prefix}_current_page"] = 0
                st.session_state[f"{prefix}_previous_sort"] = sort_option

            # Sort by Ratings
            if sort_option == "⭐ Rating (Highest to Lowest)":
                filtered_df = filtered_df.sort_values(by="vote_average", ascending=False, na_position="last")
            elif sort_option == "⭐ Rating (Lowest to Highest)":
                filtered_df = filtered_df.sort_values(by="vote_average", ascending=True, na_position="last")

            # Process sort parameters on the dataset string patterns (YYYY-MM-DD)
            elif sort_option == "📅 Release Date (Newest)":
                filtered_df = filtered_df.sort_values(by="release_date", ascending=False, na_position='last')
            elif sort_option == "📅 Release Date (Oldest)":
                filtered_df = filtered_df.sort_values(by="release_date", ascending=True, na_position='last')
                
           # --- UPDATED CASE-INSENSITIVE TITLE SORTING ---
            elif sort_option == "🔤 Title (A - Z)":
                # key=lambda col: col.str.lower() converts strings to lowercase ONLY during the sort process
                filtered_df = filtered_df.sort_values(by="title", ascending=True, key=lambda col: col.str.lower())
            elif sort_option == "🔤 Title (Z - A)":
                filtered_df = filtered_df.sort_values(by="title", ascending=False, key=lambda col: col.str.lower())
            
            # Pagination metrics calculations
            items_per_page = 20
            total_pages = math.ceil(total_results / items_per_page)
            current_page = st.session_state[f"{prefix}_current_page"]
            
            # Calculate the row indices boundaries for active chunk slicing
            start_idx = current_page * items_per_page
            end_idx = min(start_idx + items_per_page, total_results) # might less than 20 movies.
            
            # Slice our dataframe down to just the 20 target rows for this page view
            page_df = filtered_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            st.markdown(f"📊 Showing results **{start_idx + 1} - {end_idx}** of **{total_results}**:")
            
            # Display entries using your flexible grid visual components
            selected = render_movie_grid(page_df, f"{prefix}_page_{current_page}")
            if selected is not None:
                show_movie(selected)
                
            st.write("") # Spacer layout padding
            st.divider()
            
            # 5. Page Navigation Controller Buttons
            col_left, col_mid, col_right = st.columns([1, 2, 1])
            
            with col_left:
                if st.button("⬅️ Previous Page", use_container_width=True,type="primary", disabled=(current_page == 0), key=f"{prefix}_btn_prev"):
                    st.session_state[f"{prefix}_current_page"] -= 1
                    st.rerun()
                    
            with col_mid:
                st.markdown(f"<p style='text-align: center; color: gray;'>Page {current_page + 1} of {total_pages}</p>", unsafe_allow_html=True)
                
            with col_right:
                if st.button("Next Page ➡️", use_container_width=True,type="primary", disabled=(current_page >= total_pages - 1), key=f"{prefix}_btn_next"):
                    st.session_state[f"{prefix}_current_page"] += 1
                    st.rerun()
        else:
            st.error("🕵️ No matches found for this selection.")

# ============================================================================================
#                                    MOVIE DETAILS     
# ============================================================================================
# 5 Movie cards show in one row
def render_movie_grid(movie_df, key_prefix):
    """Renders a dynamic grid of movie cards wrapped into rows of 5 columns."""
    if movie_df.empty:
        st.info("🔍 No movies found matching your criteria.")
        return None
        
    selected_movie = None
    num_movies = len(movie_df)
    
    # Chunk the dataframe into rows of 5 items each
    row_size = 5
    for row_idx in range(0, num_movies, row_size):
        chunk = movie_df.iloc[row_idx : row_idx + row_size]
        
        # Create a fresh row of 5 columns for this chunk
        cols = st.columns(row_size)
        
        for idx, (_, row) in enumerate(chunk.iterrows()):
            # Calculate a globally unique index for the button key
            global_idx = row_idx + idx
            
            with cols[idx]:
                with st.container(border=True):
                    # --- FIXED POSTER FALLBACK LOGIC ---
                    poster = str(row.get("poster_url", "")).strip()
                    
                    # Check for empty string, NaN, or None equivalents
                    if poster in ["", "nan", "None"] or pd.isna(row["poster_url"]):
                        # 200x300 works perfectly for standard vertical movie cards
                        poster = "https://placehold.co"
                    
                    st.image(poster, use_container_width=True)
                    
                    # Title
                    st.markdown(f"**{row['title']}**")
                    
                    # Rating
                    rating_raw = row.get("vote_average", "N/A")
                    if pd.notna(rating_raw) and isinstance(rating_raw, (int, float)):
                        rating = f"{float(rating_raw):.1f}"
                    else:
                        rating = "N/A"
                    st.caption(f"⭐ {rating} / 10")
                    
                    st.html("<div style='min-height: 10px;'></div>")
                    
                    # Button
                    if st.button("🎬 Info", key=f"{key_prefix}_{global_idx}", use_container_width=True, type="secondary"):
                        selected_movie = row
                        
    return selected_movie 

# Movie Information 
@st.dialog("🎬 Movie Details", width="large")
def show_movie(movie):
    col1, col2 = st.columns([1, 2])
    with col1:
        # 1. Extract and clean the poster url
        poster = str(movie.get("poster_url", "")).strip()
        
        # 2. If it is empty or NaN, use a clean placeholder image
        if poster in ["", "nan", "None"] or pd.isna(movie["poster_url"]):
            # Generates a clean 300x450 grey box that says "No Poster Available"
            fallback_url = "https://placehold.co"
            st.image(fallback_url, use_container_width=True)
        else:
            st.image(poster, use_container_width=True)

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
        

# ========================================================================================================================================================================================================    
#                                                                                      UI PART 
# ========================================================================================================================================================================================================  

# =========================
# SESSION STATE CONTROL
# =========================
def logout():
    st.session_state.mode = "home"
    if "user_id" in st.session_state:
        del st.session_state.user_id
        
if "mode" not in st.session_state: # prevents data from being reset when the app re-runs from top to bottom on user interaction
    st.session_state.mode = "home"


# ============================================================================================
#                                     0. PAGE CONFIG
# ============================================================================================
st.set_page_config(page_title="MovieMatch", layout="wide")

st.title("🎬 Hybrid Movie Recommender System")
st.write("Welcome to MovieMatch! Log in with your User ID to unlock personalized recommendations or use Guest Mode.")
st.divider()

# ============================================================================================
#                                       1. HOME MODE
# ============================================================================================
if st.session_state.mode == "home":
    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.markdown("### 🔐 For Returning Members")
            st.markdown(
                """
                * 📈 **Made For You**: Get smart picks based on movies you've already rated.
                * 🎯 **Smart Predictions**: The more you rate, the better our guesses get.
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
                * 🍿 **Zero Setup**: No account or password needed—just jump straight in!
                * 🎭 **Instant Mix**: Pick a favorite film and instantly see choices just like it.
                """
            )
            if st.button("Continue as Guest", use_container_width=True,type="primary", icon="👋"):
                st.session_state.mode = "guest"
                st.rerun()

    st.divider()
    # Latest Movie Poster 
    st.subheader("✨ Latest Hits & New Releases")
    
    try:
        # 1. Ensure the release_date column is in datetime format so pandas can sort it correctly
        movies["release_date_dt"] = pd.to_datetime(movies["release_date"], dayfirst=True,errors="coerce")
        
        # 2. Quality Control: Filter out unreleased or obscure movies
        min_votes_for_new = 5000 
        qualified_new = movies[movies["vote_count"] >= min_votes_for_new]
        
        # 3. Multi-tiered Sort: Sort primarily by newest date, secondarily by popularity
        latest_movies = qualified_new.sort_values(by="release_date_dt", ascending=False).head(5).reset_index(drop=True)
    
       # 4. Clean up the temporary datetime column so it doesn't mess up your data profile
        latest_movies = latest_movies.drop(columns=["release_date_dt"])
        
        clicked_latest = render_movie_grid(latest_movies, "latest")
        if clicked_latest is not None:
            show_movie(clicked_latest)
                    
    except Exception as e:
        print(f"Error on trending section: {e}")
        pass

# ============================================================================================
#                                    2. LOGIN MODE
# ============================================================================================
elif st.session_state.mode == "login":
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("🏡 Home Page"):
            st.session_state.mode = "home"
            st.rerun()

    st.subheader("Enter User ID")
    uid = st.text_input("User ID",placeholder="e.g., 42")
    
    # 2. Password input (type="password" hides the characters)
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    if st.button("Submit", type="primary", use_container_width=True):
        if uid.isdigit() and int(uid) in ratings['userId'].values:
            # 4. Check if the password matches "123"
            if password == "123":
                st.session_state.user_id = int(uid)
                st.session_state.mode = "user"
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again!")
        else:
            st.error("❌ User not found. Please enter a valid ID!")
            
# ============================================================================================
#                                        3. USER MODE
# ============================================================================================
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
        "💬 Search by Story", 
        "🔍 Criteria Search"
    ])

    with tab1:
        st.header("Personalized Picks")
        
        current_user = st.session_state.user_id
        # Check if the user qualifies for Collaborative Filtering
        if current_user in valid_users:
            st.caption(
            """
            ### Custom tailored choices based on your historical ratings.
            Our **Item-Based Collaborative Filtering** engine analyzes the movies you've highly rated in the past.
            """
            )
            # Call your CF model using the matrix generated in the background
            recs = recommend_cf(
                current_user, 
                user_item_matrix, 
                item_topk, 
                movies
            ).reset_index(drop=True)
            
        else:
            # COLD START FALLBACK
            st.caption(
            """
            ### Welcome to the community! 
            Since you are a new user or haven't rated 25+ movies above 3.5 yet, here are some of our **most popular trending hits** to get you started! Once you rate more films, this space will personalize automatically.
            """
            )
            
            # Call your custom fallback function here
            # (Make sure get_popular_movies is imported or defined above this)
            recs = get_popular_movies(movies, min_votes=10000, top_n=5)
    
        # Render the grid for whichever recommendations were generated above
        # (recs will match format since both paths return a reset-indexed DataFrame)
        selected = render_movie_grid(recs.head(5), "user_cf")
        if selected is not None:
            show_movie(selected)

    with tab2:
        render_similar_mix_tab(prefix="user")

    with tab3:
        render_search_vibe_tab(prefix="user")

    with tab4:
        render_advanced_search_tab(prefix="user")

# ============================================================================================
#                                     4. GUEST MODE
# ============================================================================================
elif st.session_state.mode == "guest":
    col1, col2 = st.columns([8, 1])
    with col1:
        if st.button("🏡 Home Page", key="guest_home"):
            logout()
            st.rerun()

    st.title("🎥 Guest Dashboard")
    st.info("💡 Logging in with a User ID unlocks premium custom tracking and history calculations!")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 What's Hot", 
        "🎥 Similar Mix", 
        "💬 Search by Story", 
        "🔍 Criteria Search"
    ])

    with tab1:
        st.header("🔥 What's Hot")
        st.caption("The most famous blockbuster movies actively trending globally right now.")
        popular_recs = get_popular_movies(movies)
        selected = render_movie_grid(popular_recs, "guest_hot")
        if selected is not None:
            show_movie(selected)

    with tab2:
        render_similar_mix_tab(prefix="guest")

    with tab3:
        render_search_vibe_tab(prefix="guest")

    with tab4:
        render_advanced_search_tab(prefix="guest")
