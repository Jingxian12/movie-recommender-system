import streamlit as st 

if "mode" not in st.session_state:
    st.session_state.mode = None

st.title("🎬 Hybrid Movie Recommender")

col1, col2 = st.columns(2)

with col1:
    if st.button("Login"):
        st.session_state.mode = "login"

with col2:
    if st.button("Skip"):
        st.session_state.mode = "guest"


if st.session_state.mode == "login":

    user_id = st.text_input("Enter User ID")

    if st.button("Submit"):

        if int(user_id) in ratings_df["userId"].unique():

            st.session_state.user_id = int(user_id)
            st.session_state.mode = "user"

        else:
            st.error("User ID not found")

if st.session_state.mode == "user":

    tab1, tab2, tab3 = st.tabs([
        "🏠 Personalized",
        "🎥 Similar Movies",
        "🧠 Semantic Search"
    ])
