import streamlit as st 

st.set_page_config(
    page_title="Hybrid Movie Recommender",
    layout="wide"
)

st.title("🎬 Hybrid Movie Recommender System")

mode = st.sidebar.radio(
    "Select Recommendation Mode",
    [
        "Personalized Recommendation",
        "Similar Movie Search",
        "Semantic Search"
    ]
)
