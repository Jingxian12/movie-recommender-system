import streamlit as st 

st.title("🎬 Hybrid Movie Recommender")

user_id = st.text_input(
    "Enter User ID (Optional)"
)

if user_id:
    st.success(f"Welcome User {user_id}")
    recommendations = get_cf_recommendations(int(user_id))
    show_movies(recommendations)

else:
    st.info(
        "No User ID entered. "
        "You can use Similar Movie Search "
        "or Semantic Search."
    )
