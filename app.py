
import streamlit as st
from recommender import recommend_anime_modes, recommend_diverse_anime

st.set_page_config(
    page_title="Anime Recommender",
    page_icon="🎌",
    layout="wide"
)

st.title("🎌 Anime Recommender")
st.write("Discover your next favourite anime.")

st.divider()

anime_title = st.text_input(
    "Enter an anime you like",
    placeholder="e.g. Death Note, One Piece, Naruto"
)

recommendation_mode = st.selectbox(
    "Choose recommendation mode",
    [
        "Discover New Anime",
        "Find Related Anime",
        "Both"
    ]
)

top_n = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=15
)


def show_recommendations(results):
    if results is None or results.empty:
        st.warning(
            "No recommendations found. Try another anime title."
        )
        return

    # Keep the main interface simple
    display_columns = [
        col for col in ["Name", "Genres", "Score"]
        if col in results.columns
    ]

    st.dataframe(
        results[display_columns].rename(
            columns={
                "Name": "Anime",
                "Genres": "Genres",
                "Score": "Rating (/10)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    # Keep technical model scores optional
    with st.expander("🔬 Model Details"):
        detail_columns = [
            col for col in [
                "Name",
                "genre_similarity",
                "story_similarity",
                "hybrid_score"
            ]
            if col in results.columns
        ]

        if detail_columns:
            st.caption(
                "Similarity and hybrid scores are model metrics, "
                "not probabilities of enjoying an anime."
            )

            st.dataframe(
                results[detail_columns].rename(
                    columns={
                        "Name": "Anime",
                        "genre_similarity": "Genre Similarity",
                        "story_similarity": "Story Similarity",
                        "hybrid_score": "Combined Score"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


if st.button("Find Recommendations", type="primary"):
    if not anime_title.strip():
        st.warning("Please enter an anime title first.")

    else:
        with st.spinner("Finding your anime matches..."):
            try:
                if recommendation_mode == "Both":
                    related, discover = recommend_anime_modes(
                        anime_title,
                        top_n=top_n,
                        mode="both"
                    )

                    st.subheader("✨ Related Anime")
                    show_recommendations(related)

                    st.subheader("🌍 Discover New Anime")
                    show_recommendations(discover)

                elif recommendation_mode == "Find Related Anime":
                    results = recommend_anime_modes(
                        anime_title,
                        top_n=top_n,
                        mode="related"
                    )

                    st.subheader("✨ Related Anime")
                    show_recommendations(results)

                else:
                    results = recommend_diverse_anime(
                        anime_title,
                        top_n=top_n
                    )

                    st.subheader("🌍 Discover New Anime")
                    show_recommendations(results)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
