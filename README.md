# 🎌 Anime Recommender System

A hybrid anime recommendation system built with Python, Pandas, NumPy, and Scikit-learn. It combines genre similarity, synopsis-based NLP, and weighted ratings to recommend related anime and help users discover new favorites.

**🔗 Live demo:** https://ml-anime-recommender.streamlit.app/

![App screenshot](screenshot.png)

## Features

- **Genre Similarity:** Recommends anime with similar genres using TF-IDF.
- **Story Similarity:** Uses word-level and character-level TF-IDF on anime synopses.
- **Hybrid Recommendations:** Combines genre similarity, story similarity, and weighted ratings.
- **Franchise-Aware Recommendations:** Offers related anime and recommendations from different franchises.
- **Interactive Web App:** Explore recommendations through a Streamlit interface.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF and NLP
- Streamlit
- Jupyter Notebook

## Project Structure

- `app.py` — Streamlit web application
- `recommender.py` — Recommendation model and logic
- `anime_recommender_final.ipynb` — Model development and experiments
- `anime.csv` — Anime metadata dataset
- `anime_with_synopsis.csv` — Anime synopsis dataset
- `requirements.txt` — Python dependencies

## How It Works

1. **Genre similarity:** each anime's genres are converted to TF-IDF vectors, and cosine similarity measures how close two anime are.
2. **Story similarity:** synopses are vectorized two ways, word-level (1–2 grams) and character-level (3–5 grams), then combined as 70% word + 30% character.
3. **Weighted rating:** raw ratings are adjusted by number of members, so anime with very few votes don't dominate, then scaled to 0–1.
4. **Hybrid score:**

   `score = 0.40 × genre similarity + 0.40 × story similarity + 0.20 × normalized rating`

5. **Franchise-aware modes:**
   - `related`: other entries from the same franchise
   - `discover`: anime from different franchises
   - `both`: shows both lists

## Limitations & Future Improvements

- Content-based only: no user history, so recommendations aren't personalized.
- Hybrid weights (0.40 / 0.40 / 0.20) are hand-tuned, not learned.
- No formal evaluation metric yet.
- Franchise grouping uses title-cleaning rules plus a manual list, so some series may be grouped imperfectly.
- Possible next steps: evaluate with precision@k, add user-based collaborative filtering, try sentence embeddings for synopses.

## Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/kartikelearn/anime-recommender.git
   cd anime-recommender
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:

   ```bash
   streamlit run app.py
   ```

## Datasets

- [Anime Recommendations Database — CooperUnion](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)
- [Anime Recommendation Database 2020 — Hernan4444](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020)

The datasets are credited to their respective sources. Please consult the original dataset pages for licensing and attribution details.

## Author

**kartikelearn**

- [GitHub](https://github.com/kartikelearn)
