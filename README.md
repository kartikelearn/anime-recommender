# 🎌 Anime Recommender System

A hybrid anime recommendation system built with Python, Pandas, NumPy, and Scikit-learn. It combines genre similarity, synopsis-based NLP, and weighted ratings to recommend related anime and help users discover new favorites.

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