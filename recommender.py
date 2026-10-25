#!/usr/bin/env python
# coding: utf-8

# 
# # Anime Recommendation System
# 
# A content-based anime recommendation system built using Pandas, NumPy, and Scikit-learn.
# 
# The system combines genre similarity, synopsis similarity, and weighted ratings to recommend anime. It also supports franchise-aware recommendations to help users discover different series.
# 

# 
# ## 1. Imports and Dataset Loading
# 

# In[2]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


anime=pd.read_csv("anime.csv")
columns=anime.columns.to_list()


# 
# ## 2. Initial Dataset Inspection
# 

# In[4]:


columns
anime.shape


# In[5]:


print("Columns DType")
print(anime.dtypes)


# In[6]:


print("null values")
print(anime.isnull().sum())
print(anime.shape)


# In[7]:


print("duplicate values")
print(anime.duplicated().sum())


# 
# ## 3. Data Cleaning and Preparation
# 

# In[8]:


anime["genre"]=anime["genre"].fillna("")
anime["type"]=anime["type"].fillna("Unknown")


# 
# ## 4. Genre Exploration
# 

# In[9]:


# Look at some anime titles and their genres

anime[["name", "genre"]].head(15)


# In[10]:


# Find the most common genres
genre_counts = (anime["genre"].str.split(",").explode().str.strip().value_counts())


# In[11]:


genre_counts.head(15)


# 
# ## 5. Genre-Based Features
# 

# In[12]:


tfidf = TfidfVectorizer(token_pattern=r"[^,]+")

tfidf_matrix = tfidf.fit_transform(anime["genre"])

print("TF-IDF matrix shape:", tfidf_matrix.shape)
print("Unique genre features:", len(tfidf.get_feature_names_out()))


# 
# ## 6. Rating-Based Features
# 

# In[13]:


print("Total anime:", len(anime))
print("Missing ratings:", anime["rating"].isna().sum())

print("\nHighest-rated anime:")
print(
    anime[["name", "rating", "members"]].dropna(subset=["rating"])
    .sort_values("rating", ascending=False).head(10)
)


# In[14]:


# Average rating across the dataset
C = anime["rating"].mean()

# Minimum number of members required for stronger confidence
m = anime["members"].quantile(0.75)

# Calculate weighted rating
anime["weighted_rating"] = (
    (anime["members"] / (anime["members"] + m)) * anime["rating"]
    + (m / (anime["members"] + m)) * C
)

print("Average rating:", round(C, 2))
print("Minimum popularity threshold:", int(m))

print("\nTop 10 by weighted rating:")
print(
    anime[["name", "rating", "members", "weighted_rating"]].dropna(subset=["rating"])
    .sort_values("weighted_rating", ascending=False).head(10).round(2).to_string(index=False))


# In[15]:


# Normalize weighted ratings to a 0–1 scale
min_rating = anime["weighted_rating"].min()
max_rating = anime["weighted_rating"].max()

anime["normalized_rating"] = (
    (anime["weighted_rating"] - min_rating)
    / (max_rating - min_rating)
)


# 
# ## 7. Story-Based Features (synopsis)
# 

# In[16]:


synopsis = pd.read_csv("anime_with_synopsis.csv")

print("Dataset shape:", synopsis.shape)
print("\nColumns:", synopsis.columns.tolist())
print("\nMissing values:")
print(synopsis.isnull().sum())

print("\nFirst 5 anime:")
print(synopsis.head())


# In[17]:


synopsis = synopsis.rename(columns={"sypnopsis": "synopsis"})

# Fill missing synopsis text with an empty string
synopsis["synopsis"] = synopsis["synopsis"].fillna("")

# Remove leading/trailing spaces from anime titles
synopsis["Name"] = synopsis["Name"].str.strip()

print("Missing synopses:", synopsis["synopsis"].isna().sum())
print("Duplicate anime IDs:", synopsis["MAL_ID"].duplicated().sum())
print("Duplicate anime names:", synopsis["Name"].duplicated().sum())

print("\nExample synopsis:")
print(synopsis.loc[0, "synopsis"][:500])


# In[18]:


# Create normalized titles for matching
anime["match_name"] = (
    anime["name"]
    .str.casefold()
    .str.strip()
)

synopsis["match_name"] = (
    synopsis["Name"]
    .str.casefold()
    .str.strip()
)

# Count matching titles
original_titles = set(anime["match_name"])
synopsis_titles = set(synopsis["match_name"])

matched_titles = original_titles & synopsis_titles

print("Anime in original dataset:", len(original_titles))
print("Unique synopsis titles:", len(synopsis_titles))
print("Titles found in both datasets:", len(matched_titles))
print("Original titles without a synopsis match:",
      len(original_titles - synopsis_titles))


# In[19]:


# Keep one synopsis row per normalized title
synopsis_unique = synopsis.drop_duplicates(
    subset="match_name"
).copy()

# Keep only titles that exist in the original dataset
matched_synopsis = synopsis_unique[
    synopsis_unique["match_name"].isin(matched_titles)
].copy()

print("Matched anime:", len(matched_synopsis))


# In[20]:


# Map original anime titles to their row positions
original_index = {
    name: i for i, name in enumerate(anime["match_name"])
}

# Map synopsis titles to their row positions in the story matrix
story_index = {
    name: i for i, name in enumerate(matched_synopsis["match_name"])
}


# In[21]:


story_tfidf_v2 = TfidfVectorizer(
    stop_words="english",
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_df=0.85
)

story_matrix_v2 = story_tfidf_v2.fit_transform(
    matched_synopsis["synopsis"]
)

print("New matrix shape:", story_matrix_v2.shape)
print("New vocabulary size:",
      len(story_tfidf_v2.get_feature_names_out()))


# In[22]:


story_tfidf_char = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=30000,
    min_df=2,
    sublinear_tf=True,
    max_df=0.90
)

story_matrix_char = story_tfidf_char.fit_transform(
    matched_synopsis["synopsis"]
)

print("Word model shape:", story_matrix_v2.shape)
print("Character model shape:", story_matrix_char.shape)


# 
# ## 8. Franchise Processing
# 

# In[23]:


import html
import re

def clean_title(title):
    title = html.unescape(str(title)).casefold().strip()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    return re.sub(r"\s+", " ", title).strip()

def get_franchise_key(title):
    title = clean_title(title)

    for franchise in sorted(franchise_names, key=len, reverse=True):
        if title == franchise or title.startswith(franchise + " "):
            return franchise

    return title


# In[24]:


franchise_names = [
    "one piece",
    "death note",
    "naruto",
    "bleach",
    "dragon ball",
    "attack on titan",
    "shingeki no kyojin",
    "fullmetal alchemist",
    "sword art online",
    "my hero academia",
    "detective conan",
    "gintama",
    "haikyuu",
    "clannad",
    "code geass",
    "hunter x hunter",
    "steins gate",
    "toriko",
    "digimon",
    "katekyo hitman reborn",
    "tsubasa"
]


# In[25]:


for title in [
    "Shingeki no Kyojin",
    "Shingeki no Kyojin OVA",
    "Shingeki no Kyojin Movie 1: Guren no Yumiya",
    "Toriko",
    "Toriko: Jump Super Anime Tour 2009 Special",
]:
    print(title, "->", get_franchise_key(title))


# 
# ## 9. Anime Recommendation Engine
# 

# In[26]:


anime_aliases = {
    "attack on titan": "Shingeki no Kyojin",
    "your name": "Kimi no Na wa."
}


# In[27]:


# Map each matched anime to its row in the original anime dataset
candidate_names = matched_synopsis["match_name"].tolist()

candidate_original_indices = np.array([
    original_index[name] for name in candidate_names
])

# Add franchise labels to the matched synopsis dataset
matched_synopsis["franchise"] = (
    matched_synopsis["Name"].apply(get_franchise_key)
)

def recommend_anime_modes(title, top_n=10, mode="both"):
    requested_title=title.casefold().strip()
    canonical_title = anime_aliases.get(requested_title,title)
    key = canonical_title.casefold().strip()

    if key not in original_index or key not in story_index:
        print(f"'{title}' must exist in both datasets.")
        return

    original_idx = original_index[key]
    story_idx = story_index[key]
    target_franchise = get_franchise_key(canonical_title)

    # Genre similarity
    genre_scores = cosine_similarity(
        tfidf_matrix[original_idx], tfidf_matrix
    ).flatten()

    # Combined word + character text similarity
    word_scores = cosine_similarity(
        story_matrix_v2[story_idx], story_matrix_v2
    ).flatten()

    char_scores = cosine_similarity(
        story_matrix_char[story_idx], story_matrix_char
    ).flatten()

    story_scores = 0.70 * word_scores + 0.30 * char_scores

    # Align candidates with the original anime dataset
    candidate_genres = genre_scores[candidate_original_indices]

    candidate_ratings = anime.loc[
        candidate_original_indices, "normalized_rating"
    ].fillna(0).to_numpy()

    # Hybrid score: 40% genre + 40% text + 20% rating
    hybrid_scores = (
        0.40 * candidate_genres
        + 0.40 * story_scores
        + 0.20 * candidate_ratings
    )

    candidates = matched_synopsis.copy()
    candidates["genre_similarity"] = candidate_genres
    candidates["story_similarity"] = story_scores
    candidates["hybrid_score"] = hybrid_scores
    candidates["original_idx"] = candidate_original_indices

    # Exclude the exact anime being searched
    candidates = candidates[
        candidates["match_name"] != key
    ].copy()

    if mode == "related":
        candidates = candidates[
            candidates["franchise"] == target_franchise
        ]
    elif mode == "discover":
        candidates = candidates[
            candidates["franchise"] != target_franchise
        ]
    elif mode != "both":
        print("Choose mode='related', 'discover', or 'both'.")
        return

    def top_results(frame):
        frame = frame.sort_values(
            "hybrid_score", ascending=False
        ).head(top_n)

        return frame[
            [
                "Name", "Genres", "Score",
                "genre_similarity", "story_similarity",
                "hybrid_score"
            ]
        ].reset_index(drop=True).round(3)

    if mode == "both":
        related = top_results(candidates[candidates["franchise"] == target_franchise])

        discover = top_results(candidates[candidates["franchise"] != target_franchise])
        return related, discover
    return top_results(candidates)


# In[28]:


def recommend_diverse_anime(title, top_n=10):
    results = recommend_anime_modes(
        title, top_n=100, mode="discover"
    )

    if results is None or results.empty:
        return results

    results = results.copy()
    results["franchise"] = results["Name"].apply(get_franchise_key)

    # Sort by hybrid score and keep the best anime per franchise
    results = results.sort_values(
        "hybrid_score", ascending=False
    )

    diverse = results.drop_duplicates(
        subset="franchise", keep="first"
    )

    # If there aren't enough unique franchises, fill remaining slots
    if len(diverse) < top_n:
        remaining = results[
            ~results.index.isin(diverse.index)
        ]
        diverse = pd.concat([
            diverse,
            remaining.head(top_n - len(diverse))
        ])

    return diverse.head(top_n).drop(
        columns="franchise"
    ).reset_index(drop=True)
