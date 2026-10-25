# Building an Anime Recommendation System with Python, Pandas, NumPy, and Scikit-learn

### My journey from data cleaning to content-based recommendations

## Introduction

As I continued learning Python, Pandas, and NumPy, I wanted to build a project that would allow me to apply what I had learned rather than simply complete tutorials.

I decided to build an **Anime Recommendation System**.

The idea was simple: a user enters the name of an anime, and the system recommends other anime that they might enjoy.

However, as I started working with the data, I realized that building a recommendation system involves much more than finding titles with similar genres. The project introduced me to data cleaning, text preprocessing, feature extraction, cosine similarity, rating-based ranking, and the challenges of combining different recommendation signals.

My goal was not to build the most advanced recommendation engine possible. It was to build a useful project, understand each component, and learn scikit-learn along the way.

This blog documents that journey.

## 1. The Technology Stack

I chose tools that matched my current learning goals.

- **Python:** The programming language used to implement the project.
- **Pandas:** Used to load, clean, filter, merge, and manipulate datasets.
- **NumPy:** Used for numerical operations and working with arrays of recommendation scores.
- **Scikit-learn:** Used for text vectorization and similarity calculations.
- **Jupyter Notebook:** Used to experiment with the data, test individual components, and understand the results.
- **uv:** Used to manage Python packages and the development environment.

I deliberately kept the project focused on these tools instead of introducing a large machine-learning framework or a complicated backend.

## 2. Finding and Understanding the Dataset

The first step was obtaining anime data.

My primary dataset, `anime.csv`, contained approximately 12,294 records and included these columns:

| Column | Description |
|---|---|
| `anime_id` | Unique anime identifier |
| `name` | Anime title |
| `genre` | Genres associated with the anime |
| `type` | Format, such as TV or Movie |
| `episodes` | Number of episodes |
| `rating` | Available rating |
| `members` | Number of community members |

This dataset provided enough information to start building a content-based recommendation system.

However, the data was not perfectly clean. Some anime had missing genres, types, or ratings.

That meant I couldn't simply load the dataset and start calculating similarities.

## 3. Data Cleaning with Pandas

Data cleaning was one of the first important parts of the project.

I found missing values in several columns:

- 62 missing genre values.
- 25 missing type values.
- 230 missing rating values.

Instead of removing every row containing a missing value, I handled each column according to its purpose.

For genres, I replaced missing values with empty strings. For anime type, I used `"Unknown"` where the information was unavailable. I preserved missing ratings because an unknown rating should not automatically be treated as a real rating.

I also created a normalized title column:

```python
anime["genre"] = anime["genre"].fillna("")
anime["type"] = anime["type"].fillna("Unknown")

anime["match_name"] = (
    anime["name"]
    .str.casefold()
    .str.strip()
)
```

The `match_name` column made title matching more consistent by ignoring differences in capitalization and removing leading or trailing whitespace.

For example, `"Death Note"` and `"death note "` would now have the same normalized title.

This was important because the system needed to locate the selected anime reliably.

**Lesson learned:** Data cleaning is not just about removing missing values. It is about making the data usable without destroying information that might matter later.

## 4. Building a Genre-Based Recommendation System

My first recommendation approach used anime genres.

If a user enjoys an action-adventure anime, recommending other anime with similar genres is a reasonable starting point.

However, computers cannot directly calculate similarity between text labels in the way we calculate similarity between numbers. We need to represent the text numerically first.

### Introducing TF-IDF

I used `TfidfVectorizer` from scikit-learn.

TF-IDF stands for *Term Frequency–Inverse Document Frequency*. It converts text into numerical features, giving words different weights based on their occurrence and how common they are across the collection.

For my genre data, I used:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    token_pattern=r"[^,]+"
)

tfidf_matrix = tfidf.fit_transform(
    anime["genre"]
)

print(tfidf_matrix.shape)
```

The resulting matrix had the shape:

```text
(12294, 82)
```

This meant that the system represented 12,294 anime using 82 genre-related features.

The custom token pattern was necessary because genres were stored as comma-separated labels. I wanted the vectorizer to treat each genre as a complete token rather than split it into smaller pieces.

### Calculating Cosine Similarity

Once the genres had been converted into numerical vectors, I used cosine similarity to measure how similar two anime were.

Cosine similarity compares the angle between two vectors. A higher score generally indicates greater similarity in the represented features.

For example, two anime sharing several genres could receive a higher genre-similarity score than anime with completely different genres.

This gave me the foundation for ranking recommendations by genre.

**Important limitation:** Genre similarity is not the same as story similarity. Two anime can share genres but have completely different plots, characters, and themes.

That limitation became the motivation for the next stage.

## 5. Improving Recommendations with Story Information

Genres alone were not enough for the kind of recommendations I wanted.

For example, a person who enjoys a psychological thriller might want recommendations based on suspense, mystery, or the nature of the story—not simply because another anime shares the same genre labels.

To experiment with story-based recommendations, I obtained a second dataset named `anime_with_synopsis.csv`.

This dataset included anime titles, genres, ratings, and synopsis descriptions.

One small complication was that the synopsis column was named `sypnopsis`, rather than `synopsis`.

I renamed it and handled missing descriptions:

```python
synopsis = synopsis.rename(
    columns={"sypnopsis": "synopsis"}
)

synopsis["synopsis"] = (
    synopsis["synopsis"].fillna("")
)

synopsis["Name"] = (
    synopsis["Name"].str.strip()
)

synopsis["match_name"] = (
    synopsis["Name"]
    .str.casefold()
    .str.strip()
)
```

### Matching two datasets

The datasets did not contain exactly the same number of anime, and their titles were not always represented identically.

After normalizing the titles and matching the datasets, I found approximately 10,044 anime with matching records.

Around 2,248 anime in the original dataset did not have a matching synopsis under this title-matching approach.

I also removed duplicate normalized titles from the synopsis dataset so that one title would not unnecessarily create multiple candidate recommendations.

This taught me an important lesson: **combining datasets requires careful matching, and the number of rows in each dataset does not need to be the same.**

## 6. Text-Based Similarity Using TF-IDF

With the synopsis data available, I created a second recommendation signal: story-text similarity.

I experimented with two TF-IDF representations.

### Word-level TF-IDF

The word-level vectorizer used unigrams and bigrams, meaning it considered individual words and pairs of consecutive words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

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
```

Some of these settings had specific purposes:

- `stop_words="english"` removed common English words.
- `max_features=30000` limited the vocabulary size.
- `ngram_range=(1, 2)` considered individual words and two-word phrases.
- `min_df=2` ignored terms appearing in fewer than two documents.
- `sublinear_tf=True` reduced the influence of repeated occurrences of a term.
- `max_df=0.85` ignored terms appearing in more than 85% of documents.

These settings helped control the representation of the synopsis collection.

### Character-level TF-IDF

I also experimented with character-level TF-IDF using character n-grams.

```python
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
```

Instead of focusing only on complete words, this approach considered character sequences of lengths three to five within word boundaries.

I combined the word-level and character-level similarities:

```python
combined_scores = (
    0.70 * word_scores
    + 0.30 * char_scores
)
```

The 70/30 weighting was an experimental choice, not a mathematically proven optimum.

### What I learned from the results

When testing recommendations for *Death Note*, the highest combined text-similarity score was 1.0 for the anime itself. After excluding the selected anime, the strongest scores were much lower.

This was expected: the self-match is identical to itself, while other synopses contain different text.

I also noticed that some recommendations had low text-similarity scores even when they seemed plausible.

This exposed a limitation of TF-IDF: it detects patterns in shared vocabulary, but it does not truly understand story meaning.

Two descriptions can express similar ideas using different words and still receive a low similarity score. Conversely, descriptions with overlapping words may not actually describe similar plots.

**Lesson learned:** Text similarity is useful, but a numerical similarity score should not automatically be interpreted as proof that two stories are semantically alike.

## 7. Incorporating Ratings into Recommendations

Another question came up during development: should the system recommend only similar anime, or should it also consider quality and popularity?

I experimented with a weighted rating formula that combined an anime's rating with the average rating of the dataset.

The purpose was to prevent anime with relatively few community members from being ranked solely on the basis of their raw ratings.

The formula was:

\[
WR=\frac{v}{v+m}R+\frac{m}{v+m}C
\]

Where:

- \(R\) is the anime's average rating.
- \(v\) is its number of members.
- \(m\) is a chosen minimum-popularity threshold.
- \(C\) is the average rating across the dataset.

I selected the 75th percentile of the members column as the threshold for this experiment.

After calculating weighted ratings, I normalized the values to a range from zero to one so that they could be incorporated into a combined score.

Missing ratings needed special attention. An unrated anime should not be assigned an ordinary rating that it never received.

In the experimental ranking formula, missing normalized ratings were filled with zero. This was a simple fallback, but it also meant unrated anime could be penalized. A future version should explicitly compare different missing-rating strategies.

## 8. Combining Genres, Story Text, and Ratings

At this stage, I had three signals:

1. Genre similarity.
2. Story-text similarity.
3. Normalized rating.

I combined them into an experimental hybrid score:

\[
S=0.40G+0.40T+0.20R
\]

Where:

- \(G\) is genre similarity.
- \(T\) is combined word-and-character text similarity.
- \(R\) is the normalized rating.

The idea was to give genre and story similarity equal nominal weights, while allowing ratings to influence the final ranking.

However, the results revealed a subtle problem.

The text-similarity values were generally much smaller than the genre-similarity values. For example, a genre similarity of 0.90 contributes 0.36 to the final score, while a text similarity of 0.08 contributes only 0.032.

Even though both components had a weight of 40%, their actual numerical contributions could be very different.

This is why simply assigning equal weights does not guarantee that two signals will have equal influence.

A more rigorous version would evaluate the components separately, inspect their score distributions, and test alternative scaling or weighting methods against a set of relevant recommendations.

I kept the formula as an experimental baseline rather than claiming that it was optimal.

## 9. Separating Related Anime from New Discoveries

I wanted the system to support two different recommendation goals.

**Related anime:** Find other entries from the same franchise, including movies, specials, OVAs, and related releases.

**Discover new anime:** Recommend anime from different franchises that may appeal to the user.

To support this, I created a simple franchise-key function.

It normalized titles and used a list of known franchise names to group entries. For example, titles beginning with the known franchise name `One Piece` could be grouped together.

This helped distinguish another *One Piece* release from an entirely new anime recommendation.

However, I learned that franchise detection based on title prefixes is only a practical heuristic. It cannot reliably identify every sequel, spin-off, alternate title, or shared universe.

I had to update the known-franchise list when I discovered cases that were not grouped correctly, including titles related to *Tsubasa*.

This was a useful reminder that real-world data rarely follows perfectly consistent naming conventions.

### Reducing repetitive recommendations

I also created a function that selected the highest-ranked anime from each franchise before filling the remaining recommendation slots.

This encouraged diversity in the final list.

Without this step, the top ten recommendations could contain many entries from only a few franchises, especially when the input anime belonged to a popular series.

The diversity filter was a practical improvement, but it was not a guarantee that the resulting list would be the most relevant for every user.

## 10. An Unexpected Development Challenge: Python Environments

The project also taught me something beyond recommendation algorithms: Python environment management.

My notebook was using a virtual environment located in a different project folder.

I checked the active Python executable with:

```python
import sys
print(sys.executable)
```

It showed that the notebook was using the Python interpreter inside my NumPy practice project's virtual environment.

When I tried installing a package from the anime project directory, I ran:

```powershell
uv pip install scikit-learn
```

The command failed with an error indicating that no virtual environment could be found.

The problem was not necessarily that scikit-learn was unavailable. The command was looking for an environment associated with the current working directory, while my notebook was using an environment elsewhere.

I resolved the installation-target issue by explicitly specifying the Python interpreter:

```powershell
uv pip install --python "D:\Codings\Python Codes\NumPy\Practicing NumPy\numpy-100\.venv\Scripts\python.exe" sentence-transformers
```

This taught me to distinguish between the environment used by a notebook and the environment targeted by a terminal command.

In a future version, I would give the anime project its own virtual environment. This would make dependencies easier to manage and reduce the risk of one project affecting another.

## 11. Exploring Semantic Embeddings—and Knowing When to Stop

After experimenting with word-level and character-level TF-IDF, I explored sentence-transformer models.

Unlike ordinary TF-IDF, a pretrained sentence-transformer can generate dense numerical embeddings designed to capture aspects of textual meaning.

I successfully loaded the `all-MiniLM-L6-v2` model.

However, this introduced another decision: should I keep adding increasingly advanced techniques, or should I focus on understanding and completing the project with the tools I was already learning?

I chose to keep semantic embeddings as a potential future improvement.

The reason was straightforward: a more sophisticated model is not automatically a better learning choice. It introduces additional dependencies, model downloads, computation, and concepts that deserve their own study.

My immediate objective was to learn Pandas, NumPy, and scikit-learn through a project I could explain confidently.

I did not need another model just to make the project appear more advanced.

**This was an important project-management lesson: knowing what not to build is part of building a good project.**

## 12. What I Learned

This project helped me connect several concepts I had been learning independently.

### Pandas

I practiced:

- Loading and inspecting datasets.
- Handling missing values.
- Normalizing strings.
- Removing duplicate records.
- Matching records across datasets.
- Filtering and sorting recommendations.
- Creating new columns from existing data.

### NumPy

I used arrays to represent and manipulate similarity scores, align scores between datasets, and work with numerical ranking calculations.

### Scikit-learn

I explored:

- TF-IDF vectorization.
- Word and character n-grams.
- Sparse feature matrices.
- Cosine similarity.
- Text preprocessing through vectorizer settings.

### Beyond the libraries

I also learned about:

- Choosing an appropriate recommendation strategy.
- The limitations of text-based similarity.
- The difference between similarity and recommendation quality.
- Combining multiple scoring signals.
- Missing ratings and data alignment.
- Python environments and dependency management.
- The importance of inspecting results instead of assuming that working code produces good recommendations.

## 13. Current Limitations

The project is a learning-oriented, content-based recommender, and it has several limitations.

**No user preference history:** It does not learn from a user's previous ratings, clicks, or watch history.

**No true semantic understanding in the TF-IDF models:** The text component relies on statistical text features rather than a deep understanding of plot meaning.

**Limited franchise detection:** The franchise rules depend on known names and title patterns, so unusual titles may be grouped incorrectly.

**Experimental hybrid weights:** The 40/40/20 weighting has not been established as optimal through systematic evaluation.

**Dataset limitations:** Missing ratings, unavailable synopses, duplicate records, and title mismatches can affect recommendations.

**No formal recommendation evaluation yet:** Plausible-looking recommendations are encouraging, but a stronger evaluation would require a defined test set, relevance judgments, or suitable ranking metrics.

Recognizing these limitations is part of understanding what the project does and what it does not do.

## 14. Future Improvements

Rather than implementing every possible improvement immediately, I would develop the project in stages.

1. **Improve the existing model:** Review the recommendation logic, simplify the code, and make the results easier to understand.
2. **Evaluate recommendations:** Test different anime titles and compare genre, text, and hybrid rankings.
3. **Improve missing-data handling:** Avoid unintentionally penalizing anime solely because their ratings are unavailable.
4. **Build a simple interface:** Allow a user to enter an anime title and view related anime and new discoveries.
5. **Explore semantic embeddings:** Revisit sentence-transformers after understanding the current system and establishing a baseline for comparison.
6. **Consider collaborative filtering:** If suitable user-rating or interaction data becomes available, explore recommendations based on the behavior of similar users.

These improvements would build on the current project rather than replacing it with a completely different system.

## Conclusion

Building this Anime Recommendation System turned my learning of Python libraries into a practical exercise.

I started with a dataset and a straightforward idea. Along the way, I handled missing values, matched two datasets, converted text into numerical features, calculated similarities, incorporated ratings, and experimented with different recommendation strategies.

I also encountered problems that did not involve machine learning at all, such as mismatched titles and Python environment configuration.

Most importantly, I learned that building a project is not simply about adding more libraries or more advanced algorithms. It is about understanding the data, making sensible decisions, examining the output, and being honest about the limitations.

The current system is a foundation, not a finished research-grade recommendation engine. My next priority is to make it cleaner, understand every component, and evaluate how useful its recommendations actually are.

I'm continuing to learn Pandas, NumPy, and scikit-learn, and this project is one way I'm putting those skills into practice.

**The goal isn't to build the most complicated model. It's to build something useful, understand how it works, and improve it one step at a time.**
