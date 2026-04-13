# Recommender System (Collaborative Filtering)

A recommender system built on a real-world user–item ratings matrix to **predict missing ratings** and compare three collaborative-filtering approaches:

1. **Global Baseline Estimate** (global mean + user bias + item bias)
2. **User–User Collaborative Filtering** (cosine similarity between users)
3. **Item–Item Collaborative Filtering** (cosine similarity between items)

The project includes data preprocessing, exploratory data analysis (EDA), model implementation, and evaluation using **RMSE**.

---

## Project Workflow

### 1) Data Loading & Preprocessing
- The ratings dataset is loaded from: `ratings_matrix_large.csv`
- Missing ratings are kept as **NaN** so the models can predict them later.
- Basic sanity checks are performed to validate:
  - duplicates / invalid entries
  - consistency of rows (users) and columns (items)
  - overall data integrity

**Outcome:** a clean user–item matrix ready for analysis and prediction.

---

### 2) Exploratory Data Analysis (EDA)
Before modeling, visualizations are produced to understand rating behavior and sparsity patterns:

#### Rating Distribution
- A histogram of all **non-null** ratings is plotted.
- Observation: ratings are concentrated in the **mid-to-high** range, suggesting generally positive feedback.

#### User Activity
- A scatter plot shows the number of ratings per user.
- Observation: user activity follows a **long-tail** pattern—some users rate many items, many users rate only a few.
- Implication: sparse users can reduce similarity-based model quality (cold-start effect).

#### Item Popularity
- A bar chart counts ratings per item.
- Observation (per report): items are interacted with relatively evenly in this dataset.

---

## Models Implemented

### 3) Global Baseline Estimate
A simple but strong baseline using global and per-entity biases:

**Prediction formula**
\[
\hat{r}_{ui} = \mu + b_u + b_i
\]

Where:
- \(\mu\) = global mean rating
- \(b_u\) = user bias (user average − global mean)
- \(b_i\) = item bias (item average − global mean)

**Why it matters:** computationally efficient and a strong benchmark for collaborative filtering.

---

### 4) User–User Collaborative Filtering (Cosine Similarity)
- Compute cosine similarity between users using their rating vectors.
- For each missing user–item rating:
  - find the most similar users who rated that item
  - compute a similarity-weighted average of their ratings

**Prediction concept**
- Weighted average from **top similar users**.

**Strengths / limitations**
- Works well when users have overlapping rating histories.
- Can struggle when user histories are sparse.

---

### 5) Item–Item Collaborative Filtering (Cosine Similarity)
- Compute cosine similarity between items.
- For each missing user–item rating:
  - look at items the user already rated
  - compute a similarity-weighted average using the most similar items

**Prediction concept**
- Weighted average from the user’s ratings on **similar items**.

**Strengths / limitations**
- Often more stable in practice since item similarity can be more consistent than user similarity.
- Performance can vary depending on rating density and item coverage.

---

## Evaluation

### 6) Metric: RMSE on Previously Missing Entries
Models are evaluated using **Root Mean Square Error (RMSE)**.
To ensure fairness, evaluation is performed **only on the entries that were originally missing** (NaN), i.e., predicted targets.

**Results (from report):**
- **Global Baseline RMSE:** 1.28  
- **User–User CF RMSE:** 1.37  
- **Item–Item CF RMSE:** 1.57  

A bar plot is used to visually compare RMSE values across models.

**Conclusion from results:** the **Global Baseline Estimate** achieved the lowest RMSE in this experiment.

---

## Key Learnings
- The **Global Baseline** method is simple, fast, and performed best here—useful as both a baseline and a competitive option.
- **User–User CF** can capture preference similarity but is sensitive to sparse users and cold-start behavior.
- **Item–Item CF** is often more consistent due to stable item relationships, though it did not outperform the baseline in this run.
- EDA visualizations meaningfully informed model understanding and interpretation.
- Evaluating only the originally missing values avoids “inflated” performance estimates.

---

## Future Work
- Add matrix factorization methods (e.g., **SVD**) to learn latent factors.
- Use implicit feedback (clicks/views) to enrich user signals.
- Add temporal dynamics to model preference drift over time.
- Explore hybrid recommenders (collaborative + content-based).
- Optimize scalability for production-scale datasets.

---

## How to Run
1. Open the project notebook(s) in Jupyter.
2. Ensure dependencies are installed (commonly: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`).
3. Place `ratings_matrix_large.csv` in the expected directory (same folder as the notebook unless otherwise configured).
4. Run cells in order:
   - load + preprocess
   - EDA + plots
   - baseline model
   - user–user CF
   - item–item CF
   - RMSE evaluation and comparison plot
