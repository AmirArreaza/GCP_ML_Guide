# Matrix Factorization — Recommendation Engine in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Matrix Factorization?

Matrix Factorization is a **collaborative filtering** technique that decomposes a large user-item interaction matrix into two smaller latent factor matrices. From these, it reconstructs predicted ratings or interaction scores for user-item pairs that have never been observed.

```
R ≈ U × V^T

R = user-item interaction matrix  (m users × n items)
U = user latent factor matrix     (m users × k factors)
V = item latent factor matrix     (n items × k factors)
k = number of latent factors (embedding dimensions)
```

**Use cases:** Product recommendations, movie recommendations, music playlists, content personalisation, ad targeting, similar-item discovery.

---

## 2. How Collaborative Filtering Works

```
Explicit feedback  →  user directly rates items (stars, scores)
Implicit feedback  →  inferred from behaviour (clicks, views, purchases)
```

The model finds **latent patterns** — users and items are represented as vectors in a shared embedding space. Users close to items in this space are predicted to have high affinity.

- **User-User CF:** finds users with similar taste
- **Item-Item CF:** finds items liked by similar users
- **Matrix Factorisation:** learns both simultaneously via embedding vectors

---

## 3. Full Workflow

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_mf_model`
OPTIONS (
  model_type              = 'matrix_factorization',
  user_col                = 'user_id',          -- user identifier column
  item_col                = 'item_id',          -- item identifier column
  rating_col              = 'rating',           -- explicit rating (optional)
  feedback_type           = 'EXPLICIT',         -- 'EXPLICIT' or 'IMPLICIT'
  num_factors             = 16,                 -- k: latent embedding dimensions
  num_trials              = 4,                  -- Vertex AI Vizier HPO trials (optional)
  l2_reg                  = 9.83,               -- regularization on embedding weights
  learn_rate              = 0.01,               -- step size for gradient descent
  max_iterations          = 20,                 -- training iterations (ALS rounds)
  wals_alpha              = 40                  -- confidence scaling for IMPLICIT only
) AS
SELECT
  user_id,
  item_id,
  rating                                        -- omit for IMPLICIT feedback
FROM
  `project.dataset.interactions_table`;
```

### IMPLICIT Feedback Example

```sql
CREATE OR REPLACE MODEL `project.dataset.my_implicit_mf`
OPTIONS (
  model_type    = 'matrix_factorization',
  user_col      = 'user_id',
  item_col      = 'product_id',
  feedback_type = 'IMPLICIT',       -- no rating column needed
  num_factors   = 32,
  wals_alpha    = 40,               -- confidence = 1 + wals_alpha * count
  l2_reg        = 1.0,
  max_iterations = 20
) AS
SELECT
  user_id,
  product_id,
  purchase_count             -- used as implicit confidence signal
FROM
  `project.dataset.purchase_history`;
```

### Step 2 — Evaluate the Model

```sql
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.my_mf_model`);
```

> For **EXPLICIT** feedback: returns standard regression metrics (MSE, MAE, etc.) on predicted vs actual ratings.
> For **IMPLICIT** feedback: evaluation is less straightforward — mean average precision is often computed manually.

### Step 3 — Generate Recommendations

```sql
-- Recommend top 5 items for every user
SELECT *
FROM ML.RECOMMEND(
  MODEL `project.dataset.my_mf_model`
)
LIMIT 1000;

-- Recommend top 5 items for specific users only
SELECT *
FROM ML.RECOMMEND(
  MODEL `project.dataset.my_mf_model`,
  (SELECT DISTINCT user_id FROM `project.dataset.active_users`)
);
```

**Output columns:**

| Column | Description |
|--------|-------------|
| `user_id` | The user identifier |
| `item_id` | The recommended item identifier |
| `predicted_rating` | Estimated rating / affinity score |

### Step 4 — Predict Rating for Specific Pairs

```sql
-- Score specific user-item pairs
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.my_mf_model`,
  (
    SELECT user_id, item_id
    FROM `project.dataset.candidate_pairs`
  )
);
```

### Step 5 — Retrieve Learned Embeddings

```sql
-- Get user and item embedding vectors
SELECT *
FROM ML.WEIGHTS(MODEL `project.dataset.my_mf_model`)
WHERE processed_input = 'user_id';   -- or 'item_id'
```

Embeddings can be exported and used in downstream nearest-neighbour search (e.g. with ScaNN or Vertex AI Matching Engine).

---

## 4. EXPLICIT vs IMPLICIT Feedback

| Aspect | EXPLICIT | IMPLICIT |
|--------|---------|---------|
| `feedback_type` | `'EXPLICIT'` | `'IMPLICIT'` |
| `rating_col` required | Yes — actual numeric rating | No — interaction count used |
| Algorithm | ALS with ratings | WALS (Weighted ALS) |
| `wals_alpha` | Not used | Required — confidence scaling |
| Evaluation metric | RMSE, MAE on ratings | Harder — often manual MAP/NDCG |
| Data examples | Star ratings, explicit scores | Clicks, views, purchases, plays |
| Cold start issue | Worse (needs ratings) | Better (any interaction counts) |

### WALS Confidence Formula

For implicit feedback, the confidence that user u likes item i is:

```
c(u,i) = 1 + wals_alpha × r(u,i)

where r(u,i) = observed interaction count (e.g. number of purchases)
wals_alpha   = confidence scaling factor (default 40)
```

Higher `wals_alpha` → model places more weight on high-count interactions.

---

## 5. Key OPTIONS Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `model_type` | — | `'matrix_factorization'` — required |
| `user_col` | — | User identifier column — required |
| `item_col` | — | Item identifier column — required |
| `rating_col` | — | Numeric rating column — required for EXPLICIT |
| `feedback_type` | `'EXPLICIT'` | `'EXPLICIT'` or `'IMPLICIT'` |
| `num_factors` | `8` | Latent embedding dimensions (k) |
| `max_iterations` | `20` | ALS / WALS training rounds |
| `learn_rate` | auto | Step size for stochastic gradient descent |
| `l2_reg` | auto | L2 regularization on embedding weights |
| `wals_alpha` | `40` | Confidence scaling for IMPLICIT feedback only |
| `num_trials` | `1` | HPO trials via Vertex AI Vizier |
| `feedback_type` | `'EXPLICIT'` | Type of interaction signal |

---

## 6. Evaluation Metrics

### For EXPLICIT Feedback (ML.EVALUATE)

| Metric | Description | Goal |
|--------|-------------|------|
| `mean_absolute_error` | Average absolute rating error | Lower |
| `mean_squared_error` | Average squared rating error | Lower |
| `root_mean_squared_error` | RMSE — same units as rating | Lower |
| `r2_score` | Variance explained | Higher |

### For IMPLICIT Feedback (manual computation)

BQML does not natively compute ranking metrics. You must calculate these manually after using `ML.RECOMMEND` or `ML.PREDICT`:

| Metric | Description |
|--------|-------------|
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of relevant items found in top-K |
| **MAP** | Mean Average Precision across all users |
| **NDCG** | Normalised Discounted Cumulative Gain — rewards relevant items ranked higher |

---

## 7. ML Functions — Quick Reference

| Function | Description |
|----------|-------------|
| `ML.EVALUATE` | Rating error metrics for EXPLICIT; limited for IMPLICIT |
| `ML.RECOMMEND` | Generate top-N recommendations per user |
| `ML.PREDICT` | Score specific user-item pairs |
| `ML.WEIGHTS` | Retrieve learned user and item embedding vectors |

---

## 8. Cold Start Problem

**Cold start** occurs when new users or items have no interaction history.

| Type | Problem | Mitigation |
|------|---------|------------|
| New user | No user vector — cannot personalise | Use popularity-based fallback or content-based features |
| New item | No item vector — never recommended | Use item metadata / content-based approach |
| Sparse data | Too few interactions overall | Reduce `num_factors`, increase regularization (`l2_reg`) |

> Matrix Factorization cannot solve cold start alone — content-based features or hybrid models are needed for new entities.

---

## 9. Latent Factors (num_factors)

`num_factors` (k) controls the size of the embedding space:

| `num_factors` value | Effect |
|---------------------|--------|
| Too small (e.g. 2–4) | Underfitting — model cannot capture complex preferences |
| Too large (e.g. 128+) | Overfitting — memorises training interactions |
| Typical range | 8–64 for most problems |
| Increase when | Many users/items, rich interaction data |
| Decrease when | Sparse data, limited compute budget |

---

## 10. Using Embeddings Downstream

The user and item vectors from `ML.WEIGHTS` can power:

```sql
-- Export user embeddings for nearest-neighbour search
SELECT
  processed_input,
  feature,
  weight
FROM ML.WEIGHTS(MODEL `project.dataset.my_mf_model`)
WHERE processed_input = 'user_id';
```

**Downstream uses:**
- **Vertex AI Matching Engine** — approximate nearest-neighbour (ANN) search at scale
- **ScaNN (Scalable Nearest Neighbors)** — fast similarity search
- **Item-item similarity** — find similar items by comparing item embeddings
- **User segmentation** — cluster users by their latent factor vectors

---

## 11. Exam-Relevant Tips

- `model_type = 'matrix_factorization'` — the only collaborative filtering model in BQML.
- Use **`ML.RECOMMEND`** to generate top-N items — not `ML.PREDICT` (though ML.PREDICT can score specific pairs).
- **`feedback_type = 'IMPLICIT'`** does not require a `rating_col` — uses interaction counts instead.
- **`wals_alpha`** is only relevant for `IMPLICIT` feedback — it scales confidence from interaction counts.
- **Cold start** cannot be solved by matrix factorisation alone — note this limitation for the exam.
- `num_factors` is the key hyperparameter — controls the embedding dimensionality.
- For **IMPLICIT** feedback, ranking metrics (MAP, NDCG, Precision@K) must be computed **manually** — BQML does not return them from `ML.EVALUATE`.
- Embeddings from `ML.WEIGHTS` can be exported to **Vertex AI Matching Engine** for scalable ANN retrieval.
- `l2_reg` prevents overfitting of embedding weights — especially important with sparse data.
- The underlying algorithm is **ALS (Alternating Least Squares)** for EXPLICIT and **WALS (Weighted ALS)** for IMPLICIT.

---

## 12. Quick Reference Cheat Sheet

```
model_type       →  'matrix_factorization'
user_col         →  user identifier column (required)
item_col         →  item identifier column (required)
rating_col       →  required for EXPLICIT; omit for IMPLICIT
feedback_type    →  'EXPLICIT' (ratings) or 'IMPLICIT' (counts)
wals_alpha       →  confidence scaling — IMPLICIT only
num_factors      →  embedding dimensions k (default 8)
ML.RECOMMEND     →  generate top-N recommendations per user
ML.PREDICT       →  score specific user-item pairs
ML.EVALUATE      →  RMSE/MAE (EXPLICIT); limited (IMPLICIT)
ML.WEIGHTS       →  retrieve user and item embedding vectors
```

---

*Study tip: Know the difference between EXPLICIT and IMPLICIT feedback, when to use `ML.RECOMMEND` vs `ML.PREDICT`, and why ranking metrics like Precision@K and NDCG must be computed manually for implicit feedback — all common exam scenarios.*