# Regression with Boosted Trees in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What are Boosted Trees?

Boosted Trees (also known as **Gradient Boosted Decision Trees / GBDT**) are an ensemble method that builds many decision trees **sequentially**, where each new tree corrects the errors of the previous ones. BigQuery ML implements **XGBoost** under the hood.

**Key idea:** Unlike Random Forests (parallel trees), boosting trains trees one after another — each tree learns from the **residual errors** of the ensemble so far.

```
Final prediction = Tree₁ + Tree₂ + Tree₃ + ... + Treeₙ
                   (each tree corrects the previous residuals)
```

**Use cases:** Price prediction, demand forecasting, revenue estimation — particularly when relationships between features and labels are **non-linear** or **complex**.

---

## 2. Boosted Trees vs Linear Regression

| Aspect | Linear Regression | Boosted Trees |
|--------|------------------|---------------|
| Relationship | Linear only | Non-linear, complex |
| Feature engineering | Often required | Less required |
| Interpretability | High | Lower (black box) |
| Overfitting risk | Low | Higher (needs tuning) |
| Performance on tabular data | Moderate | Often superior |
| Training speed | Fast | Slower |
| Hyperparameter tuning | Minimal | Critical |

---

## 3. Full Workflow in BigQuery ML

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_boosted_tree_model`
OPTIONS (
  model_type = 'boosted_tree_regressor',
  input_label_cols = ['label_column'],
  num_parallel_tree = 1,         -- trees per boosting round
  num_leaves = 31,               -- max leaves per tree
  max_tree_depth = 6,            -- max depth of each tree
  n_estimators = 50,             -- number of boosting rounds
  learn_rate = 0.1,              -- shrinkage rate (eta)
  subsample = 0.8,               -- row sampling per tree
  colsample_bytree = 0.8,        -- feature sampling per tree
  min_tree_child_weight = 1,     -- min sum of weights in a leaf
  l1_reg = 0.0,                  -- L1 regularization on weights
  l2_reg = 1.0,                  -- L2 regularization on weights
  data_split_method = 'AUTO_SPLIT',
  early_stop = TRUE,
  min_rel_progress = 0.01
) AS
SELECT
  feature1,
  feature2,
  feature3,
  label_column
FROM
  `project.dataset.training_table`
WHERE
  label_column IS NOT NULL;
```

### Step 2 — Evaluate the Model

```sql
SELECT *
FROM ML.EVALUATE(
  MODEL `project.dataset.my_boosted_tree_model`,
  (
    SELECT feature1, feature2, feature3, label_column
    FROM `project.dataset.evaluation_table`
  )
);
```

**Key metrics returned (same as linear regression):**

| Metric | Description | Goal |
|--------|-------------|------|
| `mean_absolute_error` | Average absolute difference | Lower is better |
| `mean_squared_error` | Average squared difference | Lower is better |
| `mean_squared_log_error` | Log-scaled MSE | Lower is better |
| `r2_score` | Variance explained (0–1) | Closer to 1 is better |
| `explained_variance` | Proportion of variance explained | Higher is better |

### Step 3 — Feature Importance

```sql
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.my_boosted_tree_model`);
```

Returns **importance scores** for each feature — higher score means the feature contributed more to splits across all trees.

Also available:

```sql
-- Global Shapley-based explanation
SELECT *
FROM ML.GLOBAL_EXPLAIN(MODEL `project.dataset.my_boosted_tree_model`);
```

### Step 4 — Make Predictions

```sql
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.my_boosted_tree_model`,
  (
    SELECT feature1, feature2, feature3
    FROM `project.dataset.new_data`
  )
);
```

Output includes `predicted_<label_column>`.

---

## 4. Key `OPTIONS` Parameters

### Core Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `model_type` | — | `'boosted_tree_regressor'` — **required** |
| `input_label_cols` | — | Target column name — **required** |
| `n_estimators` | `100` | Number of boosting rounds (trees) |
| `max_tree_depth` | `6` | Max depth of each tree |
| `num_leaves` | `31` | Max number of leaves per tree |
| `learn_rate` | `0.3` | Shrinkage rate (eta); lower = more robust, slower |

### Sampling Parameters (Anti-overfitting)

| Option | Default | Description |
|--------|---------|-------------|
| `subsample` | `1.0` | Row sampling fraction per tree (0–1) |
| `colsample_bytree` | `1.0` | Feature sampling fraction per tree (0–1) |
| `colsample_bylevel` | `1.0` | Feature sampling fraction per depth level |
| `colsample_bynode` | `1.0` | Feature sampling fraction per split node |
| `min_tree_child_weight` | `1` | Minimum sum of weights in a leaf node |

### Regularization Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `l1_reg` | `0` | L1 regularization on leaf weights |
| `l2_reg` | `1.0` | L2 regularization on leaf weights |

### Early Stopping Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `early_stop` | `TRUE` | Stop if no improvement on eval set |
| `min_rel_progress` | `0.01` | Min relative improvement to continue |
| `num_parallel_tree` | `1` | Trees per round (>1 = Random Forest mode) |

### Data Split Parameters

| Option | Values | Description |
|--------|--------|-------------|
| `data_split_method` | `AUTO_SPLIT` / `RANDOM` / `SEQ` / `CUSTOM` / `NO_SPLIT` | How to split training vs eval |
| `data_split_eval_fraction` | `0.2` | Fraction used for eval |

---

## 5. How Gradient Boosting Works

```
Round 1: Train Tree₁ on original labels
         → compute residuals (actual - predicted)

Round 2: Train Tree₂ on residuals from Round 1
         → compute new residuals

Round 3: Train Tree₃ on residuals from Round 2
         ...repeat for n_estimators rounds

Final:   F(x) = η·T₁(x) + η·T₂(x) + ... + η·Tₙ(x)
         where η = learn_rate (shrinkage)
```

**Shrinkage (learn_rate):** Scales each tree's contribution, making the model more conservative and robust. Lower values need more trees (`n_estimators`).

---

## 6. Overfitting Control — Key Levers

| Problem | Solution |
|---------|----------|
| Trees too complex | Reduce `max_tree_depth` or `num_leaves` |
| Too many rounds | Reduce `n_estimators` or enable `early_stop` |
| Memorising training data | Increase `subsample` < 1 or `colsample_bytree` < 1 |
| Large leaf weights | Increase `l2_reg` or `l1_reg` |
| Learning too fast | Decrease `learn_rate` (and increase `n_estimators`) |
| Sparse leaves | Increase `min_tree_child_weight` |

---

## 7. Hyperparameter Tuning with BQML

BigQuery ML supports **automatic hyperparameter tuning** via Vertex AI:

```sql
CREATE OR REPLACE MODEL `project.dataset.my_tuned_model`
OPTIONS (
  model_type = 'boosted_tree_regressor',
  input_label_cols = ['label_column'],
  num_trials = 10,
  max_parallel_trials = 2,
  hparam_tuning_objectives = ['mean_absolute_error'],
  hparam_range_override = JSON '{"learn_rate": {"min": 0.01, "max": 0.3},
                                  "max_tree_depth": {"min": 3, "max": 8}}'
) AS
SELECT * FROM `project.dataset.training_table`;
```

| Option | Description |
|--------|-------------|
| `num_trials` | Number of hyperparameter combinations to try |
| `max_parallel_trials` | Trials to run in parallel |
| `hparam_tuning_objectives` | Metric to minimise/maximise |
| `hparam_range_override` | Custom search space as JSON |

---

## 8. Feature Preprocessing

Boosted trees are **scale-invariant** — no need to standardise numeric features. However, BQML still auto-handles:

- **NULLs** → imputed (mean for numeric, empty string for categorical)
- **Categorical strings** → ordinal encoding internally (not one-hot for trees)

You can still use `TRANSFORM` for custom feature engineering:

```sql
CREATE OR REPLACE MODEL `project.dataset.my_model`
TRANSFORM(
  feature1,
  ML.BUCKETIZE(feature2, [10, 50, 100]) AS bucketed_f2,
  EXTRACT(DAYOFWEEK FROM date_col) AS day_of_week,
  label_column
)
OPTIONS (
  model_type = 'boosted_tree_regressor',
  input_label_cols = ['label_column']
)
AS SELECT * FROM `project.dataset.training_table`;
```

---

## 9. num_parallel_tree — Boosting vs Random Forest Mode

| `num_parallel_tree` | Behaviour |
|--------------------|-----------|
| `1` (default) | Standard gradient boosting (sequential trees) |
| `> 1` | Each boosting round builds N parallel trees → acts like **Random Forest** within each boosting round |

Setting `num_parallel_tree > 1` with `n_estimators = 1` essentially gives you a **pure Random Forest**.

---

## 10. Exam-Relevant Tips

- `model_type = 'boosted_tree_regressor'` for regression; use `'boosted_tree_classifier'` for classification.
- **Boosted trees do NOT require feature scaling** — they are split-based, not gradient-based on raw values.
- `ML.FEATURE_IMPORTANCE` is specific to tree models — not available for linear regression.
- **Early stopping** (`early_stop = TRUE`) requires a validation set — ensure `data_split_method ≠ 'NO_SPLIT'`.
- Lower `learn_rate` + higher `n_estimators` = more robust but slower training.
- `subsample < 1` introduces **stochastic gradient boosting** — reduces overfitting.
- **XGBoost** is the underlying engine in BigQuery ML for boosted trees.
- `num_parallel_tree > 1` + `n_estimators = 1` ≈ **Random Forest Regressor**.
- Hyperparameter tuning via `num_trials` uses **Vertex AI Vizier** under the hood.
- The `TRANSFORM` clause is **applied automatically at prediction time** — you don't re-apply it manually.

---

## 11. Quick Reference Cheat Sheet

```
CREATE MODEL          →  train with model_type = 'boosted_tree_regressor'
ML.EVALUATE           →  MAE, MSE, MSLE, R², explained_variance
ML.FEATURE_IMPORTANCE →  per-feature importance scores (tree-specific)
ML.GLOBAL_EXPLAIN     →  Shapley-based global feature attribution
ML.PREDICT            →  generate predicted_<label> column
```

---

## 12. Comparison: Linear Regression vs Boosted Trees OPTIONS

| Option | Linear Regression | Boosted Trees |
|--------|------------------|---------------|
| `model_type` | `linear_reg` | `boosted_tree_regressor` |
| `l1_reg` / `l2_reg` | ✅ (weight penalty) | ✅ (leaf weight penalty) |
| `n_estimators` | ❌ | ✅ |
| `max_tree_depth` | ❌ | ✅ |
| `learn_rate` | ✅ (gradient descent) | ✅ (shrinkage) |
| `subsample` | ❌ | ✅ |
| `early_stop` | ✅ | ✅ |
| `num_trials` (HPO) | ✅ | ✅ |

---

*Study tip: Know which hyperparameters control overfitting vs underfitting — this is commonly tested in scenario-based exam questions.*