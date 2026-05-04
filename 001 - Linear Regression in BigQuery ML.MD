# Linear Regression in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is BigQuery ML?

BigQuery ML (BQML) enables you to **create and execute ML models directly in BigQuery using SQL**. No data movement, no separate infrastructure — training and inference happen where the data lives.

---

## 2. Linear Regression Overview

Linear regression predicts a **continuous numeric value** (label) from one or more input features.

**Formula:**
```
ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

| Term | Meaning |
|------|---------|
| `ŷ` | Predicted value |
| `β₀` | Intercept (bias) |
| `β₁...βₙ` | Coefficients (weights) |
| `x₁...xₙ` | Feature values |
| `ε` | Error term |

**Use cases:** Sales forecasting, price prediction, demand estimation, resource planning.

---

## 3. Full Workflow in BigQuery ML

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_linear_model`
OPTIONS (
  model_type = 'linear_reg',
  input_label_cols = ['label_column'],
  l1_reg = 0.01,              -- Lasso regularization (optional)
  l2_reg = 0.01,              -- Ridge regularization (optional)
  learn_rate_strategy = 'line_search',
  max_iterations = 20,
  data_split_method = 'AUTO_SPLIT'
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
  MODEL `project.dataset.my_linear_model`,
  (
    SELECT feature1, feature2, feature3, label_column
    FROM `project.dataset.evaluation_table`
  )
);
```

**Key metrics returned:**

| Metric | Description | Goal |
|--------|-------------|------|
| `mean_absolute_error` (MAE) | Average absolute difference | Lower is better |
| `mean_squared_error` (MSE) | Average squared difference | Lower is better |
| `mean_squared_log_error` (MSLE) | Log-scaled MSE | Lower is better |
| `r2_score` | Variance explained by model (0–1) | Closer to 1 is better |
| `explained_variance` | Proportion of label variance explained | Higher is better |

### Step 3 — Inspect Model Weights

```sql
SELECT *
FROM ML.WEIGHTS(MODEL `project.dataset.my_linear_model`);
```

Returns each feature's **weight (coefficient)** and its **intercept**.

### Step 4 — Make Predictions

```sql
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.my_linear_model`,
  (
    SELECT feature1, feature2, feature3
    FROM `project.dataset.new_data`
  )
);
```

Output includes a `predicted_<label_column>` field.

---

## 4. Key `OPTIONS` Parameters

| Option | Values / Default | Notes |
|--------|-----------------|-------|
| `model_type` | `'linear_reg'` | Required |
| `input_label_cols` | Column name | Required — the target variable |
| `l1_reg` | Float, default `0` | Lasso — promotes sparsity |
| `l2_reg` | Float, default `0` | Ridge — penalises large weights |
| `learn_rate` | Float | Used if strategy is `constant` |
| `learn_rate_strategy` | `'line_search'` / `'constant'` | `line_search` is adaptive |
| `max_iterations` | Int, default `20` | Max training iterations |
| `data_split_method` | `'AUTO_SPLIT'` / `'RANDOM'` / `'CUSTOM'` / `'SEQ'` / `'NO_SPLIT'` | Holdout split strategy |
| `data_split_eval_fraction` | Float `0.0–1.0`, default `0.2` | Fraction used for eval |
| `data_split_col` | Column name | Required for `CUSTOM` split |
| `ls_init_learn_rate` | Float | Initial rate for `line_search` |

---

## 5. Regularization — L1 vs L2

```
L1 (Lasso):  Loss + λ Σ|βᵢ|      → drives some weights to zero (feature selection)
L2 (Ridge):  Loss + λ Σβᵢ²        → shrinks all weights, retains all features
Elastic Net: combine both L1 + L2
```

- Use **L1** when you suspect many features are irrelevant.
- Use **L2** when all features are likely useful but overfitting is a concern.
- Both can be combined in BigQuery ML by setting both `l1_reg` and `l2_reg`.

---

## 6. Data Splitting Strategies

| Method | When to use |
|--------|-------------|
| `AUTO_SPLIT` | Default; BQML handles the split automatically |
| `RANDOM` | Random holdout; set `data_split_eval_fraction` |
| `CUSTOM` | You provide a boolean column (`TRUE` = eval set) |
| `SEQ` | Time-series data; last N% rows used for eval |
| `NO_SPLIT` | All data used for training; no automatic eval |

---

## 7. Feature Preprocessing

BigQuery ML **automatically handles**:
- Numeric features → standardised (z-score)
- String/categorical features → one-hot encoded
- NULLs → imputed with mean (numeric) or empty string (categorical)

You can **override** this using `TRANSFORM`:

```sql
CREATE OR REPLACE MODEL `project.dataset.my_model`
TRANSFORM(
  ML.STANDARD_SCALER(feature1) OVER() AS scaled_f1,
  ML.ONE_HOT_ENCODER(cat_feature) OVER() AS encoded_cat,
  label_column
)
OPTIONS (model_type = 'linear_reg', input_label_cols = ['label_column'])
AS SELECT * FROM `project.dataset.training_table`;
```

---

## 8. Global Explain (Feature Importance)

```sql
SELECT *
FROM ML.GLOBAL_EXPLAIN(MODEL `project.dataset.my_linear_model`);
```

Returns **attribution scores** showing the relative importance of each feature. Higher absolute value = more influential feature.

---

## 9. Exam-Relevant Tips

- **`ML.EVALUATE`** requires a label column in the input table.
- **`ML.PREDICT`** does **not** require the label column.
- **R² score** is the primary goodness-of-fit metric for linear regression.
- **`data_split_method = 'SEQ'`** is critical for time-ordered datasets — preserves temporal order.
- Regularization (`l1_reg`, `l2_reg`) prevents **overfitting**, especially important with many features.
- BQML linear regression uses **mini-batch gradient descent**, not closed-form OLS.
- The `TRANSFORM` clause is applied **at prediction time automatically** — no need to re-transform new data.
- You can export BQML models to **Google Cloud Storage** for use with Vertex AI or TensorFlow Serving.

---

## 10. Quick Reference Cheat Sheet

```
CREATE MODEL  →  train the model
ML.EVALUATE   →  assess model quality (needs label)
ML.WEIGHTS    →  inspect coefficients
ML.PREDICT    →  generate predictions (no label needed)
ML.GLOBAL_EXPLAIN  →  feature importance scores
```

---

*Study tip: Practice writing CREATE MODEL and ML.EVALUATE queries from scratch — the exam tests syntax awareness as well as conceptual understanding.*