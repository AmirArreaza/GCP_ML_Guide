# Feature Engineering in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Feature Engineering in BigQuery ML?

Feature engineering is the process of **transforming raw data into features** that better represent the underlying patterns for ML models. In BigQuery ML, feature engineering happens inside a **`TRANSFORM` clause**, which:

- Applies transformations **at training time** and **automatically re-applies them at prediction time**
- Eliminates the need to manually re-transform new data before calling `ML.PREDICT`
- Keeps preprocessing logic co-located with the model definition

```sql
CREATE OR REPLACE MODEL `project.dataset.my_model`
TRANSFORM(
  -- all feature transformations go here
  ML.STANDARD_SCALER(numeric_col)   OVER() AS scaled_col,
  ML.ONE_HOT_ENCODER(cat_col)       OVER() AS encoded_col,
  label_col                                             -- pass label through unchanged
)
OPTIONS (model_type = 'linear_reg', input_label_cols = ['label_col'])
AS SELECT * FROM `project.dataset.training_table`;
```

> **Key exam point:** The `TRANSFORM` clause is stored with the model and applied **automatically** during `ML.PREDICT` — you never need to re-apply transformations manually.

---

## 2. Preprocessing Functions Overview

| Function | Category | Purpose |
|----------|----------|---------|
| `ML.STANDARD_SCALER` | Numeric | Z-score normalisation (mean=0, std=1) |
| `ML.MIN_MAX_SCALER` | Numeric | Scale to [0, 1] range |
| `ML.MAX_ABS_SCALER` | Numeric | Scale by max absolute value — range [-1, 1] |
| `ML.ROBUST_SCALER` | Numeric | Scale using median and IQR — robust to outliers |
| `ML.BUCKETIZE` | Numeric | Bin into discrete buckets |
| `ML.QUANTILE_BUCKETIZE` | Numeric | Bin into equal-frequency buckets |
| `ML.ONE_HOT_ENCODER` | Categorical | Binary indicator per category |
| `ML.LABEL_ENCODER` | Categorical | Ordinal integer encoding |
| `ML.MULTI_HOT_ENCODER` | Categorical | Multi-label binary encoding |
| `ML.FEATURE_CROSS` | Interaction | Cross product of two categorical features |
| `ML.POLYNOMIAL_EXPAND` | Interaction | Polynomial feature expansion |
| `ML.NGRAMS` | Text | Character or word n-gram generation |
| `ML.IMPUTER` | Missing data | Fill NULLs with mean, median, or mode |

---

## 3. Numeric Transformations

### ML.STANDARD_SCALER — Z-Score Normalisation

Transforms a numeric feature to have **mean = 0** and **standard deviation = 1**.

```sql
ML.STANDARD_SCALER(feature_col) OVER() AS scaled_feature
```

```
z = (x - mean) / std_dev
```

**Use when:** Features have very different scales (e.g., age 0–100, salary 20k–500k). Essential for **linear regression**, **logistic regression**, **neural networks**, and **K-means clustering**.

### ML.MIN_MAX_SCALER — Range Scaling

Scales values to the **[0, 1]** range.

```sql
ML.MIN_MAX_SCALER(feature_col) OVER() AS minmax_feature
```

```
x_scaled = (x - min) / (max - min)
```

**Use when:** You need a bounded output range. Sensitive to outliers — one extreme value compresses all others.

### ML.MAX_ABS_SCALER — Absolute Scaling

Divides by the **maximum absolute value**, preserving sign, resulting in range **[-1, 1]**.

```sql
ML.MAX_ABS_SCALER(feature_col) OVER() AS maxabs_feature
```

**Use when:** Data is already centred at zero or is sparse. Does not shift the mean.

### ML.ROBUST_SCALER — Outlier-Robust Scaling

Uses **median and interquartile range (IQR)** instead of mean and std. Robust to outliers.

```sql
ML.ROBUST_SCALER(feature_col) OVER() AS robust_feature
```

```
x_scaled = (x - median) / IQR
```

**Use when:** Feature contains outliers that would distort standard or min-max scaling.

### ML.BUCKETIZE — Fixed-Boundary Binning

Converts a **continuous numeric** feature into a **categorical bucket** using manually defined boundary values.

```sql
ML.BUCKETIZE(age_col, [18, 25, 35, 50, 65]) AS age_bucket
```

Creates buckets: `(-inf, 18)`, `[18, 25)`, `[25, 35)`, `[35, 50)`, `[50, 65)`, `[65, +inf)`.

**Use when:** You know the meaningful boundaries (e.g., age groups, price tiers). Creates interpretable features.

### ML.QUANTILE_BUCKETIZE — Equal-Frequency Binning

Automatically creates **N buckets** each containing approximately the **same number of observations**.

```sql
ML.QUANTILE_BUCKETIZE(income_col, 10) OVER() AS income_decile
```

**Use when:** You want equal-density buckets without knowing the boundaries. Handles skewed distributions well.

---

## 4. Categorical Transformations

### ML.ONE_HOT_ENCODER — Binary Indicator Encoding

Creates a **binary (0/1) column for each unique category value**. The most common encoding for nominal (unordered) categories.

```sql
ML.ONE_HOT_ENCODER(colour_col) OVER() AS encoded_colour
```

**Output:** If `colour` has values `{red, blue, green}` → creates 3 binary features.

| Option | Values | Description |
|--------|--------|-------------|
| `drop` | `'MOST_POPULAR'` (default), `'LEAST_POPULAR'`, `'NONE'` | Which category to drop to avoid multicollinearity |
| `top_k` | integer | Only encode the top K most frequent categories |
| `frequency_threshold` | float | Minimum frequency to include a category |

```sql
ML.ONE_HOT_ENCODER(colour_col, 'MOST_POPULAR', 10, 0.01) OVER() AS encoded_colour
-- drop='MOST_POPULAR', top_k=10, frequency_threshold=0.01
```

### ML.LABEL_ENCODER — Ordinal Integer Encoding

Maps each category to an **integer** (0, 1, 2, ...). Preserves order implicitly — use only for **ordinal** categories.

```sql
ML.LABEL_ENCODER(size_col) OVER() AS encoded_size
```

**Use when:** Category has natural order (e.g., Small=0, Medium=1, Large=2). Avoid for nominal categories as it implies a false order.

| Option | Description |
|--------|-------------|
| `top_k` | Encode only top K categories; others become a fallback value |
| `frequency_threshold` | Minimum frequency to assign a unique label |

### ML.MULTI_HOT_ENCODER — Multi-Label Encoding

Handles features where a single observation can belong to **multiple categories** (arrays of strings).

```sql
ML.MULTI_HOT_ENCODER(tags_array_col) OVER() AS encoded_tags
```

**Use when:** A product has multiple tags, a user has multiple interests, a document has multiple topics.

---

## 5. Interaction & Derived Features

### ML.FEATURE_CROSS — Categorical Feature Interaction

Creates the **Cartesian product** of two or more categorical features, generating a new combined feature that captures interactions.

```sql
ML.FEATURE_CROSS(STRUCT(country_col AS country, device_col AS device)) AS country_device_cross
```

**Example:** `country=US` × `device=mobile` → new feature `US_mobile`

**Use when:** You suspect two categorical features interact (e.g., country × language, age group × product category).

### ML.POLYNOMIAL_EXPAND — Polynomial Feature Expansion

Generates **polynomial and interaction terms** from numeric features up to a specified degree.

```sql
ML.POLYNOMIAL_EXPAND(STRUCT(feature1 AS f1, feature2 AS f2), 2) AS poly_features
```

For degree 2 with features {f1, f2}, generates: `{f1, f2, f1², f1·f2, f2²}`

**Use when:** You suspect non-linear relationships in a linear model. Adds expressiveness without switching to a neural network.

---

## 6. Text Features

### ML.NGRAMS — N-Gram Generation

Tokenises text and generates **character or word n-grams**.

```sql
ML.NGRAMS(SPLIT(text_col, ' '), [1, 2]) AS text_ngrams
-- word unigrams and bigrams
```

```sql
ML.NGRAMS(SPLIT(text_col, ''), [2, 3]) AS char_ngrams
-- character bigrams and trigrams
```

**Use when:** Building text classification or embedding features from free-form text without full NLP.

---

## 7. Missing Data Handling

### ML.IMPUTER — Fill NULLs

Replaces NULL values with a **statistical summary** of the non-null values in that column.

```sql
ML.IMPUTER(feature_col, 'mean')   OVER() AS imputed_mean
ML.IMPUTER(feature_col, 'median') OVER() AS imputed_median
ML.IMPUTER(feature_col, 'most_frequent') OVER() AS imputed_mode
```

| Strategy | Use When |
|----------|---------|
| `'mean'` | Numeric, roughly normally distributed, no extreme outliers |
| `'median'` | Numeric, skewed distribution or outliers present |
| `'most_frequent'` | Categorical features or highly skewed numeric |

> **Exam tip:** BQML auto-imputes NULLs without `TRANSFORM` (numeric → mean, categorical → empty string). Using `ML.IMPUTER` explicitly in `TRANSFORM` gives you more control and ensures the same imputation logic is applied at prediction time.

---

## 8. The TRANSFORM Clause — Full Example

```sql
CREATE OR REPLACE MODEL `project.dataset.churn_model`
TRANSFORM(
  -- Numeric scaling
  ML.STANDARD_SCALER(age)             OVER() AS scaled_age,
  ML.STANDARD_SCALER(account_balance) OVER() AS scaled_balance,
  ML.ROBUST_SCALER(monthly_charges)   OVER() AS robust_charges,

  -- Bucketing
  ML.QUANTILE_BUCKETIZE(tenure_months, 5) OVER() AS tenure_bucket,

  -- Categorical encoding
  ML.ONE_HOT_ENCODER(contract_type)   OVER() AS encoded_contract,
  ML.LABEL_ENCODER(region)            OVER() AS encoded_region,

  -- Missing data
  ML.IMPUTER(credit_score, 'median')  OVER() AS imputed_credit,

  -- Interaction feature
  ML.FEATURE_CROSS(
    STRUCT(contract_type AS contract, region AS region)
  ) AS contract_region_cross,

  -- Pass label through unchanged
  churn_label
)
OPTIONS (
  model_type       = 'logistic_reg',
  input_label_cols = ['churn_label']
)
AS
SELECT * FROM `project.dataset.customers`;
```

---

## 9. OVER() Clause — Why It Is Required

All analytic preprocessing functions in BigQuery ML require the **`OVER()` clause** because they compute statistics (mean, std, min, max, etc.) across the **entire training dataset** — similar to window functions.

```sql
-- CORRECT
ML.STANDARD_SCALER(feature_col) OVER() AS scaled_col

-- WRONG — will throw an error
ML.STANDARD_SCALER(feature_col) AS scaled_col
```

The `OVER()` clause is **empty** because the statistic is computed globally across all training rows, not partitioned.

---

## 10. Automatic vs Manual Preprocessing

| Scenario | Auto (no TRANSFORM) | Manual (with TRANSFORM) |
|----------|-------------------|------------------------|
| Numeric features | z-score standardised automatically | Full control — choose scaler type |
| Categorical strings | One-hot encoded automatically | Choose one-hot / label / multi-hot |
| NULLs | Auto-imputed (mean / empty string) | Choose strategy explicitly |
| Feature interactions | Not created | `ML.FEATURE_CROSS`, `ML.POLYNOMIAL_EXPAND` |
| Text features | Not supported | `ML.NGRAMS` |
| Applied at prediction | N/A | Yes — automatically re-applied |

---

## 11. Choosing the Right Scaler

| Scaler | Formula | Use When |
|--------|---------|---------|
| `STANDARD_SCALER` | (x - mean) / std | Default choice — data roughly normal |
| `MIN_MAX_SCALER` | (x - min) / (max - min) | Need [0,1] range, no extreme outliers |
| `MAX_ABS_SCALER` | x / max_abs | Sparse data, data already centred at 0 |
| `ROBUST_SCALER` | (x - median) / IQR | Data has significant outliers |

---

## 12. Choosing the Right Categorical Encoder

| Encoder | Output | Use When |
|---------|--------|---------|
| `ONE_HOT_ENCODER` | Binary vectors | Nominal (unordered) categories, moderate cardinality |
| `LABEL_ENCODER` | Integer | Ordinal (ordered) categories only |
| `MULTI_HOT_ENCODER` | Binary vectors | Multi-label / array-type features |
| `FEATURE_CROSS` | Combined feature | Two features interact with each other |

---

## 13. ML Functions Applied at Prediction Time

When `TRANSFORM` is used, all transformations are **automatically applied at prediction time**:

```sql
-- No need to transform new_data — TRANSFORM handles it
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.churn_model`,
  (SELECT age, account_balance, monthly_charges, tenure_months,
          contract_type, region, credit_score
   FROM `project.dataset.new_customers`)
);
```

The raw, **untransformed** columns are passed in. BigQuery ML applies the stored `TRANSFORM` logic internally before scoring.

---

## 14. Exam-Relevant Tips

- `TRANSFORM` clause stores preprocessing **with the model** — applied automatically at `ML.PREDICT` time.
- All preprocessing functions require **`OVER()`** — they are analytic functions computing dataset-wide statistics.
- `ML.STANDARD_SCALER` is the default choice for numeric scaling in most models.
- `ML.ROBUST_SCALER` is preferred when features contain **significant outliers**.
- `ML.BUCKETIZE` uses **manual boundaries** — `ML.QUANTILE_BUCKETIZE` uses **equal-frequency** auto-boundaries.
- `ML.ONE_HOT_ENCODER` is for **nominal** categories; `ML.LABEL_ENCODER` is for **ordinal** categories.
- `ML.FEATURE_CROSS` captures **interaction effects** between two categorical features.
- `ML.POLYNOMIAL_EXPAND` adds **non-linear terms** to linear models.
- `ML.IMPUTER` gives explicit control over NULL handling — use it instead of relying on BQML defaults.
- **Boosted trees and random forests** are scale-invariant — scalers not needed but can still use `TRANSFORM` for other engineering.
- **Linear regression, logistic regression, neural networks, K-means** benefit greatly from numeric scaling.
- The label column should be **passed through** the `TRANSFORM` clause unchanged.
- `ML.MULTI_HOT_ENCODER` handles **ARRAY** type input columns — the only encoder that does.

---

## 15. Quick Reference Cheat Sheet

```
TRANSFORM clause        →  stores + auto-applies transformations at prediction time
OVER()                  →  required for all ML preprocessing functions

Numeric Scalers:
  ML.STANDARD_SCALER    →  z-score: (x - mean) / std
  ML.MIN_MAX_SCALER     →  range: (x - min) / (max - min) → [0, 1]
  ML.MAX_ABS_SCALER     →  divide by max abs → [-1, 1]
  ML.ROBUST_SCALER      →  (x - median) / IQR — outlier-robust

Binning:
  ML.BUCKETIZE          →  manual boundaries → categorical buckets
  ML.QUANTILE_BUCKETIZE →  auto equal-frequency buckets

Categorical:
  ML.ONE_HOT_ENCODER    →  binary per category (nominal)
  ML.LABEL_ENCODER      →  integer per category (ordinal)
  ML.MULTI_HOT_ENCODER  →  binary for array / multi-label inputs

Interaction:
  ML.FEATURE_CROSS      →  Cartesian product of categoricals
  ML.POLYNOMIAL_EXPAND  →  polynomial + interaction numeric terms

Text:
  ML.NGRAMS             →  word or character n-grams

Missing:
  ML.IMPUTER            →  fill NULLs with mean / median / most_frequent
```

---

*Study tip: The exam frequently tests which preprocessing function to choose for a given scenario (outliers → ROBUST_SCALER, ordinal categories → LABEL_ENCODER, interactions → FEATURE_CROSS) and whether TRANSFORM is automatically applied at prediction time (yes, it always is).*