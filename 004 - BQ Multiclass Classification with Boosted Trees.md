# Multiclass Classification with Boosted Trees in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Multiclass Classification?

Multiclass classification predicts which of **three or more classes** an observation belongs to. Unlike binary classification (2 classes), the model must learn decision boundaries separating **N distinct categories**.

**Use cases:** Product category prediction, handwriting recognition, sentiment (positive/neutral/negative), customer segment assignment, animal species identification.

```
Binary:     label ∈ {0, 1}
Multiclass: label ∈ {cat, dog, bird, fish, ...}
```

---

## 2. Boosted Trees Classifier for Multiclass

BigQuery ML uses **XGBoost** under the hood. For multiclass problems, XGBoost uses the **softmax** objective internally, producing a probability distribution across all classes that sums to 1.0.

```
P(class_1) + P(class_2) + ... + P(class_N) = 1.0
```

The model type is identical to binary classification:

```sql
model_type = 'boosted_tree_classifier'
```

> BQML **automatically detects** whether the problem is binary or multiclass based on the number of distinct values in the label column. No extra configuration is needed.

---

## 3. Full Workflow

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_multiclass_bt`
OPTIONS (
  model_type        = 'boosted_tree_classifier',
  input_label_cols  = ['label_column'],     -- STRING or INT64, 3+ distinct values
  num_parallel_tree = 1,
  max_tree_depth    = 6,
  n_estimators      = 100,
  learn_rate        = 0.1,
  subsample         = 0.8,
  colsample_bytree  = 0.8,
  min_tree_child_weight = 1,
  l1_reg            = 0.0,
  l2_reg            = 1.0,
  auto_class_weights = TRUE,               -- handles class imbalance
  data_split_method = 'AUTO_SPLIT',
  early_stop        = TRUE,
  min_rel_progress  = 0.01
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
  MODEL `project.dataset.my_multiclass_bt`,
  (
    SELECT feature1, feature2, feature3, label_column
    FROM `project.dataset.evaluation_table`
  )
);
```

### Step 3 — Inspect the Confusion Matrix

```sql
SELECT *
FROM ML.CONFUSION_MATRIX(
  MODEL `project.dataset.my_multiclass_bt`,
  (
    SELECT feature1, feature2, feature3, label_column
    FROM `project.dataset.evaluation_table`
  )
);
```

Returns an N×N matrix of predicted vs actual class counts.

### Step 4 — Feature Importance

```sql
-- Tree-based importance scores
SELECT *
FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.my_multiclass_bt`);

-- Global Shapley-based explanation
SELECT *
FROM ML.GLOBAL_EXPLAIN(MODEL `project.dataset.my_multiclass_bt`);
```

### Step 5 — Make Predictions

```sql
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.my_multiclass_bt`,
  (
    SELECT feature1, feature2, feature3
    FROM `project.dataset.new_data`
  )
);
```

**Output columns:**
- `predicted_<label_column>` — the class with the highest predicted probability
- `predicted_<label_column>_probs` — ARRAY of STRUCT `{label, prob}` for **every class**

```sql
-- Unnest probabilities for all classes per row
SELECT
  id,
  predicted_label,
  prob.label AS class_label,
  prob.prob  AS class_probability
FROM ML.PREDICT(
  MODEL `project.dataset.my_multiclass_bt`,
  (SELECT id, feature1, feature2, feature3 FROM `project.dataset.new_data`)
),
UNNEST(predicted_label_probs) AS prob
ORDER BY id, class_probability DESC;
```

---

## 4. Evaluation Metrics for Multiclass

### ML.EVALUATE Output

| Metric | Description | Goal |
|--------|-------------|------|
| `precision` | Weighted average precision across all classes | Higher |
| `recall` | Weighted average recall across all classes | Higher |
| `f1_score` | Weighted average F1 across all classes | Higher |
| `accuracy` | Overall fraction of correctly classified samples | Higher |
| `log_loss` | Cross-entropy loss across all classes | Lower |

> **Note:** For multiclass, `precision`, `recall`, and `f1_score` are reported as **weighted averages** (weighted by class support), not per-class values from `ML.EVALUATE`.

### Confusion Matrix (NxN)

For a 3-class problem (cat / dog / bird):

```
                Predicted cat   Predicted dog   Predicted bird
Actual cat   |      45        |       3        |       2       |
Actual dog   |       4        |      38        |       8       |
Actual bird  |       1        |       6        |      43       |
```

- **Diagonal cells** = correct predictions
- **Off-diagonal cells** = misclassifications
- Use this to identify **which classes the model confuses most**

### Per-Class Metrics (from ML.CONFUSION_MATRIX + manual calculation)

For each class C in a multiclass setting, treat it as a one-vs-rest binary problem:

```
Precision_C = TP_C / (TP_C + FP_C)
Recall_C    = TP_C / (TP_C + FN_C)
F1_C        = 2 * Precision_C * Recall_C / (Precision_C + Recall_C)
```

### Averaging Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Macro** | Simple mean across all classes | All classes equally important |
| **Weighted** | Mean weighted by class support (sample count) | Imbalanced classes — default in BQML |
| **Micro** | Aggregate TP/FP/FN globally, then compute | Focus on overall instance performance |

---

## 5. Key OPTIONS Parameters

### Core Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `model_type` | — | `'boosted_tree_classifier'` — **required** |
| `input_label_cols` | — | Target column — 3+ unique values for multiclass |
| `n_estimators` | `100` | Number of boosting rounds |
| `max_tree_depth` | `6` | Max depth of each tree |
| `num_leaves` | `31` | Max number of leaves per tree |
| `learn_rate` | `0.3` | Shrinkage rate — lower = more robust |

### Sampling (Anti-overfitting)

| Option | Default | Description |
|--------|---------|-------------|
| `subsample` | `1.0` | Row sampling fraction per tree |
| `colsample_bytree` | `1.0` | Feature sampling per tree |
| `colsample_bylevel` | `1.0` | Feature sampling per depth level |
| `colsample_bynode` | `1.0` | Feature sampling per split |
| `min_tree_child_weight` | `1` | Min sum of weights in a leaf |

### Regularization & Stopping

| Option | Default | Description |
|--------|---------|-------------|
| `l1_reg` | `0` | Lasso penalty on leaf weights |
| `l2_reg` | `1.0` | Ridge penalty on leaf weights |
| `early_stop` | `TRUE` | Stop if eval metric plateaus |
| `min_rel_progress` | `0.01` | Min relative improvement to continue |
| `num_parallel_tree` | `1` | Trees per round (`>1` = Random Forest mode) |

### Class Imbalance

| Option | Default | Description |
|--------|---------|-------------|
| `auto_class_weights` | `FALSE` | Auto-balance all class weights |
| `class_weights` | — | Manual per-class weight STRUCT array |

### Data Split

| Option | Values | Description |
|--------|--------|-------------|
| `data_split_method` | `AUTO_SPLIT` / `RANDOM` / `SEQ` / `CUSTOM` / `NO_SPLIT` | Training vs eval split |
| `data_split_eval_fraction` | `0.2` | Fraction used for evaluation |

---

## 6. Handling Class Imbalance in Multiclass

Multiclass datasets are often **imbalanced** — some categories have far fewer examples.

### Option 1 — Auto Class Weights (recommended)

```sql
OPTIONS (
  auto_class_weights = TRUE
)
```

BQML computes a weight for each class inversely proportional to its frequency.

### Option 2 — Manual Class Weights

```sql
OPTIONS (
  class_weights = [
    STRUCT('cat'  AS label, 1.0 AS weight),
    STRUCT('dog'  AS label, 2.0 AS weight),   -- underrepresented
    STRUCT('bird' AS label, 3.0 AS weight)    -- most underrepresented
  ]
)
```

### Option 3 — Oversample / Undersample in SQL

```sql
-- Oversample minority classes with UNION ALL before training
SELECT * FROM training_table WHERE label = 'cat'
UNION ALL
SELECT * FROM training_table WHERE label = 'dog'
UNION ALL
SELECT * FROM training_table WHERE label = 'dog'   -- duplicated
UNION ALL
SELECT * FROM training_table WHERE label = 'bird'
UNION ALL
SELECT * FROM training_table WHERE label = 'bird'
UNION ALL
SELECT * FROM training_table WHERE label = 'bird'  -- duplicated twice
```

---

## 7. Overfitting Control — Key Levers

| Symptom | Solution |
|---------|----------|
| Trees too deep | Reduce `max_tree_depth` or `num_leaves` |
| Too many rounds | Reduce `n_estimators` or use `early_stop = TRUE` |
| Memorising training data | Set `subsample < 1` or `colsample_bytree < 1` |
| Large leaf weights | Increase `l2_reg` or `l1_reg` |
| Too aggressive learning | Decrease `learn_rate`, increase `n_estimators` |
| Sparse leaves | Increase `min_tree_child_weight` |

---

## 8. Hyperparameter Tuning

```sql
CREATE OR REPLACE MODEL `project.dataset.my_tuned_multiclass`
OPTIONS (
  model_type               = 'boosted_tree_classifier',
  input_label_cols         = ['label_column'],
  num_trials               = 10,
  max_parallel_trials      = 2,
  hparam_tuning_objectives = ['accuracy'],
  hparam_range_override    = JSON '{
    "learn_rate":     {"min": 0.01, "max": 0.3},
    "max_tree_depth": {"min": 3,    "max": 10},
    "subsample":      {"min": 0.6,  "max": 1.0}
  }'
) AS
SELECT * FROM `project.dataset.training_table`;
```

> HPO is powered by **Vertex AI Vizier** under the hood.

---

## 9. Per-Row Explanations

```sql
SELECT *
FROM ML.EXPLAIN_PREDICT(
  MODEL `project.dataset.my_multiclass_bt`,
  (SELECT feature1, feature2, feature3 FROM `project.dataset.new_data`),
  STRUCT(5 AS top_k_features)
);
```

Returns **Shapley-based attributions** for each feature, per prediction. Shows which features pushed the model toward or away from the predicted class.

---

## 10. Multiclass vs Binary — Key Differences

| Aspect | Binary | Multiclass |
|--------|--------|------------|
| Label values | Exactly 2 | 3 or more |
| Internal objective | Sigmoid (logistic) | Softmax |
| Output probabilities | 1 value (pos class) | N values (one per class) |
| `roc_auc` in ML.EVALUATE | Yes | No — not returned |
| Confusion matrix | 2x2 | NxN |
| `precision` / `recall` / `f1` | Per class or binary | Weighted average across classes |
| `model_type` | `boosted_tree_classifier` | `boosted_tree_classifier` (same) |
| BQML auto-detects? | Yes | Yes — based on label cardinality |

---

## 11. ML Functions — Quick Reference

| Function | Needs Label? | Notes |
|----------|-------------|-------|
| `ML.EVALUATE` | Yes | Returns weighted avg precision, recall, F1, accuracy, log_loss |
| `ML.CONFUSION_MATRIX` | Yes | Returns NxN matrix — use to find misclassified class pairs |
| `ML.PREDICT` | No | Returns predicted label + probability array for all N classes |
| `ML.EXPLAIN_PREDICT` | No | Per-row Shapley values (top_k_features) |
| `ML.GLOBAL_EXPLAIN` | No | Global Shapley feature importance |
| `ML.FEATURE_IMPORTANCE` | No | Tree-based scores — multiclass supported |

---

## 12. Preprocessing Notes

- Boosted trees are **scale-invariant** — numeric features do not need standardisation.
- Categorical features are internally **ordinal-encoded** (not one-hot).
- NULLs are **imputed** — mean for numeric, empty string for categorical.
- Use `TRANSFORM` for custom feature engineering (applied automatically at prediction time).

```sql
CREATE OR REPLACE MODEL `project.dataset.my_model`
TRANSFORM(
  feature1,
  ML.BUCKETIZE(feature2, [10, 50, 100, 500]) AS bucketed_f2,
  EXTRACT(MONTH FROM date_col) AS month,
  label_column
)
OPTIONS (
  model_type       = 'boosted_tree_classifier',
  input_label_cols = ['label_column']
)
AS SELECT * FROM `project.dataset.training_table`;
```

---

## 13. Exam-Relevant Tips

- `model_type = 'boosted_tree_classifier'` is used for **both binary and multiclass** — BQML detects automatically.
- **`roc_auc` is NOT returned** by `ML.EVALUATE` for multiclass problems — only for binary.
- `ML.CONFUSION_MATRIX` is especially valuable for multiclass: it reveals **which classes are being confused**.
- Multiclass metrics (`precision`, `recall`, `f1`) are **weighted averages** in BQML, not per-class.
- `auto_class_weights = TRUE` works for **both binary and multiclass** imbalance.
- `ML.EXPLAIN_PREDICT` with `top_k_features` shows attributions **for the predicted class**.
- The output probability array has **one entry per class** — use `UNNEST` to flatten it.
- `num_parallel_tree > 1` + `n_estimators = 1` = **Random Forest Classifier** (works for multiclass too).
- HPO via `num_trials` uses **Vertex AI Vizier** — metric can be `accuracy`, `log_loss`, etc.
- For severe multiclass imbalance, **per-class F1 from the confusion matrix** is more informative than the weighted average.

---

## 14. Quick Reference Cheat Sheet

```sql
CREATE MODEL           →  model_type = 'boosted_tree_classifier', 3+ label values
ML.EVALUATE            →  accuracy, log_loss, weighted precision/recall/F1
                           NOTE: roc_auc NOT returned for multiclass
ML.CONFUSION_MATRIX    →  NxN matrix — identify confused class pairs
ML.PREDICT             →  predicted label + prob array (one prob per class)
ML.EXPLAIN_PREDICT     →  per-row Shapley feature attributions
ML.GLOBAL_EXPLAIN      →  global Shapley feature importance
ML.FEATURE_IMPORTANCE  →  tree-based feature scores
```

---

*Study tip: The exam often tests the difference between binary and multiclass evaluation — particularly that `roc_auc` is only available for binary, and that confusion matrices become NxN for multiclass. Know how to read one.*

## Binary vs multiclass evaluation

### Binary classification
- Simple case: each prediction is either positive or negative (spam/not spam, fraud/not fraud).
- `roc_auc` works directly because there is one ROC curve: one positive class vs one negative class.
- Confusion matrix is a 2×2 grid: `TP`, `FP`, `FN`, `TN`.

### Multiclass classification
- There are 3+ classes (cat, dog, bird), so things get more complex.
- `roc_auc` is not directly available in the standard sense.
- Workaround strategies:

| Strategy | What it does |
| --- | --- |
| One-vs-Rest (OvR) | Treats each class as "positive" and all others as "negative", computes one AUC per class, then averages |
| One-vs-One (OvO) | Computes AUC for every pair of classes, then averages |

- In scikit-learn, `roc_auc_score` requires `multi_class='ovr'` or `multi_class='ovo'` for multiclass predictions.

## Reading an NxN confusion matrix 📊

For a 3-class problem (cat, dog, bird), the matrix looks like this:

|            | Pred: Cat | Pred: Dog | Pred: Bird |
|------------|-----------|-----------|------------|
| **Cat**    | 50        | 3         | 2          |
| **Dog**    | 4         | 45        | 6          |
| **Bird**   | 1         | 2         | 47         |

How to read it:
- Diagonal = correct predictions. You want these numbers to be high.
- Off-diagonal = mistakes. The position tells you what the model confused.
  - Example: `Dog` row / `Bird` column means actual `Dog` predicted as `Bird`.
- Row sums = total actual instances of each class.
- Column sums = total predicted instances of each class.

A common exam question: "the model is confusing class A with class B — which cell shows this?" The answer is the cell at row A, column B.

## Exam-ready summary

| Concept | Binary | Multiclass |
| --- | --- | --- |
| `roc_auc` | ✅ Works directly | ⚠️ Needs OvR or OvO |
| Confusion matrix shape | 2×2 | N×N |
| Precision / Recall / F1 | Per-class or overall | Per-class, then macro/micro/weighted average |
| "Which class is confused?" | N/A | Read the off-diagonal row/column |

> Macro vs micro averaging is another common trap.
> - Macro averages each class equally (bad for imbalanced data).
> - Micro aggregates all TP/FP/FN first, then computes metrics (better for imbalanced data).
> - Use micro when one class dominates.
