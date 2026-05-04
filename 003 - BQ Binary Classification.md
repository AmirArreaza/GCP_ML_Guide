# Binary Classification in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Binary Classification?

Binary classification predicts which of **two classes** an observation belongs to. The output is a **class label** (e.g. 0/1, True/False, spam/ham) and a **probability score** between 0 and 1.

**Use cases:** Spam detection, fraud detection, churn prediction, disease diagnosis, click-through prediction.

---

## 2. Supported Model Types for Binary Classification

| `model_type` | Algorithm | Best For |
|---|---|---|
| `logistic_reg` | Logistic Regression | Linearly separable, interpretable, fast |
| `boosted_tree_classifier` | XGBoost | Non-linear, tabular data, high performance |
| `random_forest_classifier` | Random Forest | Robust, less tuning needed |
| `dnn_classifier` | Deep Neural Network | Complex patterns, large datasets |
| `automl_classifier` | AutoML Tables | Automated model selection and tuning |

> For the exam, **logistic regression** and **boosted tree classifier** are the most tested.

---

## 3. Logistic Regression — Full Workflow

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_logistic_model`
OPTIONS (
  model_type = 'logistic_reg',
  input_label_cols = ['label_column'],      -- must be STRING or INT64 with exactly 2 values
  l1_reg = 0.01,
  l2_reg = 0.01,
  learn_rate_strategy = 'line_search',
  max_iterations = 20,
  data_split_method = 'AUTO_SPLIT',
  auto_class_weights = TRUE                 -- handles class imbalance automatically
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

### Boosted Tree Classifier

```sql
CREATE OR REPLACE MODEL `project.dataset.my_bt_classifier`
OPTIONS (
  model_type = 'boosted_tree_classifier',
  input_label_cols = ['label_column'],
  max_tree_depth = 6,
  n_estimators = 50,
  learn_rate = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  l2_reg = 1.0,
  auto_class_weights = TRUE,
  data_split_method = 'AUTO_SPLIT',
  early_stop = TRUE
) AS
SELECT * FROM `project.dataset.training_table`;
```

### Step 2 — Evaluate the Model

```sql
SELECT *
FROM ML.EVALUATE(
  MODEL `project.dataset.my_logistic_model`,
  (
    SELECT feature1, feature2, feature3, label_column
    FROM `project.dataset.evaluation_table`
  )
);
```

### Step 3 — ROC Curve

```sql
SELECT *
FROM ML.ROC_CURVE(
  MODEL `project.dataset.my_logistic_model`,
  (SELECT feature1, feature2, feature3, label_column
   FROM `project.dataset.evaluation_table`)
);
```

### Step 4 — Confusion Matrix

```sql
SELECT *
FROM ML.CONFUSION_MATRIX(
  MODEL `project.dataset.my_logistic_model`,
  (SELECT feature1, feature2, feature3, label_column
   FROM `project.dataset.evaluation_table`)
);
```

### Step 5 — Make Predictions

```sql
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.my_logistic_model`,
  (SELECT feature1, feature2, feature3 FROM `project.dataset.new_data`)
);
```

**Output columns:**
- `predicted_<label_column>` — the predicted class label
- `predicted_<label_column>_probs` — ARRAY of STRUCT `{label, prob}` for each class

```sql
-- Extract probability for the positive class
SELECT
  id,
  predicted_label,
  (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = '1') AS positive_prob
FROM ML.PREDICT(
  MODEL `project.dataset.my_logistic_model`,
  (SELECT id, feature1, feature2, feature3 FROM `project.dataset.new_data`)
);
```

---

## 4. Key Binary Classification Metrics

| Metric | Formula | Description | Goal |
|--------|---------|-------------|------|
| `precision` | TP / (TP + FP) | Of predicted positives, how many are actually correct? | Higher |
| `recall` | TP / (TP + FN) | Of actual positives, how many did we successfully catch? | Higher |
| `f1_score` | 2 x (P x R) / (P + R) | Harmonic mean; balances Precision and Recall. | Higher |
| `accuracy` | (TP + TN) / Total | Ratio of total correct predictions to total observations. | Higher |
| `log_loss` | Cross-entropy loss | Penalizes confident but incorrect predictions. | Lower |
| `roc_auc` | Area under ROC curve | Ability to distinguish between classes across all thresholds. | Higher (max 1.0) |

#### Pro-Tips for Selection:

- **The Imbalance Trap:** Never rely on Accuracy if one class is rare (e.g., 1% fraud). A model that just says "No Fraud" every time will be 99% accurate but 100% useless.
- **The PR Curve vs. ROC:** If you have a massive class imbalance, the Precision-Recall (PR) Curve and the resulting AUC-PR are often more informative than ROC AUC, as ROC can be overly optimistic when the negative class is dominant.
- **F-Beta Score:** If you find yourself wanting a middle ground between F1 and Recall, look into the $F_{\beta}$ score. Setting $\beta = 2$ weighs Recall higher, while $\beta = 0.5$ weighs Precision higher.  

### Confusion Matrix

```
                    Predicted POSITIVE    Predicted NEGATIVE
Actual POSITIVE  |  TP (True Positive)  |  FN (False Negative) |
Actual NEGATIVE  |  FP (False Positive) |  TN (True Negative)  |

False Positive = Type I Error   (predicted yes, actually no)
False Negative = Type II Error  (predicted no, actually yes)
```

---

## 5. ROC Curve & AUC

```
TPR (Recall)  = TP / (TP + FN)   → y-axis
FPR           = FP / (FP + TN)   → x-axis

AUC = 1.0  → perfect classifier
AUC = 0.5  → random classifier (no skill)
AUC < 0.5  → worse than random
```

The ROC curve lets you **choose a threshold** that balances TPR vs FPR based on your business needs. `ML.ROC_CURVE` returns one row per threshold value.

---

## 6. Adjusting the Classification Threshold

Default threshold is **0.5**. Lower it to increase recall; raise it to increase precision.

```sql
SELECT
  id,
  positive_prob,
  IF(positive_prob >= 0.3, '1', '0') AS custom_label   -- lower threshold = more recall
FROM (
  SELECT
    id,
    (SELECT prob FROM UNNEST(predicted_label_probs) WHERE label = '1') AS positive_prob
  FROM ML.PREDICT(
    MODEL `project.dataset.my_logistic_model`,
    (SELECT id, feature1, feature2, feature3 FROM `project.dataset.new_data`)
  )
);
```

---

## 7. Handling Class Imbalance

| Strategy | How |
|----------|-----|
| `auto_class_weights = TRUE` | BQML computes and applies weights automatically |
| `class_weights` | Manually specify weights per class label |
| Oversample minority in SQL | UNION ALL the minority rows before training |
| Use AUC / F1 | Don't rely on accuracy as the success metric |

```sql
-- Manual class weights
OPTIONS (
  class_weights = [
    STRUCT('fraud' AS label, 10.0 AS weight),
    STRUCT('legit' AS label, 1.0 AS weight)
  ]
)
```

> **Exam tip:** With severe imbalance, a model that always predicts the majority class gets high accuracy but **zero recall**. AUC and F1 are the right metrics.

---

## 8. Precision–Recall Tradeoff by Business Context

| Scenario | Priority |
|----------|---------|
| Fraud detection | **Recall** — catch all fraud, FN is costly |
| Spam filtering | **Precision** — don't block real email, FP is costly |
| Medical diagnosis | **Recall** — don't miss disease |
| Ad click prediction | **AUC** — ranking matters across thresholds |
| Balanced classes | **F1** or Accuracy |
| Imbalanced classes | **F1** or **AUC** — never accuracy alone |

---

## 9. Explainability

```sql
-- Logistic regression: model weights / coefficients
SELECT * FROM ML.WEIGHTS(MODEL `project.dataset.my_logistic_model`);

-- Tree models: feature importance scores
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.my_bt_classifier`);

-- Global Shapley-based explanation (all model types)
SELECT * FROM ML.GLOBAL_EXPLAIN(MODEL `project.dataset.my_logistic_model`);

-- Per-row explanation (why was this individual prediction made?)
SELECT *
FROM ML.EXPLAIN_PREDICT(
  MODEL `project.dataset.my_logistic_model`,
  (SELECT feature1, feature2, feature3 FROM `project.dataset.new_data`),
  STRUCT(3 AS top_k_features)   -- top 3 contributing features per row
);
```

---

## 10. Key OPTIONS Parameters

| Option | Applies To | Description |
|--------|-----------|-------------|
| `model_type` | All | `logistic_reg`, `boosted_tree_classifier`, `random_forest_classifier`, `dnn_classifier` |
| `input_label_cols` | All | Target column — exactly 2 unique values required |
| `auto_class_weights` | All | Automatically balance class distributions |
| `class_weights` | All | Manually set per-class weights |
| `l1_reg` / `l2_reg` | All | Regularization (Lasso / Ridge) |
| `max_iterations` | Logistic Reg | Maximum training iterations |
| `learn_rate_strategy` | Logistic Reg | `line_search` (adaptive) or `constant` |
| `n_estimators` | Boosted Trees | Number of boosting rounds |
| `max_tree_depth` | Boosted Trees | Depth of each tree |
| `subsample` | Boosted Trees | Row sampling fraction per tree |
| `colsample_bytree` | Boosted Trees | Feature sampling per tree |
| `early_stop` | Boosted Trees | Stop if eval metric plateaus |
| `data_split_method` | All | `AUTO_SPLIT`, `RANDOM`, `SEQ`, `CUSTOM`, `NO_SPLIT` |
| `num_trials` | All | HPO trials (Vertex AI Vizier) |

---

## 11. ML Functions — Quick Reference

| Function | Needs Label? | Output |
|----------|-------------|--------|
| `ML.EVALUATE` | Yes | Precision, Recall, F1, Accuracy, Log Loss, AUC |
| `ML.ROC_CURVE` | Yes | Threshold, TPR, FPR data points |
| `ML.CONFUSION_MATRIX` | Yes | TP / FP / TN / FN counts |
| `ML.PREDICT` | No | Predicted label + probability array |
| `ML.EXPLAIN_PREDICT` | No | Per-row Shapley feature attributions |
| `ML.GLOBAL_EXPLAIN` | No | Global Shapley feature importance |
| `ML.FEATURE_IMPORTANCE` | No | Tree-based scores (tree models only) |
| `ML.WEIGHTS` | No | Model coefficients (logistic reg) |

---

## 12. Exam-Relevant Tips

- Label column must have **exactly 2 distinct values** — string or integer.
- `auto_class_weights = TRUE` is the simplest fix for **class imbalance**.
- **AUC = 0.5** means the model has no predictive power — equivalent to random guessing.
- `ML.ROC_CURVE` returns threshold-level data; use it to **select an operating point**.
- `ML.EXPLAIN_PREDICT` gives **per-instance Shapley values** — useful for debugging individual predictions.
- `ML.CONFUSION_MATRIX` evaluates at the **default 0.5 threshold**.
- `ML.FEATURE_IMPORTANCE` is **tree models only** — not available for logistic regression.
- `ML.WEIGHTS` is **logistic regression only** — returns per-feature coefficients.
- Lowering the threshold → **higher recall, lower precision**.
- Raising the threshold → **higher precision, lower recall**.
- `boosted_tree_classifier` and `boosted_tree_regressor` share the same hyperparameters but different `model_type` values.
- For **imbalanced datasets**: use AUC, F1, or Recall — not accuracy.

---

## 13. Quick Reference Cheat Sheet

```
CREATE MODEL           →  model_type = 'logistic_reg' or 'boosted_tree_classifier'
ML.EVALUATE            →  precision, recall, f1, accuracy, log_loss, roc_auc
ML.ROC_CURVE           →  TPR vs FPR across all thresholds
ML.CONFUSION_MATRIX    →  TP / FP / TN / FN at default threshold
ML.PREDICT             →  predicted label + probability array
ML.EXPLAIN_PREDICT     →  per-row Shapley feature attribution
ML.GLOBAL_EXPLAIN      →  global Shapley feature importance
ML.WEIGHTS             →  logistic regression coefficients only
ML.FEATURE_IMPORTANCE  →  tree model feature scores only
```

---

*Study tip: Know which metric to choose for a given business scenario — precision vs recall vs AUC is one of the most frequently tested concepts in the GCP Professional ML Engineer exam.*


## Precision vs Recall vs AUC

### Precision — "don't cry wolf" 🐺
- Precision answers: of everything I flagged as positive, how many actually were?
- Formula: `TP / (TP + FP)`
- Use precision when false positives are the bigger problem — acting on a wrong prediction is expensive or annoying.

Scenario | Why precision matters
--- | ---
Spam filter | Blocking a legitimate email (false positive) loses important messages
Fraud alert | Wrongly freezing a real customer's card causes friction
Ad targeting | Showing an irrelevant ad wastes budget

### Recall — "don't miss anything" 🎯
- Recall answers: of all the actual positives that existed, how many did I catch?
- Formula: `TP / (TP + FN)`
- Use recall when false negatives are the bigger problem — missing a real case has serious consequences.

Scenario | Why recall matters
--- | ---
Cancer screening | Missing a tumour (false negative) is far worse than a false alarm
Airport security | Missing a threat is catastrophic
Fraud detection | Letting fraud slip through costs money

### AUC-ROC — "how good is the model overall?" 📈
- AUC answers: across all possible decision thresholds, how well the model separates positives from negatives.
- A score of 1.0 = perfect separation
- A score of 0.5 = no better than random guessing
- Use AUC when you want to compare models without committing to a threshold, or when classes are imbalanced and accuracy would be misleading.

Scenario | Why AUC matters
--- | ---
Comparing two classifiers for loan default | AUC tells you which ranks risk better, regardless of threshold
Imbalanced dataset (1% positive class) | Accuracy is useless; AUC captures the full trade-off curve
Search/recommendation ranking | You care about relative ordering, not a single cut-off

### The precision–recall trade-off 🔄
These two metrics pull in opposite directions. Lowering your decision threshold catches more positives (↑ recall) but also more false alarms (↓ precision).

The F1 score (`2 × P × R / (P + R)`) is the harmonic mean of both — useful when you need a single number that respects both.