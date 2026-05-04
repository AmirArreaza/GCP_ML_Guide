# Anomaly Detection with Autoencoders in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Anomaly Detection?

Anomaly detection identifies observations that deviate significantly from the expected pattern. Unlike supervised classification, anomaly detection is typically **unsupervised** — it learns the normal distribution of the data and flags outliers as anomalies.

**Use cases:** Fraud detection, network intrusion detection, manufacturing defect detection, IoT sensor anomalies, data quality monitoring, unusual account activity.

```
Normal data    →  reconstruction error is LOW
Anomalous data →  reconstruction error is HIGH
```

---

## 2. Autoencoders — How They Work

An **autoencoder** is a neural network trained to **compress** (encode) data into a low-dimensional representation and then **reconstruct** (decode) it back to the original form.

```
Input X  →  Encoder  →  Latent Space (bottleneck)  →  Decoder  →  Reconstructed X'

Reconstruction Error = ||X - X'||²  (mean squared difference)
```

The model is trained on **normal data only**. When it encounters an anomaly, the decoder cannot reconstruct it well → **high reconstruction error** = anomaly signal.

---

## 3. Autoencoder Architecture in BQML

BigQuery ML implements autoencoders as a special variant of the `AUTOENCODER` model type, which is based on a deep neural network with a symmetric encoder-decoder structure:

```
Input Layer
     ↓
[Dense Layer 1]  — encoder hidden layer
     ↓
[Bottleneck Layer]  — compressed latent representation
     ↓
[Dense Layer 2]  — decoder hidden layer
     ↓
Output Layer (reconstruction)
```

The hidden layers are configured symmetrically using `hidden_units`.

---

## 4. Full Workflow

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_autoencoder`
OPTIONS (
  model_type          = 'AUTOENCODER',
  hidden_units        = [32, 16, 32],   -- encoder-bottleneck-decoder (symmetric)
  activation_fn       = 'RELU',          -- 'RELU', 'SIGMOID', 'TANH', 'LEAKY_RELU'
  batch_size          = 32,
  dropout             = 0.2,             -- regularization
  learn_rate          = 0.001,
  optimizer           = 'ADAM',          -- 'ADAM', 'ADAGRAD', 'RMSPROP', 'SGD'
  max_iterations      = 10,             -- training epochs
  l2_reg              = 0.01,
  data_split_method   = 'AUTO_SPLIT'
) AS
SELECT
  feature1,
  feature2,
  feature3,
  feature4
FROM
  `project.dataset.normal_data`       -- train on NORMAL data only
WHERE
  is_normal = TRUE;
```

> **Critical:** Train exclusively on **normal / non-anomalous** samples. If anomalies are included in training, the model learns to reconstruct them too — defeating the purpose.

### Step 2 — Detect Anomalies with ML.DETECT_ANOMALIES

```sql
SELECT *
FROM ML.DETECT_ANOMALIES(
  MODEL `project.dataset.my_autoencoder`,
  STRUCT(0.05 AS contamination),        -- expected fraction of anomalies
  (
    SELECT feature1, feature2, feature3, feature4
    FROM `project.dataset.data_to_score`
  )
);
```

**Output columns:**

| Column | Description |
|--------|-------------|
| `is_anomaly` | BOOL — TRUE if the row is flagged as an anomaly |
| `mean_absolute_error` | Reconstruction error for this row |
| `stddev_reconstruction_error` | Standard deviation across features |
| `reconstruction_loss` | Total reconstruction loss for the row |

### Step 3 — Reconstruct Inputs with ML.PREDICT

```sql
-- Get the reconstructed values for each input row
SELECT *
FROM ML.PREDICT(
  MODEL `project.dataset.my_autoencoder`,
  (SELECT feature1, feature2, feature3, feature4
   FROM `project.dataset.data_to_score`)
);
```

Output includes reconstructed feature values alongside originals — manually compute per-row reconstruction error if needed.

### Step 4 — Evaluate the Model

```sql
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.my_autoencoder`);
```

Returns training and evaluation reconstruction loss metrics.

---

## 5. The `contamination` Parameter

The `contamination` parameter in `ML.DETECT_ANOMALIES` sets the **expected proportion of anomalies** in the data, which determines the reconstruction error threshold:

```
contamination = 0.01   →  expect 1% anomalies  (very conservative — only extreme outliers)
contamination = 0.05   →  expect 5% anomalies  (default — reasonable for many use cases)
contamination = 0.10   →  expect 10% anomalies (more aggressive — higher recall)
```

Internally, BQML sets the anomaly threshold at the `(1 - contamination)` quantile of reconstruction errors observed on the training data.

> **Exam tip:** Higher `contamination` → lower threshold → **more rows flagged** as anomalies (higher recall, lower precision). Lower `contamination` → fewer flags (higher precision, lower recall).

---

## 6. Key OPTIONS Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `model_type` | — | `'AUTOENCODER'` — required |
| `hidden_units` | — | Array of ints defining encoder-bottleneck-decoder layer sizes |
| `activation_fn` | `'RELU'` | Activation function: `RELU`, `SIGMOID`, `TANH`, `LEAKY_RELU`, `CRELU`, `ELU` |
| `batch_size` | `32` | Mini-batch size for training |
| `dropout` | `0.0` | Dropout rate for regularization (0–1) |
| `learn_rate` | auto | Step size for gradient descent |
| `optimizer` | `'ADAM'` | Optimiser: `ADAM`, `ADAGRAD`, `RMSPROP`, `SGD` |
| `max_iterations` | `20` | Number of training epochs |
| `l1_reg` | `0` | L1 regularization on weights |
| `l2_reg` | `0` | L2 regularization on weights |
| `data_split_method` | `'RANDOM'` | How to split training vs evaluation data |
| `learn_rate_strategy` | `'CONSTANT'` | `'CONSTANT'` or `'LINE_SEARCH'` |
| `early_stop` | `TRUE` | Stop if reconstruction loss plateaus |
| `min_rel_progress` | `0.01` | Min improvement needed to continue |
| `num_trials` | `1` | HPO trials via Vertex AI Vizier |

---

## 7. hidden_units — Architecture Design

The `hidden_units` array defines the autoencoder's layers. **Symmetric design** is the standard:

```
hidden_units = [64, 32, 16, 32, 64]
                encoder     decoder
                    ↑ bottleneck ↑

Layer sizes:
[64]  →  first encoder layer
[32]  →  second encoder layer
[16]  →  bottleneck (most compressed representation)
[32]  →  first decoder layer
[64]  →  second decoder layer
```

| Design choice | Effect |
|--------------|--------|
| Small bottleneck | More compression → better anomaly detection for structured patterns |
| Large bottleneck | Less compression → better reconstruction fidelity |
| More layers | More complex patterns learned |
| Fewer layers | Faster, less prone to overfitting |
| Typical setup | 2–3 encoder layers, mirror in decoder |

---

## 8. Reconstruction Error as Anomaly Score

```
For each row: error = Σ (original_feature_i - reconstructed_feature_i)²
                       over all features

High error  →  model struggled to reconstruct → likely anomaly
Low error   →  model reconstructed easily     → likely normal
```

You can manually threshold reconstruction errors for custom sensitivity:

```sql
SELECT
  *,
  mean_absolute_error,
  CASE WHEN mean_absolute_error > 0.15 THEN TRUE ELSE FALSE END AS custom_anomaly_flag
FROM ML.DETECT_ANOMALIES(
  MODEL `project.dataset.my_autoencoder`,
  STRUCT(0.05 AS contamination),
  (SELECT * FROM `project.dataset.data_to_score`)
);
```

---

## 9. Evaluation Metrics

`ML.EVALUATE` for autoencoders returns:

| Metric | Description | Goal |
|--------|-------------|------|
| `mean_absolute_error` | Average absolute reconstruction error | Lower |
| `mean_squared_error` | Average squared reconstruction error | Lower |
| `root_mean_squared_error` | Same units as features | Lower |

> Note: There is no `precision`, `recall`, or `roc_auc` because the autoencoder is unsupervised — ground truth labels are not required at training time.
>
> To compute precision/recall, you need labelled anomaly data for post-hoc evaluation.

---

## 10. Autoencoder vs Other Anomaly Detection Approaches in BQML

| Model | Type | Best For |
|-------|------|---------|
| `AUTOENCODER` | Deep learning | Complex non-linear patterns, high-dimensional data |
| `kmeans` + distance | Clustering | Simple, interpretable, low-dimensional data |
| PCA + reconstruction | Dimensionality reduction | Linear anomalies, structured numeric data |
| ARIMA_PLUS residuals | Time series | Temporal anomalies (spikes and dips) |
| Isolation Forest | Tree-based (external) | Not natively in BQML — use Vertex AI |

> **Exam tip:** `ML.DETECT_ANOMALIES` is specific to the `AUTOENCODER` model type in BQML. It is not available for other model types.

---

## 11. Preprocessing Considerations

- **Normalise numeric features** before training — autoencoders are sensitive to scale. Use `ML.STANDARD_SCALER` inside a `TRANSFORM` clause.
- **Handle NULLs** — BQML imputes NULLs automatically but consider explicit handling for anomaly detection tasks.
- **Categorical features** — encode manually or use `ML.ONE_HOT_ENCODER` in `TRANSFORM`. Autoencoders work best with numeric inputs.

```sql
CREATE OR REPLACE MODEL `project.dataset.my_autoencoder`
TRANSFORM(
  ML.STANDARD_SCALER(feature1) OVER() AS scaled_f1,
  ML.STANDARD_SCALER(feature2) OVER() AS scaled_f2,
  ML.STANDARD_SCALER(feature3) OVER() AS scaled_f3
)
OPTIONS (
  model_type   = 'AUTOENCODER',
  hidden_units = [16, 8, 16]
)
AS SELECT feature1, feature2, feature3
   FROM `project.dataset.normal_data`;
```

---

## 12. ML Functions — Quick Reference

| Function | Description |
|----------|-------------|
| `ML.DETECT_ANOMALIES` | Flag rows as anomalies based on reconstruction error + contamination threshold |
| `ML.PREDICT` | Return reconstructed feature values per row |
| `ML.EVALUATE` | Return training reconstruction loss (MAE, MSE, RMSE) |

---

## 13. Exam-Relevant Tips

- `model_type = 'AUTOENCODER'` — the specific model type for autoencoder-based anomaly detection in BQML.
- Train on **normal data only** — including anomalies in training degrades detection quality.
- `ML.DETECT_ANOMALIES` is the primary inference function — **not `ML.PREDICT`**.
- The `contamination` parameter controls the anomaly threshold — higher → more anomalies flagged.
- `hidden_units` should be **symmetric** — encodes then mirrors for decoding.
- The **bottleneck layer** (smallest value in `hidden_units`) is where the compressed representation lives.
- There is **no `roc_auc`** or classification metric from `ML.EVALUATE` — it's unsupervised.
- **Normalise features** before training autoencoders — they are sensitive to feature scale.
- `ML.PREDICT` on an autoencoder returns **reconstructed values** — not class labels.
- `dropout` and `l2_reg` are the key regularization levers to prevent the model from perfectly reconstructing everything (which would make anomaly detection useless).
- `ML.DETECT_ANOMALIES` returns `is_anomaly` (BOOL) and `mean_absolute_error` (reconstruction error per row).

---

## 14. Quick Reference Cheat Sheet

```
model_type            →  'AUTOENCODER'
hidden_units          →  symmetric array e.g. [64, 32, 16, 32, 64]
Train on              →  NORMAL data only — never include anomalies
ML.DETECT_ANOMALIES   →  primary anomaly scoring function
  contamination       →  expected anomaly fraction (threshold control)
  is_anomaly          →  output boolean flag
  mean_absolute_error →  per-row reconstruction error
ML.PREDICT            →  returns reconstructed feature values
ML.EVALUATE           →  MAE, MSE, RMSE on reconstruction
No roc_auc / F1       →  unsupervised — no labels at train time
```

---

*Study tip: Know the distinction between `ML.DETECT_ANOMALIES` (anomaly flags) and `ML.PREDICT` (reconstructed values), the role of `contamination`, and why autoencoders must be trained exclusively on normal data — these are the most likely exam focus areas.*