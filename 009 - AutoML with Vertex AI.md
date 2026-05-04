# AutoML with Vertex AI
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is AutoML on Vertex AI?

Vertex AI AutoML is a **fully managed, no-code/low-code ML service** that automatically trains, tunes, and evaluates models for a given dataset and objective — without requiring manual algorithm selection, feature engineering, or hyperparameter tuning.

AutoML handles the entire ML pipeline:
```
Data ingestion → Feature analysis → Model search → HPO → Evaluation → Deployment
```

**Key advantages:**
- No ML expertise required to get a baseline model
- Searches across many model architectures automatically
- Integrated with Vertex AI for deployment, monitoring, and batch prediction
- Supports tabular, image, text, and video data

---

## 2. AutoML Data Types and Objectives

### Tabular Data

| Objective | Task | Description |
|-----------|------|-------------|
| `REGRESSION` | Predict a numeric value | Price, demand, score |
| `CLASSIFICATION` | Predict a category (binary or multiclass) | Churn, fraud, sentiment |
| `FORECASTING` | Predict future time-series values | Sales, traffic, energy |

### Image Data

| Objective | Description |
|-----------|-------------|
| `IMAGE_CLASSIFICATION` | Assign single or multiple labels to an image |
| `IMAGE_OBJECT_DETECTION` | Detect and locate objects within an image |
| `IMAGE_SEGMENTATION` | Classify each pixel in an image |

### Text Data

| Objective | Description |
|-----------|-------------|
| `TEXT_CLASSIFICATION` | Assign categories to text documents |
| `TEXT_ENTITY_EXTRACTION` | Extract named entities from text |
| `TEXT_SENTIMENT_ANALYSIS` | Predict sentiment score |

### Video Data

| Objective | Description |
|-----------|-------------|
| `VIDEO_CLASSIFICATION` | Label entire video clips |
| `VIDEO_OBJECT_TRACKING` | Track objects across frames |
| `VIDEO_ACTION_RECOGNITION` | Identify actions in video |

---

## 3. Creating a Vertex AI Dataset

A **Vertex AI Dataset** is a managed resource that organises training data and links it to one or more training runs.

### Via Google Cloud Console

1. Navigate to **Vertex AI → Datasets → Create Dataset**
2. Choose **data type**: Tabular / Image / Text / Video
3. Choose **objective** (classification, regression, etc.)
4. **Import data** from:
   - Google Cloud Storage (CSV, JSONL, image files)
   - BigQuery table or view
   - Local file upload (small datasets)
5. Configure **data split** (training / validation / test)
6. Review **data statistics** and feature distributions

### Via Python SDK

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Create a tabular dataset from BigQuery
dataset = aiplatform.TabularDataset.create(
    display_name="churn-dataset",
    bq_source="bq://my-project.my_dataset.customers",
    # or use gcs_source for CSV:
    # gcs_source=["gs://my-bucket/data/customers.csv"],
)

print(f"Dataset resource name: {dataset.resource_name}")
```

```python
# Create an image dataset from GCS
dataset = aiplatform.ImageDataset.create(
    display_name="product-images",
    gcs_source=["gs://my-bucket/image-import.csv"],
    import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
)
```

### Via REST API / gcloud CLI

```bash
# Create dataset via gcloud
gcloud ai datasets create \
  --region=us-central1 \
  --display-name="churn-dataset" \
  --metadata-schema-uri="gs://google-cloud-aiplatform/schema/dataset/metadata/tabular_1.0.0.yaml"
```

### Data Import Formats

| Data Type | Supported Formats |
|-----------|------------------|
| Tabular | CSV (GCS), BigQuery table/view |
| Image | JPEG, PNG, GIF, BMP via import CSV/JSONL |
| Text | TXT, JSONL with text content |
| Video | MP4, MOV, AVI via import CSV |

### Data Split Options

| Method | Description |
|--------|-------------|
| **Auto split** | Vertex AI splits 80/10/10 (train/validation/test) automatically |
| **Manual** | Specify `ml_use` column in your data (`TRAINING`, `VALIDATION`, `TEST`) |
| **Custom** | Provide explicit data split fractions |

> **Exam tip:** Use an `ml_use` column with values `TRAINING`, `VALIDATION`, `TEST` (or `UNASSIGNED`) to control splits manually.

---

## 4. Training an AutoML Model

### Via Python SDK

```python
from google.cloud import aiplatform

# Tabular classification
job = aiplatform.AutoMLTabularTrainingJob(
    display_name="churn-automl-job",
    optimization_prediction_type="classification",    # or 'regression', 'forecasting'
    optimization_objective="maximize-au-roc",         # objective metric
    column_transformations=[
        {"numeric": {"column_name": "age"}},
        {"categorical": {"column_name": "contract_type"}},
        {"text": {"column_name": "notes"}},
    ],
    # budget_milli_node_hours controls training cost/duration
)

model = job.run(
    dataset=dataset,
    target_column="churn_label",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    budget_milli_node_hours=1000,    # 1 node-hour = 1000 milli-node-hours
    model_display_name="churn-automl-model",
    disable_early_stopping=False,    # stop early if no improvement
)
```

### Key Training Parameters

| Parameter | Description |
|-----------|-------------|
| `optimization_prediction_type` | `classification`, `regression`, `forecasting` |
| `optimization_objective` | Metric to optimise (see table below) |
| `budget_milli_node_hours` | Training compute budget (1000 = 1 node-hour) |
| `target_column` | The label column to predict |
| `disable_early_stopping` | If `False`, training stops early if no improvement |
| `column_transformations` | Override auto-detected column types |

### Optimisation Objectives by Task

| Task | Objective Options |
|------|------------------|
| Binary Classification | `maximize-au-roc`, `minimize-log-loss`, `maximize-au-prc`, `maximize-precision-at-recall`, `maximize-recall-at-precision` |
| Multiclass Classification | `minimize-log-loss` |
| Regression | `minimize-rmse`, `minimize-mae`, `minimize-rmsle` |
| Forecasting | `minimize-rmse`, `minimize-mae`, `minimize-rmsle` |

### Budget Guidelines

| Budget (milli-node-hours) | Typical Use Case |
|--------------------------|-----------------|
| 1,000 | Quick prototype / feasibility check |
| 2,000–8,000 | Standard model quality |
| 8,000+ | High-quality production model |

> **Exam tip:** `budget_milli_node_hours` controls both cost and model quality. AutoML may stop early if the model converges — actual spend may be less than budget.

---

## 5. Evaluating an AutoML Model

### Via Console

Navigate to **Vertex AI → Models → [your model] → Evaluate**:
- View **confusion matrix**
- Adjust **confidence threshold** and see precision/recall update live
- View **feature importance** (for tabular models)
- Inspect **per-class metrics** for multiclass

### Via Python SDK

```python
model = aiplatform.Model("projects/my-project/locations/us-central1/models/MODEL_ID")

# List all evaluations
evaluations = model.list_model_evaluations()
for eval in evaluations:
    print(eval.metrics)

# Get specific evaluation metrics
evaluation = model.get_model_evaluation()
print(evaluation.metrics)
```

### Tabular Classification Metrics

| Metric | Description |
|--------|-------------|
| `auRoc` | Area Under ROC Curve |
| `auPrc` | Area Under Precision-Recall Curve |
| `logLoss` | Cross-entropy loss |
| `confidenceMetrics` | Per-threshold precision, recall, F1 |
| `confusionMatrix` | Actual vs predicted class counts |

### Tabular Regression Metrics

| Metric | Description |
|--------|-------------|
| `rootMeanSquaredError` | RMSE |
| `meanAbsoluteError` | MAE |
| `rSquared` | R² coefficient of determination |
| `meanAbsolutePercentageError` | MAPE |

---

## 6. Deploying a Model to an Endpoint

An **Endpoint** is a stable URL that hosts one or more deployed models and serves **online (real-time) predictions**.

### Endpoint Architecture

```
Client request → Endpoint (REST URL) → Deployed Model → Prediction response

One endpoint can host multiple model versions with traffic splits:
  Endpoint → 80% Model_v2 + 20% Model_v1   (canary deployment / A/B testing)
```

### Step 1 — Create an Endpoint

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

endpoint = aiplatform.Endpoint.create(
    display_name="churn-prediction-endpoint",
    description="Real-time churn prediction",
    labels={"env": "production", "team": "ml"},
)

print(f"Endpoint: {endpoint.resource_name}")
```

```bash
# Via gcloud
gcloud ai endpoints create \
  --region=us-central1 \
  --display-name="churn-prediction-endpoint"
```

### Step 2 — Deploy Model to Endpoint

```python
deployed_model = endpoint.deploy(
    model=model,
    deployed_model_display_name="churn-model-v1",
    machine_type="n1-standard-4",          # compute resource
    min_replica_count=1,                   # min instances (always-on)
    max_replica_count=5,                   # max for autoscaling
    traffic_split={"0": 100},             # 100% to this model (0 = new deployment)
    # For GPU:
    # accelerator_type="NVIDIA_TESLA_T4",
    # accelerator_count=1,
)
```

### Step 3 — Get Online Predictions

```python
# Single instance prediction
response = endpoint.predict(
    instances=[
        {"age": 35, "contract_type": "monthly", "tenure_months": 12}
    ]
)
print(response.predictions)

# Multiple instances
response = endpoint.predict(
    instances=[
        {"age": 35, "contract_type": "monthly", "tenure_months": 12},
        {"age": 52, "contract_type": "annual", "tenure_months": 48},
    ]
)
```

### Machine Types for Deployment

| Machine Type | vCPU | RAM | Use Case |
|-------------|------|-----|---------|
| `n1-standard-2` | 2 | 7.5 GB | Light workloads |
| `n1-standard-4` | 4 | 15 GB | Standard models |
| `n1-standard-8` | 8 | 30 GB | Heavy models |
| `n1-highmem-2` | 2 | 13 GB | Memory-intensive |
| `c2-standard-4` | 4 | 16 GB | Compute-intensive |

### Traffic Splitting (A/B Testing)

```python
# Deploy a second model version with a traffic split
endpoint.deploy(
    model=model_v2,
    deployed_model_display_name="churn-model-v2",
    machine_type="n1-standard-4",
    traffic_split={
        "deployed_model_v1_id": 70,   # 70% to v1
        "0": 30,                       # 30% to new v2
    },
)
```

### Undeploy / Delete Endpoint

```python
# Undeploy a specific model version
endpoint.undeploy(deployed_model_id="DEPLOYED_MODEL_ID")

# Delete the entire endpoint
endpoint.delete(force=True)   # force=True undeploys all models first
```

---

## 7. Batch Prediction

**Batch prediction** processes **large volumes of data asynchronously** — no always-on endpoint needed. Results are written to GCS or BigQuery.

```
Use online prediction  →  real-time, low latency, small payloads
Use batch prediction   →  high throughput, large datasets, cost-efficient, async
```

### Step 1 — Prepare Input Data

**From BigQuery:**
```sql
-- Input table must match model's expected feature schema
SELECT age, contract_type, tenure_months, monthly_charges
FROM `project.dataset.customers_to_score`
```

**From GCS (JSONL format):**
```json
{"age": 35, "contract_type": "monthly", "tenure_months": 12}
{"age": 52, "contract_type": "annual", "tenure_months": 48}
```

### Step 2 — Submit Batch Prediction Job

```python
from google.cloud import aiplatform

# Batch prediction from BigQuery to BigQuery
batch_prediction_job = model.batch_predict(
    job_display_name="churn-batch-job",
    bigquery_source="bq://my-project.my_dataset.customers_to_score",
    instances_format="bigquery",
    bigquery_destination_prefix="bq://my-project.my_dataset.churn_predictions",
    predictions_format="bigquery",
    machine_type="n1-standard-4",
    starting_replica_count=1,
    max_replica_count=5,
    generate_explanation=True,            # include Shapley explanations
    explanation_metadata=explanation_md,
    explanation_parameters=explanation_params,
)

batch_prediction_job.wait()
print(f"Output location: {batch_prediction_job.output_info}")
```

```python
# Batch prediction from GCS to GCS
batch_prediction_job = model.batch_predict(
    job_display_name="churn-batch-gcs",
    gcs_source=["gs://my-bucket/input/customers.jsonl"],
    instances_format="jsonl",
    gcs_destination_prefix="gs://my-bucket/output/predictions/",
    predictions_format="jsonl",
    machine_type="n1-standard-4",
    starting_replica_count=2,
    max_replica_count=10,
)
```

### Batch Prediction Parameters

| Parameter | Description |
|-----------|-------------|
| `job_display_name` | Human-readable name for the job |
| `bigquery_source` | Input BigQuery table URI |
| `gcs_source` | Input GCS file path(s) |
| `instances_format` | `'bigquery'`, `'jsonl'`, `'csv'` |
| `bigquery_destination_prefix` | Output BigQuery dataset URI |
| `gcs_destination_prefix` | Output GCS folder URI |
| `predictions_format` | `'bigquery'`, `'jsonl'`, `'csv'` |
| `machine_type` | Compute resource for batch workers |
| `starting_replica_count` | Initial number of parallel workers |
| `max_replica_count` | Maximum workers for autoscaling |
| `generate_explanation` | Include feature attribution with each prediction |

### Batch Prediction Output Schema

**JSONL output:**
```json
{"instance": {"age": 35, "contract_type": "monthly"}, "prediction": {"scores": [0.82, 0.18], "classes": ["churn", "no_churn"]}}
```

**BigQuery output columns:**
- All original input columns preserved
- `predicted_<label>` — winning class label
- `predicted_<label>_probs` — per-class probability scores

### Monitoring a Batch Job

```python
# Check job status
print(batch_prediction_job.state)

# Block until complete
batch_prediction_job.wait()

# Check for errors
if batch_prediction_job.has_ended:
    print(batch_prediction_job.output_info)
```

---

## 8. Online vs Batch Prediction — Decision Guide

| Aspect | Online (Endpoint) | Batch |
|--------|-------------------|-------|
| Latency | Milliseconds | Minutes to hours |
| Throughput | Low-medium | Very high |
| Cost model | Per-hour (always-on) | Per-prediction (pay-as-you-go) |
| Infrastructure | Always running | Spun up / down per job |
| Input | Single or small batch via API | Large files (GCS or BQ) |
| Output | Synchronous response | Written to GCS or BigQuery |
| Best for | Real-time apps, user-facing | Nightly scoring, bulk inference |
| Requires endpoint | Yes | No — uses model directly |

---

## 9. Explainability (Feature Attribution)

Vertex AI AutoML supports **Explainable AI** to understand model predictions.

```python
# Online prediction with explanations
response = endpoint.explain(
    instances=[{"age": 35, "contract_type": "monthly", "tenure_months": 12}]
)

for explanation in response.explanations:
    for attribution in explanation.attributions:
        print(attribution.feature_attributions)
```

**Attribution methods for AutoML Tabular:**
- **Sampled Shapley** — approximation of Shapley values
- **Integrated Gradients** — gradient-based attribution
- **XRAI** — region-based attribution (images)

---

## 10. Vertex AI AutoML vs Custom Training

| Aspect | AutoML | Custom Training |
|--------|--------|----------------|
| Code required | None (Console) / Minimal (SDK) | Full ML code |
| Control | Low | Full |
| Training time | Longer (model search) | Faster iteration |
| Model architecture | Chosen by AutoML | You choose |
| Best for | Baseline, quick value, non-ML teams | Advanced models, research |
| Budget control | `budget_milli_node_hours` | Machine type + time |
| Explainability | Built-in | Custom setup needed |

---

## 11. Key Vertex AI Resource Hierarchy

```
Project
└── Location (region, e.g. us-central1)
    ├── Dataset            ← training data
    ├── Training Job       ← AutoML or custom training
    ├── Model              ← trained model artifact
    ├── Endpoint           ← online serving
    └── Batch Prediction Job ← async large-scale inference
```

---

## 12. Exam-Relevant Tips

- A **Vertex AI Dataset** must be created before training — it manages data registration and split configuration.
- Use `ml_use` column with `TRAINING`, `VALIDATION`, `TEST` for **manual data splits**.
- `budget_milli_node_hours` controls AutoML training budget — **1000 = 1 node-hour**.
- AutoML may stop **before** the budget is consumed if the model converges early.
- An **Endpoint** is required for online (real-time) prediction — not for batch.
- One endpoint can host **multiple model versions** with configurable traffic splits (A/B testing).
- **Batch prediction** writes output to BigQuery or GCS — no endpoint needed.
- `generate_explanation=True` in batch prediction adds **Shapley feature attributions** per row.
- `min_replica_count=0` allows the endpoint to **scale to zero** (cold start latency; not available for all machine types).
- **`instances_format`** and **`predictions_format`** in batch jobs must match the source/destination types.
- For tabular data, Vertex AI AutoML automatically handles feature engineering, encoding, and scaling.
- **`endpoint.explain()`** returns per-feature attributions for online prediction.
- Model evaluation metrics are accessed via `model.get_model_evaluation()` or the Console.
- Always choose a **region** consistent across dataset, training, endpoint, and batch job.

---

## 13. Quick Reference Cheat Sheet

```
DATASET
  aiplatform.TabularDataset.create()    →  register tabular data (BQ or GCS)
  aiplatform.ImageDataset.create()      →  register image data
  ml_use column                         →  TRAINING / VALIDATION / TEST splits

TRAINING
  AutoMLTabularTrainingJob.run()        →  train with budget_milli_node_hours
  optimization_prediction_type          →  classification / regression / forecasting
  optimization_objective                →  maximize-au-roc, minimize-rmse, etc.

EVALUATION
  model.get_model_evaluation()          →  auROC, RMSE, confusionMatrix, etc.
  model.list_model_evaluations()        →  all evaluation slices

ENDPOINT (Online Prediction)
  aiplatform.Endpoint.create()          →  create serving endpoint
  endpoint.deploy(model, machine_type)  →  deploy model with traffic split
  endpoint.predict(instances)           →  synchronous real-time prediction
  endpoint.explain(instances)           →  prediction + feature attributions

BATCH PREDICTION
  model.batch_predict(                  →  async large-scale inference
    bigquery_source / gcs_source,
    predictions_format,
    generate_explanation=True
  )
```

---

*Study tip: The exam tests when to use online vs batch prediction, how to configure data splits with the `ml_use` column, and the relationship between Dataset → Training Job → Model → Endpoint. Also know that batch prediction does not require a deployed endpoint.*