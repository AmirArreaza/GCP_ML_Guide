# Vertex Explainable AI & Vertex AI Model Monitoring
> **GCP Professional ML Engineer Study Guide**

---

## Part 1 — Vertex Explainable AI (XAI)

---

### 1. What is Vertex Explainable AI?

Vertex Explainable AI answers the question: **"Why did the model make this prediction?"**

It computes **feature attributions** — a score for each input feature showing how much it pushed the prediction up or down relative to a baseline. All methods are based on variants of **Shapley values** from cooperative game theory, where each feature is treated as a "player" and receives proportional credit for the outcome.

**Why it matters:**
- Regulatory compliance (financial, healthcare, hiring decisions)
- Debugging poor predictions and catching data leakage
- Building user and stakeholder trust
- Detecting feature attribution drift in production (combined with Model Monitoring)

---

### 2. The Three Attribution Methods

| Method | Best For | How It Works |
|--------|---------|-------------|
| **Sampled Shapley** | Any model — tabular, non-differentiable, tree ensembles | Randomly samples feature subsets and averages contributions. Default for AutoML tabular. |
| **Integrated Gradients (IG)** | Differentiable models — TF/Keras neural networks | Integrates gradients along a path from baseline to input. Works only on differentiable (gradient-capable) models. |
| **XRAI** | Image models | Extends IG to rank image *regions* (superpixels) rather than individual pixels. Most intuitive for visual tasks. |

**Decision rule:**
```
Model type:          Choose:
Neural network  →    Integrated Gradients (or XRAI for images)
Tree ensemble   →    Sampled Shapley
Any other       →    Sampled Shapley
Image           →    XRAI (regions) or Integrated Gradients (pixels)
```

> **Exam tip:** Integrated Gradients does NOT work on non-differentiable models. Sampled Shapley works on all model types but is more expensive than IG on neural networks.

---

### 3. Key Concepts

**Baseline** — the reference input against which attributions are measured. Attribution = (model output for input) – (model output for baseline).

- Default: zero vector or black image
- Recommended: median values of training data, or min/max pair for stability
- More baselines → lower approximation error (but higher latency for IG/XRAI; no extra latency for Sampled Shapley)

**Attribution score** — positive = feature pushed prediction higher; negative = pushed it lower; magnitude = strength of effect.

**approximation error** — returned as `instanceOutputValue - baselineOutputValue`. If less than `0.05`, consider adjusting baselines or increasing `stepCount` / `path_count`.

**Global vs local explanations:**
- **Local** — per-prediction feature attribution (what XAI returns by default)
- **Global** — aggregate attributions over the full dataset (run via batch explanations)

---

### 4. Configuration and Code

#### Configure ExplanationSpec when uploading a model

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import explain

# --- Sampled Shapley (any model) ---
explanation_params = explain.ExplanationParameters(
    sampled_shapley_attribution=explain.SampledShapleyAttribution(
        path_count=50     # number of random paths — higher = more accurate, slower
    )
)

# --- Integrated Gradients (neural network / differentiable) ---
explanation_params = explain.ExplanationParameters(
    integrated_gradients_attribution=explain.IntegratedGradientsAttribution(
        step_count=50,    # interpolation steps along path from baseline to input
        smooth_grad_config=explain.SmoothGradConfig(
            noise_sigma=0.1,
            noisy_sample_count=3,
        ),
    )
)

# --- XRAI (images) ---
explanation_params = explain.ExplanationParameters(
    xrai_attribution=explain.XraiAttribution(
        step_count=50,
    )
)

# ExplanationMetadata — describes model inputs (required for custom models)
explanation_metadata = explain.ExplanationMetadata(
    inputs={
        "income":       explain.ExplanationMetadata.InputMetadata(),
        "credit_score": explain.ExplanationMetadata.InputMetadata(),
        "debt_ratio":   explain.ExplanationMetadata.InputMetadata(),
    },
    outputs={"probability": explain.ExplanationMetadata.OutputMetadata()},
)

# Upload model with explanations configured
model = aiplatform.Model.upload(
    display_name="loan-model-with-xai",
    artifact_uri="gs://my-bucket/models/loan-model/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
    explanation_metadata=explanation_metadata,
    explanation_parameters=explanation_params,
)
```

#### Deploy to Endpoint with explanations

```python
endpoint = model.deploy(
    deployed_model_display_name="loan-model-xai-endpoint",
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3,
    explanation_metadata=model.explanation_metadata,
    explanation_parameters=model.explanation_parameters,
)
```

#### Online prediction with explanation

```python
# Use endpoint.explain() instead of endpoint.predict()
response = endpoint.explain(
    instances=[{
        "income": 65000,
        "credit_score": 720,
        "debt_ratio": 0.35,
        "employment_years": 5,
    }]
)

prediction = response.predictions[0]
explanation = response.explanations[0]

# Feature attributions
for attr in explanation.attributions:
    print(f"Baseline output: {attr.baseline_output_value:.4f}")
    print(f"Instance output: {attr.instance_output_value:.4f}")
    for feature, score in attr.feature_attributions.items():
        print(f"  {feature}: {score:+.4f}")

# Example output:
#   income:           +0.32
#   credit_score:     +0.28
#   debt_ratio:       -0.15
#   employment_years: +0.09
```

#### Batch explanations

```python
# Use generate_explanation=True for batch jobs
batch_job = model.batch_predict(
    job_display_name="loan-batch-xai",
    gcs_source="gs://my-bucket/batch-input/instances.jsonl",
    gcs_destination_prefix="gs://my-bucket/batch-output/",
    machine_type="n1-standard-4",
    generate_explanation=True,       # ← enable explanations in batch
)
batch_job.wait()
```

---

### 5. AutoML vs Custom Model Explanations

| Aspect | AutoML Tabular | Custom Model |
|--------|---------------|-------------|
| Method | Sampled Shapley (automatic) | Configured via ExplanationSpec |
| Metadata | Auto-inferred | Required for non-TF2 models |
| Baselines | Auto-selected | Optional — you can specify |
| Global feature importance | Available in Model Registry | Via aggregated batch explanations |

> **TF2 models only:** If you use a TensorFlow 2 SavedModel with a prebuilt serving container, Vertex AI can auto-infer input/output metadata — `ExplanationMetadata` is optional.

---

## Part 2 — Vertex AI Model Monitoring

---

### 6. What is Vertex AI Model Monitoring?

Model Monitoring watches **deployed endpoints** (and batch jobs) for changes in the statistical distribution of input features over time. It alerts you when something changes significantly, so you can decide whether to retrain.

**It detects two distinct problems:**

| Problem | Description | Baseline | Root Cause |
|---------|-------------|---------|-----------|
| **Training-serving skew** | Feature distribution at serving differs from training distribution | Training data | Bug in pipeline — wrong transformation, missing feature, timezone error |
| **Inference drift** | Feature distribution at serving changes significantly *over time* | Past production data (rolling window) | World changed — seasonality, user behaviour shift, new data source |

> **Key distinction:** Skew = difference between training and serving (can exist from day 1). Drift = change within serving data over time (develops gradually).

---

### 7. How Detection Works

Model Monitoring uses **TensorFlow Data Validation (TFDV)** to compute statistical distributions and distance scores.

| Feature type | Distribution computed | Distance metric |
|-------------|----------------------|----------------|
| **Numerical** | Histogram of equal-width buckets | Jensen-Shannon divergence |
| **Categorical** | Frequency of each category | L-infinity distance |

When the distance score between two distributions exceeds your configured threshold, Monitoring raises an alert.

**Default threshold:** `0.3` (30% statistical distance). Adjust per feature.

---

### 8. Monitoring Architecture

```
Deployed Endpoint
    ↓ (prediction requests logged at sampling_rate %)
BigQuery logging table
    ↓ (monitored on schedule: every monitor_interval hours)
Distribution computed by TFDV
    ↓ (compared to baseline)
[Skew] vs training data baseline
[Drift] vs previous production window
    ↓ (if distance > threshold)
Email alert + Cloud Monitoring alert
    ↓ (view in Console)
Feature distribution histograms over time
```

---

### 9. Configuration and Code

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import model_monitoring

aiplatform.init(project="my-project", location="us-central1")

# Skew detection config (training data as baseline)
skew_config = model_monitoring.SkewDetectionConfig(
    data_source="gs://my-bucket/training-data/train.csv",
    skew_thresholds={
        "income":        model_monitoring.ThresholdConfig(value=0.2),
        "credit_score":  model_monitoring.ThresholdConfig(value=0.2),
        "debt_ratio":    model_monitoring.ThresholdConfig(value=0.3),
    },
    # Also monitor attribution skew (requires XAI configured on model)
    attribute_skew_thresholds={
        "income":        model_monitoring.ThresholdConfig(value=0.3),
        "credit_score":  model_monitoring.ThresholdConfig(value=0.3),
    },
)

# Drift detection config (production data shifts over time)
drift_config = model_monitoring.DriftDetectionConfig(
    drift_thresholds={
        "income":        model_monitoring.ThresholdConfig(value=0.3),
        "credit_score":  model_monitoring.ThresholdConfig(value=0.3),
    },
    attribute_drift_thresholds={
        "income":        model_monitoring.ThresholdConfig(value=0.3),
    },
)

# Sampling + scheduling
logging_strategy = model_monitoring.RandomSamplingConfig(sample_rate=0.1)  # 10% of requests

# Create the monitoring job
monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="loan-model-monitoring",
    endpoint=endpoint.resource_name,
    logging_sampling_strategy=logging_strategy,
    monitoring_interval=1,           # run every 1 hour (minimum)
    skew_detection_configs=[
        model_monitoring.ModelMonitoringObjectiveConfig(
            deployed_model_id=deployed_model_id,
            objective_config=model_monitoring.ModelMonitoringObjectiveConfig.ObjectiveConfig(
                training_dataset=model_monitoring.ModelMonitoringObjectiveConfig.TrainingDataset(
                    data_format="csv",
                    gcs_source=model_monitoring.GcsSource(
                        uris=["gs://my-bucket/training-data/train.csv"]
                    ),
                    target_field="loan_approved",
                ),
                training_prediction_skew_detection_config=skew_config,
                prediction_drift_detection_config=drift_config,
            ),
        )
    ],
    alert_config=model_monitoring.ModelMonitoringAlertConfig(
        email_alert_config=model_monitoring.ModelMonitoringAlertConfig.EmailAlertConfig(
            user_emails=["mlteam@company.com"],
        ),
    ),
)

print(f"Monitoring job: {monitoring_job.resource_name}")
```

---

### 10. What Gets Logged and Where

When monitoring is enabled, prediction requests are sampled and logged automatically:

| Storage location | Contains |
|-----------------|---------|
| `gs://.../model_monitoring/job-.../serving/` | Hourly production feature distributions |
| `gs://.../model_monitoring/job-.../training/` | Training baseline distribution (for skew) |
| `gs://.../model_monitoring/job-.../feature_attribution_score/` | Feature attribution distributions (if XAI enabled) |
| BigQuery table | Raw sampled prediction requests |

---

### 11. Monitoring for Batch Predictions

Model Monitoring also supports **one-time skew detection for batch jobs**:

```python
batch_job = model.batch_predict(
    job_display_name="batch-with-monitoring",
    gcs_source="gs://my-bucket/batch-input/*.jsonl",
    gcs_destination_prefix="gs://my-bucket/batch-output/",
    machine_type="n1-standard-4",
    model_monitoring_config=aiplatform.BatchPredictionJob.ModelMonitoringConfig(
        schema_file_path="gs://my-bucket/schema/model_schema.yaml",   # input schema
        bigquery_source_training_data=f"bq://my-project.dataset.training_table",
        skew_detection_config=skew_config,
    ),
)
```

---

### 12. XAI + Monitoring Together — Attribution Drift

When both Explainable AI and Model Monitoring are active, you can monitor **feature attribution drift** — changes in how much each feature contributes to predictions over time.

This is more powerful than raw feature drift because a feature's input distribution can stay stable while its *influence on the model* shifts (e.g., income stopped being predictive of loan approval after a policy change).

```
Monitor raw feature distributions  →  detect data pipeline issues
Monitor attribution distributions  →  detect changes in model behaviour
```

| Alert type | What it means |
|-----------|--------------|
| Feature skew | Training data differs from what's arriving at the endpoint |
| Feature drift | Production distribution is changing over time |
| Attribution skew | Feature *importance* at serving differs from at training time |
| Attribution drift | Feature *importance* is shifting as production data changes |

---

## Part 3 — Exam Quick Reference

### Attribution Method Selector

```
Is the model differentiable (neural network)?
  Yes + tabular → Integrated Gradients
  Yes + images  → XRAI (regions) or Integrated Gradients (pixels)
  No            → Sampled Shapley (any model type)
AutoML tabular  → Sampled Shapley (automatic, no config needed)
```

### Skew vs Drift

```
Skew  = training distribution ≠ serving distribution   →  fix the pipeline bug
Drift = serving distribution changing over time         →  consider retraining
```

### Key Parameters

| Parameter | What it controls |
|-----------|----------------|
| `path_count` | Sampled Shapley accuracy vs latency |
| `step_count` | IG / XRAI accuracy vs latency |
| `input_baselines` | Reference for attribution calculation |
| `sample_rate` | Fraction of requests logged (0–1) |
| `monitor_interval` | How often to run analysis (minimum 1 hour) |
| `skew_thresholds` | Per-feature alerting threshold (default 0.3) |
| `drift_thresholds` | Per-feature alerting threshold (default 0.3) |

### Exam Tips

- Vertex Explainable AI is **deprecated** for the stand-alone Vertex Explainable AI service (now integrated directly into Vertex AI endpoints).
- Use `endpoint.explain()` for online, `generate_explanation=True` for batch.
- **AutoML tabular models** use Sampled Shapley automatically — no configuration required.
- **IG and XRAI** only work on differentiable (gradient-based) models.
- Extra baselines reduce approximation error but increase latency for IG/XRAI — not for Sampled Shapley.
- Model Monitoring uses **TFDV** (TensorFlow Data Validation) under the hood.
- Default alert threshold is **0.3** for both skew and drift.
- Attribution skew/drift monitoring requires both **XAI** and **Model Monitoring** to be enabled together.
- Batch monitoring = one-time skew only. Online monitoring = continuous skew + drift.
- Cloud Monitoring and email alerts are both configurable for anomaly notification.