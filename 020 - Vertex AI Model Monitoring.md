# Vertex AI Model Monitoring — Study Guide
### GCP Professional Machine Learning Engineer Certification

---

## 📌 What Is Vertex AI Model Monitoring?

**Vertex AI Model Monitoring** is a managed service that continuously monitors deployed models on **Vertex AI Endpoints** for data quality issues and model degradation. It alerts you when the **statistical distribution of incoming prediction requests drifts** from the training data baseline — a signal that your model may be becoming unreliable.

> **Exam mindset:** Model Monitoring answers: *"Is my deployed model still trustworthy in production?"* It is your early-warning system against silent model decay.

---

## 🧠 Why Models Degrade — The Core Problem

Models are trained on a static snapshot of the world. Production data is not static.

| Root Cause | Description |
|---|---|
| **Data drift** | The statistical distribution of input features shifts over time |
| **Concept drift** | The relationship between features and the target label changes |
| **Data quality issues** | Missing values, schema violations, out-of-range inputs arrive at inference time |
| **Training-serving skew** | A mismatch exists between training data and live serving data — often due to feature engineering inconsistencies |

> **Exam tip:** Vertex AI Model Monitoring primarily detects **data drift** and **training-serving skew**. Concept drift (label distribution shift) requires you to also log **ground truth** labels and run separate evaluation jobs — it is not detected automatically.

---

## 🔑 Core Concepts

### 1. Training-Serving Skew
A discrepancy between the feature distributions seen **during training** and those seen **during serving**.

**Common causes:**
- Feature engineering applied differently in training vs. serving pipelines
- Data leakage or preprocessing bugs
- Different data sources used for training vs. production traffic

**How it's detected:**
Vertex AI compares the **training dataset baseline** (provided by you, e.g., a BigQuery table or GCS CSV) against **logged prediction requests** using statistical distance metrics.

---

### 2. Prediction Drift
A shift in the distribution of **incoming prediction request features** over time, compared to either:
- The training baseline, **or**
- A previous monitoring window

This is purely time-based — no training data is required, only the rolling window of recent requests.

---

### 3. Feature Attributions Monitoring *(Skew & Drift)*
In addition to raw feature distributions, Vertex AI Model Monitoring can monitor **feature attributions** — how much each feature contributes to predictions (using **Explainable AI / SHAP values**).

| Type | What It Detects |
|---|---|
| **Feature attribution skew** | Attribution distributions differ between training and serving |
| **Feature attribution drift** | Attribution distributions shift over time in production |

> **Exam tip:** Feature attribution monitoring requires **Explainable AI to be enabled** on the endpoint. It is more computationally expensive but gives deeper insight into *why* predictions are changing, not just *what* inputs changed.

---

### 4. Monitoring Job
The unit of configuration in Vertex AI Model Monitoring. A **ModelDeploymentMonitoringJob** is attached to a **Vertex AI Endpoint** and defines:
- Which deployed models to monitor
- Which features to watch
- What alert thresholds to apply
- How often to sample and evaluate

---

## 🏗️ Architecture Overview

```
Production Traffic
        │
        ▼
┌──────────────────┐
│  Vertex AI       │  ◄── Deployed Model(s)
│  Endpoint        │
└──────┬───────────┘
       │ Prediction request logs (sampled)
       ▼
┌──────────────────┐
│  Cloud Storage   │  ◄── Logged prediction inputs & outputs
│  (GCS Bucket)    │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────┐
│  Model Monitoring Job            │
│                                  │
│  Compare:                        │
│  • Current window vs. baseline   │
│  • Statistical distance metrics  │
└──────┬───────────────────────────┘
       │ Alert triggered if threshold exceeded
       ▼
┌──────────────────┐
│  Cloud Monitoring│  ──► Email / PagerDuty / Pub/Sub alerts
│  (Alerting)      │
└──────────────────┘
```

---

## 📐 Statistical Distance Metrics

Vertex AI Model Monitoring uses these metrics to quantify distribution shift:

### For Numerical Features

| Metric | Full Name | What It Measures |
|---|---|---|
| **Jensen-Shannon Divergence** | JS Divergence | Symmetric measure of similarity between two probability distributions. Range: [0, 1]. 0 = identical. |

### For Categorical Features

| Metric | Full Name | What It Measures |
|---|---|---|
| **L-infinity Distance** | Chebyshev Distance | Maximum absolute difference between two probability distributions over categories. Range: [0, 1]. |

> **Exam tip:** Know which metric applies to which feature type. JS Divergence → numerical. L-infinity → categorical.

You set an **alert threshold** per metric (e.g., `0.3`). If the distance exceeds the threshold, an alert fires.

---

## ⚙️ Setting Up Model Monitoring

### Prerequisites
1. A model deployed to a **Vertex AI Endpoint**
2. A **GCS bucket** for logging prediction inputs
3. Optionally: a **training dataset** (BigQuery table or GCS CSV) for skew detection

### Step 1 — Enable Prediction Request Logging on the Endpoint

```python
from google.cloud import aiplatform

endpoint = aiplatform.Endpoint("projects/.../endpoints/123")

# Enable request/response logging to GCS
endpoint.update(
    traffic_split={"model-id": 100},
    enable_request_response_logging=True,
    request_response_logging_sampling_rate=0.1,  # Sample 10% of traffic
    request_response_logging_bq_destination="bq://my-project.my_dataset.logs"
)
```

### Step 2 — Create a Model Monitoring Job

```python
from google.cloud.aiplatform_v1.types import (
    ModelDeploymentMonitoringJob,
    ModelDeploymentMonitoringObjectiveConfig,
    SamplingStrategy,
    ModelMonitoringAlertConfig,
    ThresholdConfig,
)

job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="fraud-model-monitor",
    endpoint=endpoint.resource_name,
    logging_sampling_strategy=SamplingStrategy(
        random_sample_config=SamplingStrategy.RandomSampleConfig(sample_rate=0.1)
    ),
    model_deployment_monitoring_objective_configs=[
        ModelDeploymentMonitoringObjectiveConfig(
            deployed_model_id="my-deployed-model-id",
            objective_config=ModelMonitoringObjectiveConfig(
                training_dataset=ModelMonitoringObjectiveConfig.TrainingDataset(
                    bigquery_source=BigQuerySource(input_uri="bq://project.dataset.train_table"),
                    target_field="label",
                ),
                training_prediction_skew_detection_config=TrainingPredictionSkewDetectionConfig(
                    skew_thresholds={
                        "feature_1": ThresholdConfig(value=0.3),
                        "feature_2": ThresholdConfig(value=0.3),
                    }
                ),
                prediction_drift_detection_config=PredictionDriftDetectionConfig(
                    drift_thresholds={
                        "feature_1": ThresholdConfig(value=0.3),
                    }
                ),
            )
        )
    ],
    model_monitoring_alert_config=ModelMonitoringAlertConfig(
        email_alert_config=ModelMonitoringAlertConfig.EmailAlertConfig(
            user_emails=["mlops-team@company.com"]
        )
    ),
    monitor_interval=Duration(seconds=3600),  # Run every hour
)
```

---

## 📊 Monitoring Schedules & Sampling

| Parameter | Description |
|---|---|
| **Monitor interval** | How frequently the monitoring job analyses collected data (e.g., every 1 hour, 24 hours) |
| **Sampling rate** | Fraction of prediction requests logged (e.g., `0.1` = 10%). Higher = more accurate stats, higher cost |
| **Analysis window** | The time window of recent requests evaluated per monitoring run |

> **Exam tip:** You do **not** need to log 100% of traffic. A representative sample is sufficient. The trade-off is statistical accuracy vs. storage/processing cost.

---

## 🚨 Alerting & Notifications

When a threshold is breached, Vertex AI Model Monitoring integrates with:

| Channel | Mechanism |
|---|---|
| **Email** | Configured directly in the monitoring job |
| **Cloud Monitoring** | Metrics published to `aiplatform.googleapis.com` namespace |
| **Pub/Sub** | For programmatic/event-driven responses |
| **PagerDuty / Slack** | Via Cloud Monitoring alerting policy integrations |

### Alert Anatomy
Each alert includes:
- Which **feature** drifted
- The **measured distance** vs. the **threshold**
- The **deployed model** and **endpoint** affected
- A link to the **monitoring dashboard** in the GCP Console

---

## 🖥️ Monitoring Dashboard

The Vertex AI Console provides a built-in dashboard showing:
- Feature distribution plots (training baseline vs. current serving window)
- Alert history and status
- Per-feature drift scores over time
- Skew scores per feature

---

## 🔁 Model Monitoring in an MLOps Pipeline

Model Monitoring is the **trigger layer** in a mature MLOps loop:

```
Deploy Model
     │
     ▼
Monitor (Vertex AI Model Monitoring)
     │
     ├── No drift detected → Continue serving
     │
     └── Drift / Skew detected
              │
              ▼
         Investigate root cause
              │
         ┌───┴────────────────────┐
         │                        │
    Retrain model           Fix data pipeline
         │                        │
         ▼                        ▼
    Deploy new model        Re-deploy model
```

---

## 🆚 Skew vs. Drift — Side-by-Side

| | Training-Serving Skew | Prediction Drift |
|---|---|---|
| **Compares** | Training data vs. current serving data | Past serving window vs. current serving window |
| **Requires training data?** | ✅ Yes | ❌ No |
| **Detects** | Pipeline inconsistencies, data leakage | Gradual input distribution changes over time |
| **Triggered by** | One-time analysis vs. baseline | Rolling window comparison |
| **Best for** | Catching launch-time issues | Catching gradual production decay |

---

## 🆚 Model Monitoring vs. Related Services

| Service | Purpose | When to Use |
|---|---|---|
| **Vertex AI Model Monitoring** | Detect feature drift and skew on deployed endpoints | Continuous production health monitoring |
| **Vertex Explainable AI** | Understand per-prediction feature importance | Debugging, fairness, required for attribution monitoring |
| **Vertex ML Metadata** | Track experiment and pipeline lineage | Reproducibility, audit trail |
| **Cloud Monitoring** | Infrastructure metrics (latency, error rate, QPS) | SRE / operational monitoring |
| **Vertex Pipelines** | Orchestrate retraining workflows | Triggered retraining on drift detection |
| **Looker / BigQuery** | Analyse logged prediction data at scale | BI-level analysis of model inputs/outputs |

---

## 💡 Common Exam Scenarios & Answers

**Scenario 1:** *Your model's accuracy has silently degraded over 3 months in production. Which Vertex AI feature should you have enabled to catch this early?*
→ **Vertex AI Model Monitoring** with prediction drift detection.

**Scenario 2:** *You notice the model performs well on your test set but poorly in production. No new concept drift is suspected. What is the likely cause?*
→ **Training-serving skew** — a difference in feature preprocessing between training and serving pipelines.

**Scenario 3:** *You want to understand not just that drift occurred, but which features are now contributing differently to predictions.*
→ Enable **feature attribution monitoring** (requires Explainable AI on the endpoint).

**Scenario 4:** *Your monitoring job fires alerts too frequently on minor fluctuations. How do you fix this?*
→ **Increase the alert threshold** values for the affected features, or **reduce the sampling rate** to smooth statistical noise.

**Scenario 5:** *You want to trigger automatic retraining when drift is detected. What is the recommended architecture?*
→ Model Monitoring alert → **Pub/Sub topic** → **Cloud Functions / Cloud Run** → trigger **Vertex Pipelines** retraining pipeline.

---

## ⚡ Exam Focus Areas

1. **Skew vs. drift** — know the difference, what each requires, and when each fires
2. **JS Divergence (numerical) vs. L-infinity (categorical)** — know which metric maps to which feature type
3. **Explainable AI dependency** — feature attribution monitoring requires XAI enabled
4. **Sampling** — 100% logging is not required; sampling is sufficient and cost-effective
5. **Alert routing** — email, Cloud Monitoring, Pub/Sub for automated responses
6. **MLOps loop** — monitoring as a trigger for retraining pipelines
7. **Concept drift is NOT auto-detected** — requires ground truth labels and separate evaluation

---

## 📝 Quick Revision Checklist

- [ ] I can explain training-serving skew vs. prediction drift
- [ ] I know JS Divergence is for numerical features and L-infinity for categorical
- [ ] I understand what is required to detect skew (training baseline) vs. drift (no baseline needed)
- [ ] I know feature attribution monitoring requires Explainable AI
- [ ] I can describe the alert routing options (email, Cloud Monitoring, Pub/Sub)
- [ ] I understand the MLOps feedback loop: monitor → alert → retrain
- [ ] I know concept drift requires ground truth labels and is not auto-detected

---

## 📚 Further Reading

- [Vertex AI Model Monitoring Overview](https://cloud.google.com/vertex-ai/docs/model-monitoring/overview)
- [Set Up Model Monitoring](https://cloud.google.com/vertex-ai/docs/model-monitoring/using-model-monitoring)
- [Vertex Explainable AI](https://cloud.google.com/vertex-ai/docs/explainable-ai/overview)
- [Model Monitoring Metrics Reference](https://cloud.google.com/vertex-ai/docs/model-monitoring/model-monitoring-metrics)
- [MLOps on Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

*Guide prepared for GCP Professional Machine Learning Engineer Certification preparation.*