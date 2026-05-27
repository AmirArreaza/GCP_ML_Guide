# Vertex AI Model Registry — Complete Guide

The **Vertex AI Model Registry** is a centralised repository for managing the full lifecycle of ML models on Google Cloud. It acts as a single source of truth for all models in an organisation — tracking versions, metadata, lineage, evaluation metrics, and deployment history.

Think of it as the **"source control for your models"**: just as Git tracks code changes, the Model Registry tracks model iterations, who trained them, how they performed, and where they are deployed.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Model versioning** | Store and compare multiple versions of the same model |
| **Metadata & lineage** | Track training datasets, pipeline runs, hyperparameters, and evaluation results |
| **Deployment tracking** | See which model version is live on which endpoint |
| **Access control** | Govern who can view, modify, or deploy models via IAM |
| **Evaluation integration** | Link model evaluation jobs directly to registry entries |
| **Alias management** | Assign human-readable aliases (e.g. `champion`, `challenger`) to versions |

---

## Core Concepts

### Model vs Model Version

```
Model (logical container)
 ├── Version 1  ← trained 2024-01-10, sklearn, accuracy 0.87
 ├── Version 2  ← trained 2024-03-05, XGBoost, accuracy 0.91
 └── Version 3  ← trained 2024-06-20, XGBoost + new features, accuracy 0.94  ← champion
```

A **Model** is a named container that holds one or more **Model Versions**. Each version is an immutable artifact corresponding to a specific training run.

### Model Aliases

Aliases are mutable pointers that can be reassigned between versions without changing endpoint configurations.

| Alias | Typical Use |
|-------|-------------|
| `champion` | Currently deployed production model |
| `challenger` | Candidate model being A/B tested |
| `latest` | Most recently trained version |
| `baseline` | Reference model for regression testing |

### Model Artifacts

Each model version stores:

- **Model artifact URI** — the Cloud Storage path to the saved model files (SavedModel, pickle, ONNX, etc.)
- **Container image** — the serving container (pre-built or custom)
- **Framework metadata** — TensorFlow, PyTorch, scikit-learn, XGBoost, etc.
- **Labels and descriptions** — free-form metadata for search and governance

---

## Uploading Models to the Registry

### Method 1: Vertex AI SDK (Python)

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

model = aiplatform.Model.upload(
    display_name="fraud-detection-xgboost",
    artifact_uri="gs://my-bucket/models/fraud-v3/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    labels={"team": "risk", "env": "production"},
    description="XGBoost fraud detection model trained on Q2 2024 data",
)

print(f"Model uploaded: {model.resource_name}")
```

### Method 2: gcloud CLI

```bash
gcloud ai models upload \
  --region=us-central1 \
  --display-name=fraud-detection-xgboost \
  --artifact-uri=gs://my-bucket/models/fraud-v3/ \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest \
  --description="XGBoost fraud model Q2 2024"
```

### Method 3: From a Vertex AI Training Job

When using Vertex AI Training, the model is automatically registered if you pass `model_display_name`:

```python
job = aiplatform.CustomTrainingJob(
    display_name="fraud-training-job",
    script_path="trainer/train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)

model = job.run(
    dataset=dataset,
    model_display_name="fraud-detection-sklearn",  # auto-registers to Model Registry
    replica_count=1,
    machine_type="n1-standard-4",
)
```

### Method 4: From a Vertex AI Pipeline

```python
from google_cloud_pipeline_components.v1.model import ModelUploadOp

upload_model_op = ModelUploadOp(
    project=project,
    display_name="pipeline-fraud-model",
    unmanaged_container_model=training_op.outputs["unmanaged_container_model"],
)
```

---

## Managing Model Versions

### Uploading a New Version to an Existing Model

```python
# Upload as a new version of an existing model (pass parent_model)
model_v2 = aiplatform.Model.upload(
    display_name="fraud-detection-xgboost",
    artifact_uri="gs://my-bucket/models/fraud-v4/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    parent_model="projects/my-project/locations/us-central1/models/MODEL_ID",
    is_default_version=False,
)
```

### Listing Model Versions

```python
model = aiplatform.Model("projects/my-project/locations/us-central1/models/MODEL_ID")

for version in model.versioning_registry.list_versions():
    print(version.version_id, version.create_time, version.version_aliases)
```

### Assigning Aliases

```python
# Promote version 3 to champion
model.versioning_registry.add_version_aliases(
    aliases=["champion"],
    version="3",
)

# Demote old champion alias from version 2
model.versioning_registry.remove_version_aliases(
    aliases=["champion"],
    version="2",
)
```

### Deleting a Version

```python
# Delete a specific version (not the parent model)
model.versioning_registry.delete_version(version="2")
```

---

## Model Evaluation in the Registry

Vertex AI allows you to run **Model Evaluation jobs** and attach the results directly to a model version in the registry.

### Running an Evaluation Job

```python
eval_job = aiplatform.ModelEvaluationJob.submit(
    model_name="projects/my-project/locations/us-central1/models/MODEL_ID",
    model_version="3",
    prediction_type="classification",
    ground_truth_gcs_source=["gs://my-bucket/eval/ground_truth.jsonl"],
    gcs_source_uri=["gs://my-bucket/eval/predictions.jsonl"],
    target_field_name="label",
    class_labels=["fraud", "legitimate"],
)
```

### Viewing Evaluation Metrics via SDK

```python
model = aiplatform.Model("projects/.../models/MODEL_ID@3")
evaluations = model.list_model_evaluations()

for eval in evaluations:
    print(eval.metrics)
    # Output: {'auRoc': 0.97, 'logLoss': 0.12, 'confidenceMetrics': [...]}
```

### Evaluation Metrics Stored per Version

| Metric Type | Examples |
|-------------|----------|
| **Classification** | AUC-ROC, AUC-PR, F1, precision, recall, log loss, confusion matrix |
| **Regression** | RMSE, MAE, R², MAPE |
| **Forecasting** | WAPE, RMSE at horizon |
| **Custom** | Any metric written to a JSON artifact |

---

## Deploying from the Registry

Models in the registry are deployed to **Endpoints**. The registry stores which versions are deployed and where.

### Deploy a Registry Model to an Endpoint

```python
endpoint = aiplatform.Endpoint.create(display_name="fraud-endpoint")

# Deploy from registry using alias
model = aiplatform.Model(
    "projects/my-project/locations/us-central1/models/MODEL_ID@champion"
)

model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="fraud-champion",
    machine_type="n1-standard-4",
    traffic_percentage=100,
)
```

### Traffic Splitting Between Versions (A/B Testing)

```python
# Deploy challenger at 20%, champion at 80%
endpoint.deploy(
    model=challenger_model,
    deployed_model_display_name="fraud-challenger",
    machine_type="n1-standard-4",
    traffic_split={
        "0": 80,                         # existing champion
        challenger_model.resource_name: 20,  # new challenger
    },
)
```

### Checking Deployment Status from Registry

```python
model = aiplatform.Model("projects/.../models/MODEL_ID")
print(model.deployed_models)
# Returns list of DeployedModel objects with endpoint and traffic info
```

---

## Model Lineage & Metadata

Vertex AI Model Registry integrates with **Vertex ML Metadata** (MLMD) to automatically capture lineage — the chain of datasets, pipelines, and experiments that produced each model.

### What Lineage Tracks

```
BigQuery Dataset ──► Vertex AI Pipeline Run ──► Training Job ──► Model Version 3
                           │
                      Hyperparameters
                      (lr=0.01, max_depth=6)
```

### Querying Lineage via SDK

```python
from google.cloud import aiplatform_v1

metadata_client = aiplatform_v1.MetadataServiceClient()

# Get all artifacts linked to a model
lineage = metadata_client.query_artifact_lineage_subgraph(
    artifact="projects/.../locations/.../metadataStores/default/artifacts/MODEL_ARTIFACT_ID",
    max_hops=10,
)
```

### Manually Adding Metadata to a Model

```python
model.update(
    labels={
        "training_dataset": "bq-fraud-v4",
        "git_commit": "abc1234",
        "experiment": "exp-42",
        "data_start": "2024-01-01",
        "data_end": "2024-06-30",
    }
)
```

---

## IAM & Access Control

The Model Registry uses standard **Google Cloud IAM** to control access.

### Key IAM Roles

| Role | What It Allows |
|------|----------------|
| `roles/aiplatform.user` | Read models, run predictions |
| `roles/aiplatform.developer` | Upload, update, and delete models |
| `roles/aiplatform.admin` | Full control including IAM policy management |
| `roles/aiplatform.viewer` | Read-only access to model metadata |

### Granting Access to a Team

```bash
# Grant developer access to the ML engineering service account
gcloud projects add-iam-policy-binding my-project \
  --member="serviceAccount:ml-engineer@my-project.iam.gserviceaccount.com" \
  --role="roles/aiplatform.developer"
```

### Resource-Level IAM (Model-Specific)

```python
# Set IAM policy on a specific model
model.set_iam_policy(
    policy={
        "bindings": [
            {
                "role": "roles/aiplatform.viewer",
                "members": ["group:data-science@company.com"],
            }
        ]
    }
)
```

---

## Monitoring Deployed Models

After deployment, you can attach **Model Monitoring** to detect drift between training data distribution and live traffic.

### Enable Monitoring

```python
job = aiplatform.ModelDeploymentMonitoringJob.create(
    display_name="fraud-model-monitoring",
    endpoint=endpoint.resource_name,
    logging_sampling_strategy=aiplatform.gapic.SamplingStrategy(
        random_sample_config=aiplatform.gapic.SamplingStrategy.RandomSampleConfig(
            sample_rate=0.1  # Monitor 10% of traffic
        )
    ),
    model_deployment_monitoring_objective_configs=[
        aiplatform.gapic.ModelDeploymentMonitoringObjectiveConfig(
            deployed_model_id=deployed_model_id,
            objective_config=aiplatform.gapic.ModelMonitoringObjectiveConfig(
                training_dataset=aiplatform.gapic.ModelMonitoringObjectiveConfig.TrainingDataset(
                    gcs_source=aiplatform.gapic.GcsSource(
                        uris=["gs://my-bucket/training/data.csv"]
                    ),
                    target_field="label",
                ),
                training_prediction_skew_detection_config=aiplatform.gapic.ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
                    skew_thresholds={"amount": aiplatform.gapic.ThresholdConfig(value=0.3)}
                ),
            ),
        )
    ],
    model_monitoring_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig(
        email_alert_config=aiplatform.gapic.ModelMonitoringAlertConfig.EmailAlertConfig(
            user_emails=["ml-ops@company.com"]
        )
    ),
    monitor_interval={"seconds": 3600},
)
```

---

## Champion/Challenger Pattern

A best-practice workflow for safe model promotion in production.

```
┌──────────────────────────────────────────────────────────┐
│                    Model Registry                        │
│                                                          │
│  fraud-detection                                         │
│  ├── v2  [alias: champion]   → 80% production traffic   │
│  └── v3  [alias: challenger] → 20% production traffic   │
└──────────────────────────────────────────────────────────┘
                          │
               Monitor metrics for 2 weeks
                          │
          ┌───────────────┴────────────────┐
          │ v3 outperforms v2?             │
          │                               │
         Yes                              No
          │                               │
  Promote v3 to champion         Roll back: 100% to v2
  Reassign alias                 Delete v3 from endpoint
  Retire v2 version
```

### Implementation Steps

```python
# Step 1: Deploy challenger at 20%
endpoint.deploy(model=v3_model, traffic_split={"deployed_v2_id": 80, "0": 20})

# Step 2: Monitor for 2 weeks (check Vertex AI Model Monitoring dashboard)

# Step 3a: Promote if challenger wins
model.versioning_registry.add_version_aliases(aliases=["champion"], version="3")
model.versioning_registry.remove_version_aliases(aliases=["champion"], version="2")
endpoint.update(traffic_split={"deployed_v3_id": 100})

# Step 3b: Rollback if champion holds
endpoint.undeploy(deployed_model_id="deployed_v3_id")
endpoint.update(traffic_split={"deployed_v2_id": 100})
```

---

## Pre-built Serving Containers

Vertex AI provides managed containers for common frameworks — no Docker knowledge needed.

| Framework | Container URI |
|-----------|---------------|
| TensorFlow 2.12 | `us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest` |
| PyTorch 1.13 | `us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest` |
| scikit-learn 1.0 | `us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest` |
| XGBoost 1.7 | `us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest` |
| LightGBM 3.3 | `us-docker.pkg.dev/vertex-ai/prediction/lightgbm-cpu.3-3:latest` |
| Custom (Prediction Routine) | Build with `google-cloud-aiplatform[prediction]` |

For custom serving logic (preprocessing, post-processing), use the **Prediction Routine** pattern with a `CustomPredictor` class.

---

## Exam Quick-Reference: Common Scenarios

| Scenario | Best Approach |
|-----------|---------------|
| Compare two model versions fairly in production | Deploy both versions to the same endpoint using **traffic splitting** |
| Promote a new model without changing client endpoint URL | Use **aliases** (`champion`) — reassign alias to new version; endpoint URL stays the same |
| Reproduce a previous model exactly | Use **model lineage** in MLMD to retrieve training dataset, pipeline run, and hyperparameters |
| Detect when live predictions drift from training data | Enable **Model Monitoring** with skew detection on the deployed endpoint |
| Prevent a junior team from deleting production models | Apply **IAM roles** — grant `aiplatform.viewer` only; restrict `aiplatform.developer` to non-prod environments |
| Register a model trained outside of Vertex AI | Use `Model.upload()` with `artifact_uri` pointing to a GCS path |
| Track which dataset version was used per model version | Store dataset references in **model labels** or link via **Vertex ML Metadata** |
| Roll back to a previous model version instantly | Keep prior version registered; reassign `champion` alias and update endpoint traffic split |
| Evaluate model on a held-out test set post-training | Run a **ModelEvaluationJob** and attach results to the specific model version |
| Share a model across multiple GCP projects | Use **cross-project model sharing** via IAM on the model resource |

---

## Key Terms Summary

| Term | Definition |
|------|------------|
| **Model** | Logical container grouping all versions of a model |
| **Model Version** | Immutable snapshot of a trained model artifact |
| **Alias** | Mutable pointer (e.g. `champion`) that can be reassigned between versions |
| **Artifact URI** | GCS path where model files (weights, binaries) are stored |
| **Serving Container** | Docker image used to serve predictions |
| **Endpoint** | Deployed serving resource that routes traffic to one or more model versions |
| **Model Evaluation** | Stored metrics (AUC, RMSE, etc.) attached to a model version |
| **Model Monitoring** | Continuous job that detects skew and drift in live traffic |
| **ML Metadata (MLMD)** | Lineage store tracking datasets, pipelines, and model relationships |
| **Traffic Split** | Percentage of endpoint traffic routed to each deployed model version |

## Feature Attributions in the Model Registry

You can enable feature attribution by configuring it when you upload model artifacts to the Vertex AI Model Registry. Vertex Explainable AI is not a standalone product — it is embedded within the Model Registry and Endpoints. If an exam question describes provisioning a separate explainability service, that framing is wrong. Explainability is a capability layered onto models you have already registered and deployed. GoogleGCP Study Hub

### Two Overarching Approaches

Vertex Explainable AI offers two overarching approaches: feature-based explanations that attribute predictions to input features, and example-based explanations that compare an input to similar training data points.   

- **Feature-based** — answers "which features drove this prediction?"
- **Example-based** — answers "what training examples is this prediction based on?"

### The 3 Feature Attribution Methods

For feature-based explanations, Vertex Explainable AI supports three methods, and which one you pick depends on the type of model and the type of input. 

|Method|Best For|How It Works|
|------|--------|------------|
|**Sampled Shapley (SHAP)**|Tree-based & ensemble models (XGBoost, Random Forest) on tabular data|Approximates Shapley values by sampling feature permutations
|**Integrated Gradients**|Differentiable models (neural networks) on tabular or image data (pixel-level)|Integrates gradients along the path from baseline to input|
|**XRAI**|Neural networks on images, when you want human-readable regions|Groups pixels into meaningful regions (shapes, textures) instead of highlighting individual pixels|