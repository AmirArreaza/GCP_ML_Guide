# Vertex AI Pipelines — Study Guide
### GCP Professional Machine Learning Engineer Certification

---

## 📌 What Are Vertex AI Pipelines?

**Vertex AI Pipelines** is a managed, serverless orchestration service for running **ML workflows as reproducible, auditable DAGs** (Directed Acyclic Graphs). Each step in the pipeline runs as an isolated container on GCP infrastructure, and the entire pipeline — inputs, outputs, parameters, and lineage — is automatically tracked via **Vertex ML Metadata**.

> **Exam mindset:** Vertex AI Pipelines answers: *"How do I make my ML workflow reproducible, scalable, and production-grade?"* It is the orchestration backbone of MLOps on GCP.

---

## 🧠 Why Pipelines? The Core Problem

Without pipelines, ML workflows are:
- **Manual** — steps run by hand, error-prone, hard to reproduce
- **Monolithic** — one large script, impossible to partially re-run
- **Untracked** — no record of which data/code produced which model
- **Not scalable** — can't parallelise steps or distribute compute

Vertex AI Pipelines solves all of this by treating each step as a **containerised, versioned, reusable component**.

---

## 🔑 Core Concepts

### 1. Pipeline
A **Pipeline** is a DAG of **components** connected by data dependencies. It is defined in Python using the **Kubeflow Pipelines (KFP) SDK v2** and compiled to a YAML/JSON spec before submission.

```
Raw Data ──► Preprocess ──► Train ──► Evaluate ──► Conditional Deploy
                                          │
                                    (if accuracy > 0.9)
```

---

### 2. Component
A **Component** is a single, self-contained step in a pipeline. It is:
- **Containerised** — runs inside a Docker image
- **Typed** — inputs and outputs have declared types
- **Reusable** — can be used across multiple pipelines
- **Tracked** — automatically logged to Vertex ML Metadata

There are two ways to define a component:

#### a) Lightweight Python Component
Quick to define — KFP wraps your function in a base Python container automatically.

```python
from kfp.v2 import dsl
from kfp.v2.dsl import component

@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.2.0", "pandas==2.0.0"]
)
def train_model(
    dataset_uri: str,
    learning_rate: float,
    model_output: dsl.Output[dsl.Model],
    metrics_output: dsl.Output[dsl.Metrics],
):
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    import pickle

    df = pd.read_csv(dataset_uri)
    X, y = df.drop("label", axis=1), df["label"]

    model = GradientBoostingClassifier(learning_rate=learning_rate)
    model.fit(X, y)

    accuracy = model.score(X, y)
    metrics_output.log_metric("accuracy", accuracy)

    with open(model_output.path, "wb") as f:
        pickle.dump(model, f)
```

#### b) Custom Container Component
Full control — you specify your own Docker image and entrypoint.

```python
@component(
    base_image="gcr.io/my-project/my-training-image:latest"
)
def train_model_custom(
    dataset_uri: str,
    model_output: dsl.Output[dsl.Model],
):
    ...
```

---

### 3. Artifact Types
KFP v2 provides typed artifacts for ML objects — these map directly to Vertex ML Metadata artifact schemas:

| KFP Artifact Type | Vertex ML Metadata Schema | Description |
|---|---|---|
| `dsl.Dataset` | `system.Dataset` | Input/output datasets |
| `dsl.Model` | `system.Model` | Trained model files |
| `dsl.Metrics` | `system.Metrics` | Scalar evaluation metrics |
| `dsl.ClassificationMetrics` | `system.ClassificationMetrics` | Confusion matrix, ROC curve |
| `dsl.HTML` | `system.HTML` | HTML reports |
| `dsl.Markdown` | `system.Markdown` | Markdown reports |

---

### 4. Pipeline Definition

```python
from kfp.v2 import dsl

@dsl.pipeline(
    name="fraud-detection-pipeline",
    description="End-to-end fraud detection training pipeline",
    pipeline_root="gs://my-bucket/pipeline-root/",
)
def fraud_pipeline(
    dataset_uri: str = "gs://my-bucket/data/train.csv",
    learning_rate: float = 0.05,
    accuracy_threshold: float = 0.90,
):
    # Step 1 — Preprocess
    preprocess_task = preprocess_data(dataset_uri=dataset_uri)

    # Step 2 — Train (depends on preprocess output)
    train_task = train_model(
        dataset_uri=preprocess_task.outputs["processed_dataset"],
        learning_rate=learning_rate,
    )

    # Step 3 — Evaluate
    eval_task = evaluate_model(
        model=train_task.outputs["model_output"],
        dataset_uri=preprocess_task.outputs["processed_dataset"],
    )

    # Step 4 — Conditional Deploy
    with dsl.Condition(
        eval_task.outputs["accuracy"] > accuracy_threshold,
        name="accuracy-gate"
    ):
        deploy_task = deploy_model(
            model=train_task.outputs["model_output"],
        )
```

---

### 5. Compiling & Running a Pipeline

```python
from kfp.v2 import compiler
from google.cloud import aiplatform

# Compile to YAML spec
compiler.Compiler().compile(
    pipeline_func=fraud_pipeline,
    package_path="fraud_pipeline.yaml",
)

# Submit to Vertex AI
aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="fraud-detection-run-001",
    template_path="fraud_pipeline.yaml",
    parameter_values={
        "dataset_uri": "gs://my-bucket/data/train_v2.csv",
        "learning_rate": 0.01,
        "accuracy_threshold": 0.92,
    },
    enable_caching=True,
)

job.submit()
```

---

## ⚡ Pipeline Caching

**Caching** is one of the most important features for iterative development. Vertex AI Pipelines caches the output of each component execution. If the component's:
- **Container image**
- **Input parameters**
- **Input artifacts**

...are all identical to a previous run, the step is **skipped** and the cached output is reused.

```python
job = aiplatform.PipelineJob(
    ...
    enable_caching=True,   # Default: True
)
```

> **Exam tip:** Caching is enabled by default. To **force re-execution** (e.g., when underlying data has changed but inputs haven't), set `enable_caching=False`. Cached steps are displayed in the console with a ⚡ icon.

---

## 🔀 Control Flow Constructs

### Conditional Execution — `dsl.Condition`

```python
with dsl.Condition(eval_task.outputs["accuracy"] > 0.90, name="deploy-gate"):
    deploy_model(model=train_task.outputs["model"])
```

### Parallel For Loop — `dsl.ParallelFor`
Run a component in parallel over a list of inputs:

```python
with dsl.ParallelFor(["us-central1", "europe-west4", "asia-east1"]) as region:
    deploy_task = deploy_to_region(model=model, region=region)
```

### Exit Handler — `dsl.ExitHandler`
Always runs a cleanup/notification step, even if the pipeline fails:

```python
exit_task = notify_pipeline_status()

with dsl.ExitHandler(exit_task):
    train_task = train_model(...)
    eval_task = evaluate_model(...)
```

---

## 🏗️ Component Resource Configuration

You can configure compute resources per component:

```python
@dsl.pipeline(...)
def my_pipeline():
    train_task = train_model(dataset_uri="gs://...")

    # Configure resources on the task
    train_task.set_cpu_limit("8")
    train_task.set_memory_limit("32G")
    train_task.set_gpu_limit("1")
    train_task.add_node_pool_label("cloud.google.com/gke-accelerator", "nvidia-tesla-t4")
```

---

## 🔗 Integration with Other Vertex AI Services

This is heavily tested. Know how Pipelines connects to the rest of the Vertex AI ecosystem:

| Service | Integration |
|---|---|
| **Vertex ML Metadata** | Automatically logs all artifact and execution metadata for every pipeline run |
| **Vertex Training** | Use `aiplatform.CustomTrainingJobOp` or `aiplatform.HyperparameterTuningJobOp` as pipeline components |
| **Vertex Batch Prediction** | Use `aiplatform.ModelBatchPredictOp` to run batch inference as a pipeline step |
| **Vertex Model Registry** | Use `aiplatform.ModelUploadOp` to register a trained model as a pipeline step |
| **Vertex Endpoints** | Use `aiplatform.ModelDeployOp` to deploy a registered model as a pipeline step |
| **Vertex Feature Store** | Read features from Feature Store as a pipeline input step |
| **Vertex Model Monitoring** | Triggered after deployment; monitoring job can trigger a new pipeline run on drift |
| **Cloud Scheduler** | Schedule pipeline runs on a cron basis for retraining |
| **Pub/Sub + Cloud Functions** | Event-driven pipeline triggers (e.g., new data landed in GCS) |
| **Artifact Registry** | Store custom component Docker images |

### Pre-built Google Cloud Pipeline Components (GCPC)

GCP provides a library of **pre-built components** for common Vertex AI operations, so you don't have to write everything from scratch:

```python
from google_cloud_pipeline_components.v1.dataset import TabularDatasetCreateOp
from google_cloud_pipeline_components.v1.automl.training_job import AutoMLTabularTrainingJobRunOp
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import ModelDeployOp

@dsl.pipeline(name="automl-tabular-pipeline")
def automl_pipeline(project: str, location: str):
    dataset_task = TabularDatasetCreateOp(
        project=project,
        display_name="my-dataset",
        gcs_source="gs://my-bucket/data.csv",
    )

    training_task = AutoMLTabularTrainingJobRunOp(
        project=project,
        display_name="automl-training",
        dataset=dataset_task.outputs["dataset"],
        target_column="label",
        predefined_split_column_name="split",
    )

    upload_task = ModelUploadOp(
        project=project,
        display_name="my-model",
        unmanaged_container_model=training_task.outputs["model"],
    )

    deploy_task = ModelDeployOp(
        model=upload_task.outputs["model"],
        endpoint=my_endpoint_resource_name,
        dedicated_resources_machine_type="n1-standard-4",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=3,
    )
```

---

## 🔄 Pipeline Scheduling & Triggers

### Scheduled Retraining via Cloud Scheduler

```bash
# Create a Cloud Scheduler job to trigger pipeline every Monday at 2am
gcloud scheduler jobs create http retrain-weekly \
  --schedule="0 2 * * 1" \
  --uri="https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/pipelineJobs" \
  --message-body=@pipeline_request.json \
  --oauth-service-account-email=pipeline-sa@my-project.iam.gserviceaccount.com
```

### Event-Driven Trigger via Pub/Sub + Cloud Functions

```
New data in GCS
      │
      ▼
GCS Event Notification
      │
      ▼
Pub/Sub Topic
      │
      ▼
Cloud Function
      │
      ▼
aiplatform.PipelineJob.submit()
```

---

## 🔐 IAM & Service Accounts

| Role | Purpose |
|---|---|
| `roles/aiplatform.user` | Submit and manage pipeline jobs |
| `roles/aiplatform.viewer` | View pipeline runs and metadata |
| `roles/storage.objectAdmin` | Read/write GCS pipeline root and artifacts |
| `roles/bigquery.dataEditor` | Read/write BigQuery datasets used in components |

> **Exam tip:** Pipelines run under a **service account**. The service account must have appropriate permissions on all resources the pipeline touches — GCS, BigQuery, Artifact Registry, Vertex AI endpoints, etc.

---

## 🆚 KFP v1 vs KFP v2

The exam may reference both. Know the key differences:

| | KFP v1 | KFP v2 |
|---|---|---|
| **Artifact system** | Untyped paths (strings) | Typed artifacts (`dsl.Model`, `dsl.Dataset`, etc.) |
| **Metadata integration** | Manual / limited | Automatic Vertex ML Metadata integration |
| **Compilation output** | JSON | YAML |
| **Recommended?** | Legacy | ✅ Current standard |
| **IR format** | Pipeline IR v1 | Pipeline IR v2 |

---

## 🆚 Vertex AI Pipelines vs. Cloud Composer

A classic exam comparison:

| | Vertex AI Pipelines | Cloud Composer (Airflow) |
|---|---|---|
| **Best for** | ML workflows | General data engineering workflows |
| **Compute** | Serverless, auto-provisioned | Managed Airflow cluster (always-on cost) |
| **ML integration** | Native Vertex AI integration | Requires custom operators |
| **Metadata tracking** | Automatic via Vertex ML Metadata | Manual |
| **Caching** | Built-in per-step caching | Not natively supported |
| **Cost model** | Pay per pipeline run | Pay for cluster uptime |
| **Use case** | Train, evaluate, deploy ML models | ETL, data pipelines, non-ML orchestration |

> **Exam tip:** If the question involves **ML workflow orchestration** → Vertex AI Pipelines. If it involves **complex data engineering DAGs with non-ML tasks** or **existing Airflow investment** → Cloud Composer.

---

## 💡 Common Exam Scenarios & Answers

**Scenario 1:** *You want to retrain your model every week with fresh data. What is the recommended architecture?*
→ **Cloud Scheduler** → triggers `PipelineJob.submit()` via HTTP or Cloud Function → Vertex AI Pipeline runs retraining, evaluation, and conditional deployment.

**Scenario 2:** *A training step takes 4 hours. During development, you keep re-running the pipeline but the data hasn't changed. How do you speed this up?*
→ Enable **pipeline caching** (`enable_caching=True`). The training step will be skipped if inputs haven't changed, reusing the cached model artifact.

**Scenario 3:** *You only want to deploy a model if its AUC exceeds 0.95. How do you implement this in a pipeline?*
→ Use `dsl.Condition` with the evaluation task's output metric as the condition expression.

**Scenario 4:** *You want to trigger a retraining pipeline automatically when Vertex AI Model Monitoring detects data drift. What is the architecture?*
→ Model Monitoring alert → **Pub/Sub** → **Cloud Function** → `aiplatform.PipelineJob.submit()`

**Scenario 5:** *Which component SDK should you use for a new Vertex AI Pipeline — KFP v1 or v2?*
→ **KFP v2** — it is the current standard with native Vertex ML Metadata integration and typed artifacts.

**Scenario 6:** *Your pipeline has a step that sends a Slack notification. Should you use Vertex AI Pipelines or Cloud Composer?*
→ This depends on the context. If the Slack step is one part of a larger **ML workflow** (train → evaluate → notify), use **Vertex AI Pipelines**. If it's part of a broader **data engineering DAG** with many non-ML tasks, use **Cloud Composer**.

---

## ⚡ Exam Focus Areas

1. **KFP v2 component types** — lightweight Python vs. custom container
2. **Typed artifacts** — `dsl.Model`, `dsl.Dataset`, `dsl.Metrics` and their Vertex ML Metadata mappings
3. **Caching** — enabled by default, bypassed with `enable_caching=False`
4. **Control flow** — `dsl.Condition`, `dsl.ParallelFor`, `dsl.ExitHandler`
5. **Pre-built Google Cloud Pipeline Components** — you don't need to write everything from scratch
6. **Trigger patterns** — Cloud Scheduler for time-based, Pub/Sub + Cloud Function for event-driven
7. **Vertex AI Pipelines vs. Cloud Composer** — ML workflows vs. general data pipelines
8. **Service account permissions** — pipelines need IAM access to all resources they touch
9. **Automatic ML Metadata logging** — every pipeline run records lineage automatically

---

## 📝 Quick Revision Checklist

- [ ] I can explain what a Component and a Pipeline are in KFP v2
- [ ] I know the difference between lightweight and custom container components
- [ ] I know the KFP v2 artifact types and their ML Metadata schema mappings
- [ ] I understand pipeline caching and when to disable it
- [ ] I can use `dsl.Condition`, `dsl.ParallelFor`, and `dsl.ExitHandler`
- [ ] I know the pre-built Google Cloud Pipeline Components library exists
- [ ] I can describe the Cloud Scheduler and Pub/Sub trigger patterns
- [ ] I can compare Vertex AI Pipelines vs. Cloud Composer for a given scenario
- [ ] I understand what IAM roles the pipeline service account needs

---

## 📚 Further Reading

- [Vertex AI Pipelines Overview](https://cloud.google.com/vertex-ai/docs/pipelines/introduction)
- [KFP v2 SDK Reference](https://www.kubeflow.org/docs/components/pipelines/v2/)
- [Google Cloud Pipeline Components](https://cloud.google.com/vertex-ai/docs/pipelines/components-introduction)
- [Pipeline Caching](https://cloud.google.com/vertex-ai/docs/pipelines/caching)
- [Vertex AI Pipelines Scheduling](https://cloud.google.com/vertex-ai/docs/pipelines/schedule-pipeline-run)
- [MLOps Architecture on GCP](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

*Guide prepared for GCP Professional Machine Learning Engineer Certification preparation.*