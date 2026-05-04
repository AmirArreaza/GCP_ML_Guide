# Vertex ML Metadata API — Study Guide
### GCP Professional Machine Learning Engineer Certification

---

## 📌 What Is Vertex ML Metadata?

**Vertex ML Metadata** is a managed service within Vertex AI that lets you **track, store, and query metadata** generated during your ML workflows. It records the *lineage* of your experiments — what data was used, which pipelines ran, which models were produced, and how they performed.

> **Exam mindset:** Think of Vertex ML Metadata as the **audit trail** for your entire ML lifecycle. It answers: *"How did this model come to be?"*

---

## 🧠 Core Concepts

### 1. Artifact
An **Artifact** is any piece of data that is consumed or produced by an ML step.

| Type | Examples |
|---|---|
| Input | Dataset, feature store snapshot, raw data URI |
| Output | Trained model, evaluation metrics file, prediction output |

Each artifact has:
- A **URI** (where the data lives, e.g., a GCS path)
- A **schema** (type identifier, e.g., `system.Dataset`, `system.Model`)
- **Metadata** (key-value properties, e.g., `{"num_rows": 10000}`)

---

### 2. Execution
An **Execution** represents a **step or run** in your ML pipeline — a unit of computation that consumes input artifacts and produces output artifacts.

Examples:
- A data preprocessing job
- A training job
- A model evaluation step

Each execution tracks:
- **State**: `RUNNING`, `COMPLETE`, `FAILED`, `CACHED`
- **Metadata**: hyperparameters, framework version, duration, etc.

---

### 3. Context
A **Context** is a **grouping mechanism**. It logically connects a set of artifacts and executions.

| Use Case | Example |
|---|---|
| Experiment | All runs under "experiment-v1" |
| Pipeline run | All steps of a single Vertex Pipeline execution |
| Model version | All artifacts tied to model v2.1 |

> **Key exam point:** Contexts enable you to **query lineage across multiple artifacts and executions** as a unit.

---

### 4. Event
An **Event** is the **link** between an Execution and an Artifact. It captures the *direction* of the relationship.

| Event Type | Meaning |
|---|---|
| `INPUT` | Artifact was consumed by the execution |
| `OUTPUT` | Artifact was produced by the execution |

Events are what enable **lineage tracing** — you can walk backwards from a model to find exactly which dataset and pipeline step produced it.

---

### 5. MetadataStore
The **MetadataStore** is the top-level container for all metadata resources in a project + region. Every project has a **default** MetadataStore, but you can create custom ones for isolation.

```
MetadataStore
├── Artifacts
├── Executions
└── Contexts
```

---

## 🔗 Lineage Graph

The relationships between these primitives form a **Directed Acyclic Graph (DAG)**:

```
[Dataset Artifact] ──INPUT──► [Training Execution] ──OUTPUT──► [Model Artifact]
                                      │
                               [Context: Experiment Run]
```

This graph is what powers **ML lineage** — reproducibility, auditability, and debugging.

---

## 🛠️ Key API Operations

### Creating an Artifact

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

artifact = aiplatform.Artifact.create(
    schema_title="system.Dataset",
    display_name="training-dataset-v1",
    uri="gs://my-bucket/datasets/train.csv",
    metadata={"num_rows": 50000, "format": "csv"},
)
```

### Creating an Execution

```python
execution = aiplatform.Execution.create(
    schema_title="system.Run",
    display_name="training-run-001",
    metadata={"learning_rate": 0.01, "epochs": 50},
)
```

### Logging Events (linking Artifact ↔ Execution)

```python
with execution.update_state(new_state=aiplatform.gapic.Execution.State.RUNNING):
    execution.assign_input_artifacts([dataset_artifact])
    execution.assign_output_artifacts([model_artifact])
```

### Creating a Context and Adding Resources

```python
context = aiplatform.ExperimentRun.create(
    run_name="experiment-v1-run-1",
    experiment="my-experiment",
)
context.log_params({"learning_rate": 0.01})
context.log_metrics({"accuracy": 0.94, "loss": 0.12})
```

---

## 🔍 Querying Lineage

You can query the lineage graph to trace the full history of any artifact.

### Get Artifact Lineage

```python
artifact = aiplatform.Artifact("projects/.../artifacts/123")

# Get all executions that used this artifact
executions = artifact.get_executions()

# Get lineage subgraph
lineage = aiplatform.utils.get_lineage_subgraph(artifact)
```

### Filter Artifacts by Metadata

```python
artifacts = aiplatform.Artifact.list(
    filter='metadata.format="csv" AND schema_title="system.Dataset"'
)
```

---

## 🧪 Vertex Experiments Integration

Vertex ML Metadata is the **backbone of Vertex Experiments**. When you use the Experiments API, metadata is recorded automatically.

```python
aiplatform.init(experiment="fraud-detection-v2")

with aiplatform.start_run("run-001"):
    aiplatform.log_params({"n_estimators": 100, "max_depth": 5})
    # ... train your model ...
    aiplatform.log_metrics({"auc": 0.97, "f1": 0.88})
```

Under the hood, each `start_run()` creates:
- An **Execution** (the run)
- A **Context** (the experiment grouping)
- **Artifacts** (datasets and models, if logged)

---

## 🔁 Vertex Pipelines Integration

When you run a **Vertex AI Pipeline**, metadata is recorded **automatically** for each pipeline step:

- Each component execution → **Execution** record
- Each input/output dataset or model → **Artifact** record
- The entire pipeline run → **Context** record

You can also log custom metadata inside a pipeline component:

```python
from kfp.v2 import dsl
from google.cloud import aiplatform

@dsl.component
def train_model(dataset_uri: str) -> str:
    aiplatform.init(project="my-project", location="us-central1")
    with aiplatform.start_run("component-run"):
        aiplatform.log_params({"dataset": dataset_uri})
        # ... training logic ...
        aiplatform.log_metrics({"accuracy": 0.95})
    return "gs://my-bucket/model/"
```

---

## 📊 Schema System

Vertex ML Metadata uses **schemas** to type artifacts and executions. GCP provides **system schemas** out of the box:

| Schema Title | Represents |
|---|---|
| `system.Dataset` | A dataset artifact |
| `system.Model` | A trained model artifact |
| `system.Metrics` | An evaluation metrics artifact |
| `system.ClassificationMetrics` | Classification-specific metrics |
| `system.Run` | A generic execution/run |
| `system.ContainerExecution` | A containerized job execution |

You can also define **custom schemas** for domain-specific metadata.

---

## 🔐 IAM & Access Control

| Role | Permissions |
|---|---|
| `roles/aiplatform.user` | Read/write metadata |
| `roles/aiplatform.viewer` | Read-only access |
| `roles/aiplatform.admin` | Full control including MetadataStore management |

MetadataStores are **regional resources** — data does not leave the specified region.

---

## ⚡ Exam Focus Areas

### High-Probability Exam Topics

1. **Lineage tracing** — Given a model, how do you find the dataset it was trained on? → Walk the Event graph from Model Artifact → Execution → Dataset Artifact.

2. **Artifacts vs. Executions vs. Contexts** — Know exactly what each represents and when to use each.

3. **Automatic metadata logging** — Vertex Pipelines logs metadata automatically; you don't need manual instrumentation for pipeline-level lineage.

4. **Experiments API** — `log_params()` records hyperparameters; `log_metrics()` records evaluation results. Both are backed by Vertex ML Metadata.

5. **MetadataStore scope** — One default store per project/region; you can create custom stores for isolation.

6. **Schema titles** — Know the common system schemas (`system.Dataset`, `system.Model`, `system.Metrics`).

---

## 🆚 Vertex ML Metadata vs. Related Services

| Service | Purpose | When to Use |
|---|---|---|
| **Vertex ML Metadata** | Track ML lineage & metadata | Reproducibility, auditing, experiment comparison |
| **Vertex Experiments** | Compare experiment runs | Hyperparameter tuning, model selection |
| **Vertex Model Registry** | Manage deployed models | Versioning, deployment staging |
| **Cloud Logging** | Infrastructure-level logs | Debugging, alerting, operational monitoring |
| **BigQuery** | Structured data analytics | Storing large-scale metrics for BI/reporting |

> **Exam tip:** Vertex Experiments *uses* Vertex ML Metadata under the hood. They are complementary, not competing.

---

## 📝 Quick Revision Checklist

- [ ] I can explain the 4 core primitives: Artifact, Execution, Context, Event
- [ ] I understand how Events link Artifacts to Executions (INPUT/OUTPUT)
- [ ] I know how Contexts group related artifacts and executions
- [ ] I can describe how Vertex Pipelines auto-logs metadata
- [ ] I can describe how Vertex Experiments maps to ML Metadata primitives
- [ ] I know the common system schema titles
- [ ] I understand lineage tracing via the Event graph
- [ ] I know the IAM roles for metadata access

---

## 📚 Further Reading

- [Vertex AI ML Metadata Overview](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction)
- [Vertex Experiments Quickstart](https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments)
- [Lineage Tracking in Vertex Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines/lineage-overview)
- [ML Metadata API Reference](https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.metadataStores)

---

*Guide prepared for GCP Professional Machine Learning Engineer Certification preparation.*