# 🧪 Vertex AI Experiments — Study Guide
### Professional Machine Learning Engineer Certification

---

## 📌 What Is Vertex AI Experiments?

Vertex AI Experiments is a managed service that helps you **track, compare, and analyse ML experiment runs**. Think of it like a lab notebook for your model development: every time you tweak a hyperparameter, change a dataset, or try a new architecture, Experiments records what you did and what happened.

> **Teacher's Analogy:** Imagine baking a cake and writing down every ingredient change alongside a score out of 10. Vertex AI Experiments does exactly that — automatically — for your ML models.

---

## 🎯 Why It Matters for the Exam

The Professional ML Engineer exam tests your ability to:

- Design **reproducible and scalable** training pipelines
- Select and track **metrics** to evaluate model performance
- Integrate experiment tracking into **Vertex AI Pipelines**
- Apply **MLOps best practices** across the ML lifecycle

Vertex AI Experiments sits squarely in the **Model Development** and **MLOps** domains of the exam blueprint.

---

## 🏗️ Core Concepts

### 1. Experiments

An **Experiment** is a named container that groups related runs together. You create one per project or modelling objective.

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

experiment = aiplatform.Experiment.create(
    experiment_name="my-classification-experiment"
)
```

- An experiment lives in a specific **project** and **region**
- It persists across notebooks, pipelines, and training jobs
- You query experiments via the Cloud Console, SDK, or REST API

---

### 2. Runs

A **Run** is a single trial within an experiment — one combination of hyperparameters, data, and code.

```python
aiplatform.start_run(run="run-lr-0.01-batch-32")
```

Each run records:

| Component | Description |
|---|---|
| **Parameters** | Inputs you control (learning rate, epochs, batch size) |
| **Metrics** | Outputs you measure (accuracy, loss, F1 score) |
| **Artifacts** | Datasets, models, or files linked to the run |
| **Metadata** | Timestamps, states, lineage |

---

### 3. Parameters vs. Metrics

This distinction is critical — the exam will test it.

- **Parameters** are things you *set before training* (hyperparameters, config values)
- **Metrics** are things you *measure after/during training* (performance outcomes)

```python
# Log parameters
aiplatform.log_params({
    "learning_rate": 0.01,
    "batch_size": 32,
    "optimizer": "adam"
})

# Log metrics
aiplatform.log_metrics({
    "accuracy": 0.94,
    "val_loss": 0.12
})
```

> ⚠️ **Exam Tip:** Parameters and metrics are kept separate intentionally. Parameters explain *what you tried*; metrics explain *what happened*.

---

### 4. Time-Series Metrics

You can log metrics at each training step, not just at the end, using `log_time_series_metrics`. This lets you visualise training curves in the Console.

```python
for step in range(num_steps):
    loss = train_one_step()
    aiplatform.log_time_series_metrics(
        {"train_loss": loss},
        step=step
    )
```

---

## 🔗 Integration with Vertex AI Pipelines

One of the most exam-relevant features is that **Vertex AI Pipelines automatically tracks experiments** when you associate a pipeline run with an experiment.

```python
job = aiplatform.PipelineJob(
    display_name="my-pipeline",
    template_path="pipeline.yaml",
    pipeline_root="gs://my-bucket/pipeline-root",
    experiment="my-classification-experiment"  # 👈 link here
)
job.run()
```

Each pipeline execution becomes a **run** in the experiment, with component-level metrics automatically captured. This is key for **reproducibility** — you can trace every model artifact back to the exact pipeline run that produced it.

---

## 🗂️ ML Metadata (MLMD) Under the Hood

Vertex AI Experiments is built on top of **ML Metadata (MLMD)**, a standard for tracking ML artefacts and their lineage.

Key entities in MLMD:

- **Artifact** — a dataset, model, or metric file with a URI
- **Execution** — a pipeline step or training job
- **Context** — groups executions together (an Experiment is a Context)
- **Event** — records which artifacts an execution used or produced

> **Teacher's Note:** You don't need to write raw MLMD code for the exam, but understanding the lineage graph concept — *which data produced which model* — is essential for MLOps questions.

### Artifact Lineage Example

```
Raw Dataset ──► Preprocessing Component ──► Processed Dataset ──► Trainer ──► Model
                       ↕                                               ↕
               (Execution metadata)                          (Execution metadata)
```

All of this is queryable:

```python
from google.cloud import aiplatform

artifact = aiplatform.Artifact.list(
    filter="display_name=my-model"
)
```

---

## 🧩 Autologging with Vertex AI Experiments

For supported frameworks, Vertex AI can **automatically log parameters and metrics** without manual `log_params()` calls. This uses the SDK's `autolog` capability.

Supported frameworks include:

- **scikit-learn** — logs model params, metrics, and the fitted model
- **XGBoost** — logs eval metrics per round
- **TensorFlow/Keras** — logs epoch-level metrics
- **LightGBM** — logs training metrics

```python
aiplatform.autolog()  # Enable autologging

# Now train normally — parameters and metrics are captured automatically
model = sklearn.linear_model.LogisticRegression(C=1.0)
model.fit(X_train, y_train)
```

> ⚠️ **Exam Tip:** Autologging reduces boilerplate but logs everything. For production, you may want manual logging for precise control over what's tracked.

---

## 📊 Comparing Experiments in the Console

The Vertex AI Console provides a **comparison view** where you can:

1. Select multiple runs from the same experiment
2. Compare parameters side-by-side
3. Plot metric values as bar charts or scatter plots
4. Filter and sort to find the best run

This is especially useful when running **hyperparameter tuning jobs** — each trial maps to a run, and you can visually identify which combination performed best.

---

## 🔄 Experiment Tracking vs. Hyperparameter Tuning

These are related but distinct services — a common source of confusion:

| Feature | Vertex AI Experiments | Vertex AI Vizier (HP Tuning) |
|---|---|---|
| **Purpose** | Log and compare runs manually or via pipelines | Automatically search for optimal hyperparameters |
| **Who drives it?** | You (or your pipeline) | The Vizier service |
| **Use case** | General experiment tracking | Systematic HP search (Bayesian, grid, random) |
| **Integration** | Standalone or in pipelines | Integrated with CustomJob or Pipelines |

> **Key point for the exam:** Vizier *generates* trial configurations; Experiments *records* them. They can be used together — Vizier trials can be tracked as Experiment runs.

---

## 📋 Run States

A run passes through the following states:

```
RUNNING ──► COMPLETE
    └──────► FAILED
```

Always end your run explicitly:

```python
aiplatform.end_run()  # Sets state to COMPLETE

# Or use a context manager (recommended)
with aiplatform.start_run("my-run"):
    aiplatform.log_params({"lr": 0.01})
    # ... training code ...
    aiplatform.log_metrics({"accuracy": 0.95})
# Run is automatically ended here
```

---

## 🛡️ IAM & Permissions

For the exam, know that working with Vertex AI Experiments requires:

| Role | Permissions Granted |
|---|---|
| `roles/aiplatform.user` | Create and log to experiments |
| `roles/aiplatform.viewer` | Read experiments and runs |
| `roles/aiplatform.admin` | Full control including deletion |

Service accounts used by training jobs or pipelines must have at minimum `aiplatform.user` to log metadata.

---

## 🏆 Best Practices (Exam Favourites)

1. **Name runs descriptively** — include key hyperparameter values in the run name (e.g., `run-lr0.01-epoch50-batchsize32`) so comparisons are readable without opening each run.

2. **Use experiments in all training jobs** — even if you're not actively tuning, logging establishes a baseline and aids future debugging.

3. **Link pipelines to experiments** — always pass the `experiment` parameter to `PipelineJob` for automatic tracking.

4. **Log early-stopping metrics** — if using early stopping, log the final epoch number as a parameter so you understand model complexity.

5. **Combine with Model Registry** — after identifying the best run, register the model in Vertex AI Model Registry and tag it with the experiment run ID for full traceability.

---

## 🔍 Key SDK Methods — Quick Reference

```python
# Initialise
aiplatform.init(project=PROJECT, location=LOCATION, experiment=EXPERIMENT_NAME)

# Start a run
aiplatform.start_run(run="run-name")

# Log parameters (call once at the start)
aiplatform.log_params({"lr": 0.01, "epochs": 50})

# Log metrics (can call multiple times)
aiplatform.log_metrics({"accuracy": 0.93, "f1": 0.91})

# Log time-series metrics (per step)
aiplatform.log_time_series_metrics({"loss": 0.25}, step=100)

# End a run
aiplatform.end_run()

# Get experiment as a DataFrame
experiment_df = aiplatform.get_experiment_df(experiment="my-experiment")
```

---

## 🎓 Practice Questions

**Q1.** You are training multiple versions of a neural network with different dropout rates. You want to compare validation accuracy across all versions in the Vertex AI Console. What should you do?

> **Answer:** Create a single Experiment and log each training run as a separate Run within it, logging `dropout_rate` as a parameter and `val_accuracy` as a metric.

---

**Q2.** A pipeline component finishes training but the run state is never set to COMPLETE. What is the most likely cause?

> **Answer:** `aiplatform.end_run()` was not called (or the context manager was not used). The run stays in RUNNING state until explicitly ended or it times out.

---

**Q3.** You want experiment tracking to happen automatically without adding logging code to every training script. What feature should you enable?

> **Answer:** `aiplatform.autolog()` — enables automatic parameter and metric capture for supported ML frameworks.

---

**Q4.** How does Vertex AI Experiments differ from Vertex AI Vizier?

> **Answer:** Experiments is a tracking and logging service for recording what happened in each run. Vizier is an optimisation service that actively suggests hyperparameter configurations to try. They are complementary — Vizier can drive trials while Experiments records them.

---

## 📚 Summary

```
Vertex AI Experiments
├── Experiment (named container)
│   ├── Run 1: params + metrics + artifacts
│   ├── Run 2: params + metrics + artifacts
│   └── Run N: ...
├── Built on ML Metadata (lineage tracking)
├── Integrates with Vertex AI Pipelines
├── Supports autologging (sklearn, XGBoost, TF, LightGBM)
└── Compare runs visually in the Cloud Console
```

Mastering Vertex AI Experiments means understanding not just the API, but the *why*: reproducibility, traceability, and collaboration are the foundations of production ML engineering.

---

*Study Tip: Hands-on practice matters most here. Create a free-tier GCP project and run at least 3–5 experiment runs with different hyperparameters. The muscle memory of the SDK will help you on scenario-based exam questions.*