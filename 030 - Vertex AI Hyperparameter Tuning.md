# Vertex AI Hyperparameter Tuning

> A comprehensive reference for automating hyperparameter optimisation using Vertex AI Vizier.
> Covers search algorithms, parameter types, trial configuration, and exam-ready scenarios.

---

## What Is Hyperparameter Tuning?

**Hyperparameters** are the configuration values you set *before* training starts — they are not learned from data. Examples: learning rate, number of trees, dropout rate, batch size.

Finding the best combination manually is slow and unreliable. **Vertex AI Hyperparameter Tuning** automates this by running multiple training **trials** in parallel, each with a different combination of hyperparameter values, and identifies which combination produces the best metric.

Under the hood, Vertex AI uses **Google Vizier** — a black-box optimisation service based on Bayesian optimisation research.

---

## Core Concepts

```
HyperparameterTuningJob
├── CustomJob (your training script, run once per trial)
├── Parameter Spec (the search space)
├── Metric Spec (what to optimise and in which direction)
├── max_trial_count (total trials to run)
├── parallel_trial_count (trials running at the same time)
└── search_algorithm (how to explore the space)
```

### Key Terms

| Term | Definition |
|------|------------|
| **Trial** | One full training run with a specific set of hyperparameter values |
| **Search space** | The range of values each hyperparameter can take |
| **Objective metric** | The metric your trials report (e.g. `val_auc`) |
| **Goal** | Whether to `maximize` or `minimize` the objective metric |
| **Vizier** | Google's internal Bayesian optimisation engine powering HP tuning |
| **Early stopping** | Automatically killing unpromising trials before they finish |

---

## Search Algorithms

| Algorithm | How It Works | When to Use |
|-----------|-------------|-------------|
| `BAYESIAN_OPTIMIZATION` | Builds a probabilistic model of the objective; uses past trials to decide where to sample next | **Default choice.** Best when trials are expensive (long training time). Finds good results with fewer trials. |
| `RANDOM_SEARCH` | Samples hyperparameter combinations uniformly at random | Good baseline; works well with many parallel trials where Bayesian has less advantage |
| `GRID_SEARCH` | Exhaustively tries every combination | Only use with small discrete search spaces; combinatorial explosion otherwise |
| `ALGORITHM_UNSPECIFIED` | Defaults to Bayesian optimisation | Same as `BAYESIAN_OPTIMIZATION` |

### Algorithm Decision Guide

```
How expensive is each trial?
├── Expensive (>10 min per trial)
│   └── Use BAYESIAN_OPTIMIZATION
│       (learns from previous trials, wastes fewer runs)
│
└── Cheap (<2 min per trial)
    ├── Large continuous search space → RANDOM_SEARCH
    └── Small discrete search space  → GRID_SEARCH
```

> **Exam tip:** `BAYESIAN_OPTIMIZATION` is almost always the right answer for ML exam scenarios. It is the most sample-efficient algorithm — it finds good hyperparameters with fewer trials.

---

## Parameter Types

Vertex AI supports four types of hyperparameter specs:

### 1. DoubleParameterSpec — Continuous float range

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

# Learning rate between 0.0001 and 0.1, sampled on a log scale
"learning_rate": hpt.DoubleParameterSpec(
    min=1e-4,
    max=1e-1,
    scale="log",      # log | linear | reverse_log
)
```

Use `scale="log"` for values that span orders of magnitude (learning rate, regularisation).
Use `scale="linear"` for values that change linearly (dropout rate, momentum).

### 2. IntegerParameterSpec — Integer range

```python
# Tree depth between 3 and 12
"max_depth": hpt.IntegerParameterSpec(
    min=3,
    max=12,
    scale="linear",
)
```

### 3. DiscreteParameterSpec — Fixed set of numeric values

```python
# Batch size from a fixed set
"batch_size": hpt.DiscreteParameterSpec(
    values=[32, 64, 128, 256],
    scale="linear",
)
```

Use when you want to test specific values rather than a continuous range.

### 4. CategoricalParameterSpec — Named string values

```python
# Optimiser choice
"optimizer": hpt.CategoricalParameterSpec(
    values=["adam", "sgd", "rmsprop"],
)
```

Use for non-numeric choices like activation functions, optimisers, or model architectures.

---

## Reporting the Metric in Your Training Script

Your training script **must** report the objective metric back to Vertex AI using the `hypertune` library. Without this, Vertex AI cannot evaluate trial performance.

### Install hypertune

```bash
pip install cloudml-hypertune
```

### Report the metric

```python
# trainer/task.py
import argparse
import hypertune
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

def get_args():
    parser = argparse.ArgumentParser()
    # Hyperparameters passed as CLI args by Vertex AI
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth",     type=int,   default=3)
    parser.add_argument("--n-estimators",  type=int,   default=100)
    parser.add_argument("--subsample",     type=float, default=1.0)
    return parser.parse_args()

def train(args):
    # Load data
    X_train, X_val, y_train, y_val = load_data()

    # Train
    model = GradientBoostingClassifier(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
    )
    model.fit(X_train, y_train)

    # Evaluate
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    # CRITICAL: report metric to Vertex AI
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="val_auc",   # must match metric_spec key
        metric_value=val_auc,
        global_step=1,                         # use epoch number for iterative training
    )

    print(f"val_auc={val_auc:.4f}")

if __name__ == "__main__":
    args = get_args()
    train(args)
```

### For iterative training (report each epoch)

```python
for epoch in range(num_epochs):
    model.fit(...)
    val_loss = evaluate(model, X_val, y_val)

    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag="val_loss",
        metric_value=val_loss,
        global_step=epoch,    # ← Vertex AI uses this for early stopping decisions
    )
```

> **Exam tip:** `global_step` is required for **early stopping** to work correctly. Vertex AI uses the step progression to decide whether a trial is improving slowly enough to kill early.

---

## Full Hyperparameter Tuning Job Example

### Tabular / sklearn / XGBoost

```python
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

aiplatform.init(project="my-project", location="us-central1")

# Define the base training job (run once per trial)
custom_job = aiplatform.CustomJob(
    display_name="fraud-hp-trial",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-4",
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
            "package_uris": ["gs://my-bucket/packages/trainer-0.1.tar.gz"],
            "python_module": "trainer.task",
        },
    }],
)

# Define and launch the tuning job
hp_job = aiplatform.HyperparameterTuningJob(
    display_name="fraud-hp-tuning",
    custom_job=custom_job,

    metric_spec={"val_auc": "maximize"},   # metric_tag: goal

    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=0.3,  scale="log"),
        "max_depth":     hpt.IntegerParameterSpec(min=2, max=10,      scale="linear"),
        "n_estimators":  hpt.DiscreteParameterSpec(values=[50, 100, 200, 500]),
        "subsample":     hpt.DoubleParameterSpec(min=0.5, max=1.0,    scale="linear"),
        "optimizer":     hpt.CategoricalParameterSpec(values=["adam", "sgd"]),
    },

    max_trial_count=30,          # run 30 trials total
    parallel_trial_count=5,      # 5 trials at a time
    max_failed_trial_count=5,    # stop the job if 5 trials fail

    search_algorithm="BAYESIAN_OPTIMIZATION",
    enable_early_stopping=True,  # kill unpromising trials early
)

hp_job.run(sync=True)
```

### Neural Network (TensorFlow / Keras)

```python
custom_job = aiplatform.CustomJob(
    display_name="nn-hp-trial",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest",
            "package_uris": ["gs://my-bucket/packages/trainer-0.1.tar.gz"],
            "python_module": "trainer.task",
        },
    }],
)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name="nn-hp-tuning",
    custom_job=custom_job,

    metric_spec={"val_loss": "minimize"},

    parameter_spec={
        "learning_rate":   hpt.DoubleParameterSpec(min=1e-5, max=1e-2, scale="log"),
        "dropout_rate":    hpt.DoubleParameterSpec(min=0.1,  max=0.5,  scale="linear"),
        "batch_size":      hpt.DiscreteParameterSpec(values=[32, 64, 128]),
        "num_layers":      hpt.IntegerParameterSpec(min=2, max=6,       scale="linear"),
        "units_per_layer": hpt.DiscreteParameterSpec(values=[64, 128, 256, 512]),
        "optimizer":       hpt.CategoricalParameterSpec(values=["adam", "rmsprop"]),
    },

    max_trial_count=40,
    parallel_trial_count=4,
    search_algorithm="BAYESIAN_OPTIMIZATION",
    enable_early_stopping=True,
)

hp_job.run(sync=True)
```

---

## Reading Trial Results

After the job completes, inspect the trials to find the best configuration:

```python
# All trials sorted by metric (best first for maximize)
for trial in hp_job.trials:
    params = {p.parameter_id: p.value for p in trial.parameters}
    metric = trial.final_measurement.metrics[0].value
    print(f"Trial {trial.id}: {params} → val_auc={metric:.4f}")

# Best trial
best = hp_job.trials[0]
best_params = {p.parameter_id: p.value for p in best.parameters}
print(f"Best params: {best_params}")
print(f"Best val_auc: {best.final_measurement.metrics[0].value:.4f}")
```

### Trial States

| State | Meaning |
|-------|---------|
| `SUCCEEDED` | Trial completed and reported a metric |
| `FAILED` | Trial crashed or timed out |
| `INFEASIBLE` | Trial reported an infeasible result (e.g. OOM) |
| `STOPPED` | Trial was stopped by early stopping |
| `ACTIVE` | Trial is currently running |

---

## Early Stopping

When `enable_early_stopping=True`, Vertex AI monitors each trial's metric progression and kills trials that are unlikely to outperform the current best — saving compute cost.

### How It Works

```
Trial A: val_auc = [0.72, 0.74, 0.75, 0.75, 0.75]  ← plateau, stopped early
Trial B: val_auc = [0.80, 0.83, 0.85, 0.87, 0.89]  ← improving, runs to completion
Trial C: val_auc = [0.60, 0.61, 0.61, ...]          ← stopped after step 3
```

### Requirements for Early Stopping

- `global_step` must be reported with each metric call in `hypertune`
- Trials must run for at least a minimum number of steps before early stopping kicks in
- Works best with `BAYESIAN_OPTIMIZATION` (not as useful with `GRID_SEARCH`)

---

## Parallel Trial Count Trade-offs

| `parallel_trial_count` | Effect |
|------------------------|--------|
| **Low (1–2)** | Bayesian learns more from each trial; best for expensive jobs. Slower wall-clock time. |
| **High (10+)** | Faster wall-clock time. Bayesian has less information per decision; approaches random search. Higher cost. |
| **Recommended** | 5–10 for most jobs. Rule of thumb: `parallel_trial_count` ≈ `max_trial_count / 5` |

---

## Using Vizier Standalone (Vertex AI Vizier)

For optimising anything beyond ML training (simulations, business metrics, A/B tests), you can use **Vertex AI Vizier** directly without a training job:

```python
from google.cloud import aiplatform_v1

vizier_client = aiplatform_v1.VizierServiceClient(
    client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
)

# Define study
study = vizier_client.create_study(
    parent=f"projects/my-project/locations/us-central1",
    study={
        "display_name": "my-vizier-study",
        "study_spec": {
            "algorithm": "GAUSSIAN_PROCESS_BANDIT",
            "parameters": [
                {"parameter_id": "x", "double_value_spec": {"min_value": -5, "max_value": 5}},
                {"parameter_id": "y", "double_value_spec": {"min_value": -5, "max_value": 5}},
            ],
            "metrics": [{"metric_id": "objective", "goal": "MAXIMIZE"}],
        },
    },
)

# Request trials
trials = vizier_client.suggest_trials(
    {"parent": study.name, "suggestion_count": 5, "client_id": "my-client"}
).result().trials

# Evaluate and report
for trial in trials:
    x = trial.parameters[0].value
    y = trial.parameters[1].value
    result = -(x**2 + y**2)  # example objective function

    vizier_client.complete_trial(
        name=trial.name,
        trial_infeasible=False,
        infeasibility_reason="",
    )
```

---

## Best Practices for Hyperparameter Tuning

### Search Space Design

| Rule | Reason |
|------|--------|
| Use `scale="log"` for learning rate and regularisation | These parameters have exponential effect; log scale samples them more uniformly |
| Keep search space as small as possible | Fewer dimensions = faster convergence for Bayesian optimisation |
| Fix obviously good values before tuning | e.g. if you know `n_estimators=500` always wins, fix it and tune others |
| Start with a wide range, then narrow | Run a coarse search first, then refine around the best region |
| Avoid tuning correlated parameters together | e.g. `num_layers` and `units_per_layer` can interact; tune one at a time first |

### Trial Count Guidelines

| Model Type | Recommended `max_trial_count` |
|------------|-------------------------------|
| sklearn / XGBoost (fast trials) | 20–50 |
| Neural network, small (5–10 min/trial) | 20–30 |
| Neural network, large (30+ min/trial) | 10–20 |
| LLM fine-tuning (hours per trial) | 5–10 |

---

## Integration with Vertex AI Pipelines

Run hyperparameter tuning as a step in an automated pipeline:

```python
from google_cloud_pipeline_components.v1.hyperparameter_tuning_job import (
    HyperparameterTuningJobRunOp,
    serialize_parameters,
    serialize_metrics,
)

@kfp.dsl.pipeline(name="hp-tuning-pipeline")
def hp_pipeline(project: str, location: str):

    hp_tuning_op = HyperparameterTuningJobRunOp(
        project=project,
        location=location,
        display_name="pipeline-hp-tuning",
        base_output_directory=pipeline_root,
        worker_pool_specs=[...],
        study_spec_metrics=serialize_metrics({"val_auc": "maximize"}),
        study_spec_parameters=serialize_parameters({
            "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=0.3, scale="log"),
            "max_depth":     hpt.IntegerParameterSpec(min=2, max=10, scale="linear"),
        }),
        max_trial_count=20,
        parallel_trial_count=5,
    )

    # Pass best trial params to next step
    best_trial = hp_tuning_op.outputs["best_study_spec"]
```

---

## Exam Quick-Reference: Common Scenarios

| Scenario | Best Approach |
|----------|---------------|
| Tune learning rate and regularisation for XGBoost efficiently | `BAYESIAN_OPTIMIZATION` with `DoubleParameterSpec(scale="log")` for both |
| Trials are expensive (1 hour each); want to minimise total cost | `BAYESIAN_OPTIMIZATION` + `enable_early_stopping=True` + low `parallel_trial_count` |
| Trials are cheap (30 seconds each); want fastest wall-clock time | `RANDOM_SEARCH` with high `parallel_trial_count` |
| Test 3 optimisers × 4 batch sizes × 3 depths exhaustively | `GRID_SEARCH` with `CategoricalParameterSpec` and `DiscreteParameterSpec` |
| Tuning job reports no metric and all trials fail | Training script is not calling `hypertune.report_hyperparameter_tuning_metric()` |
| Early stopping is not working | `global_step` is not being passed to `report_hyperparameter_tuning_metric()` |
| Want to reuse best hyperparameters from a previous tuning job | Read `hp_job.trials[0].parameters` and pass values to the production training job |
| Tune hyperparameters inside a Vertex AI Pipeline | Use `HyperparameterTuningJobRunOp` component |
| Optimise a simulation or non-ML objective | Use **Vertex AI Vizier** directly (standalone, no training job needed) |
| Too many failed trials stopping the job early | Increase `max_failed_trial_count` or fix the training script crash |
| Need to tune 10+ hyperparameters at once | Reduce search space first; dimensionality > 10 hurts Bayesian optimisation |

---

## Key Terms Summary

| Term | Definition |
|------|------------|
| **Trial** | One complete training run with a specific hyperparameter combination |
| **Vizier** | Google's Bayesian optimisation service; powers HP tuning under the hood |
| **Objective metric** | The metric your training script reports (e.g. `val_auc`, `val_loss`) |
| **Goal** | `maximize` or `minimize` — the direction Vizier optimises toward |
| **Search space** | The set of hyperparameter types and ranges Vizier can explore |
| **DoubleParameterSpec** | Continuous float hyperparameter (e.g. learning rate) |
| **IntegerParameterSpec** | Integer hyperparameter (e.g. max_depth) |
| **DiscreteParameterSpec** | Fixed set of numeric values (e.g. batch sizes) |
| **CategoricalParameterSpec** | Named string values (e.g. optimizer names) |
| **hypertune** | Python library used inside the training script to report metrics to Vizier |
| **global_step** | Training iteration/epoch counter; required for early stopping |
| **early stopping** | Killing unpromising trials before they complete to save cost |
| **parallel_trial_count** | Number of trials running simultaneously |
| **max_trial_count** | Total number of trials to run in the job |
| **max_failed_trial_count** | Number of trial failures that will abort the entire job |

---

*Guide covers Vertex AI Hyperparameter Tuning as of the 2025/2026 GCP Professional ML Engineer exam objectives.*