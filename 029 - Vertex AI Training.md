# Vertex AI Training — Complete Guide

> A comprehensive reference for running ML training jobs on Google Cloud's Vertex AI platform.
> Covers custom training, distributed training, hyperparameter tuning, and exam-ready scenarios.

---

## What Is Vertex AI Training?

**Vertex AI Training** is the managed service on Google Cloud for running ML model training jobs at scale. It provisions infrastructure, installs dependencies, executes your training code, and tears everything down when done — so you focus on the model, not the machines.

It replaces the legacy **AI Platform Training** and is the exam-current product name.

### Why Use It Instead of Training Locally?

| Need | Vertex AI Training Solution |
|------|-----------------------------|
| Train on large datasets that don't fit in memory | Scale to multi-machine distributed training |
| Experiment with GPUs/TPUs without buying hardware | On-demand accelerators, pay per second |
| Reproduce training runs reliably | Managed containers, versioned artifacts |
| Find the best hyperparameters automatically | Built-in Vizier hyperparameter tuning |
| Integrate with the rest of the ML pipeline | Native Vertex AI Pipelines, Model Registry, and Feature Store integration |

---

## Training Methods Overview

Vertex AI Training supports two main approaches:

```
Vertex AI Training
├── AutoML Training        ← No code; Google manages the model architecture
└── Custom Training        ← You write the training script
    ├── Pre-built containers  ← Google-managed runtimes (TF, PyTorch, sklearn...)
    └── Custom containers     ← Your own Docker image with full control
```

This guide focuses on **Custom Training**, which is the exam-heavy topic.

---

## Core Concepts

### Training Job Types

| Job Type | When to Use |
|----------|-------------|
| `CustomTrainingJob` | Run a Python script with a pre-built or custom container |
| `CustomContainerTrainingJob` | Run a custom Docker container directly |
| `CustomPythonPackageTrainingJob` | Upload a Python package to GCS and run it |
| `HyperparameterTuningJob` | Sweep hyperparameters across multiple training trials |
| `CustomJob` (low-level) | Full control via the API; used inside Pipelines |

### Key Resources

| Resource | Description |
|----------|-------------|
| **Worker Pool** | The set of machines used in a training job (primary + replica workers) |
| **Machine Type** | The VM spec (e.g. `n1-standard-4`, `a2-highgpu-1g`) |
| **Accelerator** | GPU or TPU attached to the machine |
| **Base Output Directory** | GCS path where checkpoints and model artifacts are saved |
| **Service Account** | Identity the training job runs as (needs GCS, BigQuery access, etc.) |

---

## Pre-built Training Containers

Google provides managed containers for common frameworks — no Dockerfile needed.

| Framework | Container URI |
|-----------|---------------|
| TensorFlow 2.12 (CPU) | `us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-12:latest` |
| TensorFlow 2.12 (GPU) | `us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest` |
| PyTorch 2.0 (CPU) | `us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0:latest` |
| PyTorch 2.0 (GPU) | `us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest` |
| scikit-learn 1.0 | `us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest` |
| XGBoost 1.7 | `us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-7:latest` |

> **Exam tip:** Pre-built containers are the fastest path to getting a training job running. Use custom containers only when you need libraries or system dependencies not available in the pre-built images.

---

## Running a Custom Training Job

### Project Structure

```
my_trainer/
├── trainer/
│   ├── __init__.py
│   └── task.py        ← your training script
└── setup.py           ← package definition (for CustomPythonPackageTrainingJob)
```

### Method 1: CustomTrainingJob (Script + Pre-built Container)

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="fraud-training-sklearn",
    script_path="trainer/task.py",                          # local script path
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
    requirements=["pandas==1.5.3", "scikit-learn==1.0.2"],  # pip packages
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)

model = job.run(
    dataset=vertex_dataset,                    # optional: Vertex AI managed dataset
    model_display_name="fraud-detection-v1",  # registers to Model Registry
    base_output_dir="gs://my-bucket/training/",
    replica_count=1,
    machine_type="n1-standard-4",
    args=["--learning-rate", "0.01", "--max-depth", "6"],  # passed to task.py
    sync=True,                                 # block until job finishes
)
```

### Method 2: CustomPythonPackageTrainingJob (Packaged Python)

```python
# First, upload your package to GCS:
# gsutil cp dist/trainer-0.1.tar.gz gs://my-bucket/packages/

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name="fraud-training-package",
    python_package_gcs_uri="gs://my-bucket/packages/trainer-0.1.tar.gz",
    python_module_name="trainer.task",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)

model = job.run(
    model_display_name="fraud-detection-v1",
    base_output_dir="gs://my-bucket/training/",
    machine_type="n1-standard-4",
)
```

### Method 3: CustomContainerTrainingJob (Your Own Docker Image)

```python
job = aiplatform.CustomContainerTrainingJob(
    display_name="fraud-training-custom",
    container_uri="us-central1-docker.pkg.dev/my-project/my-repo/trainer:latest",
    command=["python", "-m", "trainer.task"],
    model_serving_container_image_uri="us-central1-docker.pkg.dev/my-project/my-repo/serving:latest",
)

model = job.run(
    model_display_name="fraud-detection-v1",
    base_output_dir="gs://my-bucket/training/",
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
```

---

## Writing the Training Script

A minimal `trainer/task.py` for a scikit-learn model:

```python
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from google.cloud import storage

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--data-path", type=str, default="gs://my-bucket/data/train.csv")
    return parser.parse_args()

def load_data(gcs_path):
    return pd.read_csv(gcs_path)

def train(args):
    df = load_data(args.data_path)
    X = df.drop("label", axis=1)
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Validation AUC: {val_auc:.4f}")

    # CRITICAL: save to AIP_MODEL_DIR — Vertex AI mounts this as the output
    model_dir = os.environ.get("AIP_MODEL_DIR", ".")
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    args = get_args()
    train(args)
```

### Important Environment Variables Injected by Vertex AI

| Variable | Value | Use |
|----------|-------|-----|
| `AIP_MODEL_DIR` | GCS path for saving model artifacts | Always save your model here |
| `AIP_CHECKPOINT_DIR` | GCS path for saving checkpoints | Use for long training runs |
| `AIP_TENSORBOARD_LOG_DIR` | GCS path for TensorBoard logs | Point your TF/PyTorch logger here |
| `AIP_TRAINING_DATA_URI` | GCS URI of training data | When using managed datasets |
| `AIP_VALIDATION_DATA_URI` | GCS URI of validation data | When using managed datasets |
| `AIP_TEST_DATA_URI` | GCS URI of test data | When using managed datasets |
| `CLUSTER_SPEC` | JSON describing the distributed cluster | Used for distributed TF training |

---

## Machine Types & Accelerators

### CPU Machine Types

| Machine Type | vCPUs | RAM | Use Case |
|--------------|-------|-----|----------|
| `n1-standard-4` | 4 | 15 GB | Small/medium sklearn, XGBoost |
| `n1-standard-8` | 8 | 30 GB | Medium tabular, NLP preprocessing |
| `n1-standard-16` | 16 | 60 GB | Large feature engineering |
| `n1-highmem-8` | 8 | 52 GB | Memory-intensive jobs |

### GPU Machine Types

| Machine Type | GPUs | GPU Type | Use Case |
|--------------|------|----------|----------|
| `n1-standard-4` + `NVIDIA_TESLA_T4 x1` | 1x T4 | 16 GB VRAM | Small DL training |
| `n1-standard-8` + `NVIDIA_TESLA_V100 x1` | 1x V100 | 16 GB VRAM | Medium DL training |
| `a2-highgpu-1g` | 1x A100 | 40 GB VRAM | Large model fine-tuning |
| `a2-highgpu-8g` | 8x A100 | 320 GB VRAM | Large-scale distributed training |

### TPU Options

| TPU Version | Topology | Use Case |
|-------------|----------|----------|
| `TPU_V2` | 8 cores | TF model development |
| `TPU_V3` | 8 cores | Production TF training |
| `TPU_V3` | 32/128/256 cores | Large-scale distributed TF |

> **Exam tip:** TPUs only work with **TensorFlow** and **JAX**. For PyTorch distributed training, use GPU clusters.

---

## Hyperparameter Tuning

Vertex AI uses **Vizier** under the hood for hyperparameter optimisation. You define a search space and an objective metric, and Vertex AI runs multiple trials in parallel.

### Supported Search Algorithms

| Algorithm | When to Use |
|-----------|-------------|
| `GRID_SEARCH` | Small, discrete search space; exhaustive |
| `RANDOM_SEARCH` | Large continuous space; good baseline |
| `BAYESIAN_OPTIMIZATION` | Best for expensive trials; learns from previous results |

### Setting Up the Training Script for Tuning

The script must report the metric to Vertex AI using `hypertune`:

```python
import hypertune

hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag="val_auc",   # must match metric_id in job config
    metric_value=val_auc,
    global_step=epoch,
)
```

### Launching a Hyperparameter Tuning Job

```python
from google.cloud.aiplatform import hyperparameter_tuning as hpt

job = aiplatform.HyperparameterTuningJob(
    display_name="fraud-hparam-tuning",
    custom_job=aiplatform.CustomJob(
        display_name="fraud-trial",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "replica_count": 1,
            "python_package_spec": {
                "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
                "package_uris": ["gs://my-bucket/packages/trainer-0.1.tar.gz"],
                "python_module": "trainer.task",
            },
        }],
    ),
    metric_spec={"val_auc": "maximize"},    # metric_id: goal
    parameter_spec={
        "learning_rate": hpt.DoubleParameterSpec(min=1e-4, max=1e-1, scale="log"),
        "max_depth":     hpt.IntegerParameterSpec(min=3, max=10, scale="linear"),
        "n_estimators":  hpt.DiscreteParameterSpec(values=[50, 100, 200, 500]),
        "subsample":     hpt.DoubleParameterSpec(min=0.5, max=1.0, scale="linear"),
    },
    max_trial_count=20,           # total trials to run
    parallel_trial_count=5,       # trials running simultaneously
    search_algorithm="BAYESIAN_OPTIMIZATION",
)

job.run(sync=True)

# Get best trial
best_trial = job.trials[0]
print(best_trial.parameters, best_trial.final_measurement)
```

### Hyperparameter Types

| Type | Class | Example |
|------|-------|---------|
| Continuous float | `DoubleParameterSpec` | learning rate, dropout |
| Integer | `IntegerParameterSpec` | max_depth, num_leaves |
| Fixed set | `DiscreteParameterSpec` | batch sizes [32, 64, 128] |
| Categorical | `CategoricalParameterSpec` | optimizer ["adam", "sgd"] |

---

## Distributed Training

For large models or datasets, Vertex AI supports multi-machine, multi-GPU training.

### Distribution Strategies (TensorFlow)

| Strategy | Use Case |
|----------|----------|
| `MirroredStrategy` | Single machine, multiple GPUs |
| `MultiWorkerMirroredStrategy` | Multiple machines, data parallelism |
| `TPUStrategy` | TPU pods |
| `ParameterServerStrategy` | Very large models with a parameter server |

### Worker Pool Spec for Distributed Training

```python
# MultiWorkerMirrored: 1 chief + 3 workers, each with 2 GPUs
worker_pool_specs = [
    {  # Chief (index 0)
        "machine_spec": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 2,
        },
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest",
            "package_uris": ["gs://my-bucket/packages/trainer-0.1.tar.gz"],
            "python_module": "trainer.task",
        },
    },
    {  # Workers (index 1)
        "machine_spec": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 2,
        },
        "replica_count": 3,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest",
            "package_uris": ["gs://my-bucket/packages/trainer-0.1.tar.gz"],
            "python_module": "trainer.task",
        },
    },
]
```

### Parameter Server Strategy (Large Embeddings)

```python
# Chief + Workers + Parameter Servers
worker_pool_specs = [
    {"replica_count": 1, "machine_spec": {...}},   # index 0: chief
    {"replica_count": 4, "machine_spec": {...}},   # index 1: workers
    {"replica_count": 2, "machine_spec": {...}},   # index 2: parameter servers
]
```

### PyTorch Distributed Training

PyTorch uses `torch.distributed` with `nccl` backend. Vertex AI injects `MASTER_ADDR` and `MASTER_PORT` automatically:

```python
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

---

## TensorBoard Integration

Vertex AI has native TensorBoard integration — it streams training metrics in real time without manual setup.

### Create a TensorBoard Instance

```python
tensorboard = aiplatform.Tensorboard.create(
    display_name="fraud-training-tb",
    project="my-project",
    location="us-central1",
)
```

### Attach TensorBoard to a Training Job

```python
job = aiplatform.CustomTrainingJob(
    display_name="fraud-training",
    script_path="trainer/task.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest",
    tensorboard=tensorboard.resource_name,  # ← attach here
)
```

In your training script, write logs to `AIP_TENSORBOARD_LOG_DIR`:

```python
import tensorflow as tf

log_dir = os.environ.get("AIP_TENSORBOARD_LOG_DIR", "./logs")
callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]
model.fit(X_train, y_train, callbacks=callbacks, epochs=50)
```

---

## Managed Datasets Integration

When your data is in a **Vertex AI Managed Dataset**, you can pass it directly to the training job. Vertex AI splits it and injects the GCS URIs via environment variables.

```python
dataset = aiplatform.TabularDataset("projects/.../datasets/DATASET_ID")

model = job.run(
    dataset=dataset,
    target_column="label",
    training_fraction_split=0.8,
    validation_fraction_split=0.1,
    test_fraction_split=0.1,
    model_display_name="fraud-detection-v1",
    machine_type="n1-standard-4",
)
```

Inside your script, read the splits from environment variables:

```python
train_path = os.environ["AIP_TRAINING_DATA_URI"]
val_path   = os.environ["AIP_VALIDATION_DATA_URI"]
test_path  = os.environ["AIP_TEST_DATA_URI"]
```

---

## Vertex AI Pipelines Integration

Training jobs are commonly run as steps inside a **Vertex AI Pipeline** using Kubeflow Pipelines (KFP) components.

```python
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

training_op = CustomTrainingJobOp(
    project=project,
    display_name="fraud-training",
    worker_pool_specs=[{
        "machine_spec": {"machine_type": "n1-standard-4"},
        "replica_count": 1,
        "python_package_spec": {
            "executor_image_uri": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
            "package_uris": ["gs://my-bucket/packages/trainer-0.1.tar.gz"],
            "python_module": "trainer.task",
            "args": ["--n-estimators", "200"],
        },
    }],
    base_output_directory=pipeline_root,
)
```

---

## Persistent Resource (Reduce Cold-Start Time)

For iterative experimentation, you can create a **Persistent Resource** — a long-lived cluster that eliminates VM provisioning time between training runs.

```python
persistent_resource = aiplatform.PersistentResource.create(
    display_name="ml-dev-cluster",
    resource_pools=[
        aiplatform.ResourcePool(
            machine_spec=aiplatform.MachineSpec(
                machine_type="n1-standard-8",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1,
            ),
            replica_count=2,
        )
    ],
)

# Submit jobs to the persistent resource
job.run(
    persistent_resource_id=persistent_resource.resource_id,
    machine_type="n1-standard-8",
)
```

> **Exam tip:** Use Persistent Resources for **interactive development and fast iteration**. Use on-demand jobs for **production pipelines** where cost optimisation matters.

---

## Cost Optimisation

| Strategy | How |
|----------|-----|
| **Spot VMs (preemptible)** | Set `scheduling.spot=True` in worker pool spec; up to 60–91% cheaper but can be preempted |
| **Right-size machine type** | Profile memory and CPU usage; avoid over-provisioning |
| **Early stopping** | Enable in hyperparameter tuning to kill unpromising trials early |
| **Reduce parallel trials** | Fewer concurrent HP tuning trials = lower cost (but slower) |
| **Use pre-built containers** | Avoids image build time and Artifact Registry egress |
| **Cache pipeline steps** | Avoid re-running completed training steps in Vertex AI Pipelines |

### Enabling Spot VMs

```python
job = aiplatform.CustomJob(
    display_name="cost-optimised-job",
    worker_pool_specs=[{
        "machine_spec": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": 1,
        },
        "replica_count": 1,
        "scheduling": {"spot": True},   # ← enable spot pricing
        "python_package_spec": {...},
    }],
)
```

---

## Monitoring & Logging

All training job logs stream to **Cloud Logging** automatically. You can also view them in the Vertex AI console under the job detail page.

### Accessing Logs via CLI

```bash
# Stream live logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# List all training jobs
gcloud ai custom-jobs list --region=us-central1
```

### Cancelling a Job

```python
job.cancel()
```

---

## Exam Quick-Reference: Common Scenarios

| Scenario | Best Approach |
|----------|---------------|
| Train a scikit-learn model with minimal setup | `CustomTrainingJob` with pre-built sklearn container |
| Use a library not available in pre-built containers | Build a **custom Docker container** and use `CustomContainerTrainingJob` |
| Find the best learning rate and depth for XGBoost | `HyperparameterTuningJob` with `BAYESIAN_OPTIMIZATION` |
| Train a large TF model across 8 GPUs on one machine | Pre-built GPU container + `MirroredStrategy` |
| Train a TF model across 4 machines with 4 GPUs each | Multi-worker pool spec + `MultiWorkerMirroredStrategy` |
| Train a massive embedding model (e.g. recommendation) | `ParameterServerStrategy` with 3 worker pool specs |
| Reduce training cost for non-critical jobs | Enable **Spot VMs** in worker pool spec |
| Stream training metrics to TensorBoard in real time | Create a Tensorboard instance and pass it to `CustomTrainingJob` |
| Automatically register the trained model after training | Pass `model_display_name` to `job.run()` |
| Speed up iterative experiments (avoid VM cold start) | Use **Persistent Resource** cluster |
| Run training as part of an automated ML pipeline | Use `CustomTrainingJobOp` inside a Vertex AI Pipeline |
| Training script needs to know where to save the model | Read `AIP_MODEL_DIR` environment variable |
| Train on a Vertex AI Managed Dataset with auto-splits | Pass `dataset` and fraction splits to `job.run()` |
| TPU training for a TensorFlow model | Use `TPU_V3` accelerator + `TPUStrategy` in the script |

---

## Key Terms Summary

| Term | Definition |
|------|------------|
| **CustomTrainingJob** | Runs a Python script using a pre-built or custom container |
| **CustomContainerTrainingJob** | Runs your own Docker image directly |
| **CustomPythonPackageTrainingJob** | Runs a Python package uploaded to GCS |
| **HyperparameterTuningJob** | Runs multiple training trials to find optimal hyperparameters |
| **Vizier** | The underlying Bayesian optimisation service powering HP tuning |
| **Worker Pool** | A group of identical machines assigned a role in distributed training |
| **AIP_MODEL_DIR** | Environment variable pointing to the GCS output path for model artifacts |
| **MirroredStrategy** | TF strategy for single-machine multi-GPU training |
| **MultiWorkerMirroredStrategy** | TF strategy for multi-machine data-parallel training |
| **ParameterServerStrategy** | TF strategy with separate parameter servers for large embedding models |
| **Persistent Resource** | Long-lived cluster for fast iterative training without VM cold starts |
| **Spot VM** | Preemptible VM that is significantly cheaper but can be interrupted |
| **TensorBoard** | Vertex AI managed TensorBoard instance for streaming training metrics |
| **Pipeline Component** | A reusable unit of work; training jobs are a common pipeline step |

---

*Guide covers Vertex AI Training as of the 2025/2026 GCP Professional ML Engineer exam objectives.*