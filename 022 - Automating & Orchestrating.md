# Study Guide: Automating & Orchestrating ML Pipelines
**Certification Topic:** MLOps & Vertex AI Pipeline Orchestration

## 1. Core Concepts of Orchestration
Orchestration is the process of defining a directed acyclic graph (DAG) of steps. In Vertex AI, this is powered by **Kubeflow Pipelines (KFP)**.

* **Component:** A self-contained set of code that performs one step (e.g., data ingestion).
* **Pipeline:** A workflow of components connected by data dependencies.
* **Artifacts:** The outputs produced by components (Datasets, Models, Metrics).
* **Lineage:** The "family tree" of an artifact, showing exactly which data and code produced it.

---

## 2. Key Google Cloud Pipeline Components (GCPC)
To automate Vertex AI services within a pipeline, we use pre-built components from the `google_cloud_pipeline_components` library.

### A. Data Management: `TextDatasetCreateOp`
This component automates the creation of a managed Vertex AI Dataset.
* **Use Case:** When you have raw data in GCS (like a CSV or JSONL for text classification) and want to register it as a formal Vertex AI Dataset resource.
* **Key Inputs:** `display_name`, `gcs_source`, `import_schema_uri`.

### B. Training: `CustomTrainingJobOp`
Instead of running a script locally, this component launches a managed Training Job on Google’s infrastructure.
* **Mechanism:** It packages your training code (often in a Docker container or a Python script), provisions GPUs/CPUs, and executes the training.
* **Key Inputs:** `worker_pool_specs` (defines machine types/accelerators), `command`, `args`.

### C. Deployment: `ModelDeployOp`
Once a model is trained and registered in the Model Registry, this component automates its deployment to an **Endpoint** for online predictions.
* **Function:** It creates a bridge between the Model resource and the Endpoint resource.
* **Key Inputs:** `model`, `endpoint`, `dedicated_resources` (machine types for the API).

---

## 3. Workflow Example: Putting it all together
Below is a conceptual Python snippet of how these operators interact within a `@pipeline` definition.

```python
from kfp import dsl
from google_cloud_pipeline_components.v1.dataset import TextDatasetCreateOp
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp

@dsl.pipeline(name="text-classification-pipeline")
def pipeline(project: str, location: str):
    
    # 1. Create the Managed Dataset
    dataset_task = TextDatasetCreateOp(
        project=project,
        display_name="news-dataset",
        gcs_source="gs://my-bucket/data.csv",
        import_schema_uri="gs://google-cloud-aiplatform/schema/dataset/serialization/text_classification_1.0.0.yaml"
    )

    # 2. Run Custom Training
    # Note: Training code would be in a separate container/script
    training_task = CustomTrainingJobOp(
        project=project,
        display_name="train-text-model",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n1-standard-4"},
            "container_spec": {"image_uri": "gcr.io/my-project/train-image:latest"}
        }]
    )

    # 3. Create an Endpoint
    endpoint_task = EndpointCreateOp(
        project=project,
        display_name="text-model-endpoint"
    )

    # 4. Deploy the Model (Depends on training and endpoint creation)
    deploy_task = ModelDeployOp(
        model=training_task.outputs["model"],
        endpoint=endpoint_task.outputs["endpoint"],
        dedicated_resources_machine_type="n1-standard-2",
        dedicated_resources_min_replica_count=1
    )
```