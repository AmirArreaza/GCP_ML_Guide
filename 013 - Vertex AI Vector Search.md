# Vertex AI Vector Search
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Vertex AI Vector Search?

Vertex AI Vector Search (formerly known as **Vertex AI Matching Engine**) is a fully managed, high-scale, low-latency service for finding similar vectors — also known as approximate nearest neighbor (ANN) search. It is built on Google's **ScaNN (Scalable Nearest Neighbors)** algorithm, the same technology powering Google Search, YouTube, and Google Play.

**Use cases:** Semantic search, recommendation engines, image similarity, document retrieval, RAG pipelines, embedding-based classification, fraud detection, drug discovery.

```
Query embedding  →  Vector Search index  →  Top-K nearest neighbours (IDs + distances)
```

**Key characteristics:**
- Sub-10ms query latency even with billions of vectors
- Fully managed — no infrastructure to configure
- Supports both **ANN (production)** and **brute-force (ground truth / evaluation)**
- Supports **filtering** (restricts) and **hybrid search** (dense + sparse)

---

## 2. Vector Search Versions

Vertex AI now offers **two versions**:

| | Vector Search 1.0 | Vector Search 2.0 (GA, 2026) |
|---|---|---|
| Primary resource | Index + IndexEndpoint | Collection of Data Objects |
| Data model | Vectors stored separately in GCS | Vectors, metadata, and content together in Data Objects |
| Storage | External (GCS, Datastore) | Built-in replicated storage engine |
| Setup | Index → deploy to Endpoint | Simpler — Collection-first |
| Best for | Custom pipelines, RAG Engine backend | New projects, unified AI storage |
| Tuning | Manual (shard size, leaf nodes, etc.) | Auto-tuned |

> **Exam note:** The current exam primarily covers **Vector Search 1.0** (Index + Endpoint pattern). Vector Search 2.0 is GA but newer. Know both, but focus on the Index/Endpoint workflow.

---

## 3. Core Concepts (Vector Search 1.0)

| Concept | Description |
|---------|-------------|
| **Index** | The ANN structure built from your embedding vectors — stored as shards, optimised for fast lookup |
| **Index Endpoint** | The serving layer — a managed gRPC/REST endpoint that hosts one or more deployed indexes |
| **Deployed Index** | An index that has been deployed to an endpoint and is ready to serve queries |
| **Recall** | % of true nearest neighbours returned by ANN — e.g. 19/20 correct = 95% recall |
| **Restrict (Filter)** | Limit query results to a subset of vectors using Boolean or numeric rules |
| **Datapoint** | A single vector (embedding) + its ID + optional metadata |
| **Approximate Nearest Neighbour (ANN)** | Fast but approximate — trades tiny accuracy loss (~1%) for massive speed gains |
| **Brute Force** | Exact nearest neighbour — slow, used for ground truth evaluation only |

---

## 4. ANN Algorithm — Tree-AH (ScaNN)

Vertex AI Vector Search uses a **tree-based AH (Asymmetric Hashing)** index algorithm (create_tree_ah_index), powered by ScaNN:

```
Tree-AH index:
  1. Vectors are clustered into a hierarchical tree
  2. Query traverses the tree to find candidate clusters
  3. AH (asymmetric hashing) ranks candidates within each cluster
  4. Returns top-K approximate nearest neighbours

SOAR enhancement (newer):
  Assigns vectors to multiple clusters (controlled redundancy)
  → "backup" search paths → faster search with smaller indexes
```

**Key tuning parameters:**

| Parameter | Description | Tuning guidance |
|-----------|-------------|----------------|
| `approximate_neighbors_count` | Number of ANN candidates per query | Increase for higher recall; default 100–150 |
| `leaf_node_embedding_count` | Vectors per leaf node | Smaller = finer clusters, higher precision |
| `leaf_nodes_to_search_percent` | % of leaf nodes to scan per query | Higher = more recall, higher latency |

---

## 5. Full Workflow — Python SDK

### Step 1 — Prepare Embeddings

```python
# Vector data must be stored in GCS as JSONL
# Each line = one datapoint: {"id": "doc_001", "embedding": [0.1, 0.2, ...]}
# or with restricts for filtering:
# {"id": "doc_001", "embedding": [...], "restricts": [{"namespace": "category", "allow_list": ["finance"]}]}

# Generate embeddings with text-embedding-005
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

vertexai.init(project="my-project", location="us-central1")
model = TextEmbeddingModel.from_pretrained("text-embedding-005")

texts = ["Annual report Q1 2024", "Remote work policy update", "Product safety guidelines"]
inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
embeddings = model.get_embeddings(inputs)

# Write to GCS as JSONL for batch ingestion
import json
import google.cloud.storage as gcs

client = gcs.Client()
bucket = client.bucket("my-bucket")
blob = bucket.blob("vector-search/embeddings.jsonl")
with blob.open("w") as f:
    for i, emb in enumerate(embeddings):
        record = {"id": f"doc_{i:03d}", "embedding": emb.values}
        f.write(json.dumps(record) + "\n")
```

### Step 2 — Create the Index

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# ANN index (Tree-AH) — production use
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="doc-search-index",
    contents_delta_uri="gs://my-bucket/vector-search/",  # GCS path to JSONL
    dimensions=768,                          # must match embedding model output
    approximate_neighbors_count=150,         # ANN candidates per query
    leaf_node_embedding_count=500,           # vectors per leaf node
    leaf_nodes_to_search_percent=7,          # % leaf nodes scanned per query
    distance_measure_type="DOT_PRODUCT_DISTANCE",  # similarity metric
    index_update_method="STREAM_UPDATE",     # STREAM_UPDATE or BATCH_UPDATE
    description="Semantic search index for company documents",
)

print(f"Index: {my_index.resource_name}")
# Note: Creation can take 10–60+ minutes depending on dataset size
```

### Step 3 — Create an Index Endpoint

```python
# Public endpoint — accessible over internet (IAM-secured)
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="doc-search-endpoint",
    public_endpoint_enabled=True,     # public; set False for VPC private
    description="Endpoint for document semantic search",
)
```

### Step 4 — Deploy the Index to the Endpoint

```python
# Deploy the index to the endpoint
my_index_endpoint.deploy_index(
    index=my_index,
    deployed_index_id="doc_search_v1",         # user-defined ID for the deployment
    display_name="doc-search-deployed",
    min_replica_count=1,
    max_replica_count=5,                       # autoscaling range
    enable_access_logging=True,                # log query requests
    deployment_group="prod",                   # optional grouping label
)

print("Index deployed and ready to serve queries")
```

### Step 5 — Query the Index

```python
# Embed the user's query
query_input = TextEmbeddingInput(
    text="What is the company remote work policy?",
    task_type="RETRIEVAL_QUERY",              # use RETRIEVAL_QUERY for queries
)
query_embedding = model.get_embeddings([query_input])
query_vector = query_embedding[0].values

# Find nearest neighbours
response = my_index_endpoint.find_neighbors(
    deployed_index_id="doc_search_v1",
    queries=[query_vector],               # list of query vectors
    num_neighbors=10,                     # top-K results
)

# Print results
for match in response[0]:
    print(f"ID: {match.id}  Distance: {match.distance:.4f}")
```

### Step 6 — Filtered Query (Restricts)

```python
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    Namespace, NumericNamespace
)

# Text/categorical filter
response = my_index_endpoint.find_neighbors(
    deployed_index_id="doc_search_v1",
    queries=[query_vector],
    num_neighbors=10,
    filter=[
        Namespace(name="category", allow_tokens=["finance", "legal"]),
        Namespace(name="language", allow_tokens=["en"]),
    ],
)

# Numeric filter
response = my_index_endpoint.find_neighbors(
    deployed_index_id="doc_search_v1",
    queries=[query_vector],
    num_neighbors=10,
    numeric_filter=[
        NumericNamespace(name="year", value_int=2024, op="GREATER_EQUAL"),
    ],
)
```

---

## 6. Distance Measure Types

The `distance_measure_type` determines how similarity between two vectors is computed:

| Measure | Use When | Notes |
|---------|---------|-------|
| `DOT_PRODUCT_DISTANCE` | Embeddings are normalised to unit length | Equivalent to cosine similarity when vectors are normalised. **Most common choice.** |
| `COSINE_DISTANCE` | Embeddings may not be normalised | Measures angle between vectors — direction only, not magnitude |
| `SQUARED_L2_DISTANCE` | Euclidean distance | Sensitive to magnitude — use when distance in space matters |
| `L1_DISTANCE` | Manhattan / taxicab distance | Less common; robust to outlier dimensions |

> **Exam tip:** For most embedding models (including `text-embedding-005`), use `DOT_PRODUCT_DISTANCE` with pre-normalised embeddings. Cosine similarity and dot product are equivalent when vectors have unit norm.

---

## 7. Index Update Methods

| Method | Description | Best For |
|--------|-------------|---------|
| `STREAM_UPDATE` | Real-time incremental updates — new vectors appear in the index almost immediately | Live systems, e-commerce inventory, news feeds |
| `BATCH_UPDATE` | Periodic bulk updates — vectors in GCS are processed in one batch job | Weekly/monthly data refresh, offline pipelines |

```python
# Streaming update — add or update individual vectors
my_index.upsert_datapoints(
    datapoints=[
        aiplatform.gapic.IndexDatapoint(
            datapoint_id="doc_999",
            feature_vector=[0.1, 0.2, ...],  # 768-dim vector
        )
    ]
)

# Remove a vector
my_index.remove_datapoints(datapoint_ids=["doc_001", "doc_002"])
```

---

## 8. Endpoint Types

| Type | Access | Networking | Best For |
|------|--------|-----------|---------|
| **Public Endpoint** | Internet-accessible (IAM-secured) | No VPC setup needed | Development, external APIs |
| **Private VPC Endpoint** | Within your VPC network only | Requires VPC peering | Production — lowest latency, highest security |

```python
# Public endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="public-endpoint",
    public_endpoint_enabled=True,
)

# Private VPC endpoint
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="private-endpoint",
    public_endpoint_enabled=False,
    network="projects/PROJECT_NUMBER/global/networks/my-vpc",
)
```

> **Exam tip:** Private VPC endpoints require **VPC peering** between your network and Google's managed services network. They offer lower latency and are recommended for production workloads.

---

## 9. Brute Force Index — Evaluation Only

```python
# Brute force index — finds EXACT nearest neighbours
# Use to compute ground truth recall vs ANN index
brute_force_index = aiplatform.MatchingEngineIndex.create_brute_force_index(
    display_name="doc-search-brute-force",
    contents_delta_uri="gs://my-bucket/vector-search/",
    dimensions=768,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)

# Compare ANN vs brute force to compute recall:
# recall = (neighbours returned by ANN that are in brute force ground truth) / num_neighbors
```

> **Exam tip:** Brute force indexes are **never used in production** — they are slow and not scalable. They exist to measure the **recall** (accuracy) of an ANN index during tuning and evaluation.

---

## 10. Hybrid Search (Dense + Sparse)

Vector Search supports **hybrid queries** combining dense (semantic) and sparse (keyword/BM25) embeddings:

```python
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery

hybrid_queries = [
    HybridQuery(
        dense_embedding=[0.1, 0.2, ...],          # semantic embedding
        sparse_embedding_dimensions=[10, 20, 30], # token IDs
        sparse_embedding_values=[1.0, 0.5, 0.8],  # TF-IDF weights
        rrf_ranking_alpha=0.5,                    # blend ratio: 0=sparse, 1=dense
    )
]

response = my_index_endpoint.find_neighbors(
    deployed_index_id="doc_search_v1",
    hybrid_queries=hybrid_queries,
    num_neighbors=10,
)
```

**`rrf_ranking_alpha`** controls the blend:
- `0.0` = pure sparse (keyword)
- `1.0` = pure dense (semantic)
- `0.5` = balanced

---

## 11. Recall — Measuring ANN Quality

```
Recall = (True nearest neighbours returned by ANN) / num_neighbors

Example: query for 20 neighbours, ANN returns 19 correct → recall = 19/20 = 95%
```

**To improve recall:**
- Increase `approximate_neighbors_count` (more ANN candidates)
- Increase `leaf_nodes_to_search_percent` (scan more of the tree)
- Both increase latency as a tradeoff

**Recall vs Latency tradeoff:**
```
Higher recall → more accurate   → but higher latency
Lower recall  → faster queries  → but some true neighbours missed
```

---

## 12. Autoscaling

```python
my_index_endpoint.deploy_index(
    index=my_index,
    deployed_index_id="doc_search_v1",
    min_replica_count=1,     # always-on minimum
    max_replica_count=10,    # scale up to 10 nodes during traffic spikes
)
```

Vector Search automatically adds or removes nodes based on query load — no manual intervention needed.

---

## 13. Integration with RAG Engine

Vertex AI Vector Search can serve as the **custom vector DB backend** for Vertex AI RAG Engine:

```python
from vertexai.preview import rag
from google.cloud import aiplatform

# Use Vector Search as the RAG corpus backend
backend_config = rag.RagVectorDbConfig(
    vertex_vector_search=rag.VertexVectorSearch(
        index="projects/PROJECT/locations/LOCATION/indexes/INDEX_ID",
        index_endpoint="projects/PROJECT/locations/LOCATION/indexEndpoints/ENDPOINT_ID",
    ),
    rag_embedding_model_config=rag.RagEmbeddingModelConfig(
        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
            publisher_model="publishers/google/models/text-embedding-005"
        )
    ),
)

rag_corpus = rag.create_corpus(
    display_name="vs-backed-corpus",
    backend_config=backend_config,
)
```

---

## 14. Input Data Format (JSONL for GCS)

```json
// Minimal: id + embedding
{"id": "doc_001", "embedding": [0.1, 0.2, ..., 0.8]}

// With categorical restricts (for filtering)
{
  "id": "doc_002",
  "embedding": [0.1, 0.2, ..., 0.8],
  "restricts": [
    {"namespace": "category", "allow_list": ["finance"]},
    {"namespace": "language", "allow_list": ["en"]}
  ]
}

// With numeric restricts
{
  "id": "doc_003",
  "embedding": [0.1, 0.2, ..., 0.8],
  "numeric_restricts": [
    {"namespace": "year", "value_int": 2024}
  ]
}

// With crowding tag (diversification)
{
  "id": "doc_004",
  "embedding": [0.1, 0.2, ..., 0.8],
  "crowding_tag": {"crowding_attribute": "author_id_123"}
}
```

---

## 15. Key API Classes — Quick Reference

| Class / Method | Description |
|---------------|-------------|
| `MatchingEngineIndex.create_tree_ah_index()` | Create ANN index (production) |
| `MatchingEngineIndex.create_brute_force_index()` | Create brute force index (evaluation only) |
| `MatchingEngineIndexEndpoint.create()` | Create serving endpoint |
| `endpoint.deploy_index()` | Deploy an index to an endpoint |
| `endpoint.find_neighbors()` | Query for nearest neighbours |
| `index.upsert_datapoints()` | Add or update vectors (streaming) |
| `index.remove_datapoints()` | Delete vectors from the index |
| `HybridQuery` | Hybrid dense + sparse query |
| `Namespace` | Categorical filter for queries |
| `NumericNamespace` | Numeric filter for queries |

---

## 16. Vertex AI Vector Search vs RAG Engine vs Vertex AI Search

| Aspect | Vector Search | RAG Engine | Vertex AI Search |
|--------|--------------|------------|-----------------|
| Primary function | Pure ANN vector lookup | Managed RAG pipeline | Enterprise search + AI answers |
| Data input | Raw embedding vectors | Documents (chunked + embedded) | Documents, websites, BQ tables |
| Chunking/embedding | You manage | Managed | Fully managed |
| AI answers | No — returns IDs + distances | Via Gemini tool | Built-in SummarySpec |
| Use case | Custom similarity, RAG backend | Controlled RAG | Enterprise search apps |
| Control level | Maximum | Medium | Minimum |
| Setup complexity | High | Medium | Low |

---

## 17. Exam-Relevant Tips

- Vertex AI Vector Search was formerly called **Matching Engine** — you'll see both names in exam questions.
- The **Tree-AH index** (`create_tree_ah_index`) is for production ANN queries; **brute force** is for ground truth evaluation only.
- `STREAM_UPDATE` = real-time incremental; `BATCH_UPDATE` = periodic bulk update from GCS.
- **Public endpoints** are IAM-secured — not open to the internet without authentication. **Private endpoints** require VPC peering.
- `dimensions` must exactly match the output size of your embedding model (e.g. 768 for `text-embedding-005`).
- **Recall** measures ANN accuracy — higher `approximate_neighbors_count` and `leaf_nodes_to_search_percent` improve recall but increase latency.
- **Restricts** (filtering) can narrow the search space — text or numeric — without post-filtering all results.
- **Hybrid search** (`HybridQuery`) combines dense vectors + sparse vectors with `rrf_ranking_alpha` for blending.
- `DOT_PRODUCT_DISTANCE` is the most common distance measure for normalised embeddings from Google models.
- Index creation can take **10 minutes to over 1 hour** for large datasets — plan accordingly.
- Vector Search powers the vector DB backend for **Vertex AI RAG Engine** when you use `VertexVectorSearch` in the corpus config.
- **Autoscaling**: `min_replica_count` and `max_replica_count` control node scaling based on traffic.
- Vector Search 2.0 (GA 2026) uses **Collections of Data Objects** — simpler, auto-tuned, unified storage.

---

## 18. Quick Reference Cheat Sheet

```
INDEX CREATION
  create_tree_ah_index()          →  ANN production index (Tree-AH / ScaNN)
  create_brute_force_index()      →  exact NN — evaluation only (never production)
  dimensions=                     →  must match embedding model output
  distance_measure_type=          →  DOT_PRODUCT / COSINE / SQUARED_L2 / L1
  index_update_method=            →  STREAM_UPDATE (realtime) or BATCH_UPDATE (bulk)
  approximate_neighbors_count=    →  ANN candidates — higher = more recall

ENDPOINT
  MatchingEngineIndexEndpoint.create()  →  create serving endpoint
  public_endpoint_enabled=True          →  public (IAM-secured)
  network=                              →  private VPC (requires VPC peering)
  endpoint.deploy_index()               →  deploy index to endpoint

QUERYING
  endpoint.find_neighbors(              →  run ANN query
    queries=[embedding_vector],
    num_neighbors=K,
    filter=[Namespace(...)],            →  categorical filter (restricts)
    numeric_filter=[NumericNamespace],  →  numeric filter
    hybrid_queries=[HybridQuery(...)]   →  dense + sparse hybrid
  )

UPDATES
  index.upsert_datapoints()    →  add/update vectors
  index.remove_datapoints()    →  delete vectors

RECALL
  recall = ANN_correct / num_neighbors  →  measure ANN accuracy vs brute force
  higher approximate_neighbors_count     →  better recall, higher latency

V2.0 (NEW)
  Collection + Data Objects    →  unified storage, auto-tuned, simpler setup
```

---

*Study tip: Focus on the Index → Endpoint → Deploy → Query workflow, the difference between ANN (production) and brute force (evaluation), distance measure choices, STREAM vs BATCH update, and when to use private vs public endpoints — these are the most frequently tested areas for Vector Search on the GCP ML Engineer exam.*