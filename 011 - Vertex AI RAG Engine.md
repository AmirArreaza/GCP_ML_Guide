# Vertex AI RAG Engine
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Vertex AI RAG Engine?

Vertex AI RAG Engine, a component of the Vertex AI Platform, facilitates Retrieval-Augmented Generation (RAG). It is also a data framework for developing context-augmented large language model (LLM) applications. A common problem with LLMs is that they don't understand private knowledge — that is, your organisation's data. With Vertex AI RAG Engine, you can enrich the LLM context with additional private information, because the model can reduce hallucination and answer questions more accurately.

**Use cases:** Enterprise knowledge bases, document Q&A, policy auditing, personalised investment advice, drug discovery, contract review, customer support grounding.

```
Problem:  LLMs hallucinate / lack private or recent knowledge
Solution: Retrieve relevant chunks → augment the prompt → generate grounded response
```

---

## 2. RAG Pipeline — End-to-End Flow

```
Documents (GCS / Drive / Local)
        ↓
  [1] Data Ingestion
        ↓
  [2] Chunking (chunk_size, chunk_overlap)
        ↓
  [3] Embedding (text-embedding-005 or custom)
        ↓
  [4] Indexing into Corpus (managed Spanner vector DB)
        ↓
  [5] User Query → Embed query
        ↓
  [6] Retrieval (top_k + vector_distance_threshold)
        ↓
  [7] (Optional) Reranking (Vertex AI Ranking API)
        ↓
  [8] Augmented Prompt → LLM (Gemini)
        ↓
  [9] Grounded Response
```

---

## 3. Core Concepts

| Concept | Description |
|---------|-------------|
| **Corpus** | The index — a managed collection of document chunks and their embeddings. Think of it as a detailed table of contents for your knowledge base. |
| **Chunk** | A fragment of a document used as the unit of retrieval. Size is configurable. |
| **Embedding** | A numeric vector representing the semantic meaning of a text chunk. Similar texts have similar vectors. |
| **Vector DB** | Where embeddings are stored for similarity search. RAG Engine uses a managed Spanner-based DB by default. |
| **Grounding** | Using retrieved chunks as factual context to constrain LLM generation. |
| **top_k** | How many most-similar chunks to retrieve per query. |
| **vector_distance_threshold** | Only return chunks with similarity above this cutoff. |

---

## 4. Full Workflow — Python SDK

### Step 1 — Install and Initialise

```python
pip install google-cloud-aiplatform

import vertexai
from vertexai.preview import rag

vertexai.init(project="my-project", location="us-central1")
```

### Step 2 — Create a Corpus

```python
# Configure the embedding model
embedding_model_config = rag.RagEmbeddingModelConfig(
    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
        publisher_model="publishers/google/models/text-embedding-005"
    )
)

# Configure the vector DB backend
backend_config = rag.RagVectorDbConfig(
    rag_embedding_model_config=embedding_model_config
)

# Create the corpus (the managed index)
rag_corpus = rag.create_corpus(
    display_name="company-knowledge-base",
    description="Internal policies, FAQs, and product documentation",
    backend_config=backend_config,
)

print(f"Corpus name: {rag_corpus.name}")
# projects/my-project/locations/us-central1/ragCorpora/CORPUS_ID
```

### Step 3 — Import Files (with chunking config)

```python
# Import from Google Cloud Storage
rag.import_files(
    corpus_name=rag_corpus.name,
    paths=["gs://my-bucket/documents/"],          # GCS bucket or specific file
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(
            chunk_size=512,         # tokens per chunk
            chunk_overlap=100,      # overlap between consecutive chunks
        )
    ),
    max_embedding_requests_per_min=900,           # rate limit for embedding API calls
)

# Import from Google Drive
rag.import_files(
    corpus_name=rag_corpus.name,
    paths=["https://drive.google.com/drive/folders/FOLDER_ID"],
    transformation_config=rag.TransformationConfig(
        chunking_config=rag.ChunkingConfig(chunk_size=512, chunk_overlap=100)
    ),
)

# Upload a single local file
rag_file = rag.upload_file(
    corpus_name=rag_corpus.name,
    path="./local_document.pdf",
    display_name="product-manual",
)
```

### Step 4 — Retrieve Contexts (retrieve only, no generation)

```python
retrieval_config = rag.RagRetrievalConfig(
    top_k=5,
    filter=rag.Filter(vector_distance_threshold=0.5),
)

response = rag.retrieval_query(
    rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
    text="What is our refund policy for digital products?",
    rag_retrieval_config=retrieval_config,
)

for context in response.contexts.contexts:
    print(f"Source: {context.source_uri}")
    print(f"Score:  {context.score:.4f}")
    print(f"Text:   {context.text[:300]}")
    print()
```

### Step 5 — End-to-End RAG with Gemini (recommended pattern)

```python
from vertexai.generative_models import GenerativeModel, Tool

# Create a retrieval tool backed by the corpus
rag_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=5,
                filter=rag.Filter(vector_distance_threshold=0.5),
            ),
        )
    )
)

# Attach the tool to the model — Gemini retrieves and generates in one call
model = GenerativeModel(
    model_name="gemini-2.0-flash-001",
    tools=[rag_tool],
)

response = model.generate_content("What is the company refund policy?")
print(response.text)
```

> **Exam tip:** The end-to-end pattern using `Tool.from_retrieval()` is the most common production approach — Gemini handles retrieval and generation in a **single `generate_content()` call**.

---

## 5. Chunking Strategy

When documents are ingested into an index, they are split into chunks. RAG Engine provides the possibility to tune chunk size and chunk overlap and different strategies to support different types of documents.

| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| `chunk_size` | 256–1024 tokens | Size of each chunk. Smaller = more precise; larger = more context. |
| `chunk_overlap` | 50–200 tokens | Tokens shared between consecutive chunks — preserves boundary context. |

### Tuning Guidance

| Scenario | `chunk_size` | `chunk_overlap` |
|----------|-------------|----------------|
| Quick prototype | 300 | 50 |
| Standard production | 512 | 100 |
| Long-form documents needing more context | 1024 | 200 |
| Short precise answers needed | 256 | 50 |

> **Rule of thumb:** Start with 512 / 100. If answers lack context, increase. If retrieval returns irrelevant chunks, decrease.

---

## 6. Embedding Models

RAG Engine uses embedding models to convert text into vectors. The recommended model is `text-embedding-005`.

```python
# Supported publisher models:
"publishers/google/models/text-embedding-005"          # default, best balance
"publishers/google/models/textembedding-gecko@003"     # older, still supported
"publishers/google/models/text-multilingual-embedding-002"  # 100+ languages
```

### Embedding Task Types

Always specify task type for better retrieval quality:

| Task Type | Use For |
|-----------|---------|
| `RETRIEVAL_DOCUMENT` | Embedding document chunks at ingestion time |
| `RETRIEVAL_QUERY` | Embedding the user's query at retrieval time |
| `SEMANTIC_SIMILARITY` | Comparing pairs of text for similarity |
| `CLASSIFICATION` | Text classification tasks |

---

## 7. Vector Database Backends

RAG Engine works with your choice of vector database, or if you prefer, can manage the vector storage entirely for you. This flexibility ensures you're never locked into a single approach as your needs evolve.

| Backend | Description | Best For |
|---------|-------------|---------|
| **RAG Managed DB** (default) | Managed Spanner-based vector store — zero-ops | Most use cases; easiest setup |
| **Vertex AI Vector Search** | Google's scalable ANN search; high performance | Very large corpora, low latency requirements |
| **Pinecone** | External managed vector DB | Teams already using Pinecone |
| **Weaviate** | Open-source vector DB | Open-source preference |

```python
# Using Vertex AI Vector Search as backend (bring-your-own index)
backend_config = rag.RagVectorDbConfig(
    vertex_vector_search=rag.VertexVectorSearch(
        index="projects/PROJECT/locations/LOCATION/indexes/INDEX_ID",
        index_endpoint="projects/PROJECT/locations/LOCATION/indexEndpoints/ENDPOINT_ID",
    ),
    rag_embedding_model_config=embedding_model_config,
)
```

---

## 8. Retrieval Configuration

```python
retrieval_config = rag.RagRetrievalConfig(
    top_k=10,                                         # retrieve top 10 chunks
    filter=rag.Filter(
        vector_distance_threshold=0.5,                # only chunks with score >= 0.5
    ),
    hybrid_search=rag.HybridSearch(
        alpha=0.6,                                    # 0=keyword only, 1=semantic only
    ),
)
```

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 3 | Number of most relevant chunks to retrieve |
| `vector_distance_threshold` | 0.5 | Minimum similarity score (0–1); higher = stricter |
| `alpha` (hybrid search) | 1.0 (pure semantic) | Balance between keyword (0) and semantic (1) search |

### Hybrid Search Alpha Guide

| alpha | Behaviour | Best For |
|-------|-----------|---------|
| `0.0–0.3` | Favour keyword matching | Exact codes, product SKUs, error codes |
| `0.4–0.6` | Balanced | Mixed queries |
| `0.7–1.0` | Favour semantic similarity | Natural language conceptual questions |

---

## 9. Reranking

After initial retrieval, a **reranker** re-scores chunks using deeper query-document understanding, improving precision.

```python
from vertexai.preview import rag

retrieval_config = rag.RagRetrievalConfig(
    top_k=20,                                          # retrieve more candidates
    filter=rag.Filter(vector_distance_threshold=0.3),
    ranking=rag.Ranking(
        rank_service=rag.RankService(
            model_name="semantic-ranker-default@latest"  # Vertex AI Ranking API
        )
    ),
)
```

> **Exam tip:** Reranking improves quality by retrieving a large `top_k` candidate set first, then reranking with a cross-encoder model. The final answer uses the top reranked results.

---

## 10. Corpus Management

```python
# List all corpora
corpora = rag.list_corpora()
for c in corpora:
    print(c.name, c.display_name)

# Get a specific corpus
corpus = rag.get_corpus(name="projects/.../ragCorpora/CORPUS_ID")

# List files in a corpus
files = rag.list_files(corpus_name=rag_corpus.name)

# Delete a file from the corpus
rag.delete_file(name="projects/.../ragCorpora/.../ragFiles/FILE_ID")

# Delete an entire corpus
rag.delete_corpus(name=rag_corpus.name)
```

---

## 11. ADK Integration — VertexAiRagRetrieval Tool

RAG Engine integrates directly with the Agent Development Kit (ADK):

```python
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

# Create the RAG retrieval tool
rag_tool = VertexAiRagRetrieval(
    name="retrieve_company_docs",
    description="Retrieve relevant documentation and policies from the knowledge base.",
    rag_resources=[
        rag.RagResource(
            rag_corpus="projects/my-project/locations/us-central1/ragCorpora/CORPUS_ID"
        )
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)

# Attach to an ADK agent
root_agent = Agent(
    model="gemini-2.0-flash",
    name="knowledge_agent",
    instruction="Answer questions using the company knowledge base.",
    tools=[rag_tool],
)
```

> **Exam tip:** `VertexAiRagRetrieval` can only be used **by itself** in an agent instance — it cannot be combined with other tools in the same agent due to current ADK limitations.

---

## 12. Grounding Check API

The **Check Grounding API** validates whether an LLM response is actually supported by the retrieved sources:

```python
from google.cloud import discoveryengine_v1alpha as discoveryengine

client = discoveryengine.GroundedGenerationServiceClient()

response = client.check_grounding(
    request=discoveryengine.CheckGroundingRequest(
        grounding_spec=discoveryengine.CheckGroundingSpec(citation_threshold=0.6),
        answer_candidate="Our return policy allows 30 days for refunds.",
        facts=[
            discoveryengine.GroundingFact(fact_text="Returns are accepted within 30 days of purchase.")
        ],
    )
)

print(f"Support score: {response.support_score}")
print(f"Supported: {response.supported}")
```

---

## 13. DIY RAG vs Managed RAG Engine

Vertex AI RAG Engine is a managed orchestration service, streamlining the complex process of retrieving relevant information and feeding it to an LLM. This allows developers to focus on building their applications rather than managing infrastructure.

| Aspect | Vertex AI RAG Engine (Managed) | DIY RAG (Custom Pipeline) |
|--------|-------------------------------|--------------------------|
| Setup effort | Low — simple API | High — custom code |
| Control | Moderate | Full |
| Vector DB | Managed (Spanner or BYO) | You manage (Vector Search, Pinecone, etc.) |
| Chunking | Built-in, configurable | Custom logic |
| Embedding | Configured via SDK | Explicit API calls |
| Reranking | Built-in (Ranking API) | Manual integration |
| Best for | Fast prototyping to production | Complex custom pipelines |

---

## 14. Key API Methods — Quick Reference

| Method | Description |
|--------|-------------|
| `rag.create_corpus()` | Create a new managed corpus (index) |
| `rag.import_files()` | Ingest files from GCS or Google Drive with chunking |
| `rag.upload_file()` | Upload a single local file to the corpus |
| `rag.retrieval_query()` | Retrieve relevant chunks only (no generation) |
| `Tool.from_retrieval()` | Create a Gemini-compatible RAG tool for end-to-end generation |
| `rag.list_corpora()` | List all corpora in the project |
| `rag.list_files()` | List files within a corpus |
| `rag.delete_corpus()` | Delete a corpus |
| `VertexAiRagRetrieval` | ADK tool for private data retrieval via RAG Engine |

---

## 15. Exam-Relevant Tips

- A **corpus** is the managed index — it holds chunked documents and their embeddings.
- The default vector DB is a **managed Spanner-based store** — zero infrastructure needed.
- The recommended embedding model is **`text-embedding-005`**.
- `chunk_size=512, chunk_overlap=100` is the standard production starting point.
- `top_k` controls how many chunks are retrieved; `vector_distance_threshold` filters by minimum similarity.
- **Hybrid search** (`alpha`) balances keyword vs semantic retrieval — lower alpha = more keyword, higher = more semantic.
- **Reranking** improves precision: retrieve a large `top_k`, then re-score with the Ranking API.
- `Tool.from_retrieval()` + `GenerativeModel.generate_content()` is the **end-to-end RAG pattern** — one call does both retrieval and generation.
- `rag.retrieval_query()` is for **retrieval only** — returns chunks without calling the LLM.
- The **Check Grounding API** validates whether a response is actually supported by retrieved facts.
- In ADK, use `VertexAiRagRetrieval` — it **cannot be combined** with other tools in the same agent.
- `max_embedding_requests_per_min` limits embedding API quota consumption during large imports.
- RAG Engine supports **multiple corpora** per project — useful for isolating data by domain or security boundary.
- Multiple distinct corpuses provide a security and isolation mechanism, ensuring a model working on sensitive data is restricted to that corpus.

---

## 16. Quick Reference Cheat Sheet

```
CORPUS MANAGEMENT
  rag.create_corpus()          →  create the managed index
  rag.import_files()           →  ingest from GCS or Drive with chunking
  rag.upload_file()            →  upload single local file
  rag.list_files()             →  list files in corpus
  rag.delete_corpus()          →  remove entire corpus

CHUNKING
  chunk_size=512               →  tokens per chunk (start here)
  chunk_overlap=100            →  token overlap between chunks

RETRIEVAL
  rag.retrieval_query()        →  chunks only (no LLM)
  top_k=5                      →  number of chunks to return
  vector_distance_threshold    →  minimum similarity cutoff
  alpha (hybrid_search)        →  0=keyword, 1=semantic

END-TO-END RAG
  Tool.from_retrieval()        →  wrap corpus as Gemini tool
  model.generate_content()     →  retrieve + generate in one call

RERANKING
  rag.Ranking(rank_service)    →  Vertex AI Ranking API post-retrieval

ADK INTEGRATION
  VertexAiRagRetrieval         →  ADK tool for corpus retrieval
                                  (cannot combine with other tools)

GROUNDING VALIDATION
  Check Grounding API          →  verify response is supported by sources
```

---

*Study tip: Know the difference between `rag.retrieval_query()` (chunks only) and `Tool.from_retrieval()` with `generate_content()` (end-to-end). Also know the chunking defaults (512/100), the managed vector DB (Spanner-based), and why multi-corpus isolation is a security best practice — all common exam scenarios.*