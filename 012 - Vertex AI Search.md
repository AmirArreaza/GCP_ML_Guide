# Vertex AI Search
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Vertex AI Search?

Vertex AI Search (formerly Enterprise Search on Generative AI App Builder) is a **fully managed search service** that lets you build powerful enterprise-grade search applications over your private data — with the quality of Google Search applied to your own documents, websites, and structured data.

It supports:
- **Semantic and keyword search** over private content
- **AI-generated answers** (grounded summaries) for conversational queries
- **Grounding** of Gemini model responses against private data stores
- **Recommendations** for content personalisation

**Use cases:** Internal knowledge base search, enterprise document search, e-commerce search, healthcare FHIR search, media content discovery, website search, customer support Q&A.

---

## 2. Core Architecture — Apps and Data Stores

With Vertex AI Search, you create a **search or recommendations app** and connect it to a **data store**.

```
Data Store  (the indexed knowledge)
    ↕  connect
Search App  (the serving layer — query, config, UI widget)
```

### Relationship Rules

| App Type | Data Store Relationship |
|----------|------------------------|
| **Custom Search** | Many-to-many — one app can connect to multiple data stores (blended search) |
| **Custom Recommendations** | One-to-one — one app per data store |
| **Media App** | Many-to-one — multiple apps can share one data store |

> **Blended search:** When multiple data stores are connected to a single custom search app, Vertex AI Search searches across all of them and merges results.

---

## 3. Data Store Types

| Data Type | Document Format | Use Case |
|-----------|----------------|---------|
| **Structured** | Table rows / JSON records with schema | Searchable databases, product catalogues, FAQs |
| **Unstructured** | TXT, PDF, HTML, DOCX, PPTX, XLSX, XLSM | Document repositories, knowledge bases |
| **Website** | Crawled web pages | Internal portals, public websites |
| **Structured (Media)** | JSON with title, URI, categories, duration | Videos, podcasts, news articles |
| **Healthcare FHIR** | FHIR R4 resources | Clinical search, patient records |

---

## 4. Data Ingestion Sources

Vertex AI Search can ingest data from multiple sources:

| Source | Details |
|--------|---------|
| **Google Cloud Storage (GCS)** | CSV, JSON, PDF, HTML, DOCX and more |
| **BigQuery** | Import from BQ tables directly |
| **Google Drive** | Documents, spreadsheets, presentations |
| **Inline JSON** | Direct API upload |
| **Website crawling** | Sitemap-based or manual crawl |
| **Third-party connectors** | Confluence, Jira, Salesforce, SharePoint, Slack, etc. |
| **Healthcare FHIR** | HL7 FHIR R4 bundles |

---

## 5. Full Workflow — Python SDK

### Install

```python
pip install google-cloud-discoveryengine
```

### Step 1 — Create a Data Store

```python
from google.cloud import discoveryengine_v1 as discoveryengine

client = discoveryengine.DataStoreServiceClient()

data_store = client.create_data_store(
    parent=f"projects/my-project/locations/global/collections/default_collection",
    data_store=discoveryengine.DataStore(
        display_name="company-knowledge-base",
        industry_vertical=discoveryengine.IndustryVertical.GENERIC,
        content_config=discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED,
        solution_types=[discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH],
    ),
    data_store_id="company-kb",
)
```

### Step 2 — Import Documents

```python
# Import from GCS
operation = client.import_documents(
    request=discoveryengine.ImportDocumentsRequest(
        parent=f"projects/my-project/locations/global/collections/default_collection/dataStores/company-kb/branches/default_branch",
        gcs_source=discoveryengine.GcsSource(
            input_uris=["gs://my-bucket/documents/*"],
            data_schema="document",       # or "content" for unstructured files
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.INCREMENTAL,
    )
)
operation.result()  # Wait for import to complete
```

```python
# Import from BigQuery
operation = client.import_documents(
    request=discoveryengine.ImportDocumentsRequest(
        parent=f"projects/my-project/locations/global/collections/default_collection/dataStores/company-kb/branches/default_branch",
        bigquery_source=discoveryengine.BigQuerySource(
            project_id="my-project",
            dataset_id="my_dataset",
            table_id="knowledge_articles",
            data_schema="custom",
        ),
        reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
    )
)
```

### Step 3 — Create a Search App

```python
engine_client = discoveryengine.EngineServiceClient()

engine = engine_client.create_engine(
    parent="projects/my-project/locations/global/collections/default_collection",
    engine=discoveryengine.Engine(
        display_name="company-search-app",
        solution_type=discoveryengine.SolutionType.SOLUTION_TYPE_SEARCH,
        data_store_ids=["company-kb"],
        search_engine_config=discoveryengine.Engine.SearchEngineConfig(
            search_tier=discoveryengine.SearchTier.SEARCH_TIER_ENTERPRISE,
            search_add_ons=[discoveryengine.SearchAddOn.SEARCH_ADD_ON_LLM],
        ),
    ),
    engine_id="company-search-engine",
)
```

### Step 4 — Query the Search App

```python
search_client = discoveryengine.SearchServiceClient()

response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=f"projects/my-project/locations/global/collections/default_collection/engines/company-search-engine/servingConfigs/default_config",
        query="What is the company remote work policy?",
        page_size=10,
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            # Snippet config — short text extracts
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True,
                max_snippet_count=3,
            ),
            # Summary config — AI-generated answer
            summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                summary_result_count=5,                  # docs to summarise from
                include_citations=True,
                ignore_adversarial_query=True,
                ignore_non_summary_seeking_query=True,
                model_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelSpec(
                    version="stable",
                ),
            ),
            # Extractive answers — verbatim passages from documents
            extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                max_extractive_answer_count=1,
                max_extractive_segment_count=3,
            ),
        ),
    )
)

# AI-generated summary answer
print(response.summary.summary_text)

# Individual document results
for result in response.results:
    print(result.document.derived_struct_data)
```

---

## 6. Search Response Types

Vertex AI Search can return three types of content per query:

| Type | Description | Config |
|------|-------------|--------|
| **Snippets** | Short text extracts highlighting relevant passages | `SnippetSpec` |
| **Extractive Answers** | Verbatim exact passages from documents | `ExtractiveContentSpec` |
| **Summary (AI Answer)** | AI-generated grounded answer synthesising multiple documents | `SummarySpec` |

```python
# In the response:
response.summary.summary_text          # AI-generated answer
response.summary.safety_attributes     # safety classifications
response.results[0].document           # ranked document result
response.results[0].document.derived_struct_data["snippets"]  # text snippets
```

---

## 7. Grounding Gemini with Vertex AI Search

Vertex AI Search can be used as a **grounding data source** for Gemini — allowing the model to generate answers anchored to your private data.

### Grounding Pattern 1 — Vertex AI Search as a Gemini Tool

```python
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, grounding

vertexai.init(project="my-project", location="us-central1")

# Connect Gemini to a Vertex AI Search data store
data_store_path = "projects/my-project/locations/global/collections/default_collection/dataStores/company-kb"

grounding_tool = Tool.from_retrieval(
    grounding.Retrieval(
        source=grounding.VertexAISearch(datastore=data_store_path)
    )
)

model = GenerativeModel(
    model_name="gemini-2.0-flash-001",
    tools=[grounding_tool],
)

response = model.generate_content("What is the company's parental leave policy?")
print(response.text)

# Access grounding metadata
for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
    print(chunk.retrieved_context.uri)
    print(chunk.retrieved_context.text)
```

### Grounding Pattern 2 — Google Search Grounding

```python
from vertexai.generative_models import GenerativeModel, Tool, grounding

# Ground with Google's public web index
grounding_tool = Tool.from_google_search_retrieval(
    grounding.GoogleSearchRetrieval()
)

model = GenerativeModel(
    model_name="gemini-2.0-flash-001",
    tools=[grounding_tool],
)

response = model.generate_content("What are the latest developments in Vertex AI?")
print(response.text)
print(response.candidates[0].grounding_metadata.web_search_queries)
```

### Grounding Metadata in Response

```python
grounding_metadata = response.candidates[0].grounding_metadata

grounding_metadata.grounding_chunks        # source documents / web pages
grounding_metadata.grounding_supports      # which text segments are supported by which sources
grounding_metadata.web_search_queries      # queries issued to Google Search
grounding_metadata.search_entry_point      # rendered Search Suggestion chip HTML
```

---

## 8. Get Answers API (Conversational)

The **Answer API** enables multi-turn conversational search with follow-up questions:

```python
answer_client = discoveryengine.ConversationalSearchServiceClient()

# Initial question
response = answer_client.answer_query(
    discoveryengine.AnswerQueryRequest(
        serving_config=f"projects/my-project/locations/global/collections/default_collection/engines/company-search-engine/servingConfigs/default_config",
        query=discoveryengine.Query(text="What is the refund policy?"),
        answer_generation_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
            include_citations=True,
            model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                model_version="gemini-2.0-flash-001/answer_gen/v1",
            ),
        ),
    )
)

print(response.answer.answer_text)
session_id = response.session.name      # store for follow-up

# Follow-up question
followup_response = answer_client.answer_query(
    discoveryengine.AnswerQueryRequest(
        serving_config=...,
        query=discoveryengine.Query(text="What about digital purchases?"),
        session=session_id,              # pass session for context continuity
    )
)
```

---

## 9. Filtering and Boosting

### Filter by Metadata

```python
response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=...,
        query="remote work policy",
        filter='category: ANY("HR") AND year >= 2024',     # metadata filter
    )
)
```

### Boost Specific Results

```python
response = search_client.search(
    discoveryengine.SearchRequest(
        serving_config=...,
        query="annual leave",
        boost_spec=discoveryengine.SearchRequest.BoostSpec(
            condition_boost_specs=[
                discoveryengine.SearchRequest.BoostSpec.ConditionBoostSpec(
                    condition='document_type: ANY("policy")',
                    boost=0.5,         # boost value: -1 (bury) to +1 (promote)
                )
            ]
        ),
    )
)
```

---

## 10. Serving Tiers and Add-ons

| Tier / Add-on | Description |
|--------------|-------------|
| `SEARCH_TIER_STANDARD` | Basic keyword + semantic search |
| `SEARCH_TIER_ENTERPRISE` | Advanced: spell correction, synonyms, custom ranking |
| `SEARCH_ADD_ON_LLM` | Adds AI-generated summaries and extractive answers |

> **Exam tip:** `SEARCH_ADD_ON_LLM` must be enabled to use `SummarySpec` (AI answers). Without it, you only get document snippets.

---

## 11. Schema — Structured vs Unstructured

### Auto-detect schema (unstructured)

```python
discoveryengine.DataStore.ContentConfig.CONTENT_REQUIRED
# Vertex AI Search automatically parses and indexes documents
```

### Provide your own schema (structured)

```python
# Schema is provided as a JSON Schema definition
# Vertex AI Search uses it to understand field types for filtering and faceting
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "title":    {"type": "string", "keyPropertyMapping": "title"},
    "content":  {"type": "string", "keyPropertyMapping": "content"},
    "category": {"type": "string"},
    "year":     {"type": "integer"}
  }
}
```

---

## 12. Website Search

```python
# Website data store — indexed from URLs or sitemaps
data_store = discoveryengine.DataStore(
    display_name="company-website",
    industry_vertical=discoveryengine.IndustryVertical.GENERIC,
    content_config=discoveryengine.DataStore.ContentConfig.PUBLIC_WEBSITE,
)
```

**Advanced Website Indexing:**
- Submit and maintain sitemaps for automatic re-crawling
- Verify domains for security
- Add structured metadata via meta tags, PageMap, or schema.org

---

## 13. ADK Integration — Vertex AI Search as Grounding Tool

Vertex AI Search integrates directly as an ADK grounding tool:

```python
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_search import VertexAiSearchRetrieval

search_tool = VertexAiSearchRetrieval(
    name="search_company_docs",
    description="Search company policies and documentation.",
    data_store_id="projects/my-project/locations/global/collections/default_collection/dataStores/company-kb",
)

agent = Agent(
    model="gemini-2.0-flash",
    name="support_agent",
    instruction="Answer employee questions using the company knowledge base.",
    tools=[search_tool],
)
```

---

## 14. Vertex AI Search vs RAG Engine vs Vector Search

| Aspect | Vertex AI Search | RAG Engine | Vertex AI Vector Search |
|--------|-----------------|------------|------------------------|
| Setup | Low — UI or simple API | Medium — create corpus, import files | High — custom embedding + index |
| Search type | Hybrid (semantic + keyword) + AI answers | Semantic vector + optional hybrid | Pure vector (ANN) |
| AI answers | Built-in SummarySpec | Via Gemini tool | Not built-in |
| Grounding | Native Gemini integration | Via Tool.from_retrieval() | Manual integration |
| Data sources | GCS, BQ, Drive, websites, connectors | GCS, Drive, local | Custom (any vector source) |
| Best for | Enterprise search apps, document Q&A | Controlled RAG pipelines | Custom similarity search at scale |
| LLM coupling | Tight — grounding is a first-class feature | Via tool in generate_content() | Manual RAG pipeline |

---

## 15. Key API Classes — Quick Reference

| Class / Method | Description |
|---------------|-------------|
| `DataStoreServiceClient.create_data_store()` | Create a new data store |
| `DocumentServiceClient.import_documents()` | Ingest documents from GCS, BQ, etc. |
| `EngineServiceClient.create_engine()` | Create a search or recommendations app |
| `SearchServiceClient.search()` | Issue a search query |
| `ConversationalSearchServiceClient.answer_query()` | Multi-turn conversational search |
| `SnippetSpec` | Return short text extracts |
| `SummarySpec` | Return AI-generated grounded answer |
| `ExtractiveContentSpec` | Return verbatim passages |
| `BoostSpec` | Promote or bury specific results |
| `Tool.from_retrieval(grounding.VertexAISearch(...))` | Ground Gemini with private data store |
| `Tool.from_google_search_retrieval()` | Ground Gemini with public Google Search |

---

## 16. Exam-Relevant Tips

- A **data store** holds indexed documents; an **app** is the serving layer — they are separate resources.
- **Blended search** = one search app connected to multiple data stores.
- `SEARCH_ADD_ON_LLM` is required to use `SummarySpec` for AI-generated answers.
- `SummarySpec` → AI-generated answer; `SnippetSpec` → short text extracts; `ExtractiveContentSpec` → verbatim passages.
- Grounding Gemini with a Vertex AI Search data store uses `Tool.from_retrieval(grounding.VertexAISearch(...))`.
- Grounding with public web uses `Tool.from_google_search_retrieval()`.
- The **Answer API** (`answer_query`) supports multi-turn conversational search with session continuity.
- `grounding_metadata` in the response includes `grounding_chunks` (sources), `grounding_supports` (segment-level citations), and `web_search_queries`.
- **Boost values** range from **-1 (bury) to +1 (promote)** — not from 0 to 1.
- `ReconciliationMode.INCREMENTAL` adds new docs; `ReconciliationMode.FULL` replaces all docs.
- Third-party connectors (Confluence, Salesforce, Slack, etc.) are supported without custom ETL.
- Healthcare FHIR data is a supported first-class data type with its own schema.
- Vertex AI Search is the **simplest path** to grounded enterprise search — use RAG Engine when you need more control over chunking and retrieval pipelines.

---

## 17. Quick Reference Cheat Sheet

```
DATA STORE
  create_data_store()         →  create the indexed knowledge store
  import_documents()          →  ingest from GCS, BigQuery, Drive, connectors
  ReconciliationMode.INCREMENTAL → add new docs
  ReconciliationMode.FULL     →  replace all docs

SEARCH APP
  create_engine()             →  create search or recommendations app
  SEARCH_TIER_ENTERPRISE      →  advanced features (synonyms, custom ranking)
  SEARCH_ADD_ON_LLM           →  enables AI summary answers

SEARCH QUERY
  search()                    →  issue a query, returns ranked results
  SummarySpec                 →  AI-generated answer (needs ADD_ON_LLM)
  SnippetSpec                 →  short text extracts
  ExtractiveContentSpec       →  verbatim passages
  BoostSpec                   →  promote (+1) or bury (-1) results
  filter=                     →  metadata filtering expression

CONVERSATIONAL
  answer_query()              →  multi-turn search with session continuity
  session=                    →  pass session name for follow-up context

GROUNDING GEMINI
  Tool.from_retrieval(        →  private data grounding (Vertex AI Search)
    grounding.VertexAISearch(datastore=...))
  Tool.from_google_search_retrieval() → public web grounding

GROUNDING RESPONSE
  grounding_metadata.grounding_chunks    →  cited source documents
  grounding_metadata.grounding_supports  →  segment-level citations
  grounding_metadata.web_search_queries  →  Google Search queries issued
```

---

*Study tip: Know the relationship between data stores and apps (many-to-many for custom search), the three response content types (snippets, extractive answers, AI summary), when to use Vertex AI Search vs RAG Engine, and how to ground Gemini with private data vs public Google Search — all common exam scenarios.*

# Vertex AI Search & Grounding — Study Guide

> **Exam tip:** Know the relationship between data stores and apps (many-to-many for
> custom search), the three response content types (snippets, extractive answers, AI
> summary), when to use Vertex AI Search vs RAG Engine, and how to ground Gemini with
> private data vs public Google Search — all common exam scenarios.

---

## 1. Data stores and apps — many-to-many relationship

A **data store** holds your indexed content (websites, structured data, unstructured
documents). An **app** defines the search or recommendation experience served to users.

The relationship is **many-to-many**:

- One app can connect to multiple data stores (e.g. a single search app querying both
  a product catalogue and a support docs store simultaneously).
- One data store can be attached to multiple apps (e.g. the same document corpus
  powering both an internal and an external-facing search experience).

| Entity | What it holds | Can connect to |
|---|---|---|
| Data store | Indexed content (docs, website, BigQuery, etc.) | Many apps |
| App | Search/recommendation config & UI | Many data stores |

**Exam trap:** Don't assume one-to-one. A question describing a single corpus serving
multiple surfaces is testing whether you know data stores are reusable.

---

## 2. The three response content types

When Vertex AI Search returns results, it can surface content in three ways:

### Snippets
- Short excerpts extracted from the source document, similar to Google Search results.
- Best for **quick scanning** — the user sees a preview and decides whether to click through.
- No synthesis; just raw fragments from matching passages.

### Extractive answers
- A specific passage pulled verbatim from the document that directly answers the query.
- More targeted than a snippet — the system identifies the *most answer-like* segment.
- Best when the document contains a clear, self-contained answer to the question.

### AI summary (grounded generation)
- A generative response synthesised from one or more retrieved documents.
- Includes citations linking back to source passages.
- Best for **conversational interfaces** or when the answer spans multiple documents.
- Uses grounding to reduce hallucination — the model is constrained to retrieved content.

| Type | Verbatim? | Synthesised? | Cites sources? | Best for |
|---|---|---|---|---|
| Snippet | Partial | No | No | Result previews |
| Extractive answer | Yes | No | Implicit | Direct Q&A |
| AI summary | No | Yes | Yes | Conversational / multi-doc |

---

## 3. Vertex AI Search vs RAG Engine

Both retrieve information and ground generative responses, but they serve different needs.

| | Vertex AI Search | RAG Engine |
|---|---|---|
| **Primary use case** | Enterprise search & discovery | Custom RAG pipelines |
| **Who configures it** | Low-code / console-driven | Developers building custom apps |
| **Retrieval control** | Managed (you configure, Google handles retrieval) | Full control over chunking, embedding, retrieval logic |
| **Data sources** | Websites, GCS, BigQuery, Firestore, FHIR | Any vector store (Vertex AI Vector Search, Spanner, AlloyDB, etc.) |
| **Response types** | Snippets, extractive answers, AI summary | Raw retrieved chunks + generation (you assemble the prompt) |
| **Grounding** | Built-in, with citation support | You control how context is injected into the prompt |
| **Best for** | Search-as-a-service, internal knowledge bases, retail/media search | Fine-grained RAG with custom preprocessing, hybrid retrieval, or non-standard data |

**Decision rule for the exam:**
- Need a managed, out-of-the-box search experience with a UI? → **Vertex AI Search**
- Need programmatic control over the full retrieval-augmented generation pipeline? → **RAG Engine**

---

## 4. Grounding Gemini — private data vs public Google Search

Grounding connects Gemini's responses to a source of truth, reducing hallucination.
There are two distinct grounding mechanisms:

### Grounding with private data (Vertex AI Search)
- Connects Gemini to **your own documents, databases, or website content**.
- The model's response is constrained to information in your data store.
- Returns citations pointing back to your source documents.
- Use when the answer must come from **proprietary or domain-specific content**
  (internal policies, product manuals, medical records, etc.).

### Grounding with Google Search
- Connects Gemini to **live, public web content** via Google Search.
- Keeps responses current beyond the model's training cut-off date.
- Use when the query requires **up-to-date public information**
  (news, current events, general knowledge).
- Not appropriate for confidential or proprietary data.

| | Private data grounding | Google Search grounding |
|---|---|---|
| **Data source** | Your Vertex AI Search data store | Live public web |
| **Freshness** | As current as your last index update | Real-time |
| **Privacy** | Your data stays private | Public content only |
| **Citations** | Links to your documents | Links to web pages |
| **Use case** | Internal knowledge, proprietary content | Current events, general knowledge |

**Exam trap:** A question may describe a scenario where a company wants Gemini to answer
questions about *its own products* — that is private data grounding, not Google Search
grounding, even if the phrasing sounds like "searching the internet."

---

## Quick-reference cheat sheet

```
Data stores ──── many-to-many ──── Apps

Response types:
  Snippet          → fragment preview, no synthesis
  Extractive answer → verbatim passage, direct Q&A
  AI summary       → generated, cited, multi-doc

Search vs RAG:
  Vertex AI Search → managed, low-code, search-as-a-service
  RAG Engine       → developer-controlled, custom pipelines

Grounding:
  Private data  → Vertex AI Search data store, proprietary content
  Google Search → live web, current events, public knowledge
```