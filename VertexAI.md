# 🤖 Google Cloud — Vertex AI Products & APIs
## Complete TL;DR Reference Guide

> **How to use this guide:** Organized by category. Each product has a one-liner TL;DR, a slightly deeper "What it does", and an exam-relevant tip where applicable.

---

## 📌 Category Overview

| Category | Products Covered |
|---|---|
| 🏗️ Core ML Platform | Workbench, Pipelines, Training, Prediction, Experiments, Model Registry |
| 🤖 AutoML | AutoML Tabular, Image, Text, Video |
| 🧠 Generative AI | Gemini API, Imagen, Embeddings, Model Garden, Codey |
| 🔍 Search & Retrieval | Vertex AI Search, Vertex AI RAG Engine, Matching Engine |
| 📦 Data & Features | Feature Store, Dataset service, Data Labeling |
| 🛡️ Responsible AI | Explainable AI, Model Monitoring, Model Evaluation |
| 🏪 Model Management | Model Registry, Model Garden, Endpoints |
| ⚙️ MLOps | Vertex AI Pipelines, Metadata, TensorBoard, Comet |
| 🌐 Agents | Vertex AI Agent Builder, Agent Engine |

---

## 🏗️ CORE ML PLATFORM

---

### 🖥️ Vertex AI Workbench
> **TL;DR:** Managed Jupyter notebooks in the cloud, pre-loaded with GCP integrations.

**What it does:** A hosted notebook environment (JupyterLab) where data scientists can explore data, build models, and connect directly to BigQuery, GCS, and other GCP services — without managing infrastructure.

- Two modes: **Instances** (fully managed) and **User-managed Notebooks** (more control)
- Pre-installed with common ML libraries (TensorFlow, PyTorch, scikit-learn, XGBoost)
- 📝 **Exam tip:** This is the go-to for *interactive* ML development on GCP.

---

### 🎓 Vertex AI Training
> **TL;DR:** Run custom ML training jobs at scale on managed infrastructure.

**What it does:** Submit your training scripts (TensorFlow, PyTorch, scikit-learn, XGBoost, or any framework) and GCP handles provisioning machines, GPUs/TPUs, and distributed training.

- Supports **custom containers** for any framework
- Supports **hyperparameter tuning** (Vizier-powered)
- Can run as a one-off job or as part of a pipeline
- 📝 **Exam tip:** Use this when AutoML isn't flexible enough and you need a custom training loop.

---

### 🚀 Vertex AI Prediction (Online & Batch)
> **TL;DR:** Deploy trained models to get predictions — either in real-time or in bulk.

**What it does:** Hosts your trained model behind a REST endpoint.

- **Online Prediction:** Low-latency, real-time predictions via HTTP (e.g., scoring a single transaction)
- **Batch Prediction:** Process millions of records asynchronously from GCS or BigQuery
- Supports auto-scaling and multiple model versions per endpoint
- 📝 **Exam tip:** Online = latency-sensitive; Batch = high-volume, cost-efficient.

---

### 🧪 Vertex AI Experiments
> **TL;DR:** Track, compare, and manage ML training runs and their metrics.

**What it does:** Logs parameters, metrics, and artifacts from training runs so you can compare experiments side-by-side and reproduce results.

- Works with Vertex AI Training, custom code, or even local runs
- Built on top of **Vertex ML Metadata**
- Similar to: MLflow Tracking, Weights & Biases

---

### 📋 Vertex AI Model Registry
> **TL;DR:** A central catalog to version, organize, and manage all your trained models.

**What it does:** Store trained model artifacts with versioning, metadata, and lineage tracking. Deploy directly from the registry to endpoints.

- Tracks which dataset/pipeline produced each model version
- Supports promotion workflows (dev → staging → prod)
- 📝 **Exam tip:** Think of it as Git, but for ML models.

---

### 🔗 Vertex AI Pipelines
> **TL;DR:** Orchestrate end-to-end ML workflows as reproducible, automated pipelines.

**What it does:** Define multi-step ML workflows (data prep → training → evaluation → deployment) as DAGs (Directed Acyclic Graphs) that run on managed infrastructure.

- Compatible with **Kubeflow Pipelines (KFP)** and **TFX (TensorFlow Extended)** SDKs
- Each step runs in its own container
- 📝 **Exam tip:** Vertex AI Pipelines = the orchestration layer. Use it to automate and reproduce your entire ML workflow.

---

### 🗂️ Vertex ML Metadata
> **TL;DR:** Automatically track lineage of datasets, models, and metrics across all your ML runs.

**What it does:** Records the "who, what, when, and how" of every artifact in your ML system — enabling full lineage tracing from raw data to deployed model.

- Answers: *"Which dataset was used to train model v3?"* or *"What hyperparameters produced the best AUC?"*
- Automatically populated when using Vertex AI Pipelines

---

### 📊 Vertex AI TensorBoard
> **TL;DR:** Hosted TensorBoard for visualizing training metrics and model graphs.

**What it does:** The managed version of TensorBoard — visualize loss curves, accuracy, histograms, embeddings, and computational graphs without running your own server.

- Integrates natively with Vertex AI Training jobs
- Persistent storage of experiment visualizations

---

## 🤖 AUTOML

---

### 📊 Vertex AI AutoML Tabular
> **TL;DR:** Train high-quality ML models on structured/tabular data with zero code.

**What it does:** Upload a CSV or point to a BigQuery table, choose your target column, and AutoML handles feature engineering, model selection, and hyperparameter tuning automatically.

- Supports: **Regression**, **Classification**, **Forecasting**
- Uses an ensemble of models internally (neural architecture search + boosted trees)
- 📝 **Exam tip:** Best choice when you have tabular data and don't want to write model code.

---

### 🖼️ Vertex AI AutoML Image
> **TL;DR:** Train image classification or object detection models with no ML expertise required.

**What it does:** Provide labeled images → AutoML trains a custom vision model for you.

- Tasks: **Image Classification** (single/multi-label), **Object Detection**, **Image Segmentation**
- Handles data augmentation and transfer learning internally
- 📝 **Exam tip:** Needs labeled images — pair with **Data Labeling Service** if labels are missing.

---

### 📝 Vertex AI AutoML Text
> **TL;DR:** Build NLP models for text classification, extraction, or sentiment — no code needed.

**What it does:** Train custom NLP models on your own text data.

- Tasks: **Text Classification**, **Entity Extraction** (NER), **Sentiment Analysis**
- Works well with domain-specific vocabulary where general models underperform

---

### 🎬 Vertex AI AutoML Video
> **TL;DR:** Train models to classify, track objects, or recognize actions in video content.

**What it does:** Upload labeled video data and AutoML builds video understanding models.

- Tasks: **Video Classification**, **Object Tracking**, **Action Recognition**
- Use case: surveillance, sports analytics, media content tagging

---

## 🧠 GENERATIVE AI

---

### 💎 Vertex AI Gemini API
> **TL;DR:** Access Google's Gemini family of multimodal LLMs (text, image, audio, video, code).

**What it does:** The primary API for interacting with Gemini models (Pro, Flash, Ultra variants) for generation, summarization, Q&A, reasoning, and multimodal tasks.

- Supports **streaming**, **function calling**, **system instructions**, and **grounding**
- Models: Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini Ultra
- 📝 **Exam tip:** Default choice for generative AI tasks on GCP.

---

### 🖌️ Vertex AI Imagen
> **TL;DR:** Generate, edit, and upscale images using Google's text-to-image models.

**What it does:** Generate photorealistic images from text prompts, edit existing images (inpainting/outpainting), or upscale low-resolution images.

- Use cases: marketing assets, product mockups, creative content
- Includes safety filters and watermarking (SynthID)

---

### 🔢 Vertex AI Embeddings API
> **TL;DR:** Convert text (or images) into dense numerical vectors that capture semantic meaning.

**What it does:** Generate embedding vectors for text or multimodal content. These vectors enable semantic search, clustering, classification, and RAG retrieval.

- Text model: `text-embedding-004` and `text-multilingual-embedding`
- Multimodal model: `multimodalembedding`
- 📝 **Exam tip:** Embeddings are the foundation of semantic search and RAG pipelines.

---

### 🛠️ Codey (Code Generation APIs)
> **TL;DR:** LLMs specialized for generating, completing, and chatting about code.

**What it does:** Provides code-specific generative models:

- `code-bison` → generate code from a description
- `code-gecko` → inline code completion (IDE-style)
- `codechat-bison` → multi-turn chat about code
- Use cases: developer tools, code review assistants, documentation generation

---

### 🌿 Vertex AI Model Garden
> **TL;DR:** A catalog of 150+ curated, ready-to-deploy foundation models (Google + open source).

**What it does:** Browse, deploy, and fine-tune foundation models directly in Vertex AI — including Google's own models (Gemini, Imagen, Codey), Hugging Face models, and open-source models (Llama 3, Mistral, Falcon, etc.).

- One-click deployment to Vertex AI endpoints
- Some models support **fine-tuning** directly from the Garden
- 📝 **Exam tip:** Model Garden = the "app store" for foundation models on GCP.

---

### ✏️ Vertex AI Model Tuning (Fine-Tuning)
> **TL;DR:** Customize a foundation model on your own data to improve task-specific performance.

**What it does:** Fine-tune Gemini and PaLM models using your own labeled examples.

- **Supervised Fine-Tuning (SFT):** Train on prompt-response pairs
- **RLHF:** Reinforcement Learning from Human Feedback (for alignment)
- **Adapter tuning / LoRA:** Efficient fine-tuning with fewer compute resources
- 📝 **Exam tip:** Use fine-tuning when prompt engineering alone isn't enough.

---

## 🔍 SEARCH & RETRIEVAL

---

### 🔎 Vertex AI Search (formerly Enterprise Search / Gen App Builder)
> **TL;DR:** Add Google-quality search to your apps — over your own documents, websites, or data.

**What it does:** Build a semantic search engine over your enterprise content (PDFs, websites, BigQuery, GCS) without building the indexing/retrieval infrastructure yourself.

- Supports **keyword + semantic + hybrid search**
- Can be grounded with Gemini for **RAG-powered Q&A**
- Use cases: internal knowledge bases, e-commerce search, document search portals
- 📝 **Exam tip:** Fastest path to production-ready enterprise search on GCP.

---

### 🧩 Vertex AI RAG Engine
> **TL;DR:** Fully managed Retrieval-Augmented Generation pipeline — connect LLMs to your own data.

**What it does:** Manages the full RAG pipeline: ingest documents → chunk → embed → store in vector DB → retrieve relevant chunks → pass to Gemini for generation.

- Handles: **document ingestion, chunking, embedding, vector indexing, retrieval, and generation**
- Supports Google Drive, GCS, and inline uploads as data sources
- Removes the need to manually wire together embedding models + vector DBs + LLMs
- 📝 **Exam tip:** RAG Engine = managed alternative to building your own LangChain/LlamaIndex pipeline.

---

### ⚡ Vertex AI Matching Engine (Vector Search)
> **TL;DR:** Ultra-fast, scalable approximate nearest neighbor (ANN) search for embedding vectors.

**What it does:** Store billions of embedding vectors and find the most semantically similar ones in milliseconds. The retrieval backbone behind semantic search and RAG.

- Uses **ScaNN** (Google's ANN algorithm) under the hood
- Supports real-time updates (streaming upserts)
- 📝 **Exam tip:** Matching Engine = the vector database layer. RAG Engine uses it internally. Use it directly when you need custom vector search control.

---

## 📦 DATA & FEATURES

---

### 🗄️ Vertex AI Feature Store
> **TL;DR:** Centralized repository to store, share, and serve ML features consistently across training and serving.

**What it does:** Solves the **training-serving skew** problem by ensuring the exact same feature values used during training are also served at prediction time.

- **Online Store:** Low-latency feature lookup for real-time inference
- **Offline Store:** Bulk feature retrieval for training jobs (backed by BigQuery)
- Features can be shared across teams and models
- 📝 **Exam tip:** If an exam question mentions "training-serving skew" or "reusing features across models" → Feature Store.

---

### 🏷️ Vertex AI Data Labeling Service
> **TL;DR:** Get human-labeled training data for your ML datasets (images, text, video, audio).

**What it does:** Send your unlabeled data to human labelers (Google-managed workforce or your own specialists) to generate ground truth labels for supervised learning.

- Supports: image bounding boxes, classification labels, text annotations, video labels
- Can be combined with **AutoML** or **custom training**

---

### 📁 Vertex AI Datasets
> **TL;DR:** Managed dataset resource that registers your data for use in AutoML or training jobs.

**What it does:** A logical container that connects your raw data (in GCS or BigQuery) to Vertex AI training. Tracks data versions and manages train/validation/test splits.

---

## 🛡️ RESPONSIBLE AI & MONITORING

---

### 💡 Vertex Explainable AI
> **TL;DR:** Understand *why* your model made a prediction — feature attributions for any model.

**What it does:** After a prediction, returns an explanation showing which input features contributed most to that output.

- Methods: **SHAP**, **Integrated Gradients**, **XRAI** (for images), **Sampled Shapley**
- Works with AutoML and custom-trained models
- 📝 **Exam tip:** Required for regulated industries (finance, healthcare) where predictions must be explainable.

---

### 👁️ Vertex AI Model Monitoring
> **TL;DR:** Automatically detect when your deployed model's performance degrades over time.

**What it does:** Continuously monitors live prediction traffic for:

- **Training-serving skew:** Input distributions at serving differ from training data
- **Prediction drift:** Model output distributions are shifting over time
- Sends alerts when drift exceeds configurable thresholds
- 📝 **Exam tip:** Essential for production MLOps — models decay as the real world changes.

---

### ✅ Vertex AI Model Evaluation
> **TL;DR:** Compute standard ML metrics for your trained models on evaluation datasets.

**What it does:** Run offline evaluation jobs against a labeled test set and compute metrics like AUC, F1, RMSE, BLEU (for text), etc. Results are stored in Vertex ML Metadata.

---

### 🧯 Vertex AI Managed Datasets + Data Validation
> **TL;DR:** Detect anomalies and schema violations in your training/serving data.

**What it does:** Validate that incoming data matches the expected schema and statistical profile. Catches issues like missing columns, unexpected value distributions, or type mismatches before they break your model.

---

## 🌐 AGENTS & CONVERSATIONAL AI

---

### 🤝 Vertex AI Agent Builder
> **TL;DR:** Build, deploy, and manage AI agents (chatbots, search apps, RAG apps) with a no-code/low-code interface.

**What it does:** A unified platform to create:

- **Conversational agents** (powered by Gemini + Dialogflow CX)
- **Search apps** (powered by Vertex AI Search)
- **Custom RAG apps** (powered by Vertex AI RAG Engine)
- 📝 **Exam tip:** The front-end builder that orchestrates Vertex Search, RAG Engine, and Gemini together.

---

### 🔧 Vertex AI Agent Engine (formerly Reasoning Engine)
> **TL;DR:** Deploy and manage custom LLM-powered agents (LangChain, custom) on fully managed infrastructure.

**What it does:** Run your own agentic applications — built with LangChain, LangGraph, or custom agent frameworks — without managing servers. Handles scaling, sessions, and logging.

- Use case: custom multi-step agents that use tools, APIs, and memory
- 📝 **Exam tip:** Agent Engine = managed runtime for *custom* agents. Agent Builder = managed UI for *pre-built* agent patterns.

---

### 💬 Vertex AI Conversation (Dialogflow CX)
> **TL;DR:** Build enterprise-grade conversational AI (chatbots/IVR) with visual flow design.

**What it does:** Design and deploy multi-turn conversational agents for customer service, virtual assistants, and IVR systems — using a visual state machine flow builder.

- Supports voice and text channels
- Now enhanced with Gemini for generative fallback and summarization
- Different from: Dialogflow ES (simpler, legacy version)

---

## ⚙️ SPECIALIZED APIS

---

### 🌡️ Vertex AI Forecast
> **TL;DR:** AutoML-powered time-series forecasting purpose-built for business metrics.

**What it does:** Forecast multiple related time series simultaneously (e.g., sales for 10,000 SKUs at once) with automatic handling of trends, seasonality, and holidays.

- Handles **hierarchical forecasting** (store → region → country)
- Uses **temporal fusion transformers** under the hood
- 📝 **Exam tip:** Use Vertex Forecast for large-scale multi-series forecasting. Use BigQuery ML ARIMA_PLUS for simpler single-series cases.

---

### 🧬 Vertex AI for Natural Language (NL API)
> **TL;DR:** Pre-trained NLP API for entity recognition, sentiment, classification, and syntax — no training needed.

**What it does:** Send any text and get back structured NLP results instantly using Google's pre-trained models.

- Entity extraction, sentiment score, content classification, syntax analysis
- 📝 **Exam tip:** Use when you need NLP on general text and *don't* need a custom model. For domain-specific text → AutoML Text.

---

### 👁️ Vertex AI Vision (Cloud Vision API)
> **TL;DR:** Pre-trained image analysis — labels, faces, objects, OCR, safe search — no training needed.

**What it does:** Send an image, get back rich structured metadata: detected objects, labels, faces, landmarks, text (OCR), logos, and safe-search classifications.

- Use when you need general image understanding without a custom model
- For *custom* image models → use AutoML Image

---

### 🔊 Vertex AI Speech APIs
> **TL;DR:** Convert speech to text (STT) and text to speech (TTS) with Google's audio models.

**What it does:**

- **Speech-to-Text:** Transcribe audio in 125+ languages with high accuracy, supports streaming
- **Text-to-Speech:** Generate natural-sounding voice audio from text, with custom voice options (WaveNet, Neural2, Studio)

---

### 🔄 Vertex AI Translation API
> **TL;DR:** Translate text between 100+ languages using Google's Neural Machine Translation.

**What it does:** Instant high-quality translation, with options for AutoML custom translation models if you need domain-specific terminology (legal, medical, etc.).

---

## 🏪 GROUNDING & SAFETY

---

### 🌐 Grounding with Google Search
> **TL;DR:** Give your Gemini model access to real-time Google Search results to reduce hallucinations.

**What it does:** When generating a response, the model can retrieve and cite up-to-date web search results — anchoring answers in verifiable, current information.

- 📝 **Exam tip:** Use grounding when freshness matters or when hallucination risk is high.

---

### 🛡️ Vertex AI Safety Filters & Guardrails
> **TL;DR:** Built-in content moderation to block harmful, unsafe, or off-topic LLM outputs.

**What it does:** Configurable safety thresholds across categories: hate speech, harassment, sexually explicit content, dangerous content. Can be tuned per use case.

---

## 📊 QUICK EXAM CHEAT SHEET

| Scenario | Product to Use |
|---|---|
| Interactive notebook development | Vertex AI Workbench |
| Train custom model (any framework) | Vertex AI Training |
| AutoML on a CSV / BigQuery table | AutoML Tabular |
| Classify images with no code | AutoML Image |
| NLP on your own labeled text | AutoML Text |
| Call Gemini in your app | Vertex AI Gemini API |
| Generate images from text | Imagen |
| Convert text → vectors | Embeddings API |
| Browse & deploy open-source models | Model Garden |
| Customize Gemini on your data | Model Tuning (Fine-Tuning) |
| Semantic search over your documents | Vertex AI Search |
| Build a Q&A bot over your own docs | Vertex AI RAG Engine |
| Store/serve features consistently | Feature Store |
| Explain a prediction | Explainable AI |
| Detect model drift in production | Model Monitoring |
| Orchestrate ML steps as a workflow | Vertex AI Pipelines |
| Track experiments and metrics | Vertex AI Experiments |
| Version and catalog models | Model Registry |
| Deploy & manage custom agents | Agent Engine |
| Build chatbots / voice agents | Agent Builder + Dialogflow CX |
| Multi-series business forecasting | Vertex AI Forecast |
| General image analysis (no training) | Cloud Vision API |
| General NLP (no training) | Natural Language API |
| Real-time grounded answers | Grounding with Google Search |
| Vector similarity search at scale | Matching Engine (Vector Search) |

---

*📌 Remember for the exam: Vertex AI is the unified ML platform — it covers everything from data prep → training → evaluation → deployment → monitoring. Know when to use managed/AutoML vs. custom, and when to use pre-trained APIs vs. fine-tuned models.*