# GCP Professional ML Engineer — Study Guide Index
> All topics covered in this series. Use this as your revision checklist and navigation map.

---

## BigQuery ML (BQML)

| # | Topic | Key Concepts |
|---|-------|-------------|
| 01 | **Linear Regression** | `linear_reg`, ML.EVALUATE, ML.WEIGHTS, ML.PREDICT, R², MAE, MSE, L1/L2 regularisation, data split methods |
| 02 | **Regression with Boosted Trees** | `boosted_tree_regressor`, XGBoost, n_estimators, max_tree_depth, learn_rate, subsample, early_stop, ML.FEATURE_IMPORTANCE |
| 03 | **Binary Classification** | `logistic_reg`, `boosted_tree_classifier`, precision, recall, F1, AUC-ROC, confusion matrix, ML.ROC_CURVE, threshold tuning, class imbalance |
| 04 | **Multiclass Classification (Boosted Trees)** | `boosted_tree_classifier` auto-detects 3+ labels, softmax, weighted avg metrics, NxN confusion matrix, **no roc_auc for multiclass** |
| 05 | **Time-Series Forecasting** | `ARIMA_PLUS`, `ARIMA_PLUS_XREG`, ML.FORECAST (not ML.PREDICT), ML.EXPLAIN_FORECAST, horizon, data_frequency, holiday_region, MAPE, AIC, multiple series via time_series_id_col |
| 06 | **Matrix Factorization** | `matrix_factorization`, ML.RECOMMEND, EXPLICIT vs IMPLICIT feedback, wals_alpha, num_factors, ALS/WALS algorithm, cold start, ML.WEIGHTS for embeddings |
| 07 | **Anomaly Detection (Autoencoders)** | `AUTOENCODER`, hidden_units symmetric, ML.DETECT_ANOMALIES, contamination parameter, train on normal data only, no roc_auc, unsupervised |
| 08 | **Feature Engineering** | TRANSFORM clause (auto-applied at predict), ML.STANDARD_SCALER, ML.ROBUST_SCALER (outliers), ML.BUCKETIZE vs ML.QUANTILE_BUCKETIZE, ML.ONE_HOT_ENCODER (nominal) vs ML.LABEL_ENCODER (ordinal), ML.FEATURE_CROSS, ML.IMPUTER |

---

## Vertex AI AutoML & Managed Services

| # | Topic | Key Concepts |
|---|-------|-------------|
| 09 | **AutoML with Vertex AI** | Dataset → Training Job → Model → Endpoint → Prediction, `ml_use` column for splits, budget_milli_node_hours (1000=1 hr), SummarySpec needs ADD_ON_LLM, batch prediction needs no endpoint, generate_explanation=True for Shapley |
| 10 | **Vertex AI RAG Engine** | Corpus (managed index), ARIMA chunking 512/100, text-embedding-005, rag.retrieval_query() (chunks only) vs Tool.from_retrieval() (E2E), hybrid search alpha, reranking, multi-corpus isolation |
| 11 | **Vertex AI Search** | Data Store + App (separate resources), blended search (1 app → many stores), SnippetSpec / ExtractiveContentSpec / SummarySpec, answer_query() for conversational, SEARCH_ADD_ON_LLM required for AI summaries, Tool.from_retrieval(VertexAISearch) for private grounding |
| 12 | **Vertex AI Vector Search** | Formerly Matching Engine, Tree-AH ANN index (ScaNN), brute force = eval only (never prod), STREAM vs BATCH update, DOT_PRODUCT distance, public vs private VPC endpoint, Recall = ANN_correct/num_neighbors, HybridQuery for dense+sparse |
| 13 | **Vertex AI Feature Store** | V2: BigQuery = offline store + source of truth, FeatureOnlineStore (Bigtable) → FeatureView, scheduled vs continuous sync, continuous = new inserts only (no updates/deletes), feature_timestamp for point-in-time correct training, Feature Registry backed by Dataplex |
| 14 | **Sampled Shapley** | how much each input feature (like age, income, or location) contributed to a specific prediction |

---

## Agent Development

| # | Topic | Key Concepts |
|---|-------|-------------|
| 14 | **Agent Development Kit (ADK)** | LlmAgent (dynamic), SequentialAgent (ordered), ParallelAgent (concurrent), LoopAgent (iterative), FunctionTool (docstring = auto schema), AgentTool (sub-agent as tool), MCPToolset, session state vs Memory Bank (cross-session), Callbacks (non-None return = short-circuit), AdkApp → agent_engines.create() |
| 15 | **Gemini File API** | 3 methods: Files API (2 GB, 48-hr TTL), inline data (~100 MB), GCS URI / HTTPS URL (persistent), files cannot be downloaded, context caching: 90% discount (Gemini 2.5+), 75% (2.0), min 2,048 tokens, audio=32 tok/sec, video=~5 tok/frame |
| 16 | **Gemini AI Studio** | Browser-based, free, API key auth (not ADC), temperature=0 is greedy, Gemini 3 default temp=1.0 (don't lower), top_p/top_k narrow candidate tokens, structured output = response_mime_type + response_schema, BLOCK_LOW_AND_ABOVE = strictest safety, use google-genai SDK (not google-generativeai — deprecated), thinking_level (Gemini 3) replaces thinking_budget (2.5) |

---

## Key Cross-Topic Exam Traps

| Trap | Correct Answer |
|------|---------------|
| roc_auc for multiclass | ❌ Not returned — multiclass only |
| ML.PREDICT for time-series | ❌ Use ML.FORECAST |
| ML.PREDICT for recommendations | ❌ Use ML.RECOMMEND |
| ML.PREDICT for anomalies | ❌ Use ML.DETECT_ANOMALIES (ML.PREDICT returns reconstructed values) |
| Brute force index in production | ❌ Evaluation only — use Tree-AH ANN |
| Optimized online serving (Feature Store) | ❌ Deprecated Feb 2027 — use Bigtable |
| Legacy Feature Store V1 (EntityType) | ❌ Deprecated Feb 2027 — use V2 |
| `google-generativeai` SDK | ❌ Deprecated — use `google-genai` |
| Continuous sync picks up updates/deletes | ❌ New inserts only |
| SummarySpec without SEARCH_ADD_ON_LLM | ❌ Requires ADD_ON_LLM tier |
| Files API files downloadable | ❌ Write-once, read via URI only |
| thinking_level + thinking_budget together | ❌ Use one, not both |

---

## Quick Metric Cheat Sheet

| Task | Primary Metric | When Imbalanced |
|------|---------------|-----------------|
| Regression | R², RMSE, MAE | — |
| Binary classification | AUC-ROC, F1 | AUC or F1 (not accuracy) |
| Multiclass classification | Accuracy, weighted F1 | Per-class F1 from confusion matrix |
| Ranking / recommendations | Precision@K, NDCG, MAP | — |
| Time-series forecast | MAPE (scale-independent), RMSE, AIC | — |
| Anomaly detection | MAE (reconstruction error), contamination | — |

---

## Key "Always/Never" Rules

- **Always** check `file.state == ACTIVE` before using an uploaded Gemini file
- **Always** specify `MIME type` in multimodal Gemini requests
- **Always** include `OVER()` clause on BQML TRANSFORM functions
- **Always** pass label column through TRANSFORM unchanged
- **Never** use brute force index in production (Vector Search)
- **Never** train an autoencoder on anomalous data
- **Never** use `roc_auc` metric for multiclass problems
- **Never** use `ML.PREDICT` for time-series — use `ML.FORECAST`
- **Never** mix `thinking_level` and `thinking_budget` in the same request
- **Never** use the deprecated `google-generativeai` SDK — use `google-genai`

---

*Good luck with the GCP Professional Machine Learning Engineer exam! 🎯*