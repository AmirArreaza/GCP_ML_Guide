# Machine Learning & AI — Concepts & Acronyms Dictionary
> **GCP Professional ML Engineer Reference Glossary**
> Sorted alphabetically. Use Ctrl+F / Cmd+F to find terms quickly.

---

## A

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **A/B Testing** | Split Testing | Serving two model versions simultaneously to a fraction of traffic to compare performance before full rollout. |
| **Accuracy** | Classification Accuracy | Fraction of all predictions that are correct: (TP + TN) / Total. Misleading on imbalanced datasets. |
| **ADK** | Agent Development Kit | Google's open-source framework for building, evaluating, and deploying AI agents. Supports Python and Java. |
| **ADC** | Application Default Credentials | GCP authentication mechanism that automatically finds credentials from the environment. Used by Vertex AI SDK. |
| **Agent** | AI Agent | An autonomous system that perceives its environment, reasons, uses tools, and takes actions to achieve goals. |
| **AgentTool** | Agent-as-Tool | ADK construct that wraps a specialised sub-agent so a root agent can delegate tasks to it as a tool. |
| **AIC** | Akaike Information Criterion | Model selection metric that balances goodness-of-fit against complexity. Lower AIC = better ARIMA model. |
| **ALS** | Alternating Least Squares | Optimisation algorithm for matrix factorisation with explicit feedback (ratings). |
| **ANN** | Approximate Nearest Neighbour | Fast algorithm that finds vectors close to a query in high-dimensional space with slight accuracy trade-off. Used by Vector Search / ScaNN. |
| **ARIMA** | AutoRegressive Integrated Moving Average | Classical time-series forecasting model combining AR (past values), I (differencing), and MA (past errors). |
| **ARIMA_PLUS** | Enhanced ARIMA | Google's managed ARIMA implementation in BigQuery ML, with auto order selection, holiday effects, and anomaly handling. |
| **ARIMA_PLUS_XREG** | ARIMA with External Regressors | BQML time-series model that includes exogenous (external) variables alongside time-series history. |
| **Artifact** | ML Artifact | Any file or binary output produced during ML workflows — models, datasets, reports, embeddings, etc. |
| **Attention** | Attention Mechanism | Component in transformers that allows the model to weigh the importance of different input tokens when generating each output token. |
| **AUC** | Area Under the Curve | Area under the ROC curve. Measures overall classifier performance across all thresholds. 1.0 = perfect, 0.5 = random. |
| **AUC-PRC** | Area Under Precision-Recall Curve | Alternative to AUC-ROC for imbalanced datasets; focuses on precision and recall trade-offs. |
| **AutoML** | Automated Machine Learning | Automatically searches for the best model architecture, features, and hyperparameters for a given dataset and objective. |
| **Autoencoder** | Autoencoder | Neural network trained to compress (encode) then reconstruct (decode) input. High reconstruction error = anomaly signal. |

---

## B

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Backpropagation** | Backward Propagation of Errors | Algorithm for training neural networks by computing gradients of the loss function and updating weights. |
| **Batch Normalisation** | Batch Normalisation | Technique that normalises layer inputs during training to stabilise and accelerate learning. |
| **Batch Prediction** | Batch Inference | Asynchronous scoring of large datasets. Output written to GCS or BigQuery. No always-on endpoint needed. |
| **Bagging** | Bootstrap Aggregating | Ensemble method that trains multiple models on random data subsets and averages predictions. Used by Random Forest. |
| **Bias** | Model Bias | Systematic error from incorrect assumptions in the learning algorithm; causes underfitting. |
| **Bias-Variance Trade-off** | Bias-Variance Trade-off | Balance between a model that is too simple (high bias) and too complex (high variance/overfitting). |
| **BigQuery ML** | BigQuery Machine Learning | Google service to create and run ML models using SQL directly inside BigQuery. |
| **Blended Search** | Blended Search | Vertex AI Search feature where one app connects to multiple data stores and merges results. |
| **Boosting** | Gradient Boosting | Ensemble method that trains trees sequentially, each correcting the errors of the previous one. |
| **BQML** | BigQuery ML | Abbreviation for BigQuery Machine Learning. |
| **Brute Force Index** | Exact Nearest Neighbour Index | Vector Search index type that finds exact nearest neighbours. Slow; used only for ground truth evaluation, never production. |

---

## C

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Callback** | Lifecycle Callback | In ADK, a function that intercepts agent execution at defined points (before/after agent, tool, or model calls). |
| **Categorical Feature** | Categorical Variable | Feature with discrete, named values (e.g. colour, country, product category). |
| **Chain-of-Thought** | Chain-of-Thought Prompting | Technique that encourages the model to reason step-by-step before giving a final answer. Improves accuracy on complex tasks. |
| **Classification** | Classification Task | Predicting which category/class an observation belongs to. Binary (2 classes) or multiclass (3+ classes). |
| **CMEK** | Customer-Managed Encryption Key | Allows customers to control their own encryption keys via Cloud KMS in GCP services. |
| **Cold Start** | Cold Start Problem | In recommendation systems, the inability to make predictions for new users or items with no interaction history. |
| **Collaborative Filtering** | Collaborative Filtering | Recommendation technique that finds patterns from user-item interactions — no content features required. |
| **Confidence Interval** | Confidence Interval | In forecasting, a range around the point prediction that reflects forecast uncertainty. |
| **Contamination** | Contamination Parameter | In BQML autoencoders, the expected proportion of anomalies in the dataset. Controls the anomaly detection threshold. |
| **Context Caching** | Context Caching | Storing pre-processed tokens from large repeated inputs so they can be reused across requests at discounted cost. |
| **Context Window** | Context Window | The maximum number of tokens a model can process in a single request (input + output combined). |
| **Corpus** | RAG Corpus | In Vertex AI RAG Engine, the managed index containing chunked documents and their embeddings. |
| **Cosine Similarity** | Cosine Similarity | Measures the angle between two vectors. A value of 1 means identical direction; 0 means orthogonal. |
| **CoT** | Chain-of-Thought | Abbreviation for Chain-of-Thought Prompting. |
| **Cross-Entropy Loss** | Cross-Entropy Loss | Loss function for classification tasks. Penalises confident wrong predictions more heavily. Also called log loss. |
| **Cross-Validation** | Cross-Validation | Technique to evaluate model performance by splitting data into K folds and training/testing on each. |
| **Crowding Tag** | Crowding Tag | In Vector Search, a metadata attribute that diversifies results by limiting how many results share the same attribute. |

---

## D

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Data Augmentation** | Data Augmentation | Creating new training examples by transforming existing ones (flip, rotate, crop images) to improve generalisation. |
| **Data Drift** | Data Drift / Feature Drift | Change in the statistical distribution of input features over time. Can degrade model performance. |
| **Data Store** | Vertex AI Search Data Store | The indexed knowledge base in Vertex AI Search containing documents, structured data, or website content. |
| **DCG** | Discounted Cumulative Gain | Ranking metric that rewards relevant items higher in the ranking list more than lower positions. |
| **Dense Vector** | Dense Embedding | A vector where most dimensions have non-zero values. Used for semantic/neural retrieval. |
| **Differencing** | Time-Series Differencing | Subtracting consecutive time-series values to remove trends and achieve stationarity. The `d` parameter in ARIMA. |
| **Dimensionality Reduction** | Dimensionality Reduction | Reducing the number of features while retaining important information. PCA is a common method. |
| **DNN** | Deep Neural Network | Neural network with many hidden layers, capable of learning complex non-linear patterns. |
| **Dropout** | Dropout Regularisation | Regularisation technique that randomly disables neurons during training to prevent overfitting. |
| **Dot Product** | Dot Product | Mathematical operation: sum of element-wise products of two vectors. Used as similarity measure in vector search. |

---

## E

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Early Stopping** | Early Stopping | Halting training when the validation metric stops improving, to prevent overfitting. |
| **Elastic Net** | Elastic Net Regularisation | Combination of L1 (Lasso) and L2 (Ridge) regularisation. |
| **Embedding** | Feature Embedding | Dense numerical vector representing an entity (word, user, item, document) in a continuous vector space. |
| **Encoder** | Encoder (Autoencoder) | The compression part of an autoencoder that reduces input to a low-dimensional latent representation. |
| **Endpoint** | Serving Endpoint | In Vertex AI, a stable URL that hosts one or more deployed models and serves online predictions. |
| **Ensemble** | Ensemble Method | Combining multiple models to produce better predictions than any single model (e.g. Random Forest, Gradient Boosting). |
| **Epoch** | Training Epoch | One complete pass through the entire training dataset during model training. |
| **ETL** | Extract, Transform, Load | Pipeline process for moving and transforming data from source systems to a target data store. |
| **Exogenous Variable** | Exogenous Regressor | External variable used to help predict a time series (e.g. temperature to predict ice cream sales). Used in ARIMA_PLUS_XREG. |
| **Explainability** | Model Explainability | Ability to understand and explain why a model made a specific prediction. See: Shapley values, LIME, SHAP. |
| **Explicit Feedback** | Explicit Ratings | Direct user ratings or scores (stars, thumbs up) used in recommendation systems. |

---

## F

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **F1 Score** | F1 Score | Harmonic mean of precision and recall: 2 × (P × R) / (P + R). Best single metric when both matter. |
| **False Negative** | Type II Error | Model predicted negative (no) but actual was positive (yes). Costly in medical diagnosis, fraud detection. |
| **False Positive** | Type I Error | Model predicted positive (yes) but actual was negative (no). Costly in spam filtering. |
| **Feature** | ML Feature | An individual measurable property or characteristic used as input to a machine learning model. |
| **Feature Cross** | Feature Interaction | Creating a new feature by combining two existing features (Cartesian product). Captures interaction effects. |
| **Feature Engineering** | Feature Engineering | Process of transforming raw data into meaningful features that improve model performance. |
| **Feature Group** | Feature Group (Feature Store) | In Vertex AI Feature Store V2, a metadata registration of a BigQuery source grouping related features. |
| **Feature Importance** | Feature Importance | Score indicating how much each feature contributes to model predictions. Available for tree models. |
| **Feature Monitoring** | Feature Drift Monitoring | Detecting when feature distributions shift over time. Requires Feature Groups in Vertex AI Feature Store. |
| **Feature Registry** | Feature Registry | Vertex AI Feature Store's searchable catalogue of all features, backed by Dataplex Universal Catalog. |
| **Feature Store** | Feature Store | Centralised repository to store, manage, and serve ML features consistently across training and serving. |
| **Feature View** | Feature View (Feature Store) | In Vertex AI Feature Store V2, a logical view of BigQuery data synced to the online store for serving. |
| **Few-Shot** | Few-Shot Learning | Providing 2–5 input/output examples in the prompt to guide the model toward the desired output format. |
| **File API** | Gemini File API | Service for uploading and managing files (images, video, audio, PDFs) for use in Gemini multimodal prompts. |
| **Fine-Tuning** | Model Fine-Tuning | Adapting a pre-trained model to a specific task using a smaller, task-specific dataset. |
| **FN** | False Negative | See False Negative. |
| **FP** | False Positive | See False Positive. |
| **FPR** | False Positive Rate | FP / (FP + TN). The x-axis of the ROC curve. |
| **Function Calling** | Function Calling | Feature allowing a Gemini model to invoke developer-defined functions/tools and use their responses. |

---

## G

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **GBDT** | Gradient Boosted Decision Trees | Ensemble method where trees are built sequentially on residuals. Same as XGBoost / boosted trees. |
| **GCS** | Google Cloud Storage | Google's object storage service. Used for storing embeddings, model artefacts, training data. |
| **Gemini** | Gemini Model Family | Google's family of multimodal foundation models (Flash, Pro, Ultra variants). |
| **Generative AI** | Generative Artificial Intelligence | AI that creates new content (text, images, code, audio, video) based on patterns learned from training data. |
| **Gradient Descent** | Gradient Descent | Iterative optimisation algorithm that updates model weights by moving in the direction that reduces the loss function. |
| **Grounding** | Grounding | Connecting LLM outputs to verifiable external sources to reduce hallucinations. Sources can be search, private data, or maps. |

---

## H

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Hallucination** | Model Hallucination | When an LLM generates plausible-sounding but factually incorrect information not supported by its training data. |
| **Hidden Layer** | Hidden Layer (Neural Network) | Layers between input and output in a neural network that learn intermediate representations. |
| **HNSW** | Hierarchical Navigable Small World | Graph-based ANN algorithm used in some vector databases. Alternative to ScaNN. |
| **HPO** | Hyperparameter Optimisation | Systematic search for the best hyperparameter combination. Vertex AI uses Vizier under the hood. |
| **Hybrid Search** | Hybrid Search | Combining dense (semantic) and sparse (keyword) retrieval to improve relevance. Uses alpha blending. |
| **Hyperparameter** | Hyperparameter | Configuration value set before training that controls the training process (e.g. learning rate, number of trees). |

---

## I

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **IAM** | Identity and Access Management | GCP's access control system that manages who can do what on which resources. |
| **IMPLICIT Feedback** | Implicit User Feedback | Inferred user preferences from behaviour (clicks, views, purchases) — no explicit rating given. |
| **Imputation** | Missing Value Imputation | Filling in missing values using strategies like mean, median, or mode. |
| **Inference** | Model Inference | Using a trained model to make predictions on new data. Also called serving or scoring. |
| **IQR** | Interquartile Range | The difference between the 75th and 25th percentiles. Used in robust scaling and outlier detection. |

---

## J

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **JSON Mode** | Structured JSON Output | Forcing an LLM to respond only in valid JSON, defined by a schema. In Gemini: `response_mime_type="application/json"`. |
| **JSONL** | JSON Lines | File format where each line is a valid JSON object. Used for batch prediction inputs and vector index data. |

---

## K

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **K-Means** | K-Means Clustering | Unsupervised algorithm that partitions data into K clusters by minimising within-cluster variance. |
| **Kernel** | Kernel Function | Function mapping data to a higher-dimensional space to make it linearly separable (used in SVMs). |
| **KMS** | Key Management Service | GCP service for managing cryptographic keys, used for CMEK encryption. |
| **KNN** | K-Nearest Neighbours | Algorithm that classifies or predicts by majority vote of the K closest training examples. |
| **KPSS Test** | Kwiatkowski-Phillips-Schmidt-Shin Test | Statistical test used by ARIMA_PLUS to determine if a time series is stationary (determines `d` parameter). |

---

## L

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **L1 Regularisation** | Lasso Regularisation | Regularisation that adds absolute value of weights to the loss. Drives some weights to zero (feature selection). |
| **L2 Regularisation** | Ridge Regularisation | Regularisation that adds squared weights to the loss. Shrinks all weights but keeps all features. |
| **Label** | Target Variable | The output variable a model is trained to predict. Also called target, response variable, or dependent variable. |
| **Label Leakage** | Target Leakage | Including information in training that would not be available at prediction time. Leads to inflated metrics. |
| **Latent Factor** | Latent Factor | Hidden representation learned by a model. In Matrix Factorisation, users and items are represented as latent factor vectors. |
| **Latent Space** | Embedding Space | The lower-dimensional space where embeddings (vectors) are positioned. Similar items cluster together. |
| **Leaf Node** | Decision Tree Leaf | Terminal node of a decision tree that contains the final prediction for that branch. |
| **Learn Rate** | Learning Rate | Step size for gradient descent updates. Too high = overshooting; too low = slow convergence. |
| **LIME** | Local Interpretable Model-agnostic Explanations | Explainability technique that fits a simple interpretable model around a single prediction. |
| **LLM** | Large Language Model | A neural network trained on vast text data capable of generating, understanding, and reasoning about language. |
| **LlmAgent** | LLM Agent (ADK) | ADK agent type that uses an LLM as its reasoning engine to dynamically decide actions and tool use. |
| **Log Loss** | Logarithmic Loss | Cross-entropy loss for classification. Penalises confident wrong predictions heavily. |
| **LoopAgent** | Loop Agent (ADK) | ADK workflow agent that iterates sub-agents until max_iterations is reached. |

---

## M

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **MAE** | Mean Absolute Error | Average absolute difference between predictions and actual values. Same units as the label. |
| **MAP** | Mean Average Precision | Average of precision@K across all users/queries. Used to evaluate recommendation systems. |
| **MAPE** | Mean Absolute Percentage Error | Scale-independent forecast error. Good for comparing accuracy across series with different magnitudes. |
| **Matrix Factorisation** | Matrix Factorisation | Collaborative filtering technique that decomposes a user-item matrix into latent factor matrices for recommendations. |
| **MCP** | Model Context Protocol | Open standard for connecting AI agents to external tools, data sources, and services. |
| **Memory Bank** | Vertex AI Memory Bank | Long-term cross-session memory service for ADK agents using Vertex AI Agent Engine. |
| **MIME Type** | Multipurpose Internet Mail Extensions Type | Standard identifier for file format and nature (e.g. `image/jpeg`, `application/pdf`, `video/mp4`). |
| **Mini-Batch** | Mini-Batch Gradient Descent | Gradient descent using a small subset of training data per update. More stable than stochastic, faster than full-batch. |
| **MLOps** | Machine Learning Operations | Practices for deploying, monitoring, and maintaining ML models in production reliably and efficiently. |
| **Model Drift** | Model Performance Drift | Degradation of model accuracy over time due to changes in data distributions (see: data drift, concept drift). |
| **MSE** | Mean Squared Error | Average squared difference between predictions and actual values. Penalises large errors more than MAE. |
| **Multi-Agent System** | Multi-Agent System | Architecture where multiple specialised agents collaborate and delegate tasks to each other. |
| **Multiclass** | Multiclass Classification | Classification problem with 3 or more output categories. |
| **Multimodal** | Multimodal AI | AI that processes and generates multiple types of data simultaneously (text, images, audio, video). |

---

## N

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **NDCG** | Normalised Discounted Cumulative Gain | Ranking metric that rewards relevant items appearing higher in result lists. Normalised between 0 and 1. |
| **Neural Network** | Artificial Neural Network | Computing system inspired by biological neurons, consisting of interconnected layers of nodes. |
| **NLP** | Natural Language Processing | AI field focused on enabling machines to understand, interpret, and generate human language. |
| **Nucleus Sampling** | Top-P Sampling | Token selection method that samples from the smallest set of tokens whose cumulative probability ≥ P. |
| **Null Hypothesis** | Null Hypothesis | Assumption that there is no effect or relationship. Rejected if statistical evidence is strong enough. |
| **Num Factors** | Number of Latent Factors | In Matrix Factorisation, the dimensionality `k` of the user and item embedding vectors. |

---

## O

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Offline Store** | Feature Offline Store | Historical feature data used for training. In Feature Store V2, BigQuery is the offline store. |
| **One-Hot Encoding** | One-Hot Encoding | Encoding categorical variables as binary vectors — one dimension per category. Used for nominal (unordered) variables. |
| **Online Store** | Feature Online Store | Low-latency store for serving the latest feature values during real-time predictions. Bigtable-backed in Feature Store V2. |
| **Online Serving** | Online Prediction | Real-time, low-latency model serving via an endpoint. Synchronous request-response pattern. |
| **Ordinal Encoding** | Label/Ordinal Encoding | Encoding ordered categorical variables as integers (0, 1, 2...). Preserves rank order. |
| **Overfitting** | Overfitting | Model performs well on training data but poorly on unseen data. Caused by excessive complexity. |

---

## P

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **ParallelAgent** | Parallel Agent (ADK) | ADK workflow agent that runs sub-agents concurrently for independent tasks. |
| **PCA** | Principal Component Analysis | Unsupervised dimensionality reduction technique that projects data onto orthogonal components of maximum variance. |
| **Perplexity** | Perplexity | Measure of how well a language model predicts a text. Lower perplexity = better model. |
| **Point-in-Time Correctness** | Point-in-Time Correct Join | Ensuring training features are computed using only data available before the label date. Prevents label leakage. |
| **Polynomial Expansion** | Polynomial Feature Expansion | Generating polynomial and interaction terms from numeric features to add non-linear expressiveness to linear models. |
| **Precision** | Precision | TP / (TP + FP). Of all predicted positives, how many are actually positive. |
| **Precision@K** | Precision at K | Fraction of the top-K recommendations that are relevant. |
| **Prompt** | Prompt | Input text (and optionally files) provided to an LLM to elicit a desired output. |
| **Prompt Engineering** | Prompt Engineering | The discipline of crafting effective inputs (prompts) to guide LLM outputs toward desired results. |

---

## Q

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Quantile** | Quantile | Value below which a given percentage of data falls. Q25 = 25th percentile, Q75 = 75th percentile. |

---

## R

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **R²** | Coefficient of Determination | Proportion of variance in the label explained by the model. 1.0 = perfect; 0.5 = random baseline. |
| **RAG** | Retrieval-Augmented Generation | Technique combining retrieval of relevant documents with LLM generation to produce grounded, accurate responses. |
| **Ranking** | Re-Ranking | Post-retrieval step that re-scores candidate documents using a deeper query-document model to improve precision. |
| **Recall** | True Positive Rate | TP / (TP + FN). Of all actual positives, how many did the model catch. Also = Sensitivity = TPR. |
| **Recall@K** | Recall at K | Fraction of all relevant items found in the top-K recommendations. |
| **Reconstruction Error** | Reconstruction Error | In autoencoders, the difference between original and reconstructed input. High error = likely anomaly. |
| **Regularisation** | Regularisation | Technique to reduce overfitting by adding a penalty term to the loss function (L1, L2, or Elastic Net). |
| **Restrict** | Vector Search Restrict | Filter in Vertex AI Vector Search that limits results to a subset of vectors using metadata attributes. |
| **RFE** | Recursive Feature Elimination | Feature selection technique that iteratively removes the least important features. |
| **RMSE** | Root Mean Squared Error | Square root of MSE. Same units as the label. More sensitive to large errors than MAE. |
| **ROC Curve** | Receiver Operating Characteristic Curve | Plot of TPR vs FPR across all classification thresholds. AUC = area under this curve. |
| **Role Prompting** | Role Prompting | Assigning a persona to the model ("You are a senior engineer...") to shape response style and expertise. |
| **RRF** | Reciprocal Rank Fusion | Algorithm for blending results from multiple ranked lists. Used in hybrid search (`rrf_ranking_alpha`). |

---

## S

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Safety Settings** | Safety Filters | In Gemini, configurable thresholds that control how strictly the model blocks potentially harmful content. |
| **ScaNN** | Scalable Nearest Neighbours | Google's ANN algorithm used by Vertex AI Vector Search. Powers Google Search, YouTube, and Google Play. |
| **Seed** | Random Seed | Fixed value for random number generation to produce reproducible (though not guaranteed deterministic) model outputs. |
| **Semantic Search** | Semantic Search | Search based on meaning/intent rather than exact keyword matching. Uses embedding vectors. |
| **SequentialAgent** | Sequential Agent (ADK) | ADK workflow agent that runs sub-agents in a fixed order. Used for deterministic pipelines. |
| **Session** | Agent Session | In ADK, a managed container holding conversation history and state for one user interaction. |
| **SHAP** | SHapley Additive exPlanations | Explainability method based on game theory that assigns each feature a contribution value for a given prediction. |
| **Shapley Values** | Shapley Values | Fair attribution of prediction contribution to each feature, based on the SHAP/Shapley framework. |
| **Shrinkage** | Boosting Shrinkage | Scaling each boosting tree's contribution by `learn_rate` to make the ensemble more robust. |
| **Sigmoid** | Sigmoid Function | Activation function that squashes values to (0, 1). Used in binary classification output layers. |
| **sMAPE** | Symmetric MAPE | Symmetric version of MAPE that treats over- and under-prediction equally. Used in ARIMA_PLUS evaluation. |
| **Softmax** | Softmax Function | Converts raw model scores into a probability distribution across all classes. Used in multiclass output. |
| **Sparse Vector** | Sparse Embedding | A vector where most dimensions are zero. Used for keyword/BM25 retrieval in hybrid search. |
| **Stationarity** | Time-Series Stationarity | Property of a time series with constant mean, variance, and autocorrelation over time. Required by ARIMA. |
| **Stop Sequence** | Stop Sequence | A string that instructs the model to stop generating text when encountered. |
| **Structured Output** | Structured Output / JSON Mode | Forcing model responses to conform to a predefined schema. Enables reliable downstream parsing. |
| **Sub-Agent** | Sub-Agent (ADK) | Specialised agent called by a root/orchestrator agent to handle a specific sub-task. |
| **Subsample** | Subsampling (Boosted Trees) | Fraction of training rows used to fit each tree. <1.0 introduces stochastic gradient boosting (anti-overfitting). |
| **SVM** | Support Vector Machine | Algorithm that finds the optimal hyperplane to separate classes in high-dimensional space. |
| **System Instruction** | System Prompt | In Gemini, instructions that define model behaviour, persona, and rules and persist across all conversation turns. |

---

## T

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Temperature** | Sampling Temperature | Controls randomness in token selection. 0 = greedy/deterministic; higher = more diverse/creative. |
| **Tensor** | Tensor | Multi-dimensional array used in deep learning frameworks. Generalises scalars, vectors, and matrices. |
| **TF-IDF** | Term Frequency–Inverse Document Frequency | Statistical measure for keyword relevance. Foundation of sparse (BM25) retrieval. |
| **Thinking Mode** | Thinking Mode (Gemini) | Gemini capability where the model reasons internally before producing a final answer. Controlled by `thinking_level`. |
| **TN** | True Negative | Model correctly predicted the negative class. |
| **Token** | Token | The basic unit of text processed by an LLM. Approximately 4 characters or 0.75 words in English. |
| **Top-K** | Top-K Sampling | Token selection method that considers only the K most probable next tokens. |
| **Top-P** | Top-P / Nucleus Sampling | Token selection method that samples from tokens whose cumulative probability ≥ P. |
| **TP** | True Positive | Model correctly predicted the positive class. |
| **TPR** | True Positive Rate | TP / (TP + FN). Equivalent to Recall and Sensitivity. The y-axis of the ROC curve. |
| **Training-Serving Skew** | Training-Serving Skew | When features used in training are computed differently from how they're computed at serving time. Leads to poor production performance. |
| **TRANSFORM** | BQML TRANSFORM Clause | SQL clause in BigQuery ML that defines feature preprocessing. Stored with the model and auto-applied at prediction time. |
| **Transfer Learning** | Transfer Learning | Reusing a model pre-trained on one task as the starting point for a different but related task. |
| **TTL** | Time-To-Live | Duration after which a resource (file, cache, session) is automatically deleted. |

---

## U

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Underfitting** | Underfitting | Model is too simple to capture the underlying patterns in the data. High bias, low variance. |
| **Unstructured Data** | Unstructured Data | Data with no predefined schema — documents, images, video, audio. |

---

## V

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Validation Set** | Validation Set | Held-out data used during training to tune hyperparameters and detect overfitting. Not used for final evaluation. |
| **Variance** | Model Variance | Sensitivity of model predictions to fluctuations in training data. High variance = overfitting. |
| **Vector** | Embedding Vector | A list of numbers representing an entity (text, image, user) in a multi-dimensional space. |
| **Vector Database** | Vector Database | Storage and retrieval system optimised for similarity search over high-dimensional embedding vectors. |
| **Vector Search** | Vertex AI Vector Search | Google's fully managed ANN search service. Formerly Matching Engine. Uses ScaNN algorithm. |
| **Vertex AI** | Vertex AI | Google Cloud's unified ML platform for building, deploying, and managing ML models at scale. |
| **Vertex AI Search** | Vertex AI Search | Fully managed enterprise search service using semantic + keyword search over private data. |
| **Vizier** | Vertex AI Vizier | Google's hyperparameter optimisation service used internally by BQML `num_trials` and AutoML. |
| **VPC** | Virtual Private Cloud | Isolated network within GCP for private, secure communication between services. |
| **VPC Peering** | VPC Network Peering | Private connection between two VPC networks. Required for Vector Search private endpoints. |

---

## W

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **WALS** | Weighted Alternating Least Squares | Optimisation algorithm for matrix factorisation with implicit feedback. Uses confidence weighting via `wals_alpha`. |
| **Weight** | Model Weight | Learnable parameter in a neural network or linear model that scales feature contributions. |
| **Weighted Average** | Weighted Average Metric | Metric averaged across classes proportionally to their support (number of samples). Default in BQML multiclass. |

---

## X

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **XGBoost** | Extreme Gradient Boosting | High-performance gradient boosting framework. The underlying engine for BQML boosted tree models. |

---

## Z

| Term / Acronym | Full Name | Definition |
|---------------|-----------|------------|
| **Zero-Shot** | Zero-Shot Learning | Asking a model to perform a task with no examples provided — relying entirely on its pre-trained knowledge. |
| **Z-Score** | Standard Score | Number of standard deviations from the mean. Used in STANDARD_SCALER: z = (x − mean) / std_dev. |

---

## Symbols & Notation Quick Reference

| Symbol | Meaning |
|--------|---------|
| `P(y\|x)` | Probability of y given x |
| `ŷ` | Predicted value |
| `y` | Actual / ground truth value |
| `η` (eta) | Learning rate in gradient boosting |
| `λ` (lambda) | Regularisation strength |
| `k` | Number of latent factors / clusters / neighbours |
| `d` | ARIMA differencing order |
| `p` | ARIMA autoregressive order |
| `q` | ARIMA moving average order |
| `α` (alpha) | Hybrid search blend ratio (0=keyword, 1=semantic) |
| `top-k` | Top-K nearest neighbours retrieved |

---

*Reference last updated: April 2026. Always verify model-specific limits and parameters against the latest Google Cloud documentation.*