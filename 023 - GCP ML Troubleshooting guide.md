# GCP ML Professional Certification — Model Troubleshooting Guide

> A quick-reference table covering the most common exam scenarios across all major ML model types.
> Use this to diagnose symptoms and apply the correct fix on the exam.

---

## 🔁 Data & Splitting Issues

| Situation | Solution |
|-----------|----------|
| Randomly shuffled splits → 92% validation accuracy, but 58% production accuracy (time-series data) | Use **chronological splits**: train on past, validate on future. Random shuffling causes data leakage across time. |
| Validation accuracy is high but the model fails on a new demographic group in production | The dataset has **sampling bias**. Stratify splits by demographic group and ensure all subgroups are represented in train/val/test. |
| Model performs well in the lab but degrades rapidly after deployment (concept drift) | Implement **continuous evaluation** with monitoring on Vertex AI. Schedule periodic retraining triggered by performance thresholds. |
| Training data contains duplicates that span train and validation sets | Run **deduplication** before splitting. Leaking near-identical samples inflates validation metrics. |
| Dataset has severe class imbalance (e.g. 99% negative, 1% positive) | Use **stratified sampling** for splits. Apply class weights, oversampling (SMOTE), or undersampling. Evaluate with F1/AUC-PR instead of accuracy. |

---

## 📉 Underfitting (High Bias)

| Situation | Solution |
|-----------|----------|
| Training accuracy and validation accuracy are both low | Model is **underfitting**. Increase model complexity (more layers/neurons, higher-degree polynomial features, deeper trees). |
| Linear model fails to capture a non-linear relationship | Switch to a **non-linear model** (neural network, gradient boosting, kernel SVM) or add polynomial/interaction features. |
| Decision tree with max_depth=2 has poor train and val accuracy | Increase `max_depth` or switch to **Random Forest / XGBoost** for better expressiveness. |
| Model trained with very high regularization (e.g. L2 λ=100) underperforms | **Reduce regularization strength**. High regularization constrains the model too aggressively. Tune λ via cross-validation. |
| BERT fine-tuned for only 1 epoch on a small dataset shows poor results | Train for more epochs and use a **lower learning rate** with warm-up; fine-tuning large models requires careful scheduling. |

---

## 📈 Overfitting (High Variance)

| Situation | Solution |
|-----------|----------|
| Training accuracy is 99%, validation accuracy is 70% | Model is **overfitting**. Add Dropout, L1/L2 regularization, reduce model size, or collect more training data. |
| A deep neural network memorises the training set within 2 epochs | Apply **early stopping** (monitor val loss with patience), Dropout layers, and data augmentation. |
| Random Forest has 500 trees and overfits on a small dataset | Reduce `n_estimators`, limit `max_depth`, increase `min_samples_leaf`, or use a **simpler ensemble**. |
| XGBoost achieves 100% train accuracy but poor test accuracy | Tune `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`. Use `early_stopping_rounds` on val set. |
| Fine-tuned LLM outputs training examples verbatim | The model has **memorised** training data. Reduce fine-tuning epochs, add regularization, and apply differential privacy if needed. |

---

## 🧠 Neural Networks

| Situation | Solution |
|-----------|----------|
| Loss is NaN after the first few iterations | **Exploding gradients**. Apply gradient clipping (`clipnorm` or `clipvalue`), reduce learning rate, or use batch normalisation. |
| Loss does not decrease at all (flat loss curve) | **Vanishing gradients** or dead learning rate. Use ReLU activations, He initialisation, check learning rate (try 1e-3), and verify labels. |
| Model trains but all predictions are the same class | **Dead neurons or bad initialisation**. Check for label encoding errors, use Xavier/He init, and ensure the output layer uses appropriate activation (softmax/sigmoid). |
| Validation loss decreases then suddenly spikes and diverges | Learning rate is **too high** for later training. Use a **learning rate scheduler** (cosine decay, reduce-on-plateau). |
| Very deep network trains worse than a shallower one | Use **residual connections** (ResNet-style skip connections) to allow gradients to flow through deep layers. |
| Training is extremely slow on a large dataset in Vertex AI | Enable **GPU/TPU acceleration**, increase batch size, use `tf.data` pipeline with prefetch and caching, or use distributed training. |

---

## 🖼️ Computer Vision (CNNs)

| Situation | Solution |
|-----------|----------|
| CNN trained on studio images fails on real-world photos | **Domain shift**. Apply data augmentation (rotation, flips, colour jitter, random crop) and consider **domain adaptation** or fine-tuning on in-domain data. |
| Small image dataset leads to overfitting despite dropout | Use **transfer learning** (e.g. EfficientNet, ResNet pretrained on ImageNet). Freeze early layers, fine-tune later layers. |
| Model cannot detect small objects in images | Use **Feature Pyramid Networks (FPN)** or multi-scale detection architectures (e.g. YOLO, Faster R-CNN with FPN). |
| Object detection model has high mAP but slow inference in production | Apply **model quantisation** (INT8), pruning, or switch to a lightweight architecture (MobileNet, EfficientDet-Lite) and serve on Vertex AI endpoints. |
| Image classification model is biased toward background colour | The model learned a **spurious correlation**. Use **GradCAM** to inspect attention, then apply random background augmentation or patch-based training. |

---

## 📝 NLP / Text Models

| Situation | Solution |
|-----------|----------|
| Sentiment model performs well on formal text but poorly on social media slang | **Domain mismatch**. Fine-tune on in-domain data or use a model pre-trained on social media (e.g. BERTweet). |
| NER model misses entities not seen in training | Add more labelled examples for unseen entity types or use **few-shot prompting** with a foundation model via Vertex AI. |
| Text classifier ignores word order and loses contextual meaning | Switch from Bag-of-Words/TF-IDF to a **transformer-based model** (BERT, T5) that captures sequential context. |
| LLM generates hallucinated facts confidently | Implement **Retrieval-Augmented Generation (RAG)** using Vertex AI Search + Matching Engine to ground outputs in factual documents. |
| Token sequence exceeds model's max context window | **Chunk** the document with overlap and aggregate predictions, or use a long-context model (e.g. Gemini 1.5 Pro with 1M token window). |
| Translation model produces fluent but inaccurate outputs (hallucinations) | Use **constrained decoding**, increase beam search width, or add faithfulness metrics (BERTScore, BLEURT) to the evaluation pipeline. |

---

## 📊 Tabular / Structured Data Models

| Situation | Solution |
|-----------|----------|
| XGBoost model has poor performance on a feature-heavy dataset with many irrelevant columns | Apply **feature selection** (permutation importance, SHAP values) to drop low-signal features before training. |
| Linear regression has high residuals for large values only | The relationship is **heteroscedastic**. Apply log-transform to the target variable or use quantile regression. |
| Logistic regression convergence warning; model doesn't converge | Increase `max_iter`, scale features with `StandardScaler`, or switch solver (e.g. `lbfgs` → `saga`). |
| Feature with high cardinality (e.g. ZIP code, user ID) causes overfitting | Use **target encoding** with cross-validation, **embedding layers**, or dimensionality reduction instead of one-hot encoding. |
| Model accuracy drops after new categorical values appear in production | **Unseen categories at inference**. Handle with a default/unknown bucket during preprocessing or use **frequency encoding**. |
| AutoML Tables model underperforms a manual model | Check that **feature engineering** (interactions, date decomposition, lag features) has been applied; AutoML benefits from well-engineered inputs. |

---

## ⏱️ Time Series Models

| Situation | Solution |
|-----------|----------|
| Forecast model trained on 2 years of data has high error in the last 3 months | Likely **concept drift** or seasonality change. Retrain on recent data and add trend/seasonality decomposition (e.g. Prophet, ARIMA with exogenous vars). |
| Walk-forward validation shows degrading performance over time | Model is **not adapting**. Use **online learning** or rolling-window retraining on Vertex AI Pipelines. |
| Time-series model with stationary assumption applied to a non-stationary series | Apply **differencing** or **log transformation** to make the series stationary before modelling. |
| LSTM for forecasting is slow to train and underperforms XGBoost on tabular time-series | For tabular time-series, prefer **gradient boosting with lag features** (LightGBM/XGBoost). LSTMs shine on raw sequence data without handcrafted features. |
| Model does not capture weekly/yearly seasonality | Add **Fourier terms** or use **seasonal decomposition** (STL). Prophet handles multiple seasonalities out of the box. |

---

## 🤖 Recommendation Systems

| Situation | Solution |
|-----------|----------|
| Collaborative filtering returns popular items to every user (popularity bias) | Add **diversity regularisation** or use **two-tower models** with user-specific embeddings. Explore re-ranking with novelty scores. |
| Cold-start problem: new users receive irrelevant recommendations | Use **content-based filtering** for new users using profile/demographic features until enough interaction data is collected. |
| Matrix factorisation model fails to scale to millions of users | Use **approximate nearest-neighbour search** (Vertex AI Matching Engine / ScaNN) on pre-computed embeddings for scalable retrieval. |
| Model recommends items the user already purchased | Add a **post-processing filter** to exclude already-interacted items before serving. |
| Embedding model trained offline drifts from live user preferences | Implement **near-real-time feature updates** using Vertex AI Feature Store with streaming ingestion. |

---

## ☁️ GCP / Vertex AI Specific

| Situation | Solution |
|-----------|----------|
| Vertex AI Training job runs out of memory on a large model | Use **gradient checkpointing**, reduce batch size, switch to a larger machine type (A100 GPU), or use **model parallelism** across multiple devices. |
| Vertex AI Pipelines job fails intermittently with no clear error | Enable **step caching** and check **Cloud Logging** for OOM errors or quota limits. Add retry logic to fragile pipeline components. |
| Online prediction latency is too high in production | Use **batch prediction** for offline use cases. For online, enable **autoscaling**, choose a smaller/quantised model, or use a regional endpoint closer to users. |
| BigQuery ML model predictions differ from the same model trained in sklearn | Check **feature preprocessing** differences (scaling, encoding). BQML applies transformations inside SQL — ensure they match your sklearn pipeline exactly. |
| Vertex AI Workbench notebook runs slow on a large Pandas DataFrame | Switch to **BigQuery** for data processing or use **Spark on Dataproc** for distributed computation; avoid Pandas for >10 GB datasets. |
| Model deployed on Vertex AI returns stale predictions after retraining | The old model version is still serving. **Deploy the new model version** to the endpoint and shift traffic using traffic-splitting controls. |
| Explainability (SHAP) values on Vertex AI are inconsistent across runs | Set a **fixed random seed** for the SHAP background dataset sampler and use a consistent baseline for integrated gradients. |

---

## 🔍 Evaluation & Metrics

| Situation | Solution |
|-----------|----------|
| Accuracy is 95% but the model is useless (imbalanced classes) | Use **AUC-ROC, F1-score, precision-recall curve** instead of accuracy. Accuracy is misleading when classes are skewed. |
| Model has high precision but very low recall for fraud detection | The decision **threshold is too high**. Lower the classification threshold or optimise directly for recall/F1 using threshold tuning on the validation set. |
| RMSE is low but the model consistently underestimates high values | Check for **target skewness**. Log-transform the target, or use **quantile loss** to capture high-value tail predictions. |
| ROC-AUC is 0.98 but PR-AUC is 0.20 | The dataset is **highly imbalanced**. PR-AUC is the more informative metric; investigate false negatives and recalibrate the model. |
| Two models have equal AUC but different business outcomes | Choose based on **business-relevant metrics**: use confusion matrix analysis at your operating threshold, and align with cost of false positives vs false negatives. |

---

*Last updated for GCP Professional Machine Learning Engineer exam objectives — 2025/2026.*
