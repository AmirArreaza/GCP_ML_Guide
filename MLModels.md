# 🧠 GCP Machine Learning Professional Certification
## Model Types — Study Guide

> **How to use this guide:** Each model type includes a plain-English explanation, a memory hook, real-world scenarios, and GCP-specific notes. Study one section at a time.

---

## 📌 Quick Reference Table

| Model Type | Output | Key Question It Answers |
|---|---|---|
| Linear Regression | Continuous number | "How much / how many?" |
| Regression with Boosted Trees | Continuous number (complex) | "How much?" (with nonlinear patterns) |
| Binary Classification | One of two classes | "Is it A or B?" |
| Multiclass Classification | One of many classes | "Which category does this belong to?" |
| Time-Series Forecasting | Future values over time | "What will happen next?" |
| Matrix Factorization | Hidden relationships | "What would this user like?" |
| Anomaly Detection | Normal or anomalous | "Is this unusual?" |

---

## 1. 📈 Linear Regression

### What is it?
Linear Regression finds the **straight-line relationship** between input features and a **continuous numerical output**.  
Think of it as drawing the best possible line through a scatter plot of data points.

### 🧲 Memory Hook
> *"Draw a line, predict a number."*

### Formula Intuition
```
y = (w1 * x1) + (w2 * x2) + ... + b
```
- `y` = the value you want to predict  
- `x` = your input features  
- `w` = weights (learned during training)  
- `b` = bias (the intercept)

### ✅ When to Use It
- The output is a **real number** (price, temperature, score)
- The relationship between inputs and output is roughly **linear**
- You need a **simple, interpretable** model

### 🌍 Scenarios

**Scenario 1 — House Price Prediction**  
You have data: square footage, number of rooms, neighborhood score.  
You want to predict: **sale price in dollars**.  
→ Linear Regression maps those features to a price.

**Scenario 2 — Fuel Consumption**  
You have: engine size, car weight, speed.  
You want to predict: **liters per 100km**.  
→ A linear model can approximate this relationship well.

**Scenario 3 — Student Exam Score**  
You have: hours studied, sleep hours, practice tests completed.  
You want to predict: **final exam score (0–100)**.

### 🔧 GCP Context
- Available in **BigQuery ML** → `CREATE MODEL ... OPTIONS(model_type='linear_reg')`
- Available in **Vertex AI AutoML Tabular** (regression task)
- Evaluation metric: **RMSE, MAE, R²**

---

## 2. 🌲 Regression with Boosted Trees (Gradient Boosted Trees)

### What is it?
Instead of one straight line, this builds **many decision trees sequentially**, where each tree **corrects the mistakes** of the previous one.  
The result is a very powerful model for structured/tabular data with **nonlinear and complex patterns**.

### 🧲 Memory Hook
> *"A team of trees, each fixing the last one's mistakes."*

### How Boosting Works (Simplified)
```
Tree 1 predicts → has errors
Tree 2 focuses on those errors → has smaller errors
Tree 3 focuses on Tree 2's errors → even smaller
... repeat N times → very accurate ensemble
```

### ✅ When to Use It
- Output is **continuous** (like linear regression) but the data is complex
- Features have **nonlinear relationships**
- You have **structured/tabular data** and want high accuracy
- Data may have **missing values or outliers**

### 🌍 Scenarios

**Scenario 1 — Taxi Fare Prediction**  
You have: pickup location, dropoff location, time of day, day of week, traffic index.  
You want to predict: **fare amount**.  
→ Relationships are nonlinear (rush hour effects, weekend patterns). Boosted Trees captures this.

**Scenario 2 — Insurance Premium Pricing**  
You have: age, medical history, lifestyle factors, region.  
You want to predict: **annual premium cost**.  
→ Many interactions between features that a straight line can't model well.

**Scenario 3 — Energy Consumption Forecasting**  
You have: building size, occupancy, weather, hour of day.  
You want to predict: **kilowatt-hours consumed**.

### 🔧 GCP Context
- Available in **BigQuery ML** → `model_type='boosted_tree_regressor'`
- Available in **Vertex AI AutoML Tabular** and **custom training with XGBoost**
- Better accuracy than linear regression for complex datasets but less interpretable
- Evaluation metric: **RMSE, MAE, R²**

---

## 3. ⚖️ Binary Classification

### What is it?
Predicts **one of exactly two outcomes** — Yes/No, True/False, Spam/Not Spam.  
The model outputs a **probability score** (0 to 1), and a threshold (usually 0.5) decides the final label.

### 🧲 Memory Hook
> *"Flip a coin — but a very smart, data-trained coin."*

### Output Example
```
Input: email content features
Output: 0.87 → "Spam" (above 0.5 threshold)
Output: 0.12 → "Not Spam" (below 0.5 threshold)
```

### ✅ When to Use It
- There are **only 2 possible outcomes**
- You want to know the **probability** of something happening
- Examples: fraud detection, disease diagnosis, churn prediction

### 🌍 Scenarios

**Scenario 1 — Credit Card Fraud Detection**  
You have: transaction amount, location, merchant type, time since last transaction.  
You want to predict: **Fraud (1) or Not Fraud (0)**.  
→ Binary Classification outputs a fraud probability per transaction.

**Scenario 2 — Customer Churn**  
You have: usage frequency, support tickets, subscription age, last login date.  
You want to predict: **Will this customer cancel? Yes or No**.

**Scenario 3 — Medical Diagnosis**  
You have: patient vitals, lab results, age, medical history.  
You want to predict: **Has disease (Positive) or Doesn't have disease (Negative)**.  
→ The probability score lets doctors set a risk threshold.

### 🔧 GCP Context
- **BigQuery ML** → `model_type='logistic_reg'` (for logistic regression-based binary classification)
- **BigQuery ML** → `model_type='boosted_tree_classifier'`
- **Vertex AI AutoML Tabular** → Classification task with 2 classes
- Evaluation metrics: **AUC-ROC, Precision, Recall, F1-Score, Confusion Matrix**
- ⚠️ Watch out for **class imbalance** (e.g., 99% not fraud, 1% fraud)

---

## 4. 🎨 Multiclass Classification

### What is it?
Like Binary Classification, but now the model must pick **one label from three or more possible classes**.  
Each input gets assigned to exactly one category.

### 🧲 Memory Hook
> *"Sort items into one of many labeled buckets."*

### Output Example
```
Input: an image of a piece of clothing
Output probabilities:
  - T-shirt: 0.72 ✅ (winner)
  - Pants: 0.15
  - Shoes: 0.08
  - Dress: 0.05
```

### ✅ When to Use It
- There are **3+ possible categories**
- Each input belongs to **exactly one** category
- Examples: topic classification, product categorization, image labeling

### 🌍 Scenarios

**Scenario 1 — Support Ticket Routing**  
You have: text content of a customer support message.  
You want to predict: **which department handles it** → Billing / Technical / Returns / General.  
→ Multiclass classifies each ticket into one department.

**Scenario 2 — Crop Disease Identification**  
You have: image of a crop leaf.  
You want to predict: **disease type** → Healthy / Rust / Blight / Mildew / Mosaic Virus.

**Scenario 3 — News Article Categorization**  
You have: article headline and body text.  
You want to predict: **topic** → Sports / Politics / Technology / Finance / Entertainment.

### 🔧 GCP Context
- **BigQuery ML** → `model_type='logistic_reg'` automatically handles multiclass when target has 3+ classes
- **BigQuery ML** → `model_type='boosted_tree_classifier'`
- **Vertex AI AutoML** → Text classification, Image classification, Tabular classification
- Evaluation metrics: **Per-class Precision/Recall, Confusion Matrix, Macro/Micro F1**
- ⚠️ Difference from Binary: uses **Softmax** activation (not Sigmoid)

---

## 5. 📅 Time-Series Forecasting

### What is it?
Predicts **future values** based on **historical patterns over time**.  
Unlike standard regression, the order and timing of data points matters — past values influence future ones.

### 🧲 Memory Hook
> *"Learn from the past to predict the future — time is a feature."*

### Key Concepts
| Concept | Meaning |
|---|---|
| **Trend** | Long-term increase or decrease |
| **Seasonality** | Repeating patterns (daily, weekly, yearly) |
| **Noise** | Random fluctuations |
| **Lag features** | Using past values (t-1, t-7) as inputs |

### ✅ When to Use It
- Your data is **ordered by time**
- You want to **predict future values** of a metric
- Examples: sales forecasting, demand planning, stock trends, weather

### 🌍 Scenarios

**Scenario 1 — Retail Sales Forecasting**  
You have: 3 years of weekly sales data per product per store.  
You want to predict: **sales for the next 4 weeks**.  
→ The model picks up on holiday seasonality, weekend peaks, and long-term trends.

**Scenario 2 — Server CPU Usage Prediction**  
You have: per-minute CPU utilization logs for 6 months.  
You want to predict: **CPU usage for the next 2 hours**.  
→ Helps auto-scaling systems prepare before traffic spikes happen.

**Scenario 3 — Electricity Demand Forecasting**  
You have: hourly electricity consumption for a city.  
You want to predict: **demand for the next 24 hours**.  
→ Utility companies use this for grid management.

### 🔧 GCP Context
- **BigQuery ML** → `model_type='arima_plus'` (AutoRegressive Integrated Moving Average, enhanced)
- **Vertex AI Forecast** → Purpose-built AutoML for time-series (supports multiple series at once)
- `ARIMA_PLUS` auto-detects trend, seasonality, and holiday effects
- Evaluation metrics: **MAE, RMSE, MAPE (Mean Absolute Percentage Error)**
- ⚠️ Key exam concept: `TIME_SERIES_TIMESTAMP_COL` and `TIME_SERIES_DATA_COL` in BigQuery ML

---

## 6. 🤝 Matrix Factorization

### What is it?
Discovers **hidden (latent) factors** that explain the relationship between two entities (e.g., users and products).  
It works by **decomposing a large sparse matrix** into two smaller dense matrices.

### 🧲 Memory Hook
> *"You liked these movies → you probably like these other movies too."*

### How It Works (Intuition)
```
Large sparse matrix (Users × Movies, mostly empty):
         Movie A  Movie B  Movie C  Movie D
User 1:    5        ?        3        ?
User 2:    ?        4        ?        5
User 3:    3        ?        ?        4

Matrix Factorization fills in the "?" values by finding
hidden patterns like genre preferences, style affinities, etc.
```

### ✅ When to Use It
- You have a **user-item interaction matrix** (ratings, clicks, purchases)
- The matrix is **sparse** (most users haven't interacted with most items)
- Goal: **recommend items** a user hasn't seen yet

### 🌍 Scenarios

**Scenario 1 — Movie Recommendation (Classic)**  
You have: user-movie rating data (1–5 stars), but each user only rated ~50 of 10,000 movies.  
You want to: **predict what rating a user would give unseen movies**.  
→ Recommend top-N movies with highest predicted ratings.

**Scenario 2 — E-commerce Product Recommendations**  
You have: user purchase history (user bought product → implicit rating of 1).  
You want to: **suggest new products** the user hasn't purchased yet.

**Scenario 3 — Music Playlist Personalization**  
You have: play counts per user per song.  
You want to: **recommend songs the user hasn't played** but would likely enjoy.

### 🔧 GCP Context
- **BigQuery ML** → `model_type='matrix_factorization'`
- Requires: `user_col`, `item_col`, `rating_col`
- Supports **explicit feedback** (star ratings) and **implicit feedback** (clicks, plays)
- Use `ML.RECOMMEND` to generate predictions after training
- Evaluation metrics: **Mean Average Precision (MAP), NDCG**
- ⚠️ Exam tip: This is **collaborative filtering** — it does NOT use item content features

---

## 7. 🚨 Anomaly Detection

### What is it?
Identifies data points that are **significantly different from the norm**.  
The model learns what "normal" looks like, then flags anything that deviates too far.

### 🧲 Memory Hook
> *"If it doesn't look like the others, it's suspicious."*

### Two Main Approaches
| Approach | How It Works |
|---|---|
| **Statistical** | Points outside N standard deviations are anomalies |
| **Model-based** | Train on normal data; high reconstruction error = anomaly |

### ✅ When to Use It
- You want to find **rare, unusual, or suspicious events**
- You may have **very few or no labeled anomaly examples** (unsupervised)
- Examples: fraud, equipment failure, network intrusion, data quality issues

### 🌍 Scenarios

**Scenario 1 — Manufacturing Equipment Monitoring**  
You have: sensor readings (temperature, vibration, pressure) from a machine over months.  
You want to: **detect when readings suggest the machine is about to fail**.  
→ The model learns normal operating ranges and alerts on deviations.

**Scenario 2 — Financial Transaction Anomalies**  
You have: transaction history for thousands of accounts (normal behavior).  
You want to: **flag transactions that don't fit a user's normal pattern**.  
→ A $3,000 transaction at 3am in a foreign country for a user who typically spends $50 locally.

**Scenario 3 — Log-Based Intrusion Detection**  
You have: server access logs (normal login patterns).  
You want to: **detect unusual access patterns** that could indicate a breach.

**Scenario 4 — Data Pipeline Quality Checks**  
You have: daily row counts, null rates, and value distributions for a dataset.  
You want to: **alert when today's data looks statistically different** from historical data.

### 🔧 GCP Context
- **BigQuery ML** → `model_type='kmeans'` (cluster-based anomaly detection)
- **Vertex AI** → **Anomaly Detection** in Vertex AI Tabular workflows
- **Vertex AI** → `AUTOENCODER` style models for complex anomaly detection
- For time-series anomalies: `ARIMA_PLUS` includes anomaly detection capabilities
- Evaluation: **Precision/Recall at threshold**, or **anomaly score distributions**
- ⚠️ Exam tip: Often **unsupervised** because anomalies are rare and hard to label

---

## 🔁 Model Selection Decision Tree

```
Is your output a number?
├── YES → Is your data ordered by time?
│         ├── YES → ⏱️ Time-Series Forecasting
│         └── NO  → Is the relationship complex/nonlinear?
│                   ├── YES → 🌲 Regression with Boosted Trees
│                   └── NO  → 📈 Linear Regression
│
└── NO → Is it a category?
          ├── YES → How many categories?
          │         ├── 2 → ⚖️ Binary Classification
          │         └── 3+ → 🎨 Multiclass Classification
          │
          └── NO → What's the goal?
                    ├── Recommend items → 🤝 Matrix Factorization
                    └── Find outliers  → 🚨 Anomaly Detection
```

---

## 📝 GCP Exam Quick-Reference Cheat Sheet

| Model Type | BigQuery ML `model_type` | Vertex AI Equivalent | Key Metric |
|---|---|---|---|
| Linear Regression | `linear_reg` | AutoML Tabular (Regression) | RMSE, R² |
| Boosted Trees Regression | `boosted_tree_regressor` | AutoML / XGBoost | RMSE, MAE |
| Binary Classification | `logistic_reg` / `boosted_tree_classifier` | AutoML Tabular (Classification) | AUC-ROC, F1 |
| Multiclass Classification | `logistic_reg` / `boosted_tree_classifier` | AutoML Text/Image/Tabular | F1, Confusion Matrix |
| Time-Series Forecasting | `arima_plus` | Vertex AI Forecast | MAPE, RMSE |
| Matrix Factorization | `matrix_factorization` | — | MAP, NDCG |
| Anomaly Detection | `kmeans` / `autoencoder` | Vertex AI Tabular Anomaly | Precision/Recall |

---

*Happy studying! 🎓 Focus on knowing **when** to use each model, not just **what** it is — that's what the GCP exam tests.*