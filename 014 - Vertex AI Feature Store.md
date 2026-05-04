# Vertex AI Feature Store
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Vertex AI Feature Store?

Vertex AI Feature Store is a managed, cloud-native feature store service that is integral to Vertex AI. It streamlines ML feature management and online serving by letting you manage your feature data in a **BigQuery table or view**, and then serve feature values online at low latencies for real-time predictions.

**Key benefits:**
- **No separate offline store** — BigQuery is both the offline store and the source of truth
- **No data duplication** — the feature data lives in BigQuery; Feature Store acts as a metadata and serving layer
- **Reusability** — features defined once can be shared across multiple models and teams
- **Consistency** — same features used in training and serving eliminates training-serving skew
- **Point-in-time correctness** — historical feature values preserved via `feature_timestamp` column

**Use cases:** Real-time fraud detection, personalised recommendations, churn prediction, credit scoring, any application needing low-latency feature serving.

---

## 2. Versions and Deprecations (Critical Exam Awareness)

| Version | Status | Sunset Date |
|---------|--------|------------|
| **Feature Store (V2) — Current** | Active — use this | N/A |
| **Feature Store (Legacy / V1)** | Deprecated May 2026 | Feb 17, 2027 |
| **Optimized online serving** | Deprecated May 2026 | Feb 17, 2027 — migrate to Bigtable |

> **Exam tip:** Focus on **Feature Store V2** (BigQuery-backed) with **Bigtable online serving**. The legacy Featurestore (EntityType/FeatureType hierarchy) is being phased out. Optimized online serving is also deprecated — migrate to Bigtable.

---

## 3. Core Architecture

```
BigQuery Table / View
  (offline store — source of truth — history + latest values)
         ↓  sync (scheduled or continuous)
Feature View
  (logical view over BigQuery data — defines what gets served)
         ↓  served via
FeatureOnlineStore
  (Bigtable-backed serving cluster — low-latency reads)
         ↓  accessed via
fetch_feature_values(entity_id)
  (real-time lookup in model serving pipeline)
```

**Optional registration layer:**

```
BigQuery Table
       ↓ registered as
Feature Group  (metadata registration — groups related features)
       ↓ contains
Features       (individual feature columns with metadata)
       ↓ referenced by
Feature View   (can aggregate features from multiple Feature Groups)
```

---

## 4. Core Resources (V2)

| Resource | Description |
|----------|-------------|
| **FeatureOnlineStore** | The online serving cluster (backed by Bigtable). Manages nodes and scaling. |
| **FeatureView** | A logical view of feature data synced from BigQuery into the online store. Defines which features to serve. |
| **FeatureGroup** | Optional metadata layer that registers a BigQuery source and groups related features. Required for monitoring, continuous sync, and null serving. |
| **Feature** | An individual column within a Feature Group — represents one feature variable. |

---

## 5. Full Workflow — Python SDK

### Step 1 — Prepare BigQuery Data Source

```sql
-- Create the feature table in BigQuery
-- Required: entity_id column (STRING type)
-- Optional: feature_timestamp column (TIMESTAMP) for historical data
CREATE TABLE `my-project.ml_features.user_features` (
  user_id           STRING NOT NULL,         -- entity ID column
  feature_timestamp TIMESTAMP,              -- enables point-in-time queries
  total_purchases   INT64,
  avg_order_value   FLOAT64,
  days_since_last   INT64,
  preferred_category STRING,
  lifetime_value    FLOAT64
);
```

### Step 2 — Create a FeatureOnlineStore (Bigtable)

```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Create the online store backed by Bigtable
feature_online_store = aiplatform.FeatureOnlineStore.create_bigtable_store(
    name="user-feature-store",
    min_node_count=1,               # minimum Bigtable nodes (always-on)
    max_node_count=3,               # maximum nodes for autoscaling
    cpu_utilization_target=70,      # autoscale when CPU > 70%
)

print(f"FeatureOnlineStore: {feature_online_store.resource_name}")
```

### Step 3 — (Optional) Register a Feature Group

```python
# Register the BigQuery source as a Feature Group in the Feature Registry
feature_group = aiplatform.FeatureGroup.create(
    name="user-features",
    source=aiplatform.feature_store.utils.FeatureGroupBigQuerySource(
        uri="bq://my-project.ml_features.user_features",
        entity_id_columns=["user_id"],     # column(s) that identify unique entities
    ),
    labels={"team": "ml", "domain": "ecommerce"},
)

# Register individual features within the group
total_purchases_feature = feature_group.create_feature(
    name="total_purchases",
    description="Total number of purchases by this user",
)

avg_order_value_feature = feature_group.create_feature(
    name="avg_order_value",
    description="Average order value in USD",
)
```

### Step 4 — Create a Feature View

```python
# Option A: Feature View directly from BigQuery (no feature group registration)
feature_view = feature_online_store.create_feature_view(
    name="user-feature-view",
    source=aiplatform.feature_store.utils.FeatureViewBigQuerySource(
        uri="bq://my-project.ml_features.user_features",
        entity_id_columns=["user_id"],
    ),
    sync_config=aiplatform.feature_store.utils.FeatureViewSyncConfig(
        cron="0 */6 * * *",    # sync every 6 hours (cron format)
    ),
)

# Option B: Feature View from Feature Groups (enables monitoring + continuous sync)
feature_view = feature_online_store.create_feature_view(
    name="user-feature-view-v2",
    source=aiplatform.feature_store.utils.FeatureViewFeatureRegistrySource(
        features=[
            aiplatform.feature_store.utils.FeatureViewFeatureRegistrySource.Feature(
                feature_group_id="user-features",
                feature_ids=["total_purchases", "avg_order_value", "days_since_last"],
            )
        ]
    ),
    sync_config=aiplatform.feature_store.utils.FeatureViewSyncConfig(
        cron="0 2 * * *",      # sync daily at 2am
    ),
)
```

### Step 5 — Sync Data to Online Store

```python
# Manually trigger a sync (skips waiting for scheduled cron)
sync_response = feature_view.sync()
feature_view.wait_for_sync()     # blocks until sync completes

print("Sync complete — online store is up to date")
```

### Step 6 — Fetch Feature Values Online

```python
# Fetch latest feature values for a single entity
response = feature_view.fetch_feature_values(
    id="user_12345",           # the entity ID to look up
)
print(response)

# Fetch for multiple entity IDs
response = feature_view.fetch_feature_values(
    id=["user_12345", "user_67890"],
)
```

---

## 6. Offline Serving (Batch / Training)

Because all feature data lives in BigQuery, offline serving is just a BigQuery query:

```python
from google.cloud import bigquery

client = bigquery.Client()

# Point-in-time correct feature lookup for training
query = """
SELECT
  f.user_id,
  f.total_purchases,
  f.avg_order_value,
  f.days_since_last,
  l.churned
FROM
  `my-project.ml_features.user_features` f
JOIN
  `my-project.ml_labels.churn_labels` l USING (user_id)
WHERE
  -- Use the feature values as of the label date (point-in-time)
  f.feature_timestamp <= l.label_date
  AND f.feature_timestamp = (
    SELECT MAX(feature_timestamp)
    FROM `my-project.ml_features.user_features`
    WHERE user_id = f.user_id AND feature_timestamp <= l.label_date
  )
"""

training_df = client.query(query).to_dataframe()
```

---

## 7. Data Sync Types

| Sync Type | Description | Requirements |
|-----------|-------------|-------------|
| **Scheduled** | Syncs on a cron schedule. Works with all online store types. | Any FeatureOnlineStore + any FeatureView |
| **Continuous** | Syncs whenever BigQuery source is updated — near real-time | Bigtable online serving + Feature Groups + source in eu / us / us-central1 |

> **Exam tip:** Continuous sync requires **Bigtable** online serving and feature data **registered via Feature Groups** (not direct BigQuery source). Also, continuous sync does not sync deleted or updated records — only new inserts.

---

## 8. Online Serving Types

| Type | Backend | Latency | Status | Notes |
|------|---------|---------|--------|-------|
| **Bigtable** | Cloud Bigtable | ~30ms server-side | **Active** | Recommended — supports large volumes, continuous sync, null serving |
| **Optimized** | Proprietary | Ultra-low | **Deprecated** (Feb 2027) | Migrate to Bigtable |

### Bigtable Autoscaling

```python
# Bigtable nodes autoscale based on CPU utilisation
# min_node_count  = always-on baseline
# max_node_count  = upper limit during traffic spikes
# cpu_utilization_target = trigger threshold (e.g. 70%)
```

---

## 9. Feature Registry

The Feature Registry provides a **searchable, governed catalogue** of all features across the organisation. It is backed by **Dataplex Universal Catalog**.

```
Feature Registry
├── Feature Group A  (maps to a BigQuery table/view)
│   ├── Feature 1 (column: total_purchases)
│   ├── Feature 2 (column: avg_order_value)
│   └── Feature 3 (column: days_since_last)
└── Feature Group B  (maps to another BigQuery table/view)
    ├── Feature 4 (column: device_type)
    └── Feature 5 (column: app_version)
```

**Why register features?**
- Enable **feature monitoring** (drift detection, statistics)
- Enable **continuous data sync**
- Enable **null value serving**
- Aggregate features from **multiple BigQuery sources** into one FeatureView
- Improve **discoverability and governance** across teams

---

## 10. Feature Monitoring

Feature monitoring detects **drift** and **anomalies** in feature distributions over time.

- **Requires** feature registration via Feature Groups
- Computes statistics (mean, std, quantiles) per feature
- Alerts when distributions shift significantly vs. a baseline

```python
# Enable monitoring when creating a Feature Group
feature_group = aiplatform.FeatureGroup.create(
    name="monitored-features",
    source=aiplatform.feature_store.utils.FeatureGroupBigQuerySource(
        uri="bq://my-project.ml_features.user_features",
        entity_id_columns=["user_id"],
    ),
)

# Feature monitoring runs as a scheduled job
# Results visible in the Google Cloud Console → Feature Store → Feature Groups
```

---

## 11. Point-in-Time Correctness

A critical feature store concept. Point-in-time correct training prevents **label leakage** (using future data to train a model that should only know past data).

```
Without point-in-time correctness:
  Model trains with features computed AFTER the label date
  → model sees the future → artificially high accuracy → model fails in production

With point-in-time correctness (feature_timestamp):
  Only feature values BEFORE OR AT the label date are used
  → realistic, leakage-free training data
```

The `feature_timestamp` column in the BigQuery source enables historical lookups:

```sql
-- Point-in-time join: get each user's features as of their churn event date
SELECT
  user_id,
  feature_timestamp,
  total_purchases,
  avg_order_value
FROM (
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY feature_timestamp DESC
    ) AS rn
  FROM `my-project.ml_features.user_features`
  WHERE feature_timestamp <= '2024-06-01'   -- as of the label date
)
WHERE rn = 1;                                -- latest value before cutoff
```

---

## 12. Training-Serving Skew Prevention

Training-serving skew = model trains on features computed differently from how they're served at prediction time. Feature Store prevents this by:

1. **Single source of truth** — training data and serving data both come from the same BigQuery table
2. **Same transformations** — features computed once in BigQuery, not recomputed in serving
3. **Consistent entity IDs** — the same `user_id` maps to the same features in both contexts

---

## 13. Key API Classes — Quick Reference

| Class / Method | Description |
|---------------|-------------|
| `FeatureOnlineStore.create_bigtable_store()` | Create Bigtable-backed online serving cluster |
| `FeatureOnlineStore.create_feature_view()` | Add a feature view to the online store |
| `FeatureGroup.create()` | Register a BigQuery source in the Feature Registry |
| `feature_group.create_feature()` | Register an individual feature column |
| `feature_view.sync()` | Manually trigger a data sync |
| `feature_view.fetch_feature_values(id=...)` | Fetch latest feature values by entity ID |
| `FeatureViewBigQuerySource` | Direct BigQuery source (no feature group required) |
| `FeatureViewFeatureRegistrySource` | Source from registered Feature Groups |
| `FeatureViewSyncConfig(cron=...)` | Set sync schedule using cron expression |

---

## 14. Feature Store V2 vs Legacy (V1)

| Aspect | Feature Store V2 (Current) | Feature Store Legacy (V1 — Deprecated) |
|--------|---------------------------|---------------------------------------|
| Offline store | BigQuery (no separate store) | Managed offline store in Vertex AI |
| Online store | Bigtable-backed FeatureOnlineStore | Featurestore → EntityType serving |
| Core resources | FeatureGroup, Feature, FeatureView, FeatureOnlineStore | Featurestore, EntityType, Feature |
| Sync | Scheduled or continuous (BQ → Bigtable) | Batch ingest from GCS or BQ |
| Data import | BQ is source of truth | Must import data into Featurestore |
| Status | Active | Deprecated — sunset Feb 2027 |
| Historical serving | BQ query with feature_timestamp | Point-in-time lookup via SDK |

---

## 15. Exam-Relevant Tips

- Feature Store V2 uses **BigQuery as both offline store and source of truth** — no separate offline store within Vertex AI.
- The resource hierarchy is: **FeatureOnlineStore → FeatureView** (with optional **FeatureGroup → Feature** registration).
- **Bigtable online serving** is the active, recommended serving type — Optimized is deprecated (Feb 2027).
- **Continuous sync** requires Bigtable online serving + Feature Groups registration + BQ in eu/us/us-central1 regions.
- **Continuous sync only picks up new records** — it does NOT sync updates or deletes.
- **Feature monitoring** requires feature registration via FeatureGroups.
- **Serving null values** requires Feature Groups + Bigtable + scheduled sync.
- `feature_timestamp` enables **point-in-time correct** feature lookups — critical for leakage-free training.
- Training-serving skew is prevented because **both training and serving read from the same BigQuery source**.
- `fetch_feature_values(entity_id)` is the online serving call — returns the latest synced feature values.
- Feature Registry is backed by **Dataplex Universal Catalog** — provides cross-team discoverability.
- One FeatureView can aggregate features from **multiple Feature Groups** — enabling cross-source feature joins.
- Legacy Feature Store (V1) had an **EntityType** hierarchy — this is deprecated, do not confuse with V2.

---

## 16. Quick Reference Cheat Sheet

```
CORE RESOURCES
  FeatureOnlineStore        →  Bigtable-backed serving cluster
  FeatureView               →  logical view of BQ data served online
  FeatureGroup              →  registered BQ source (optional metadata layer)
  Feature                   →  individual column within a Feature Group

WORKFLOW
  1. Prepare BQ table (entity_id col required, feature_timestamp optional)
  2. create_bigtable_store() → FeatureOnlineStore
  3. FeatureGroup.create()   → register source (optional)
  4. create_feature_view()  → link BQ to online store + set cron sync
  5. feature_view.sync()    → trigger initial sync
  6. fetch_feature_values() → serve features at prediction time

SYNC TYPES
  Scheduled (cron)         →  works with all store types
  Continuous               →  Bigtable + Feature Groups + restricted regions
                              new inserts only (no updates/deletes)

ONLINE SERVING
  Bigtable                 →  ~30ms latency, large volumes, active ✅
  Optimized                →  deprecated — migrate to Bigtable ❌

OFFLINE SERVING
  BigQuery query           →  direct BQ query for training data
  feature_timestamp        →  enables point-in-time correct joins

KEY CONCEPTS
  Training-serving skew    →  prevented by single BQ source of truth
  Point-in-time correct    →  use feature_timestamp for leakage-free training
  Feature Registry         →  Dataplex-backed catalogue for governance
  Feature monitoring       →  drift detection — requires Feature Groups
```

---

*Study tip: Know the two-layer architecture (FeatureGroup/Feature → optional registration; FeatureOnlineStore/FeatureView → mandatory for serving), the difference between scheduled and continuous sync, why BigQuery is the offline store, and how point-in-time correctness prevents label leakage — all heavily tested areas for the GCP ML Engineer exam.*