# Time-Series Forecasting in BigQuery ML
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is Time-Series Forecasting?

Time-series forecasting predicts **future numeric values** based on historical observations ordered in time. Unlike standard regression, the temporal ordering of data is fundamental — past values directly influence future ones.

**Use cases:** Demand forecasting, sales prediction, capacity planning, financial forecasting, energy consumption, anomaly detection in time series.

---

## 2. ARIMA_PLUS — The Primary Model

BigQuery ML implements **ARIMA_PLUS**, an enhanced version of the classical ARIMA model. It extends traditional ARIMA with:

- **Automatic model selection** (auto-ARIMA) — searches for optimal (p, d, q) parameters
- **Holiday effects** — models known calendar events
- **Seasonal decomposition** — handles multiple seasonality patterns
- **Trend modelling** — linear, non-linear drift
- **Spike and dip anomaly** — detects and adjusts for outliers

```
ARIMA = AutoRegressive Integrated Moving Average
p = AR order  (how many past values to use)
d = I  order  (differencing to make series stationary)
q = MA order  (how many past forecast errors to use)
```

---

## 3. ARIMA_PLUS_XREG — With External Regressors

`ARIMA_PLUS_XREG` extends `ARIMA_PLUS` by allowing **exogenous (external) variables** — additional features that help explain the time series beyond its own history.

```
ARIMA_PLUS      →  univariate (time + label only)
ARIMA_PLUS_XREG →  multivariate (time + label + external features)
```

**Example:** Forecasting ice cream sales using temperature and day-of-week as external regressors.

---

## 4. Full Workflow

### Step 1 — Create the Model

```sql
CREATE OR REPLACE MODEL `project.dataset.my_arima_model`
OPTIONS (
  model_type        = 'ARIMA_PLUS',
  time_series_timestamp_col = 'date_col',      -- TIMESTAMP, DATE or DATETIME
  time_series_data_col      = 'sales',         -- the numeric label to forecast
  time_series_id_col        = 'store_id',      -- optional: one model per series
  horizon                   = 30,              -- how many future periods to forecast
  auto_arima                = TRUE,            -- auto-search (p,d,q) params
  data_frequency            = 'AUTO_FREQUENCY',-- or 'DAILY', 'WEEKLY', etc.
  decompose_time_series     = TRUE,            -- extract trend + seasonality
  clean_spikes_and_dips     = TRUE,            -- handle anomalies in training data
  adjust_step_changes       = TRUE,            -- handle structural breaks
  holiday_region            = 'US'             -- model public holidays
) AS
SELECT
  date_col,
  store_id,
  sales
FROM
  `project.dataset.sales_history`
ORDER BY date_col;
```

### ARIMA_PLUS_XREG (with external features)

```sql
CREATE OR REPLACE MODEL `project.dataset.my_arima_xreg_model`
OPTIONS (
  model_type                = 'ARIMA_PLUS_XREG',
  time_series_timestamp_col = 'date_col',
  time_series_data_col      = 'sales',
  time_series_id_col        = 'store_id',
  horizon                   = 14,
  auto_arima                = TRUE,
  data_frequency            = 'DAILY',
  holiday_region            = 'GB'
) AS
SELECT
  date_col,
  store_id,
  sales,
  temperature,          -- external regressor 1
  is_promotion          -- external regressor 2
FROM
  `project.dataset.sales_history`
ORDER BY date_col;
```

### Step 2 — Evaluate the Model

```sql
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.my_arima_model`);
-- No input table needed for ARIMA_PLUS — uses held-out window from training data
```

### Step 3 — Inspect Model Components

```sql
-- View ARIMA coefficients and model fit statistics
SELECT *
FROM ML.ARIMA_COEFFICIENTS(MODEL `project.dataset.my_arima_model`);
```

```sql
-- Decompose the series into trend + seasonal + residual
SELECT *
FROM ML.EXPLAIN_FORECAST(
  MODEL `project.dataset.my_arima_model`,
  STRUCT(30 AS horizon, 0.9 AS confidence_level)
);
```

### Step 4 — Make Forecasts

```sql
SELECT *
FROM ML.FORECAST(
  MODEL `project.dataset.my_arima_model`,
  STRUCT(
    30  AS horizon,           -- periods ahead to forecast
    0.8 AS confidence_level   -- prediction interval width (0–1)
  )
);
```

**Output columns:**

| Column | Description |
|--------|-------------|
| `forecast_timestamp` | Future timestamp for this prediction |
| `forecast_value` | Point forecast (predicted value) |
| `standard_error` | Standard error of the forecast |
| `confidence_level` | The confidence level specified |
| `prediction_interval_lower_bound` | Lower bound of the interval |
| `prediction_interval_upper_bound` | Upper bound of the interval |
| `confidence_interval_lower_bound` | Alias for lower bound |
| `confidence_interval_upper_bound` | Alias for upper bound |

### ARIMA_PLUS_XREG Forecast (requires future regressor values)

```sql
SELECT *
FROM ML.FORECAST(
  MODEL `project.dataset.my_arima_xreg_model`,
  (
    -- Must supply future values of external regressors
    SELECT date_col, store_id, temperature, is_promotion
    FROM `project.dataset.future_features`
  ),
  STRUCT(0.9 AS confidence_level)
);
```

---

## 5. Evaluation Metrics

```sql
SELECT *
FROM ML.EVALUATE(MODEL `project.dataset.my_arima_model`);
```

| Metric | Description | Goal |
|--------|-------------|------|
| `mean_absolute_error` (MAE) | Average absolute deviation | Lower |
| `mean_squared_error` (MSE) | Average squared deviation | Lower |
| `root_mean_squared_error` (RMSE) | Square root of MSE — same units as label | Lower |
| `mean_absolute_percentage_error` (MAPE) | % deviation from actual — scale-independent | Lower |
| `symmetric_mean_absolute_percentage_error` (sMAPE) | Symmetric MAPE — treats over/under equally | Lower |
| `median_absolute_error` | Median of absolute errors — robust to outliers | Lower |
| `r2_score` | Variance explained | Higher (closer to 1) |
| `AIC` | Akaike Information Criterion — model complexity penalty | Lower |

> **MAPE** is particularly useful for comparing forecast accuracy across series with different scales.
> **AIC** helps compare competing ARIMA configurations — lower AIC = better model fit penalised for complexity.

---

## 6. Key OPTIONS Parameters

### Required Parameters

| Option | Description |
|--------|-------------|
| `model_type` | `'ARIMA_PLUS'` or `'ARIMA_PLUS_XREG'` |
| `time_series_timestamp_col` | Column containing the time dimension (DATE, DATETIME, TIMESTAMP) |
| `time_series_data_col` | Column containing the numeric value to forecast |

### Optional but Important Parameters

| Option | Default | Description |
|--------|---------|-------------|
| `time_series_id_col` | — | Group column — trains one model per unique value (multiple series) |
| `horizon` | `1000` | Number of future periods to forecast |
| `auto_arima` | `TRUE` | Automatically search for optimal (p, d, q) orders |
| `auto_arima_max_order` | `5` | Max sum of p + q when searching |
| `data_frequency` | `'AUTO_FREQUENCY'` | Granularity: `DAILY`, `WEEKLY`, `MONTHLY`, `QUARTERLY`, `YEARLY`, `AUTO_FREQUENCY` |
| `decompose_time_series` | `TRUE` | Extract and model trend, seasonality separately |
| `clean_spikes_and_dips` | `TRUE` | Detect and smooth anomalies before fitting |
| `adjust_step_changes` | `TRUE` | Detect and adjust for structural level shifts |
| `holiday_region` | — | Model public holidays: `'US'`, `'GB'`, `'JP'`, etc. |
| `is_test_set` | — | Boolean column to mark rows as test (for custom evaluation) |

### ARIMA Orders (manual override of auto-ARIMA)

| Option | Default | Description |
|--------|---------|-------------|
| `p` | auto | AR order — number of lag observations |
| `d` | auto | Differencing order — to achieve stationarity |
| `q` | auto | MA order — number of lagged forecast errors |

---

## 7. Multiple Time Series (time_series_id_col)

One of the most powerful BQML features: train **one model per time series in a single SQL statement**.

```sql
CREATE OR REPLACE MODEL `project.dataset.multi_series_model`
OPTIONS (
  model_type                = 'ARIMA_PLUS',
  time_series_timestamp_col = 'date_col',
  time_series_data_col      = 'revenue',
  time_series_id_col        = 'product_id',   -- one model per product
  horizon                   = 7,
  auto_arima                = TRUE,
  data_frequency            = 'DAILY'
) AS
SELECT date_col, product_id, revenue
FROM `project.dataset.revenue_history`;
```

Then forecast all series at once:

```sql
SELECT *
FROM ML.FORECAST(
  MODEL `project.dataset.multi_series_model`,
  STRUCT(7 AS horizon, 0.9 AS confidence_level)
)
ORDER BY time_series_id, forecast_timestamp;
```

> **Exam tip:** Each unique value in `time_series_id_col` gets its own independent ARIMA model. Output includes a `time_series_id` column identifying which series each forecast row belongs to.

---

## 8. ML.EXPLAIN_FORECAST — Decomposition

`ML.EXPLAIN_FORECAST` breaks the forecast into interpretable components:

```sql
SELECT *
FROM ML.EXPLAIN_FORECAST(
  MODEL `project.dataset.my_arima_model`,
  STRUCT(30 AS horizon, 0.9 AS confidence_level)
);
```

**Output components:**

| Component | Description |
|-----------|-------------|
| `trend` | Long-term direction of the series |
| `seasonal_period_yearly` | Annual seasonal pattern |
| `seasonal_period_weekly` | Weekly seasonal pattern |
| `seasonal_period_daily` | Intra-day seasonal pattern |
| `holiday_effect` | Impact of public holidays |
| `spikes_and_dips` | Detected anomaly adjustments |
| `step_changes` | Structural level-shift adjustments |
| `residual` | Unexplained remainder |
| `prediction_interval_lower_bound` | Lower confidence bound |
| `prediction_interval_upper_bound` | Upper confidence bound |

---

## 9. data_frequency Values

| Value | Description |
|-------|-------------|
| `AUTO_FREQUENCY` | BQML infers frequency from the data |
| `PER_MINUTE` | Minute-level granularity |
| `HOURLY` | Hourly |
| `DAILY` | Daily |
| `WEEKLY` | Weekly |
| `MONTHLY` | Monthly |
| `QUARTERLY` | Quarterly |
| `YEARLY` | Annual |

---

## 10. Stationarity and Differencing

ARIMA requires the time series to be **stationary** (constant mean, variance, autocorrelation over time). The `d` parameter controls the number of times the series is differenced to achieve stationarity.

```
d = 0  →  series is already stationary
d = 1  →  first-order differencing (removes linear trend)
d = 2  →  second-order differencing (removes quadratic trend)
```

With `auto_arima = TRUE`, BQML runs the **KPSS test** to determine the required differencing order automatically.

---

## 11. Holiday Regions Supported

| Code | Region |
|------|--------|
| `US` | United States |
| `GB` | Great Britain |
| `JP` | Japan |
| `IN` | India |
| `CA` | Canada |
| `AU` | Australia |
| `DE` | Germany |
| `FR` | France |
| `IT` | Italy |
| `BR` | Brazil |

---

## 12. ML Functions — Quick Reference

| Function | Description |
|----------|-------------|
| `ML.EVALUATE` | Returns MAE, MSE, RMSE, MAPE, sMAPE, AIC, R² |
| `ML.FORECAST` | Generates future predictions with confidence intervals |
| `ML.EXPLAIN_FORECAST` | Decomposes forecast into trend, seasonal, holiday, residual components |
| `ML.ARIMA_COEFFICIENTS` | Returns fitted AR, MA, differencing coefficients |
| `ML.ARIMA_EVALUATE` | Evaluates multiple candidate ARIMA orders (when `auto_arima = TRUE`) |

---

## 13. ARIMA_PLUS vs ARIMA_PLUS_XREG

| Aspect | ARIMA_PLUS | ARIMA_PLUS_XREG |
|--------|-----------|-----------------|
| External features | No | Yes |
| Forecast input | No input table needed | Future regressor values required |
| Use case | Pure time-series | Time-series + external drivers |
| Complexity | Lower | Higher |
| Future data requirement | None | Must provide future regressors |

---

## 14. Exam-Relevant Tips

- `time_series_timestamp_col` must be a **DATE, DATETIME, or TIMESTAMP** type — not a STRING.
- `ML.EVALUATE` for ARIMA_PLUS **does not require an input table** — it uses a held-out window from training.
- `ML.FORECAST` requires `STRUCT(horizon AS horizon, confidence_level AS confidence_level)`.
- `time_series_id_col` enables training **multiple independent ARIMA models** in a single `CREATE MODEL`.
- `ARIMA_PLUS_XREG` requires future values of external regressors to be passed into `ML.FORECAST`.
- `data_frequency = 'AUTO_FREQUENCY'` lets BQML infer the cadence — use when unsure.
- `clean_spikes_and_dips = TRUE` adjusts training data anomalies **before** fitting — improves model quality.
- `decompose_time_series = TRUE` enables `ML.EXPLAIN_FORECAST` component breakdown.
- `holiday_region` only affects the model if `decompose_time_series = TRUE`.
- **MAPE** is scale-independent — best for comparing accuracy across series with different magnitudes.
- **AIC** (Akaike Information Criterion) is used by `auto_arima` to select the best (p, d, q) combination.
- The `SEQ` data split method (from other BQML models) is **not applicable** to ARIMA_PLUS — it has its own internal time-aware evaluation.
- ARIMA_PLUS does **not** use `ML.PREDICT` — it uses `ML.FORECAST` instead.
- Output of `ML.FORECAST` always includes **prediction interval bounds** — not just a point estimate.

---

## 15. Quick Reference Cheat Sheet

```
model_type              →  'ARIMA_PLUS' or 'ARIMA_PLUS_XREG'
time_series_timestamp_col →  required: DATE/DATETIME/TIMESTAMP column
time_series_data_col    →  required: numeric column to forecast
time_series_id_col      →  optional: one model per unique ID value
horizon                 →  how many future periods to predict
ML.FORECAST             →  generate future predictions (NOT ML.PREDICT)
ML.EVALUATE             →  MAE, RMSE, MAPE, sMAPE, AIC — no input table needed
ML.EXPLAIN_FORECAST     →  decompose into trend, seasonal, holiday, residual
ML.ARIMA_COEFFICIENTS   →  view fitted (p,d,q) coefficients
```

---

*Study tip: Time-series in BigQuery ML has unique functions (`ML.FORECAST`, `ML.EXPLAIN_FORECAST`, `ML.ARIMA_COEFFICIENTS`) not used in other model types — and critically uses `ML.FORECAST` instead of `ML.PREDICT`. This distinction is frequently tested.*