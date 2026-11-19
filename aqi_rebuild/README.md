# AQI Intelligence - Rebuilt Production Platform

A complete, production-grade ambient air quality monitoring and machine learning forecasting platform for Delhi NCR.

---

## 🌟 Key Capabilities

1. **Live OpenAQ Ingestion Pipeline**: Connects directly to verified OpenAQ API v3 endpoints across Delhi Continuous Ambient Air Quality Monitoring Stations (CAAQMS).
2. **Exact 45-Feature Engineering**: Reconstructs the exact temporal memory, circular wind components, missingness indicators, and calendar cycles required by the current models.
3. **Multi-Horizon ML Forecasting**: Serves 1-hour ahead and 3-hour ahead probabilistic AQI predictions using the trained Gradient Boosting classification pipelines.
4. **Pollution Alert & Early Warning System**: Real-time alerts for Severe AQI emergencies, acute rate-of-rise surges, and impending deterioration.
5. **Interactive Streamlit Analytics Dashboard**:
   - **Live City Monitoring**: Delhi-wide KPIs, interactive spatial station map, and live sensor grid.
   - **Station Deep-Dive**: 1h & 3h predicted category probabilities, multi-pollutant radar profile, and health advisories.
   - **Station Comparisons & Regional Analytics**: Cross-station pollutant benchmarks and regional rankings.
   - **Historical Trends**: Continuous timeseries with 6h, 12h, 24h rolling means and diurnal cycle analysis.
   - **ML Leaderboard & Diagnostics**: 6-model comparison, confusion matrices, and 45-feature importance charts.
   - **Pollution Alerts**: Active warning feed and Graded Response Action Plan (GRAP) recommendations.

---

## 🏗️ Architecture & Directory Layout

```
aqi_rebuild/
├── config/
│   ├── settings.py           # Paths, OpenAQ credentials, AQI breakpoints, styling
│   └── stations.json         # Complete 15+ Delhi station metadata & sensor IDs
├── pipeline/
│   ├── openaq_fetcher.py     # Live OpenAQ API v3 client with sensor lookup & rate-limiting
│   ├── preprocessor.py       # Manages continuous timeseries buffer and ingestion
│   ├── feature_engineer.py   # Computes the EXACT 45 features expected by the models
│   └── predictor.py          # 1h & 3h model inference, confidence scores, and trajectory
├── services/
│   ├── alert_service.py      # Severity thresholds, rapid spikes, and health advisories
│   └── leaderboard_service.py# Model leaderboard, confusion matrices, and feature importance
├── data/
│   ├── live_readings_buffer.csv
│   ├── latest_predictions.json
│   └── alert_history.json
├── dashboard/
│   ├── app.py                # Main interactive Streamlit analytics dashboard
│   └── components.py         # Reusable Plotly charts, maps, and UI metric cards
├── tests/
│   ├── test_features.py      # Verifies 45 features match models strictly
│   ├── test_models.py        # Verifies 1h and 3h model inference
│   ├── test_alerts.py        # Verifies emergency alerts & spike detection
│   └── test_openaq_live.py   # Verifies live OpenAQ API query
├── run_pipeline.py           # CLI runner for live fetch -> feature engineering -> prediction
├── start_dashboard.py        # Shortcut runner to launch the Streamlit dashboard
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Run the Live Pipeline

To fetch the latest verified sensor data from OpenAQ, calculate all 45 features, and run 1-hour and 3-hour forecasts:

```bash
python aqi_rebuild/run_pipeline.py
```

*To run using local buffered data without an external API query:*
```bash
python aqi_rebuild/run_pipeline.py --no-fetch
```

### 2. Launch the Streamlit Dashboard

Launch the analytics dashboard in your browser:

```bash
streamlit run aqi_rebuild/dashboard/app.py
```
*Or via the helper script:*
```bash
python aqi_rebuild/start_dashboard.py
```

### 3. Run the Test Suite

Verify pipeline integrity, feature alignment, and inference:

```bash
python aqi_rebuild/tests/test_features.py
python aqi_rebuild/tests/test_models.py
python aqi_rebuild/tests/test_alerts.py
```

---

## 🔬 Model & Feature Specifications

- **Features (45)**:
  - Pollutants: `co`, `no`, `no2`, `nox`, `o3`, `pm10`, `pm25`, `so2`
  - Meteorology: `temperature`, `relative_humidity`, `wind_speed`, `wind_direction_sin`, `wind_direction_cos`
  - Missingness Flags: `PM25_Missing`, `co_missing`, `no_missing`, `no2_missing`, `nox_missing`, `o3_missing`, `pm10_missing`, `relative_humidity_missing`, `so2_missing`, `temperature_missing`, `wind_direction_missing`, `wind_speed_missing`
  - Temporal & Cycles: `hour`, `day`, `day_of_week`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`
  - Lags & Memory: `pm25_lag_1`, `pm25_lag_3`, `pm25_lag_6`, `pm25_lag_12`, `pm25_lag_24`
  - Rolling Stats: `pm25_rolling_mean_6h`, `pm25_rolling_mean_12h`, `pm25_rolling_mean_24h`, `pm25_rolling_std_24h`, `pm25_24h_avg`
  - Categorical: `candidate_station`
- **Classes (6)**: Good, Satisfactory, Moderate, Poor, Very Poor, Severe
- **Horizons**: 1-hour ahead (`best_model_1h.joblib`), 3-hours ahead (`best_model_3h.joblib`)
