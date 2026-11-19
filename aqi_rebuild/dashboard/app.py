"""
AQI Intelligence Analytics Dashboard.
Modern, interactive, multi-view Streamlit dashboard for real-time Delhi air quality monitoring,
45-feature ML forecasts (1-hour & 3-hour), station comparisons, historical trends,
model diagnostics & leaderboard, and practical pollution alerts.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure path so dashboard can import pipeline and services
DASHBOARD_DIR = Path(__file__).resolve().parent
REBUILD_DIR = DASHBOARD_DIR.parent
PROJECT_ROOT = REBUILD_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REBUILD_DIR) not in sys.path:
    sys.path.insert(0, str(REBUILD_DIR))

from aqi_rebuild.config.settings import (
    LIVE_READINGS_BUFFER,
    LATEST_PREDICTIONS_JSON,
    AQI_CATEGORIES,
    ORDERED_CATEGORIES,
    TRAINED_STATIONS,
    CONFUSION_GB_1H,
    CONFUSION_GB_3H,
    CONFUSION_RF_1H,
    CONFUSION_RF_3H,
)
from aqi_rebuild.pipeline.preprocessor import TimeseriesPreprocessor
from aqi_rebuild.services.alert_service import PollutionAlertService
from aqi_rebuild.services.leaderboard_service import LeaderboardService
from aqi_rebuild.dashboard.components import (
    get_category_color,
    render_aqi_badge,
    create_delhi_map,
    create_probability_distribution_chart,
    create_historical_trend_chart,
    create_pollutant_radar_chart,
    create_feature_importance_chart,
)

# Page configuration
st.set_page_config(
    page_title="AQI Intelligence | Delhi Air Quality Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished typography, metric cards, and clean styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1a1a1a;
        margin-top: 4px;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #495057;
        margin-top: 2px;
    }
    .alert-box {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_predictions_data():
    """Loads latest forecasts and station measurements."""
    if not LATEST_PREDICTIONS_JSON.exists():
        return {}
    try:
        with open(LATEST_PREDICTIONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_buffer_history():
    """Loads timeseries buffer for trend and comparison charts."""
    preprocessor = TimeseriesPreprocessor()
    return preprocessor.load_buffer()


# Initialize services
alert_service = PollutionAlertService()
leaderboard_service = LeaderboardService()

# Load data
pred_data = load_predictions_data()
history_df = load_buffer_history()
stations_list = pred_data.get("stations", [])

# ==========================================
# SIDEBAR CONTROLS & PIPELINE TRIGGER
# ==========================================
st.sidebar.title("🌫️ AQI Intelligence")
st.sidebar.caption("Delhi Continuous Air Quality & ML Forecasting System")

# Refresh / Pipeline Execution Button
if st.sidebar.button("🔄 Sync Live OpenAQ Pipeline", use_container_width=True):
    with st.sidebar.status("Connecting to OpenAQ API v3...", expanded=True) as status:
        try:
            from aqi_rebuild.run_pipeline import run_full_pipeline
            st.write("Fetching sensor feeds for Delhi stations...")
            new_preds = run_full_pipeline(fetch_live=True)
            status.update(label="Pipeline synchronized successfully!", state="complete", expanded=False)
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            status.update(label=f"Pipeline error: {e}", state="error")
            st.sidebar.error(str(e))

st.sidebar.markdown("---")

# Pipeline status info
gen_time = pred_data.get("generated_at", "Not yet executed")
st.sidebar.write(f"**Last Sync:** `{gen_time}`")
trained_count = sum(1 for s in stations_list if s.get("is_seen_in_training"))
unseen_count = len(stations_list) - trained_count
st.sidebar.write(f"**Monitoring Stations:** `{len(stations_list)}` ({trained_count} ML-Forecasted, {unseen_count} Live Only)")
st.sidebar.write(f"**Active ML Pipeline:** `45-Feature GBDT`")
st.sidebar.write(f"**Horizons:** `1-Hour` & `3-Hour`")

# Station & Region Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filter & Focus")
regions = ["All Regions"] + sorted(list({s.get("region", "Delhi") for s in stations_list if s.get("region")}))
selected_region = st.sidebar.selectbox("Filter by Region", regions)

filtered_stations = stations_list
if selected_region != "All Regions":
    filtered_stations = [s for s in stations_list if s.get("region") == selected_region]

station_names = [s.get("candidate_station", s.get("station_name")) for s in filtered_stations]

def format_station_label(name: str) -> str:
    if not name or name == "None":
        return name
    is_seen = any(s.get("candidate_station") == name and s.get("is_seen_in_training") for s in filtered_stations)
    return f"{name} [ML Forecast]" if is_seen else f"{name} [Live Only]"

selected_station_name = st.sidebar.selectbox(
    "Select Station Deep-Dive",
    station_names if station_names else ["None"],
    format_func=format_station_label
)

# Find selected station object
selected_station = next((s for s in filtered_stations if s.get("candidate_station") == selected_station_name or s.get("station_name") == selected_station_name), None)

# Auto-refresh note
st.sidebar.markdown("---")
st.sidebar.caption("Data source: Verified OpenAQ API v3 CAAQMS sensors. ML models: 45-feature gradient boosted decision trees.")

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
tabs = st.tabs([
    "🏙️ Live City Monitoring",
    "🔮 Station Forecasts",
    "📊 Station Comparisons",
    "📈 Historical Trends",
    "🏆 Model Leaderboard & Diagnostics",
    "🚨 Pollution Alerts & Health Advisories"
])

# ------------------------------------------------------------------
# TAB 1: LIVE CITY MONITORING & DELHI MAP
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Delhi Ambient Air Quality Live Monitoring")
    st.caption("Real-time measurements from Continuous Ambient Air Quality Monitoring Stations (CAAQMS)")

    if not stations_list:
        st.warning("No live predictions found. Click 'Sync Live OpenAQ Pipeline' in the sidebar to initialize.")
    else:
        # Citywide KPI summary (strictly based on verified live feeds)
        live_stations = [s for s in stations_list if s.get("is_current_live") and s.get("current_pm25") is not None]
        valid_pm25 = [s["current_pm25"] for s in live_stations]
        avg_pm25 = np.mean(valid_pm25) if valid_pm25 else 0.0

        severe_count = sum(1 for s in live_stations if s.get("current_aqi_category") == "Severe")
        very_poor_count = sum(1 for s in live_stations if s.get("current_aqi_category") == "Very Poor")

        sorted_by_pm25 = sorted(live_stations, key=lambda x: x["current_pm25"], reverse=True)
        worst_st = sorted_by_pm25[0] if sorted_by_pm25 else None
        best_st = sorted_by_pm25[-1] if sorted_by_pm25 else None

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Citywide Average PM2.5</div>
                <div class="metric-value">{avg_pm25:.1f} <span style="font-size:1rem; font-weight:normal;">µg/m³</span></div>
                <div class="metric-sub">Across {len(live_stations)} verified live stations</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Highest Pollution Hotspot</div>
                <div class="metric-value" style="color: #FF0000;">{worst_st['candidate_station'] if worst_st else 'N/A'}</div>
                <div class="metric-sub">{worst_st['current_pm25']:.1f} µg/m³ ({worst_st['current_aqi_category']})</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Cleanest Station</div>
                <div class="metric-value" style="color: #00A000;">{best_st['candidate_station'] if best_st else 'N/A'}</div>
                <div class="metric-sub">{best_st['current_pm25']:.1f} µg/m³ ({best_st['current_aqi_category']})</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Active Live Network</div>
                <div class="metric-value">{len(live_stations)} <span style="font-size:1rem; font-weight:normal;">/ {len(stations_list)}</span></div>
                <div class="metric-sub">{severe_count} Severe | {very_poor_count} Very Poor</div>
            </div>
            """, unsafe_allow_html=True)

        # Spatial Map
        st.subheader("Spatial Air Quality Map across Delhi NCR")
        map_fig = create_delhi_map(filtered_stations)
        st.plotly_chart(map_fig, use_container_width=True)

        # Live Station Grid Table
        st.subheader("Live Station Sensor Readings & Predictions")
        table_rows = []
        for s in filtered_stations:
            pm25 = s.get("current_pm25")
            freshness_label = s.get("freshness_label", "⚪ Offline")
            training_status = "✅ ML Forecast (13)" if s.get("is_seen_in_training") else "📊 Monitoring Only"
            cur_cat = s.get("current_aqi_category", "N/A")
            f1 = s.get("forecast_1h", {})
            f3 = s.get("forecast_3h", {})
            meas = s.get("measurements", {})
            table_rows.append({
                "Station": s.get("candidate_station"),
                "Feed Freshness": freshness_label,
                "Observed (IST)": str(s.get("last_observed_at_ist", ""))[:16],
                "Feed Age": s.get("observation_age_text", "N/A"),
                "Training Model": training_status,
                "PM2.5 (µg/m³)": round(pm25, 1) if pm25 is not None else np.nan,
                "Current AQI": cur_cat,
                "1h Forecast": f1.get("predicted_category", "N/A"),
                "1h Confidence": f"{f1['confidence']*100:.1f}%" if f1.get('confidence') is not None else "N/A",
                "3h Forecast": f3.get("predicted_category", "N/A"),
                "3h Confidence": f"{f3['confidence']*100:.1f}%" if f3.get('confidence') is not None else "N/A",
                "Trend": s.get("trajectory", "Stable"),
                "PM10": meas.get("pm10"),
                "NO2": meas.get("no2"),
                "Temp (°C)": meas.get("temperature"),
                "Humidity (%)": meas.get("relative_humidity")
            })

        tbl_df = pd.DataFrame(table_rows).sort_values("PM2.5 (µg/m³)", ascending=False)
        st.dataframe(tbl_df, use_container_width=True, height=360)

# ------------------------------------------------------------------
# TAB 2: STATION FORECASTS & DEEP-DIVE
# ------------------------------------------------------------------
with tabs[1]:
    if not selected_station:
        st.info("Please select a station in the sidebar to view detailed forecasts.")
    else:
        st.header(f"Station Deep-Dive: {selected_station['candidate_station']}")
        st.caption(f"Full Name: {selected_station['station_name']} | Region: {selected_station.get('region')} | Coordinates: ({selected_station.get('latitude'):.4f}, {selected_station.get('longitude'):.4f})")

        # 1. Freshness & Inactivity Warnings
        if selected_station.get("is_delayed"):
            st.warning(
                f"🟡 **Delayed Feed Notice:** The latest reading for this station was observed at "
                f"**{selected_station.get('last_observed_at_ist')}** ({selected_station.get('observation_age_text')}), "
                "which exceeds the 3-hour live freshness threshold. "
                "The observation is displayed for analytical reference but is not real-time current."
            )
        elif selected_station.get("is_severely_stale") or not selected_station.get("current_pm25"):
            st.error(
                "🔴 **Inactive / Stale Sensor Notice:** This station has no verified OpenAQ feed within the last "
                "48 hours. Historical reference dataset rows are strictly segregated as background baseline "
                "context and are never presented as live measurements."
            )

        # 2. Training Coverage Status
        is_seen = selected_station.get("is_seen_in_training", True)
        if not is_seen:
            st.info(
                f"📊 **Live Monitoring Station (Not Included in ML Training):** Station **'{selected_station.get('candidate_station')}'** "
                "is an active CAAQMS monitoring site with live pollutant measurements and alerts. "
                "However, the ML models were trained strictly on 13 specific reference stations. To maintain model integrity, "
                "1-hour and 3-hour ML forward forecasts are disabled for this station."
            )

        f1 = selected_station.get("forecast_1h", {})
        f3 = selected_station.get("forecast_3h", {})
        curr_pm25 = selected_station.get("current_pm25")
        curr_cat = selected_station.get("current_aqi_category", "Unavailable")
        p1_cat = f1.get("predicted_category", "Unavailable")
        p3_cat = f3.get("predicted_category", "Unavailable")
        trajectory = selected_station.get("trajectory", "Stable")
        age_text = selected_station.get("observation_age_text", "N/A")
        obs_ist = selected_station.get("last_observed_at_ist", "N/A")

        # 4 Key Metrics for this station
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            pm_display = f"{curr_pm25:.1f} <span style='font-size:1rem; font-weight:normal;'>µg/m³</span>" if curr_pm25 is not None else "<span style='color:#888;'>Offline</span>"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Current PM2.5 Concentration</div>
                <div class="metric-value">{pm_display}</div>
                <div class="metric-sub">Observed: <b>{obs_ist}</b> ({age_text})</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            if is_seen and p1_cat != "Offline":
                p1_disp = p1_cat
                p1_color = get_category_color(p1_cat)
                p1_sub = f"Confidence: <b>{f1['confidence']*100:.1f}%</b>" if f1.get("confidence") is not None else "Confidence: N/A"
            elif is_seen and p1_cat == "Offline":
                p1_disp = "<span style='font-size:1.1rem; color:#888;'>Offline</span>"
                p1_color = "#888888"
                p1_sub = "Sensor inactive"
            else:
                p1_disp = "<span style='font-size:0.95rem; color:#6c757d; font-weight:600;'>Forecast Unavailable</span>"
                p1_color = "#6c757d"
                p1_sub = "Station not in training dataset"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">1-Hour Ahead Forecast</div>
                <div class="metric-value" style="color: {p1_color};">{p1_disp}</div>
                <div class="metric-sub">{p1_sub}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            if is_seen and p3_cat != "Offline":
                p3_disp = p3_cat
                p3_color = get_category_color(p3_cat)
                p3_sub = f"Confidence: <b>{f3['confidence']*100:.1f}%</b>" if f3.get("confidence") is not None else "Confidence: N/A"
            elif is_seen and p3_cat == "Offline":
                p3_disp = "<span style='font-size:1.1rem; color:#888;'>Offline</span>"
                p3_color = "#888888"
                p3_sub = "Sensor inactive"
            else:
                p3_disp = "<span style='font-size:0.95rem; color:#6c757d; font-weight:600;'>Forecast Unavailable</span>"
                p3_color = "#6c757d"
                p3_sub = "Station not in training dataset"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">3-Hour Ahead Forecast</div>
                <div class="metric-value" style="color: {p3_color};">{p3_disp}</div>
                <div class="metric-sub">{p3_sub}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            if is_seen:
                traj_color = "#D9534F" if trajectory == "Deteriorating" else ("#5CB85C" if trajectory == "Improving" else "#F0AD4E")
                train_badge = "✅ Seen in Training"
                traj_val = trajectory
            else:
                traj_color = "#17a2b8"
                train_badge = "📊 Monitoring Only"
                traj_val = "Monitoring Only"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Trajectory & Model Status</div>
                <div class="metric-value" style="color: {traj_color}; font-size:1.3rem;">{traj_val}</div>
                <div class="metric-sub">{train_badge}</div>
            </div>
            """, unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("Forecasted Category Probability Distributions")
            if is_seen and f1.get("probabilities") and f3.get("probabilities"):
                prob_fig = create_probability_distribution_chart(f1, f3)
                st.plotly_chart(prob_fig, use_container_width=True)
            elif is_seen:
                st.info("Probability distributions unavailable while station sensor is offline.")
            else:
                st.info(
                    "🔮 **ML Forecasting Disabled for Unseen Station**\n\n"
                    f"Station **'{selected_station_name}'** was not among the 13 reference monitoring stations "
                    "in the model training dataset (Alipur, Anand Vihar, Bawana, Dwarka, ITO, Mandir Marg, Mundka, "
                    "Patparganj, Punjabi Bagh, Pusa, RK Puram, Rohini, Vivek Vihar).\n\n"
                    "• **1-Hour Horizon:** *Forecast unavailable — station not in training*\n"
                    "• **3-Hour Horizon:** *Forecast unavailable — station not in training*\n\n"
                    "No fake forecasts, probabilities, or confidence scores are generated. "
                    "Please refer to the live multi-pollutant profile on the right and historical trends."
                )

        with col_right:
            st.subheader("Current Multi-Pollutant Profile")
            radar_fig = create_pollutant_radar_chart(selected_station.get("measurements", {}), selected_station["candidate_station"])
            st.plotly_chart(radar_fig, use_container_width=True)

        # Health Advisory & Protective Guidance
        st.subheader("Health Advisory & Actionable Protection")
        advisory_meta = AQI_CATEGORIES.get(curr_cat, {})

        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            st.info(f"**Current Health Impact ({curr_cat}):** {advisory_meta.get('description', '')}\n\n**General Public:** {advisory_meta.get('advisory', '')}")
        with adv_col2:
            if is_seen and p1_cat != "Offline":
                worst_forecast_cat = p3_cat if AQI_CATEGORIES.get(p3_cat, {}).get("severity", 0) > AQI_CATEGORIES.get(p1_cat, {}).get("severity", 0) else p1_cat
                forecast_meta = AQI_CATEGORIES.get(worst_forecast_cat, {})
                st.warning(f"**Upcoming Risk Window ({worst_forecast_cat}):**\n\nExpected condition in next 1 to 3 hours is **{worst_forecast_cat}**.\n\n{forecast_meta.get('advisory', '')}")
            else:
                st.info(
                    "**Upcoming Risk Window:**\n\n"
                    "Forward ML projections are reserved strictly for the 13 model-trained stations. "
                    f"Health guidance for this station is driven by current verified CAAQMS observations ({curr_cat})."
                )

# ------------------------------------------------------------------
# TAB 3: STATION COMPARISONS & REGIONAL ANALYTICS
# ------------------------------------------------------------------
with tabs[2]:
    st.header("Cross-Station Comparisons & Regional Analysis")

    compare_col1, compare_col2 = st.columns([1, 1])
    with compare_col1:
        st.subheader("Side-by-Side Pollutant Comparison")
        all_candidate_names = [s.get("candidate_station") for s in stations_list]
        default_compare = [n for n in ["Anand Vihar", "ITO", "Punjabi Bagh", "RK Puram"] if n in all_candidate_names]
        selected_compare = st.multiselect("Select Stations to Compare", all_candidate_names, default=default_compare)

        if selected_compare:
            comp_rows = []
            for s in stations_list:
                if s.get("candidate_station") in selected_compare:
                    meas = s.get("measurements", {})
                    comp_rows.append({
                        "Station": s.get("candidate_station"),
                        "PM2.5": s.get("current_pm25"),
                        "PM10": meas.get("pm10"),
                        "NO2": meas.get("no2"),
                        "SO2": meas.get("so2"),
                        "CO": meas.get("co"),
                        "O3": meas.get("o3")
                    })
            comp_df = pd.DataFrame(comp_rows)

            fig_comp = px.bar(
                comp_df,
                x="Station",
                y=["PM2.5", "PM10"],
                barmode="group",
                title="Particulate Matter Comparison (PM2.5 & PM10)",
                labels={"value": "Concentration (µg/m³)", "variable": "Pollutant"}
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    with compare_col2:
        st.subheader("Regional Air Quality Breakdown")
        region_rows = []
        for s in stations_list:
            if s.get("current_pm25") is not None:
                region_rows.append({
                    "Region": s.get("region", "Delhi"),
                    "PM2.5": s.get("current_pm25")
                })
        if region_rows:
            reg_df = pd.DataFrame(region_rows)
            reg_summary = reg_df.groupby("Region")["PM2.5"].agg(["mean", "max", "min"]).reset_index()
            reg_summary.columns = ["Region", "Average PM2.5", "Peak Station", "Cleanest Station"]

            fig_reg = px.bar(
                reg_summary,
                x="Region",
                y="Average PM2.5",
                color="Average PM2.5",
                color_continuous_scale="Reds",
                title="Average PM2.5 by Delhi Geographical Region",
                text=[f"{v:.1f} µg/m³" for v in reg_summary["Average PM2.5"]]
            )
            st.plotly_chart(fig_reg, use_container_width=True)

    # Dedicated Forecast Comparison across Model-Trained Stations
    st.markdown("---")
    st.subheader("Forecast Trajectory Comparison across Model-Trained Stations")
    st.caption("Forward ML predictions (1-hour & 3-hour horizons) strictly for the 13 CAAQMS stations present in model training. The other 12 stations remain active for live sensor monitoring only.")

    trained_rows = []
    for s in stations_list:
        if s.get("is_seen_in_training"):
            f1 = s.get("forecast_1h", {})
            f3 = s.get("forecast_3h", {})
            pm25 = s.get("current_pm25")
            trained_rows.append({
                "Station": s.get("candidate_station"),
                "Feed Freshness": s.get("freshness_label", "⚪ Offline")[:18],
                "Current PM2.5 (µg/m³)": round(pm25, 1) if pm25 is not None else np.nan,
                "Current Category": s.get("current_aqi_category", "N/A"),
                "1h Forecast": f1.get("predicted_category", "N/A"),
                "1h Confidence": f"{f1['confidence']*100:.1f}%" if f1.get("confidence") is not None else "N/A",
                "3h Forecast": f3.get("predicted_category", "N/A"),
                "3h Confidence": f"{f3['confidence']*100:.1f}%" if f3.get("confidence") is not None else "N/A",
                "Trajectory": s.get("trajectory", "Stable")
            })

    if trained_rows:
        trained_df = pd.DataFrame(trained_rows).sort_values("Current PM2.5 (µg/m³)", ascending=False)
        st.dataframe(trained_df, use_container_width=True)

# ------------------------------------------------------------------
# TAB 4: HISTORICAL TRENDS & DIURNAL PATTERNS
# ------------------------------------------------------------------
with tabs[3]:
    st.header("Historical Timeseries & Temporal Dynamics")

    if history_df.empty:
        st.info("Historical buffer is currently initializing.")
    else:
        st.subheader(f"Hourly Timeseries & Rolling Standards: {selected_station_name}")
        trend_fig = create_historical_trend_chart(history_df, selected_station_name)
        st.plotly_chart(trend_fig, use_container_width=True)

        st.subheader("Diurnal Pollution Cycle (Hour of Day Analysis)")
        st.caption("Analyzes typical diurnal peaks (e.g. morning and evening vehicular traffic spikes)")

        h_copy = history_df.copy()
        if "observed_at_ist" in h_copy.columns:
            h_copy["hour"] = pd.to_datetime(h_copy["observed_at_ist"]).dt.hour
        else:
            h_copy["hour"] = pd.to_datetime(h_copy["observed_at"]).dt.hour

        hourly_avg = h_copy.groupby("hour")["pm25"].mean().reset_index()
        fig_diurnal = px.line(
            hourly_avg,
            x="hour",
            y="pm25",
            markers=True,
            title="Citywide Mean PM2.5 by Hour of Day (24-Hour Diurnal Cycle)",
            labels={"hour": "Hour of Day (IST 0-23)", "pm25": "Mean PM2.5 (µg/m³)"}
        )
        fig_diurnal.add_vrect(x0=7, x1=10, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0, annotation_text="Morning Rush")
        fig_diurnal.add_vrect(x0=18, x1=22, fillcolor="rgba(255, 0, 0, 0.1)", line_width=0, annotation_text="Evening Inversion Peak")
        st.plotly_chart(fig_diurnal, use_container_width=True)

# ------------------------------------------------------------------
# TAB 5: MODEL LEADERBOARD & DIAGNOSTICS
# ------------------------------------------------------------------
with tabs[4]:
    st.header("Machine Learning Model Leaderboard & Diagnostics")
    st.caption("Benchmark of evaluated classifiers trained on the reconstructed 45-feature pipeline")

    lb_1h = leaderboard_service.get_leaderboard_by_horizon("1-hour")
    lb_3h = leaderboard_service.get_leaderboard_by_horizon("3-hour")

    col_lb1, col_lb2 = st.columns(2)
    with col_lb1:
        st.subheader("1-Hour Horizon Leaderboard")
        if not lb_1h.empty:
            display_1h = lb_1h[["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]].copy()
            for col in ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]:
                display_1h[col] = (display_1h[col] * 100).round(2).astype(str) + "%"
            st.dataframe(display_1h, use_container_width=True)

    with col_lb2:
        st.subheader("3-Hour Horizon Leaderboard")
        if not lb_3h.empty:
            display_3h = lb_3h[["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]].copy()
            for col in ["Accuracy", "Precision", "Recall", "Macro F1", "Weighted F1"]:
                display_3h[col] = (display_3h[col] * 100).round(2).astype(str) + "%"
            st.dataframe(display_3h, use_container_width=True)

    # Feature Importance
    st.markdown("---")
    st.subheader("45-Feature Importance & Predictive Drivers")
    fi_horizon = st.radio("Select Horizon for Feature Importance", ["1-Hour Ahead (1h)", "3-Hours Ahead (3h)"], horizontal=True)
    h_key = "1h" if "1-Hour" in fi_horizon else "3h"
    fi_df = leaderboard_service.get_feature_importances(horizon=h_key, top_n=15)
    fi_fig = create_feature_importance_chart(fi_df, top_n=15)
    st.plotly_chart(fi_fig, use_container_width=True)

    # Confusion Matrices
    st.markdown("---")
    st.subheader("Confusion Matrices & Classification Breakdown")
    cf_col1, cf_col2 = st.columns(2)
    with cf_col1:
        st.write("**Gradient Boosting (1-Hour Ahead)**")
        if CONFUSION_GB_1H.exists():
            st.image(str(CONFUSION_GB_1H), use_container_width=True)
    with cf_col2:
        st.write("**Gradient Boosting (3-Hours Ahead)**")
        if CONFUSION_GB_3H.exists():
            st.image(str(CONFUSION_GB_3H), use_container_width=True)

    # Cost-Sensitive Analysis
    st.markdown("---")
    st.subheader("Cost-Sensitive Evaluation")
    st.caption("In air quality management, false negatives on severe pollution (predicting 'Good' or 'Moderate' when actual is 'Severe') cause dangerous public exposure and carry high asymmetric penalties.")
    cost_df = leaderboard_service.get_cost_comparison()
    if not cost_df.empty:
        st.dataframe(cost_df, use_container_width=True)

# ------------------------------------------------------------------
# TAB 6: POLLUTION ALERTS & HEALTH ADVISORIES
# ------------------------------------------------------------------
with tabs[5]:
    st.header("🚨 Pollution Alerts & Early Warning System")
    st.caption("Automated threat detection across Delhi continuous air monitoring network")

    active_alerts = alert_service.load_active_alerts()

    if not active_alerts:
        st.success("No active pollution alerts recorded. Air quality is currently within standard bounds.")
    else:
        st.write(f"Displaying **{len(active_alerts)}** active alert events:")
        for alert in active_alerts[:15]:
            severity_color = alert.get("level_color", "#FF0000")
            title = alert.get("title", "Pollution Alert")
            msg = alert.get("message", "")
            action = alert.get("action", "")
            impact = alert.get("impact", "")
            vuln = alert.get("vulnerable_groups", "")
            ts = alert.get("timestamp", "")

            st.markdown(f"""
            <div class="alert-box" style="border-left-color: {severity_color}; background-color: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.08);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: {severity_color};">{title}</h4>
                    <span style="font-size: 0.8rem; color: #888;">{ts}</span>
                </div>
                <p style="margin: 8px 0 6px 0; font-size: 0.95rem; font-weight: 500;">{msg}</p>
                <div style="font-size: 0.88rem; color: #495057; margin-bottom: 4px;"><b>Impact:</b> {impact}</div>
                <div style="font-size: 0.88rem; color: #155724; background-color: #d4edda; padding: 6px 10px; border-radius: 4px; margin: 6px 0;">
                    <b>Recommended Action:</b> {action}
                </div>
                <div style="font-size: 0.84rem; color: #721c24;"><b>Vulnerable Groups:</b> {vuln}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Graded Response Action Plan (GRAP) Guidance")
    grap_col1, grap_col2, grap_col3, grap_col4 = st.columns(4)
    with grap_col1:
        st.info("**Stage I (Poor: 201-300)**\n\n- Strict dust control on roads\n- Regular water sprinkling\n- Ban on open waste burning")
    with grap_col2:
        st.warning("**Stage II (Very Poor: 301-400)**\n\n- Ban on diesel generator sets\n- Enhanced parking fees to discourage private vehicles\n- Increased bus and metro frequency")
    with grap_col3:
        st.error("**Stage III (Severe: 401-450)**\n\n- Halt non-essential construction\n- Restrict BS-III petrol & BS-IV diesel cars\n- Shift primary schools to online")
    with grap_col4:
        st.error("**Stage IV (Severe+: >450)**\n\n- Entry ban on non-Delhi commercial trucks\n- Closure of all schools & colleges\n- Odd-Even vehicle scheme trigger")
