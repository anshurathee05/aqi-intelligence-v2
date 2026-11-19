"""
Reusable UI Components and Visualizations for the AQI Rebuild Dashboard.
Provides Plotly maps, probability gauges, timeseries charts, metric tiles, and alert cards.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aqi_rebuild.config.settings import AQI_CATEGORIES, ORDERED_CATEGORIES


def get_category_color(category: str) -> str:
    """Returns the hex color code for an AQI category."""
    return AQI_CATEGORIES.get(category, {}).get("color", "#888888")


def render_aqi_badge(category: str) -> str:
    """Returns an HTML badge styled for the given AQI category."""
    meta = AQI_CATEGORIES.get(category, {
        "color": "#666", "bg_color": "rgba(100,100,100,0.1)", "border_color": "#888"
    })
    return (
        f'<span style="background-color: {meta["bg_color"]}; color: {meta["color"]}; '
        f'border: 1px solid {meta["border_color"]}; padding: 3px 10px; border-radius: 12px; '
        f'font-weight: 700; font-size: 0.88rem; display: inline-block;">'
        f'{category}</span>'
    )


def create_delhi_map(stations_data: List[Dict[str, Any]]) -> go.Figure:
    """
    Creates an interactive map of Delhi with stations colored by current AQI category or PM2.5.
    """
    if not stations_data:
        fig = go.Figure()
        fig.update_layout(title="No station spatial data available")
        return fig

    color_map = {cat: AQI_CATEGORIES[cat]["color"] for cat in ORDERED_CATEGORIES if cat in AQI_CATEGORIES}
    color_map["Offline / No Feed"] = "#9E9E9E"
    color_map["Unavailable"] = "#9E9E9E"

    rows = []
    for s in stations_data:
        is_live = s.get("is_live", True) and s.get("current_pm25") is not None
        pm25 = s.get("current_pm25") if is_live else None
        cat = s.get("current_aqi_category", "Offline / No Feed") if is_live else "Offline / No Feed"
        p1 = s.get("forecast_1h", {}).get("predicted_category", "N/A") if is_live else "Offline"
        p3 = s.get("forecast_3h", {}).get("predicted_category", "N/A") if is_live else "Offline"
        rows.append({
            "station": s.get("candidate_station", s.get("station_name")),
            "lat": s.get("latitude"),
            "lon": s.get("longitude"),
            "region": s.get("region"),
            "pm25": pm25 if pm25 is not None else 0,
            "pm25_label": f"{pm25:.1f} µg/m³" if pm25 is not None else "Offline (No Live Feed)",
            "category": cat,
            "forecast_1h": p1,
            "forecast_3h": p3,
            "trajectory": s.get("trajectory", "Stable") if is_live else "Offline",
            "color": color_map.get(cat, "#9E9E9E")
        })

    df = pd.DataFrame(rows)

    # Plotly Express Map: Supports Plotly 7.0+ (scatter_map) and legacy (scatter_mapbox)
    common_map_kwargs = dict(
        data_frame=df,
        lat="lat",
        lon="lon",
        hover_name="station",
        hover_data={
            "region": True,
            "pm25_label": True,
            "category": True,
            "forecast_1h": True,
            "forecast_3h": True,
            "trajectory": True,
            "lat": False,
            "lon": False
        },
        color="category",
        color_discrete_map=color_map,
        size=[18] * len(df),
        size_max=18,
        zoom=10.2,
        center={"lat": 28.6448, "lon": 77.2167},
        title="Delhi Continuous Ambient Air Quality Monitoring Network (CAAQMS)"
    )

    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(**common_map_kwargs, map_style="carto-positron")
    else:
        fig = px.scatter_mapbox(**common_map_kwargs, mapbox_style="carto-positron")

    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend_title_text="Current AQI Category",
        height=520
    )
    return fig


def create_probability_distribution_chart(forecast_1h: Dict[str, Any], forecast_3h: Dict[str, Any]) -> go.Figure:
    """
    Renders grouped bar chart comparing predicted class probabilities for 1-hour and 3-hour horizons.
    """
    p1 = forecast_1h.get("probabilities", {})
    p3 = forecast_3h.get("probabilities", {})

    categories = [cat for cat in ORDERED_CATEGORIES if cat in p1 or cat in p3]
    vals_1h = [p1.get(cat, 0.0) * 100 for cat in categories]
    vals_3h = [p3.get(cat, 0.0) * 100 for cat in categories]

    fig = go.Figure(data=[
        go.Bar(
            name="1-Hour Ahead",
            x=categories,
            y=vals_1h,
            marker_color="#2E91E5",
            text=[f"{v:.1f}%" for v in vals_1h],
            textposition="auto"
        ),
        go.Bar(
            name="3-Hours Ahead",
            x=categories,
            y=vals_3h,
            marker_color="#E15F99",
            text=[f"{v:.1f}%" for v in vals_3h],
            textposition="auto"
        )
    ])

    fig.update_layout(
        barmode="group",
        title="Forecasted AQI Category Probabilities (1-Hour vs 3-Hour)",
        xaxis_title="AQI Health Category",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=380,
        margin={"t": 40, "b": 30, "l": 40, "r": 20}
    )
    return fig


def create_historical_trend_chart(history_df: pd.DataFrame, station_name: str) -> go.Figure:
    """
    Renders historical PM2.5 timeseries with 6h, 12h, 24h rolling averages and CPCB threshold bands.
    """
    if history_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No timeseries data available")
        return fig

    st_df = history_df[history_df["station_name"] == station_name].copy()
    if st_df.empty:
        st_df = history_df[history_df["candidate_station"] == station_name].copy()
    if st_df.empty:
        st_df = history_df.copy()

    st_df["dt"] = pd.to_datetime(st_df.get("observed_at_ist", st_df["observed_at"]))
    st_df = st_df.sort_values("dt").reset_index(drop=True)

    fig = go.Figure()

    # Threshold background bands
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0, 228, 0, 0.12)", line_width=0, layer="below", annotation_text="Good (0-30)")
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(146, 208, 80, 0.12)", line_width=0, layer="below", annotation_text="Satisfactory (30-60)")
    fig.add_hrect(y0=60, y1=90, fillcolor="rgba(255, 255, 0, 0.12)", line_width=0, layer="below", annotation_text="Moderate (60-90)")
    fig.add_hrect(y0=90, y1=120, fillcolor="rgba(255, 126, 0, 0.15)", line_width=0, layer="below", annotation_text="Poor (90-120)")
    fig.add_hrect(y0=120, y1=250, fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, layer="below", annotation_text="Very Poor (120-250)")
    fig.add_hrect(y0=250, y1=700, fillcolor="rgba(126, 0, 35, 0.2)", line_width=0, layer="below", annotation_text="Severe (>250)")

    # Raw PM2.5
    fig.add_trace(go.Scatter(
        x=st_df["dt"],
        y=st_df["pm25"],
        mode="lines+markers",
        name="Hourly PM2.5",
        line=dict(color="#1f77b4", width=2),
        marker=dict(size=4)
    ))

    # 24h rolling avg if present
    if "pm25_24h_avg" in st_df.columns:
        fig.add_trace(go.Scatter(
            x=st_df["dt"],
            y=st_df["pm25_24h_avg"],
            mode="lines",
            name="24h Rolling Mean (NAAQS Standard)",
            line=dict(color="#d62728", width=3, dash="dot")
        ))
    elif "pm25_rolling_mean_24h" in st_df.columns:
        fig.add_trace(go.Scatter(
            x=st_df["dt"],
            y=st_df["pm25_rolling_mean_24h"],
            mode="lines",
            name="24h Prior Rolling Mean",
            line=dict(color="#d62728", width=3, dash="dot")
        ))

    fig.update_layout(
        title=f"PM2.5 Concentration Timeseries & Standard Bands: {station_name}",
        xaxis_title="Observation Time (IST)",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_pollutant_radar_chart(measurements: Dict[str, Any], station_name: str) -> go.Figure:
    """
    Renders multi-pollutant radar chart comparing current concentrations.
    """
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    labels = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO2 (ppb)", "SO2 (ppb)", "CO (ppm/ppb)", "O3 (µg/m³)"]
    values = [measurements.get(p) or 0.0 for p in pollutants]

    # Normalize roughly to 0-100 scale for radar visual balance
    # PM2.5 / 2.5, PM10 / 5.0, NO2 / 1.0, SO2 / 1.0, CO * 20, O3 / 1.5
    scales = [2.5, 5.0, 1.0, 1.0, 1.0, 1.5]
    scaled_values = [min(100.0, v / s) if s else v for v, s in zip(values, scales)]

    fig = go.Figure(data=go.Scatterpolar(
        r=scaled_values + [scaled_values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.3)',
        line=dict(color='#1f77b4', width=2),
        text=[f"Actual: {v:.1f}" for v in values] + [f"Actual: {values[0]:.1f}"],
        hoverinfo='theta+text'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=f"Multi-Pollutant Profile: {station_name}",
        height=380,
        margin={"t": 40, "b": 30, "l": 30, "r": 30}
    )
    return fig


def create_feature_importance_chart(fi_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    Renders feature importance bar chart colored by domain.
    """
    if fi_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No feature importances available")
        return fig

    sub_df = fi_df.head(top_n).sort_values("importance", ascending=True)

    domain_colors = {
        "PM2.5 Temporal Memory": "#1f77b4",
        "Meteorological": "#2ca02c",
        "Co-Pollutants": "#ff7f0e",
        "Diurnal & Seasonal Cycles": "#9467bd",
        "Sensor Quality / Missingness": "#8c564b",
        "Spatial / Station Location": "#e377c2",
        "Other": "#7f7f7f"
    }

    fig = px.bar(
        sub_df,
        x="relative_pct",
        y="feature",
        color="domain",
        color_discrete_map=domain_colors,
        orientation="h",
        labels={"relative_pct": "Relative Importance (%)", "feature": "Engineered Predictor"},
        title=f"Top {top_n} Predictive Drivers (Gradient Boosting Classifier)",
        text=[f"{v:.1f}%" for v in sub_df["relative_pct"]]
    )

    fig.update_layout(
        template="plotly_white",
        height=480,
        margin={"t": 40, "b": 30, "l": 150, "r": 20},
        legend_title_text="Feature Domain"
    )
    return fig
