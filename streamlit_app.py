"""Public portfolio dashboard for the M5 demand forecasting project."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-top: 2.4rem;}
    [data-testid="stMetric"] {
        background: #f5f7fa;
        border: 1px solid #e6e9ef;
        border-radius: 14px;
        padding: 18px;
    }
    h1, h2, h3 {letter-spacing: -0.025em;}
    .eyebrow {color:#52606d; font-size:.85rem; font-weight:700; letter-spacing:.08em;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="eyebrow">DATA SCIENCE · FORECASTING · INVENTORY</p>', unsafe_allow_html=True)
st.title("Retail demand forecasting")
st.write(
    "A portfolio case that turns five years of daily retail sales into a "
    "28-day forecast and a practical inventory decision framework."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Demand series", "30,490")
metric_columns[1].metric("Products", "3,049")
metric_columns[2].metric("Stores", "10")
metric_columns[3].metric("Historical days", "1,941")

st.divider()
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("The business question")
    st.write(
        "How many units should each store expect to sell during the next 28 days, "
        "and how can that forecast reduce stockouts and excess inventory?"
    )
    st.subheader("Dataset quality")
    st.markdown(
        """
        - Daily coverage from **2011-01-29 to 2016-06-19**
        - **6.84 million** store-item-week price records
        - No duplicate calendar or price keys
        - No missing `sell_price` values
        """
    )

with right:
    st.subheader("Project workflow")
    st.markdown(
        """
        1. Audit calendar, sales and price tables
        2. Define a leakage-free temporal holdout
        3. Establish statistical baselines
        4. Engineer lag, rolling, price and event features
        5. Train and compare machine-learning models
        6. Translate error into inventory decisions
        """
    )

st.divider()
st.subheader("Baseline backtest")
st.caption("Train: d_1–d_1913 · Test: d_1914–d_1941 · Horizon: 28 days")

results = pd.DataFrame(
    {
        "Model": ["Mean · last 28 days", "Seasonal · 7 days", "Seasonal · 28 days", "Last value"],
        "WAPE": [73.86, 86.22, 89.00, 95.16],
        "RMSSE": [0.9240, 1.2010, 1.2445, 1.2063],
        "MAE": [1.0657, 1.2440, 1.2840, 1.3730],
    }
)

chart = px.bar(
    results.sort_values("WAPE", ascending=False),
    x="WAPE",
    y="Model",
    orientation="h",
    text="WAPE",
    color="WAPE",
    color_continuous_scale=["#1f9d8a", "#f2b134", "#e35d6a"],
    labels={"WAPE": "WAPE (%)"},
)
chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
chart.update_layout(
    height=390,
    margin=dict(l=10, r=40, t=10, b=10),
    coloraxis_showscale=False,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(chart, use_container_width=True)

best = results.loc[results["WAPE"].idxmin()]
st.success(
    f"Current benchmark: {best['Model']} with {best['WAPE']:.2f}% WAPE. "
    "The machine-learning model must beat this result on the same holdout period."
)

with st.expander("Metric definitions"):
    st.markdown(
        """
        - **MAE:** average absolute unit error.
        - **WAPE:** total absolute error divided by total actual demand.
        - **RMSSE:** scaled error that allows comparison across products with different demand levels.
        """
    )

st.divider()
st.subheader("Next experiment")
st.write(
    "Train a global LightGBM model using demand lags, rolling statistics, price changes, "
    "calendar events and product/store identifiers. Evaluate it on the unchanged 28-day holdout."
)
st.caption("Built by Eric Villegas · Data Science & AI portfolio project")

