import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_connection
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Credit Card Fraud Dashboard",
    page_icon  = "💳",
    layout     = "wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #e74c3c;
    }
    .stMetric { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Load Data (cached so it only loads once) ───────────────────────
@st.cache_data
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"]    = df["trans_date_trans_time"].dt.hour
    df["month"]   = df["trans_date_trans_time"].dt.to_period("M").astype(str)
    df["day_type"] = df["trans_date_trans_time"].dt.dayofweek.apply(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )
    return df

# ── Load ───────────────────────────────────────────────────────────
with st.spinner("Loading 1.3M transactions..."):
    df = load_data()

fraud = df[df["is_fraud"] == 1]
legit = df[df["is_fraud"] == 0]


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
st.title("💳 Credit Card Fraud Analytics Dashboard")
st.markdown("**Analyzing 1.3 million transactions | 2019–2020 | Built with SQL + Python**")
st.divider()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ══════════════════════════════════════════════════════════════════
st.sidebar.title("🔍 Filters")
st.sidebar.markdown("---")

selected_category = st.sidebar.multiselect(
    "Category",
    options = sorted(df["category"].unique()),
    default = sorted(df["category"].unique())
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    options = df["gender"].unique(),
    default = df["gender"].unique()
)

amount_range = st.sidebar.slider(
    "Transaction Amount ($)",
    min_value = 0,
    max_value = 1000,
    value     = (0, 1000)
)

selected_fraud = st.sidebar.radio(
    "Transaction Type",
    options = ["All", "Fraud Only", "Legitimate Only"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Built by:** Jeet Dave")
st.sidebar.markdown("**Dataset:** 1.3M Credit Card Transactions")
st.sidebar.markdown("**Tools:** PostgreSQL · Python · Streamlit")

# ── Apply Filters ──────────────────────────────────────────────────
filtered = df[
    (df["category"].isin(selected_category)) &
    (df["gender"].isin(selected_gender)) &
    (df["amt"] >= amount_range[0]) &
    (df["amt"] <= amount_range[1])
]
if selected_fraud == "Fraud Only":
    filtered = filtered[filtered["is_fraud"] == 1]
elif selected_fraud == "Legitimate Only":
    filtered = filtered[filtered["is_fraud"] == 0]


# ══════════════════════════════════════════════════════════════════
# KPI METRICS ROW
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Total Transactions",
        f"{len(filtered):,}"
    )
with k2:
    st.metric(
        "Fraud Transactions",
        f"{filtered['is_fraud'].sum():,}",
        delta=f"{filtered['is_fraud'].mean()*100:.2f}%"
    )
with k3:
    st.metric(
        "Total Revenue",
        f"${filtered['amt'].sum():,.0f}"
    )
with k4:
    st.metric(
        "Avg Transaction",
        f"${filtered['amt'].mean():.2f}"
    )
with k5:
    fraud_amt = filtered[filtered["is_fraud"]==1]["amt"].sum()
    st.metric(
        "Total Fraud Amount",
        f"${fraud_amt:,.0f}"
    )

st.divider()


# ══════════════════════════════════════════════════════════════════
# ROW 1 — Fraud Overview + Category Analysis
# ══════════════════════════════════════════════════════════════════
st.subheader("🔍 Fraud Overview")
col1, col2 = st.columns(2)

with col1:
    fraud_counts = filtered["is_fraud"].value_counts()
    fig = px.pie(
        values = fraud_counts.values,
        names  = ["Legitimate", "Fraud"],
        color_discrete_sequence = ["#2ecc71", "#e74c3c"],
        title  = "Fraud vs Legitimate Transactions",
        hole   = 0.4
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    cat_fraud = (
        filtered.groupby("category")
        .agg(total=("is_fraud","count"), fraud=("is_fraud","sum"))
        .assign(fraud_rate=lambda x: x["fraud"]/x["total"]*100)
        .sort_values("fraud_rate", ascending=True)
        .reset_index()
    )
    fig = px.bar(
        cat_fraud,
        x     = "fraud_rate",
        y     = "category",
        title = "Fraud Rate by Category (%)",
        color = "fraud_rate",
        color_continuous_scale = "Reds",
        orientation = "h"
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
# ROW 2 — Time Analysis
# ══════════════════════════════════════════════════════════════════
st.subheader("⏰ Time-Based Fraud Patterns")
col3, col4 = st.columns(2)

with col3:
    hour_data = (
        filtered.groupby("hour")
        .agg(total=("is_fraud","count"), fraud=("is_fraud","sum"))
        .assign(fraud_rate=lambda x: x["fraud"]/x["total"]*100)
        .reset_index()
    )
    fig = px.line(
        hour_data,
        x      = "hour",
        y      = "fraud_rate",
        title  = "Fraud Rate by Hour of Day",
        markers= True,
        color_discrete_sequence=["#e74c3c"]
    )
    fig.update_layout(xaxis_title="Hour (0=Midnight)", yaxis_title="Fraud Rate (%)")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    monthly = (
        filtered.groupby("month")
        .agg(total_amt=("amt","sum"), fraud_count=("is_fraud","sum"))
        .reset_index()
    )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=monthly["month"], y=monthly["total_amt"],
               name="Total Spending", marker_color="#3498db"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=monthly["month"], y=monthly["fraud_count"],
                   name="Fraud Count", line=dict(color="#e74c3c", width=2),
                   mode="lines+markers"),
        secondary_y=True
    )
    fig.update_layout(title="Monthly Spending vs Fraud Count")
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
# ROW 3 — Geographic + Amount Analysis
# ══════════════════════════════════════════════════════════════════
st.subheader("🗺️ Geographic & Amount Analysis")
col5, col6 = st.columns(2)

with col5:
    state_fraud = (
        filtered[filtered["is_fraud"]==1]
        .groupby("state")["amt"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    fig = px.bar(
        state_fraud,
        x     = "state",
        y     = "amt",
        title = "Top 15 States by Fraud Amount Lost",
        color = "amt",
        color_continuous_scale = "OrRd"
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    fig = px.histogram(
        filtered[filtered["amt"] < 500],
        x      = "amt",
        color  = "is_fraud",
        title  = "Transaction Amount Distribution",
        nbins  = 50,
        barmode= "overlay",
        opacity= 0.7,
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"}
    )
    fig.update_layout(legend_title="Is Fraud")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
# ROW 4 — Customer Analysis + Scatter
# ══════════════════════════════════════════════════════════════════
st.subheader("👥 Customer & Merchant Analysis")
col7, col8 = st.columns(2)

with col7:
    gender_data = (
        filtered.groupby("gender")
        .agg(
            total   = ("is_fraud","count"),
            fraud   = ("is_fraud","sum"),
            avg_amt = ("amt","mean")
        )
        .assign(fraud_rate=lambda x: x["fraud"]/x["total"]*100)
        .reset_index()
    )
    fig = px.bar(
        gender_data,
        x      = "gender",
        y      = ["fraud_rate"],
        title  = "Fraud Rate by Gender (%)",
        color  = "gender",
        color_discrete_map={"M":"#3498db","F":"#e91e8c"}
    )
    st.plotly_chart(fig, use_container_width=True)

with col8:
    day_data = (
        filtered.groupby("day_type")
        .agg(total=("is_fraud","count"), fraud=("is_fraud","sum"))
        .assign(fraud_rate=lambda x: x["fraud"]/x["total"]*100)
        .reset_index()
    )
    fig = px.bar(
        day_data,
        x      = "day_type",
        y      = "fraud_rate",
        title  = "Fraud Rate: Weekday vs Weekend",
        color  = "day_type",
        color_discrete_map={"Weekday":"#3498db","Weekend":"#e74c3c"}
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# ══════════════════════════════════════════════════════════════════
# RAW DATA TABLE
# ══════════════════════════════════════════════════════════════════
st.subheader("📋 Raw Data Explorer")
show_fraud_only = st.checkbox("Show fraud transactions only", value=False)

display_df = filtered[filtered["is_fraud"]==1] if show_fraud_only else filtered

st.dataframe(
    display_df[[
        "trans_date_trans_time", "cc_num", "merchant",
        "category", "amt", "first", "last",
        "state", "is_fraud"
    ]].head(100),
    use_container_width=True
)

st.caption(f"Showing top 100 of {len(display_df):,} filtered rows")