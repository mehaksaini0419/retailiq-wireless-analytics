"""
RetailIQ - Wireless Store Performance Dashboard
===============================================
An interactive dashboard that turns raw store transaction data into the
KPIs a wireless retail manager actually needs:

  1. Store overview      - conversion, ARPU, attach rates
  2. Pace-to-goal        - are we on track to hit the monthly target?
  3. Revenue-at-risk     - $ recoverable if weak performers reach median
  4. Rep scorecard       - who needs coaching, ranked fairly
  5. Multi-store compare  - fair, traffic-aware comparison

Run it with:  streamlit run dashboard.py
(Make sure store_sales.csv is in the same folder.)
"""

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="RetailIQ", page_icon="📱", layout="wide")
st.title("📱 RetailIQ — Wireless Store Performance Engine")
st.caption("Simulated data for portfolio demonstration. Modeled on wireless retail operations.")

# ---------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------
# @st.cache_data tells Streamlit to load the file once and remember it,
# so the app stays fast.
@st.cache_data
def load_data():
    df = pd.read_csv("store_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# ---------------------------------------------------------------
# SIDEBAR FILTER - let the user pick a store (or all stores)
# ---------------------------------------------------------------
st.sidebar.header("Filters")
store_options = ["All Stores"] + sorted(df["store_name"].unique().tolist())
selected_store = st.sidebar.selectbox("Select store", store_options)

# Apply the filter. If "All Stores", we keep everything.
if selected_store == "All Stores":
    data = df.copy()
else:
    data = df[df["store_name"] == selected_store].copy()

# ---------------------------------------------------------------
# SECTION 1: TOP-LINE KPIs
# ---------------------------------------------------------------
st.header("Store Overview")

# Total activations = total number of sales (each row is one sale).
total_activations = len(data)

# Conversion rate = activations / total traffic.
# Traffic is logged per store per day, so we take it once per store-day
# to avoid counting the same day's traffic many times.
traffic_per_day = data.groupby(["store_id", "date"])["daily_traffic"].first()
total_traffic = traffic_per_day.sum()
conversion_rate = (total_activations / total_traffic * 100) if total_traffic else 0

# Attach rates = % of sales that included the add-on.
accessory_attach_rate = data["accessory_attach"].mean() * 100
insurance_attach_rate = data["insurance_attach"].mean() * 100

# ARPU = average plan revenue per activation.
arpu = data["plan_revenue"].mean()

# Total monthly revenue (plans + accessories + insurance).
total_revenue = (
    data["plan_revenue"].sum()
    + data["accessory_revenue"].sum()
    + data["insurance_revenue"].sum()
)

# Show them as metric cards in a row.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Activations", f"{total_activations:,}")
c2.metric("Conversion Rate", f"{conversion_rate:.1f}%")
c3.metric("Accessory Attach", f"{accessory_attach_rate:.1f}%")
c4.metric("Insurance Attach", f"{insurance_attach_rate:.1f}%")
c5.metric("ARPU", f"${arpu:.2f}")

st.metric("Total Revenue (incl. add-ons)", f"${total_revenue:,.0f}")

# ---------------------------------------------------------------
# SECTION 2: PACE-TO-GOAL
# ---------------------------------------------------------------
st.header("📈 Pace-to-Goal Tracker")

# This only makes sense per-store (each store has its own goal),
# so we compute it for every store and display a table.
pace_rows = []
for store_id in df["store_id"].unique():
    store_data = df[df["store_id"] == store_id]
    store_name = store_data["store_name"].iloc[0]
    goal = store_data["monthly_store_goal"].iloc[0]

    actual = len(store_data)                      # sales so far
    days_elapsed = store_data["date"].nunique()   # days of data we have
    days_in_month = 30                            # November

    # Project the full month based on the current daily run-rate.
    daily_rate = actual / days_elapsed if days_elapsed else 0
    projected = daily_rate * days_in_month

    # How far ahead/behind goal are we projected to land?
    gap = projected - goal
    pct_to_goal = (projected / goal * 100) if goal else 0

    pace_rows.append({
        "Store": store_name,
        "Goal": goal,
        "Sales so far": actual,
        "Projected month-end": round(projected),
        "% of Goal (projected)": round(pct_to_goal, 1),
        "Status": "✅ On track" if gap >= 0 else "⚠️ Behind",
    })

pace_df = pd.DataFrame(pace_rows)
st.dataframe(pace_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# SECTION 3: REVENUE-AT-RISK
# ---------------------------------------------------------------
# The business question: if our weaker reps attached accessories at the
# same rate as the median rep, how much more revenue would we make?
st.header("💸 Revenue-at-Risk (Accessory Attach)")

# Average accessory attach rate per rep.
rep_attach = df.groupby("rep_id")["accessory_attach"].mean()
median_attach = rep_attach.median()

# Average revenue per attached accessory (so we can value the gap).
avg_accessory_value = df[df["accessory_attach"] == 1]["accessory_revenue"].mean()

# For each rep below the median, how many extra attaches would they get,
# and what is that worth?
recoverable = 0
for rep_id, attach in rep_attach.items():
    if attach < median_attach:
        rep_sales = len(df[df["rep_id"] == rep_id])
        missed_attaches = (median_attach - attach) * rep_sales
        recoverable += missed_attaches * avg_accessory_value

st.metric(
    "Monthly revenue recoverable if below-median reps reach median attach",
    f"${recoverable:,.0f}",
)
st.caption(
    "This is the coaching opportunity: closing the attach-rate gap on weaker "
    "reps converts directly into this much additional monthly revenue."
)

# ---------------------------------------------------------------
# SECTION 4: REP SCORECARD
# ---------------------------------------------------------------
st.header("🧑‍💼 Rep Performance Scorecard")

rep_scorecard = data.groupby("rep_id").agg(
    Sales=("transaction_id", "count"),
    Accessory_Attach=("accessory_attach", "mean"),
    Insurance_Attach=("insurance_attach", "mean"),
    ARPU=("plan_revenue", "mean"),
).reset_index()

# Convert rates to percentages and round for readability.
rep_scorecard["Accessory_Attach"] = (rep_scorecard["Accessory_Attach"] * 100).round(1)
rep_scorecard["Insurance_Attach"] = (rep_scorecard["Insurance_Attach"] * 100).round(1)
rep_scorecard["ARPU"] = rep_scorecard["ARPU"].round(2)

# Flag coaching priority: bottom-third accessory attach gets flagged.
attach_cutoff = rep_scorecard["Accessory_Attach"].quantile(0.34)
rep_scorecard["Coaching Flag"] = rep_scorecard["Accessory_Attach"].apply(
    lambda x: "🔴 Coach" if x <= attach_cutoff else "🟢 OK"
)

rep_scorecard = rep_scorecard.sort_values("Accessory_Attach", ascending=False)
st.dataframe(rep_scorecard, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------
# SECTION 5: MULTI-STORE COMPARISON (fair, traffic-aware)
# ---------------------------------------------------------------
st.header("🏪 Multi-Store Comparison")

store_compare = []
for store_id in df["store_id"].unique():
    sd = df[df["store_id"] == store_id]
    name = sd["store_name"].iloc[0]
    sales = len(sd)
    traffic = sd.groupby("date")["daily_traffic"].first().sum()
    conv = (sales / traffic * 100) if traffic else 0
    store_compare.append({
        "Store": name,
        "Sales": sales,
        "Conversion %": round(conv, 1),       # the FAIR metric
        "Accessory Attach %": round(sd["accessory_attach"].mean() * 100, 1),
        "ARPU": round(sd["plan_revenue"].mean(), 2),
    })

compare_df = pd.DataFrame(store_compare).sort_values("Conversion %", ascending=False)

# Chart: conversion rate by store (fair comparison, not raw sales).
fig = px.bar(
    compare_df, x="Store", y="Conversion %",
    title="Conversion Rate by Store (traffic-adjusted — the fair comparison)",
    text="Conversion %",
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(compare_df, use_container_width=True, hide_index=True)
