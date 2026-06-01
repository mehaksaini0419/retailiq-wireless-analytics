"""
RetailIQ - Wireless Store Performance Engine (Metro by T-Mobile edition)
========================================================================
Built around the FIVE core Metro sales targets:
  Magenta migrations | Voice lines | Tablet lines | Home Internet (HSI) | Insurance (P360)

Sections:
  01 Overview            - the 5 core KPIs + conversion + revenue
  02 ML Attach Predictor - predicts accessory-attach likelihood
  03 Pace-to-Goal        - on track to hit monthly voice-line targets?
  04 Coaching            - revenue-at-risk + named rep scorecard
  05 Store Benchmark     - traffic-adjusted fair comparison
  06 Insights            - plain-English recommendations

Run with:  python -m streamlit run dashboard_pro.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===============================================================
# PAGE CONFIG + DESIGN
# ===============================================================
st.set_page_config(page_title="RetailIQ", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');
.stApp {
    background: radial-gradient(circle at 20% 0%, #16213e 0%, #0f1729 55%, #0a0f1f 100%);
    color: #e6ebf5;
}
h1, h2, h3 { font-family: 'Sora', sans-serif !important; color: #f0f4fc !important; letter-spacing: -0.5px; }
p, span, div, label { font-family: 'Sora', sans-serif; }
.kpi-card {
    background: linear-gradient(145deg, rgba(34,48,82,0.9), rgba(22,33,62,0.7));
    border: 1px solid rgba(120,160,255,0.18);
    border-radius: 16px; padding: 16px 16px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35); height: 100%;
}
.kpi-label { font-size: 12px; color: #8ea3c8; text-transform: uppercase; letter-spacing: 0.5px; font-weight:600; white-space:nowrap;}
.kpi-value { font-size: 28px; font-weight: 800; color: #ffffff; font-family:'Sora'; line-height:1.1; margin-top:6px; white-space:nowrap;}
.kpi-sub   { font-size: 11px; color: #6fcf97; margin-top:4px; font-family:'DM Mono'; }
.section-tag {
    display:inline-block; font-family:'DM Mono'; font-size:12px; color:#5b8cff;
    border:1px solid rgba(91,140,255,0.4); border-radius:20px; padding:3px 12px; margin-bottom:6px;
}
.insight {
    background: rgba(91,140,255,0.08); border-left: 3px solid #5b8cff;
    border-radius: 8px; padding: 14px 18px; margin: 8px 0;
    font-size: 15px; color:#d7e1f5;
}
.insight b { color:#fff; }
</style>
""", unsafe_allow_html=True)

# ===============================================================
# LOAD DATA
# ===============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("store_sales.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.markdown("# 📡 RetailIQ")
st.markdown("<p style='color:#8ea3c8; font-size:17px; margin-top:-10px;'>Metro by T-Mobile — Store Performance & Revenue Intelligence Engine</p>", unsafe_allow_html=True)
st.markdown("<span class='section-tag'>SIMULATED DATA · PORTFOLIO DEMO</span>", unsafe_allow_html=True)
st.write("")

st.sidebar.header("⚙️ Filters")
store_options = ["All Stores"] + sorted(df["store_name"].unique().tolist())
selected_store = st.sidebar.selectbox("Store", store_options)
data = df if selected_store == "All Stores" else df[df["store_name"] == selected_store].copy()

def kpi_card(col, label, value, sub=""):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

# ===============================================================
# SECTION 1: CORE KPIs (the 5 Metro targets)
# ===============================================================
st.markdown("<span class='section-tag'>01 · CORE TARGETS</span>", unsafe_allow_html=True)
st.markdown("## Store Performance Snapshot")

total_activations = len(data)
traffic = data.groupby(["store_name", "date"])["daily_traffic"].first().sum()
conversion = (total_activations / traffic * 100) if traffic else 0

magenta = int(data["magenta_migration"].sum())
voice = int(data["voice_lines"].sum())
tablets = int(data["tablet_lines"].sum())
hsi = int(data["home_internet"].sum())
insurance = int(data["insurance_attach"].sum())

total_rev = (data["plan_revenue"].sum() + data["accessory_revenue"].sum()
             + data["insurance_revenue"].sum() + data["hsi_revenue"].sum())

# Row of the 5 core targets
c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, "Magenta Migr.", f"{magenta:,}", "to T-Mobile")
kpi_card(c2, "Voice Lines", f"{voice:,}", "activations")
kpi_card(c3, "Tablet Lines", f"{tablets:,}", "device adds")
kpi_card(c4, "Home Internet", f"{hsi:,}", "HSI adds")
kpi_card(c5, "Insurance P360", f"{insurance:,}", "protection")
st.write("")

# Secondary row: conversion, attach %, revenue
ins_rate = data["insurance_attach"].mean() * 100
acc_rate = data["accessory_attach"].mean() * 100
d1, d2, d3, d4 = st.columns(4)
kpi_card(d1, "Conversion", f"{conversion:.1f}%", "of traffic")
kpi_card(d2, "Insurance Rate", f"{ins_rate:.1f}%", "of sales")
kpi_card(d3, "Accessory Rate", f"{acc_rate:.1f}%", "of sales")
kpi_card(d4, "Total Revenue", f"${total_rev:,.0f}", "monthly, all lines")
st.write("")

# Mini bar of the 5 targets
target_df = pd.DataFrame({
    "Target": ["Magenta", "Voice", "Tablet", "Home Internet", "Insurance"],
    "Count": [magenta, voice, tablets, hsi, insurance],
})
fig_t = px.bar(target_df, x="Target", y="Count", text="Count",
               title="Core Sales Targets — units sold")
fig_t.update_traces(marker_color="#5b8cff", textposition="outside")
fig_t.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#cdd8ee", title_font_color="#f0f4fc", height=300,
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
st.plotly_chart(fig_t, use_container_width=True)

# ===============================================================
# SECTION 2: ML ATTACH PREDICTOR
# ===============================================================
st.markdown("<span class='section-tag'>02 · PREDICTIVE MODEL</span>", unsafe_allow_html=True)
st.markdown("## 🤖 Accessory Attach Predictor")
st.markdown("<p style='color:#8ea3c8'>A machine-learning model trained on past sales learns which factors drive accessory purchases — so reps know which customers to focus the pitch on.</p>", unsafe_allow_html=True)

@st.cache_resource
def train_model(dataframe):
    m = dataframe.copy()
    m["is_magenta"] = m["magenta_migration"]
    age = pd.get_dummies(m["age_band"], prefix="age")
    feats = pd.concat([m[["device_price", "hour", "is_weekend", "plan_revenue", "is_magenta"]], age], axis=1)
    target = m["accessory_attach"]
    Xtr, Xte, ytr, yte = train_test_split(feats, target, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42).fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    imp = pd.DataFrame({"Factor": feats.columns, "Importance": clf.feature_importances_}).sort_values("Importance", ascending=False)
    return clf, auc, imp, feats.columns

model, auc, importance, feature_cols = train_model(df)

cL, cR = st.columns(2)
with cL:
    kpi_card(st.container(), "Model Accuracy (AUC)", f"{auc:.2f}", "1.0 = perfect · 0.5 = guessing")
    st.write("")
    fig_imp = px.bar(importance.head(7).sort_values("Importance"),
                     x="Importance", y="Factor", orientation="h",
                     title="What drives an accessory sale?")
    fig_imp.update_traces(marker_color="#5b8cff")
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#cdd8ee", title_font_color="#f0f4fc", height=320,
                          margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)

with cR:
    st.markdown("#### 🎯 Try it: will this customer buy an accessory?")
    in_price = st.select_slider("Device price ($)", [99, 199, 299, 499, 699, 999], value=699)
    in_age = st.selectbox("Customer age band", ["18-25", "26-40", "41-60", "60+"])
    in_weekend = st.radio("Weekend?", ["No", "Yes"], horizontal=True)
    in_magenta = st.radio("Magenta migration?", ["No", "Yes"], horizontal=True)

    row = {c: 0 for c in feature_cols}
    row["device_price"] = in_price
    row["hour"] = 15
    row["is_weekend"] = 1 if in_weekend == "Yes" else 0
    row["plan_revenue"] = 75 if in_magenta == "Yes" else 60
    row["is_magenta"] = 1 if in_magenta == "Yes" else 0
    if f"age_{in_age}" in row:
        row[f"age_{in_age}"] = 1
    X_one = pd.DataFrame([row])[feature_cols]
    prob = model.predict_proba(X_one)[0][1] * 100
    color = "#6fcf97" if prob >= 50 else "#f2c94c" if prob >= 30 else "#eb5757"
    st.markdown(f"""
    <div class='kpi-card' style='text-align:center; margin-top:10px;'>
        <div class='kpi-label'>Predicted Attach Probability</div>
        <div style='font-size:52px;font-weight:800;color:{color};font-family:Sora'>{prob:.0f}%</div>
        <div class='kpi-sub' style='color:#8ea3c8'>{'High - prioritize the pitch' if prob>=50 else 'Medium - worth trying' if prob>=30 else 'Low - focus elsewhere'}</div>
    </div>""", unsafe_allow_html=True)
st.write("")

# ===============================================================
# SECTION 3: PACE-TO-GOAL
# ===============================================================
st.markdown("<span class='section-tag'>03 · TARGETS</span>", unsafe_allow_html=True)
st.markdown("## 📈 Pace-to-Goal (voice-line activations)")

pace_rows = []
for s in df["store_name"].unique():
    sd = df[df["store_name"] == s]
    goal = sd["monthly_store_goal"].iloc[0]
    actual = len(sd)
    days = sd["date"].nunique()
    projected = (actual / days) * 30 if days else 0
    pace_rows.append({
        "Store": s, "Goal": goal, "Sales so far": actual,
        "Projected": round(projected), "% of Goal": round(projected / goal * 100, 1),
        "Status": "✅ On track" if projected >= goal else "⚠️ Behind",
    })
st.dataframe(pd.DataFrame(pace_rows), use_container_width=True, hide_index=True)
st.write("")

# ===============================================================
# SECTION 4: COACHING (revenue-at-risk + rep scorecard)
# ===============================================================
st.markdown("<span class='section-tag'>04 · COACHING</span>", unsafe_allow_html=True)
st.markdown("## 💸 Revenue-at-Risk & Rep Scorecard")

rep_attach = df.groupby("rep_name")["accessory_attach"].mean()
median_attach = rep_attach.median()
avg_acc_value = df[df["accessory_attach"] == 1]["accessory_revenue"].mean()
recoverable = sum((median_attach - a) * len(df[df["rep_name"] == r]) * avg_acc_value
                  for r, a in rep_attach.items() if a < median_attach)

cA, cB = st.columns([1, 2])
kpi_card(cA, "Monthly Revenue at Risk", f"${recoverable:,.0f}", "recoverable via coaching")

scorecard = df.groupby("rep_name").agg(
    Sales=("transaction_id", "count"),
    Magenta=("magenta_migration", "sum"),
    Insurance=("insurance_attach", "mean"),
    Accessory=("accessory_attach", "mean"),
    HSI=("home_internet", "sum"),
).reset_index()
scorecard["Insurance"] = (scorecard["Insurance"] * 100).round(1)
scorecard["Accessory"] = (scorecard["Accessory"] * 100).round(1)
cutoff = scorecard["Accessory"].quantile(0.34)
scorecard["Flag"] = scorecard["Accessory"].apply(lambda x: "🔴 Coach" if x <= cutoff else "🟢 OK")
scorecard = scorecard.sort_values("Accessory", ascending=False)
scorecard.columns = ["Rep", "Sales", "Magenta", "Insurance %", "Accessory %", "HSI", "Flag"]
cB.dataframe(scorecard, use_container_width=True, hide_index=True)
st.write("")

# ===============================================================
# SECTION 5: STORE BENCHMARK
# ===============================================================
st.markdown("<span class='section-tag'>05 · BENCHMARK</span>", unsafe_allow_html=True)
st.markdown("## 🏪 Store Comparison (traffic-adjusted)")

comp = []
for s in df["store_name"].unique():
    sd = df[df["store_name"] == s]
    t = sd.groupby("date")["daily_traffic"].first().sum()
    comp.append({
        "Store": s, "Sales": len(sd),
        "Conversion %": round(len(sd) / t * 100, 1),
        "Magenta %": round(sd["magenta_migration"].mean() * 100, 1),
        "Insurance %": round(sd["insurance_attach"].mean() * 100, 1),
    })
comp_df = pd.DataFrame(comp).sort_values("Conversion %", ascending=False)
fig = px.bar(comp_df, x="Store", y="Conversion %", text="Conversion %",
             title="Conversion Rate by Store — the fair ranking")
fig.update_traces(marker_color="#5b8cff", textposition="outside")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#cdd8ee", title_font_color="#f0f4fc",
                  yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
st.plotly_chart(fig, use_container_width=True)
st.dataframe(comp_df, use_container_width=True, hide_index=True)
st.write("")

# ===============================================================
# SECTION 6: INSIGHTS
# ===============================================================
st.markdown("<span class='section-tag'>06 · INSIGHTS</span>", unsafe_allow_html=True)
st.markdown("## 🧠 What the Data Is Telling You")

best = comp_df.iloc[0]
high_vol = comp_df.sort_values("Sales", ascending=False).iloc[0]
weak_rep = scorecard.sort_values("Accessory %").iloc[0]
top_factor = importance.iloc[0]["Factor"]
magenta_rate = df["magenta_migration"].mean() * 100

insights = [
    f"<b>{best['Store']}</b> is your best-run store at <b>{best['Conversion %']}%</b> conversion — even though <b>{high_vol['Store']}</b> has the most raw sales ({high_vol['Sales']:,}). Volume ≠ efficiency; {high_vol['Store']} is coasting on foot traffic.",
    f"<b>${recoverable:,.0f}/month</b> is recoverable by coaching below-median reps up to the team median accessory-attach rate. <b>{weak_rep['Rep']}</b> is the priority at only {weak_rep['Accessory %']}% attach.",
    f"Magenta migration rate is <b>{magenta_rate:.1f}%</b> of activations — the migration funnel from Metro prepaid to T-Mobile postpaid is the biggest lever on long-term ARPU and retention.",
    f"The predictive model (AUC <b>{auc:.2f}</b>) finds <b>{str(top_factor).replace('_',' ')}</b> is the strongest driver of an accessory sale — coach reps to pitch hardest on those transactions.",
]
for ins in insights:
    st.markdown(f"<div class='insight'>{ins}</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<p style='color:#5b7099;font-size:13px;text-align:center;font-family:DM Mono'>RetailIQ · built with Python, pandas, scikit-learn, Plotly & Streamlit</p>", unsafe_allow_html=True)
