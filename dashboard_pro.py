"""
RetailIQ - Wireless Store Performance Engine (PRO version)
==========================================================
Adds three things over the basic version:
  1. Professional custom design (fonts, colors, KPI cards, layout)
  2. A machine-learning model that PREDICTS accessory-attach likelihood
  3. An automatic "Insights & Recommendations" section in plain English

Run it with:  python -m streamlit run dashboard_pro.py
(store_sales.csv must be in the same folder.)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===============================================================
# PAGE CONFIG + CUSTOM DESIGN
# ===============================================================
st.set_page_config(page_title="RetailIQ", page_icon="📡", layout="wide")

# Custom CSS — this is what turns plain Streamlit into something that looks
# deliberately designed. We define a dark, modern "command-center" theme.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap');

/* App background: deep navy with a subtle gradient */
.stApp {
    background: radial-gradient(circle at 20% 0%, #16213e 0%, #0f1729 55%, #0a0f1f 100%);
    color: #e6ebf5;
}
h1, h2, h3 { font-family: 'Sora', sans-serif !important; color: #f0f4fc !important; letter-spacing: -0.5px; }
p, span, div, label { font-family: 'Sora', sans-serif; }

/* Our custom KPI card */
.kpi-card {
    background: linear-gradient(145deg, rgba(34,48,82,0.9), rgba(22,33,62,0.7));
    border: 1px solid rgba(120,160,255,0.18);
    border-radius: 16px;
    padding: 18px 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    height: 100%;
}
.kpi-label { font-size: 12px; color: #8ea3c8; text-transform: uppercase; letter-spacing: 0.5px; font-weight:600; white-space:nowrap; }
.kpi-value { font-size: 30px; font-weight: 800; color: #ffffff; font-family:'Sora'; line-height:1.1; margin-top:6px; white-space:nowrap;}
.kpi-sub   { font-size: 12px; color: #6fcf97; margin-top:4px; font-family:'DM Mono'; }

/* Section divider accent */
.section-tag {
    display:inline-block; font-family:'DM Mono'; font-size:12px; color:#5b8cff;
    border:1px solid rgba(91,140,255,0.4); border-radius:20px; padding:3px 12px; margin-bottom:6px;
}
/* Insight box */
.insight {
    background: rgba(91,140,255,0.08);
    border-left: 3px solid #5b8cff;
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

# ---- Header ----
st.markdown("# 📡 RetailIQ")
st.markdown("<p style='color:#8ea3c8; font-size:17px; margin-top:-10px;'>Wireless Store Performance & Revenue Intelligence Engine</p>", unsafe_allow_html=True)
st.markdown("<span class='section-tag'>SIMULATED DATA · PORTFOLIO DEMO</span>", unsafe_allow_html=True)
st.write("")

# Sidebar filter
st.sidebar.header("⚙️ Filters")
store_options = ["All Stores"] + sorted(df["store_name"].unique().tolist())
selected_store = st.sidebar.selectbox("Store", store_options)
data = df if selected_store == "All Stores" else df[df["store_name"] == selected_store].copy()

# ===============================================================
# HELPER: render a KPI card
# ===============================================================
def kpi_card(col, label, value, sub=""):
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value'>{value}</div>
        <div class='kpi-sub'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ===============================================================
# SECTION 1: KPIs
# ===============================================================
st.markdown("<span class='section-tag'>01 · OVERVIEW</span>", unsafe_allow_html=True)
st.markdown("## Store Performance Snapshot")

total_activations = len(data)
traffic = data.groupby(["store_id", "date"])["daily_traffic"].first().sum()
conversion = (total_activations / traffic * 100) if traffic else 0
acc_attach = data["accessory_attach"].mean() * 100
ins_attach = data["insurance_attach"].mean() * 100
arpu = data["plan_revenue"].mean()
total_rev = data["plan_revenue"].sum() + data["accessory_revenue"].sum() + data["insurance_revenue"].sum()

c1, c2, c3, c4, c5 = st.columns(5)
kpi_card(c1, "Activations", f"{total_activations:,}", "total sales")
kpi_card(c2, "Conversion", f"{conversion:.1f}%", "of traffic")
kpi_card(c3, "Acc. Attach", f"{acc_attach:.1f}%", "of sales")
kpi_card(c4, "Ins. Attach", f"{ins_attach:.1f}%", "of sales")
kpi_card(c5, "ARPU", f"${arpu:.0f}", "avg plan rev")
st.write("")
c6, _ = st.columns([1, 4])
kpi_card(c6, "Total Revenue", f"${total_rev:,.0f}", "incl. add-ons")
st.write("")

# ===============================================================
# SECTION 2: MACHINE LEARNING — PREDICT ACCESSORY ATTACH
# ===============================================================
st.markdown("<span class='section-tag'>02 · PREDICTIVE MODEL</span>", unsafe_allow_html=True)
st.markdown("## 🤖 Accessory Attach Predictor")
st.markdown("<p style='color:#8ea3c8'>A machine-learning model trained on past sales learns which factors drive accessory purchases — so reps know which customers to focus the pitch on.</p>", unsafe_allow_html=True)

@st.cache_resource
def train_model(dataframe):
    # Features the model learns from. We convert text columns to numbers.
    model_df = dataframe.copy()
    model_df["is_premium_plan"] = (model_df["plan_type"] == "Premium").astype(int)
    # One-hot encode age band (turn categories into 0/1 columns).
    age_dummies = pd.get_dummies(model_df["age_band"], prefix="age")
    features = pd.concat([
        model_df[["device_price", "hour", "is_weekend", "plan_revenue", "is_premium_plan"]],
        age_dummies
    ], axis=1)
    target = model_df["accessory_attach"]

    # Split into training and test sets so we can measure honest accuracy.
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # AUC: how well the model separates buyers from non-buyers (0.5=random, 1.0=perfect).
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Feature importance: which factors matter most.
    importance = pd.DataFrame({
        "Factor": features.columns,
        "Importance": clf.feature_importances_
    }).sort_values("Importance", ascending=False)
    return clf, auc, importance, features.columns

model, auc, importance, feature_cols = train_model(df)

cL, cR = st.columns([1, 1])

with cL:
    kpi_card(st.container(), "Model Accuracy (AUC)", f"{auc:.2f}",
             "1.0 = perfect · 0.5 = guessing")
    st.write("")
    # Feature importance chart
    fig_imp = px.bar(
        importance.head(7).sort_values("Importance"),
        x="Importance", y="Factor", orientation="h",
        title="What drives an accessory sale?",
    )
    fig_imp.update_traces(marker_color="#5b8cff")
    fig_imp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cdd8ee", title_font_color="#f0f4fc", height=320,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with cR:
    st.markdown("#### 🎯 Try it: will this customer buy an accessory?")
    st.markdown("<p style='color:#8ea3c8;font-size:14px'>Adjust the customer/sale details and the model predicts the attach probability.</p>", unsafe_allow_html=True)
    in_price = st.select_slider("Device price ($)", [99, 199, 299, 499, 699, 999], value=699)
    in_age = st.selectbox("Customer age band", ["18-25", "26-40", "41-60", "60+"])
    in_weekend = st.radio("Weekend?", ["No", "Yes"], horizontal=True)
    in_premium = st.radio("Premium plan?", ["No", "Yes"], horizontal=True)

    # Build a single-row input matching the model's columns.
    row = {c: 0 for c in feature_cols}
    row["device_price"] = in_price
    row["hour"] = 15
    row["is_weekend"] = 1 if in_weekend == "Yes" else 0
    row["plan_revenue"] = 80 if in_premium == "Yes" else 60
    row["is_premium_plan"] = 1 if in_premium == "Yes" else 0
    age_key = f"age_{in_age}"
    if age_key in row:
        row[age_key] = 1
    X_one = pd.DataFrame([row])[feature_cols]
    prob = model.predict_proba(X_one)[0][1] * 100

    color = "#6fcf97" if prob >= 50 else "#f2c94c" if prob >= 30 else "#eb5757"
    st.markdown(f"""
    <div class='kpi-card' style='text-align:center; margin-top:10px;'>
        <div class='kpi-label'>Predicted Attach Probability</div>
        <div style='font-size:52px;font-weight:800;color:{color};font-family:Sora'>{prob:.0f}%</div>
        <div class='kpi-sub' style='color:#8ea3c8'>{'High - prioritize the pitch' if prob>=50 else 'Medium - worth trying' if prob>=30 else 'Low - focus elsewhere'}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ===============================================================
# SECTION 3: PACE-TO-GOAL
# ===============================================================
st.markdown("<span class='section-tag'>03 · TARGETS</span>", unsafe_allow_html=True)
st.markdown("## 📈 Pace-to-Goal")

pace_rows = []
for store_id in df["store_id"].unique():
    sd = df[df["store_id"] == store_id]
    goal = sd["monthly_store_goal"].iloc[0]
    actual = len(sd)
    days_elapsed = sd["date"].nunique()
    projected = (actual / days_elapsed) * 30 if days_elapsed else 0
    pace_rows.append({
        "Store": sd["store_name"].iloc[0],
        "Goal": goal, "Sales so far": actual,
        "Projected": round(projected),
        "% of Goal": round(projected / goal * 100, 1),
        "Status": "✅ On track" if projected >= goal else "⚠️ Behind",
    })
st.dataframe(pd.DataFrame(pace_rows), use_container_width=True, hide_index=True)
st.write("")

# ===============================================================
# SECTION 4: REVENUE-AT-RISK + REP SCORECARD
# ===============================================================
st.markdown("<span class='section-tag'>04 · COACHING</span>", unsafe_allow_html=True)
st.markdown("## 💸 Revenue-at-Risk & Rep Scorecard")

rep_attach = df.groupby("rep_id")["accessory_attach"].mean()
median_attach = rep_attach.median()
avg_acc_value = df[df["accessory_attach"] == 1]["accessory_revenue"].mean()
recoverable = sum(
    (median_attach - a) * len(df[df["rep_id"] == r]) * avg_acc_value
    for r, a in rep_attach.items() if a < median_attach
)

cA, cB = st.columns([1, 2])
kpi_card(cA, "Monthly Revenue at Risk", f"${recoverable:,.0f}",
         "recoverable via coaching")

scorecard = df.groupby("rep_id").agg(
    Sales=("transaction_id", "count"),
    Accessory=("accessory_attach", "mean"),
    Insurance=("insurance_attach", "mean"),
    ARPU=("plan_revenue", "mean"),
).reset_index()
scorecard["Accessory"] = (scorecard["Accessory"] * 100).round(1)
scorecard["Insurance"] = (scorecard["Insurance"] * 100).round(1)
scorecard["ARPU"] = scorecard["ARPU"].round(0)
cutoff = scorecard["Accessory"].quantile(0.34)
scorecard["Flag"] = scorecard["Accessory"].apply(lambda x: "🔴 Coach" if x <= cutoff else "🟢 OK")
scorecard = scorecard.sort_values("Accessory", ascending=False)
cB.dataframe(scorecard, use_container_width=True, hide_index=True)
st.write("")

# ===============================================================
# SECTION 5: MULTI-STORE COMPARISON
# ===============================================================
st.markdown("<span class='section-tag'>05 · BENCHMARK</span>", unsafe_allow_html=True)
st.markdown("## 🏪 Store Comparison (traffic-adjusted)")

comp = []
for store_id in df["store_id"].unique():
    sd = df[df["store_id"] == store_id]
    t = sd.groupby("date")["daily_traffic"].first().sum()
    comp.append({
        "Store": sd["store_name"].iloc[0],
        "Sales": len(sd),
        "Conversion %": round(len(sd) / t * 100, 1),
        "Accessory %": round(sd["accessory_attach"].mean() * 100, 1),
        "ARPU": round(sd["plan_revenue"].mean(), 0),
    })
comp_df = pd.DataFrame(comp).sort_values("Conversion %", ascending=False)
fig = px.bar(comp_df, x="Store", y="Conversion %", text="Conversion %",
             title="Conversion Rate by Store — the fair ranking")
fig.update_traces(marker_color="#5b8cff", textposition="outside")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                  font_color="#cdd8ee", title_font_color="#f0f4fc",
                  yaxis=dict(gridcolor="rgba(255,255,255,0.06)"))
st.plotly_chart(fig, use_container_width=True)
st.write("")

# ===============================================================
# SECTION 6: AUTOMATIC INSIGHTS & RECOMMENDATIONS
# ===============================================================
st.markdown("<span class='section-tag'>06 · INSIGHTS</span>", unsafe_allow_html=True)
st.markdown("## 🧠 What the Data Is Telling You")

# Best/worst store by conversion
best_store = comp_df.iloc[0]
worst_store = comp_df.iloc[-1]
# Highest-volume store (raw sales) to contrast with fair ranking
high_vol = comp_df.sort_values("Sales", ascending=False).iloc[0]
# Weakest rep
weak_rep = scorecard.sort_values("Accessory").iloc[0]
top_factor = importance.iloc[0]["Factor"]

insights = [
    f"<b>{best_store['Store']}</b> is your best-run store at <b>{best_store['Conversion %']}%</b> conversion — even though <b>{high_vol['Store']}</b> has the most raw sales ({high_vol['Sales']:,}). Volume ≠ efficiency; {high_vol['Store']} is coasting on foot traffic.",
    f"<b>${recoverable:,.0f}/month</b> is recoverable by coaching below-median reps up to the team median accessory-attach rate. <b>{weak_rep['rep_id']}</b> is the priority — high sales volume but only {weak_rep['Accessory']}% attach.",
    f"The predictive model (AUC <b>{auc:.2f}</b>) finds that <b>{top_factor.replace('_',' ')}</b> is the strongest driver of an accessory sale — train reps to pitch hardest on those transactions.",
    f"Insurance attach sits at <b>{df['insurance_attach'].mean()*100:.1f}%</b> — typically the highest-margin recurring add-on, so even a few points of improvement compounds monthly.",
]
for ins in insights:
    st.markdown(f"<div class='insight'>{ins}</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<p style='color:#5b7099;font-size:13px;text-align:center;font-family:DM Mono'>RetailIQ · built with Python, pandas, scikit-learn, Plotly & Streamlit</p>", unsafe_allow_html=True)
