# 📡 RetailIQ — Wireless Store Performance & Revenue Intelligence Engine

A Python analytics tool that turns wireless retail transaction data into
manager-ready intelligence: conversion, add-on/insurance attach, pace-to-goal,
revenue-at-risk, rep coaching scorecards, fair multi-store benchmarking — plus a
**machine-learning model** that predicts which customers will buy accessories,
and an **automatic insights** section that states the business story in plain English.

> Built to solve real operational problems observed in wireless retail
> (Metro by T-Mobile / Total Wireless style operations). Uses clearly-labeled
> simulated data.

---

## Features

| # | Section | Business question it answers |
|---|---|---|
| 01 | Overview | How are conversion, ARPU, and attach rates doing right now? |
| 02 | **ML Attach Predictor** | Which customers are most likely to buy an accessory? |
| 03 | Pace-to-Goal | Will each store hit its monthly target at current pace? |
| 04 | Revenue-at-Risk + Scorecard | How much money is recoverable via coaching, and who? |
| 05 | Store Comparison | Which store performs best *after adjusting for traffic*? |
| 06 | **Auto Insights** | What should a manager actually do about all this? |

---

## How to run it

### 1. Install the libraries (one time)
```
python -m pip install pandas numpy scikit-learn streamlit plotly
```

### 2. Generate the data
```
python generate_data.py
```

### 3. Launch the PRO dashboard
```
python -m streamlit run dashboard_pro.py
```
Your browser opens at `http://localhost:8501`.

> `dashboard.py` is the simpler original version; `dashboard_pro.py` is the
> upgraded one with design, ML, and insights.

---

## Tech stack
Python | pandas | NumPy | scikit-learn (Random Forest) | Plotly | Streamlit

---

## Putting this on GitHub (step by step)

1. Create a free account at **https://github.com**
2. Click the **+** (top right) then **New repository**
3. Name it `retailiq-wireless-analytics`, set to **Public**, check **"Add a README"**, click **Create**
4. On the repo page click **Add file then Upload files**
5. Drag in: `generate_data.py`, `dashboard.py`, `dashboard_pro.py`, `store_sales.csv`, and this `README.md`
6. Click **Commit changes**

Your project is now live and you can put the link on your resume and LinkedIn.

---

## Resume bullet points

- Built **RetailIQ**, a Python retail-analytics engine (pandas, scikit-learn, Streamlit, Plotly) that diagnoses wireless-store performance and quantifies **$6K+/month** in recoverable revenue from low accessory-attach rates.
- Trained a **Random Forest model (AUC 0.67)** to predict accessory-attach likelihood per transaction, surfacing device price and customer demographics as the top sales drivers.
- Developed a **pace-to-goal forecasting** module projecting month-end sales vs. target, enabling mid-month course-correction instead of end-of-month surprises.
- Designed a **traffic-normalized multi-store benchmark** that replaced misleading raw-sales rankings, plus per-rep coaching scorecards flagging the highest-ROI coaching targets.
