# 📡 RetailIQ — Metro by T-Mobile Store Performance & Revenue Intelligence Engine

A Python analytics engine that turns wireless retail transaction data into
manager-ready intelligence, built around the **five core Metro sales targets**:

| KPI | What it measures |
|---|---|
| Magenta migrations | Metro prepaid → T-Mobile postpaid conversions |
| Voice lines | Primary phone activations |
| Tablet lines | Secondary connected-device adds |
| Home Internet (HSI) | High-value home internet attach |
| Insurance (P360) | Protection 360 recurring add-on |

Plus a **machine-learning model** that predicts which customers will buy
accessories, fair traffic-adjusted store benchmarking, a named-rep coaching
scorecard, pace-to-goal forecasting, and an auto-generated insights section.

> Built to solve real operational challenges observed in wireless retail.
> **All data is fully simulated** — no real customer, store, or employer data is used.

---

## Sections

| # | Section | Business question |
|---|---|---|
| 01 | Core Targets | How are the 5 KPIs + conversion + revenue tracking? |
| 02 | ML Attach Predictor | Which customers are most likely to buy accessories? |
| 03 | Pace-to-Goal | Will each store hit its monthly voice-line target? |
| 04 | Coaching | How much revenue is recoverable, and which rep is the priority? |
| 05 | Store Benchmark | Which store performs best *after adjusting for traffic*? |
| 06 | Insights | What should a manager actually do about it? |

---

## How to run

```
python -m pip install pandas numpy scikit-learn streamlit plotly
python generate_data.py
python -m streamlit run dashboard_pro.py
```
Opens at `http://localhost:8501`.

---

## Tech stack
Python | pandas | NumPy | scikit-learn (Random Forest) | Plotly | Streamlit

---

## Resume bullet points

- Built **RetailIQ**, a Python retail-analytics engine (pandas, scikit-learn, Streamlit, Plotly) modeling Metro by T-Mobile store operations across the five core sales targets (Magenta migrations, voice/tablet lines, Home Internet, P360 insurance).
- Trained a **Random Forest model (AUC ~0.65)** to predict accessory-attach likelihood per transaction, surfacing device price and customer demographics as the strongest sales drivers.
- Quantified **$K/month recoverable revenue** from below-median accessory-attach rates and built a named-rep coaching scorecard flagging the highest-ROI coaching target.
- Developed **pace-to-goal forecasting** and **traffic-normalized store benchmarking** to replace misleading raw-sales rankings and enable mid-month course-correction.
