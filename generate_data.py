"""
RetailIQ - Data Generator
==========================
This script creates realistic (but SIMULATED) wireless retail store
transaction data, modeled on Metro by T-Mobile / Total Wireless operations.

Why we simulate: real store data is confidential. For a portfolio project,
clearly-labeled simulated data is completely acceptable and standard practice.

Output: a CSV file called 'store_sales.csv' that the dashboard will read.

Run it with:  python generate_data.py
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------------
# A "seed" makes the random data the same every time you run it,
# so your numbers don't change unexpectedly while building.
np.random.seed(42)

# Our simulated stores. Each has a different traffic level and skill,
# so the analysis has realistic variation to find.
STORES = {
    "MTR_001": {"name": "Downtown",   "daily_traffic": 90, "skill": 1.10},
    "MTR_002": {"name": "Mall",       "daily_traffic": 130, "skill": 1.00},
    "MTR_003": {"name": "Suburb",     "daily_traffic": 60, "skill": 0.95},
    "MTR_004": {"name": "Highway",    "daily_traffic": 45, "skill": 0.85},
}

# Reps assigned to each store (rep_id -> store_id).
# Each rep has a personal "skill" multiplier that affects their attach rates.
REPS = {
    "R_01": {"store": "MTR_001", "skill": 1.20},
    "R_02": {"store": "MTR_001", "skill": 0.90},
    "R_03": {"store": "MTR_002", "skill": 1.05},
    "R_04": {"store": "MTR_002", "skill": 0.80},  # weak rep - coaching target
    "R_05": {"store": "MTR_003", "skill": 1.00},
    "R_06": {"store": "MTR_004", "skill": 0.95},
}

# Monthly activation goals per store (for pace-to-goal tracking).
STORE_GOALS = {
    "MTR_001": 320,
    "MTR_002": 420,
    "MTR_003": 200,
    "MTR_004": 150,
}

# Plan options: name, monthly revenue (ARPU), and how likely it is to be sold.
PLANS = [
    {"plan": "Basic",       "revenue": 40, "weight": 0.45},
    {"plan": "Unlimited",   "revenue": 60, "weight": 0.40},
    {"plan": "Premium",     "revenue": 80, "weight": 0.15},
]

# Date range: one full month of data.
DATES = pd.date_range("2025-11-01", "2025-11-30", freq="D")

# ---------------------------------------------------------------
# 2. BUILD THE TRANSACTIONS
# ---------------------------------------------------------------
rows = []                # we'll collect each sale as a dictionary here
transaction_counter = 1  # gives every sale a unique ID

for date in DATES:
    for store_id, store in STORES.items():

        # Daily traffic varies +/- 25% randomly around the store's average.
        traffic = int(store["daily_traffic"] * np.random.uniform(0.75, 1.25))

        # Conversion: roughly 35% of visitors buy, adjusted by store skill.
        base_conversion = 0.35 * store["skill"]
        activations = int(traffic * base_conversion)

        # Find which reps work at this store.
        store_reps = [r for r, info in REPS.items() if info["store"] == store_id]

        # Create one row per activation (a completed sale).
        for _ in range(activations):
            rep_id = np.random.choice(store_reps)
            rep_skill = REPS[rep_id]["skill"]

            # Pick a plan based on the weighted probabilities above.
            plan = np.random.choice(
                [p["plan"] for p in PLANS],
                p=[p["weight"] for p in PLANS],
            )
            plan_revenue = next(p["revenue"] for p in PLANS if p["plan"] == plan)

            # Device price: random realistic phone price.
            device_price = round(np.random.choice([99, 199, 299, 499, 699, 999]), 2)

            # --- Extra features that realistically influence buying behavior ---
            # (These give the machine-learning model real patterns to learn.)

            # Hour of sale (stores open ~10am-8pm).
            hour = int(np.random.choice(range(10, 21)))

            # Weekend flag (Sat/Sun shoppers behave a bit differently).
            is_weekend = int(date.weekday() >= 5)

            # Customer age band (younger buyers attach accessories more).
            age_band = np.random.choice(["18-25", "26-40", "41-60", "60+"],
                                        p=[0.30, 0.35, 0.25, 0.10])

            # Accessory attach probability is driven by REAL, learnable factors.
            # We make these effects strong and clear so the ML model can detect them.
            attach_prob = 0.18 * rep_skill
            attach_prob += 0.00045 * device_price          # pricier phone -> much higher
            attach_prob += 0.20 if age_band == "18-25" else 0.0
            attach_prob += 0.10 if age_band == "26-40" else 0.0
            attach_prob -= 0.08 if age_band == "60+" else 0.0
            attach_prob += 0.07 if is_weekend else 0.0
            attach_prob += 0.10 if plan == "Premium" else 0.0  # premium buyers spend more
            attach_prob = float(np.clip(attach_prob, 0.03, 0.95))

            accessory_attach = int(np.random.random() < attach_prob)
            accessory_revenue = round(np.random.uniform(25, 120), 2) if accessory_attach else 0.0

            # Insurance (TWP) attach: base 30% chance, scaled by rep skill.
            insurance_attach = int(np.random.random() < (0.30 * rep_skill))
            insurance_revenue = 15.0 if insurance_attach else 0.0  # monthly recurring

            # Upgrade flag: 20% of sales are upgrades to existing customers.
            upgrade_flag = int(np.random.random() < 0.20)

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "store_id": store_id,
                "store_name": store["name"],
                "rep_id": rep_id,
                "transaction_id": f"T_{transaction_counter:05d}",
                "daily_traffic": traffic,
                "plan_type": plan,
                "plan_revenue": plan_revenue,
                "device_price": device_price,
                "hour": hour,
                "is_weekend": is_weekend,
                "age_band": age_band,
                "accessory_attach": accessory_attach,
                "accessory_revenue": accessory_revenue,
                "insurance_attach": insurance_attach,
                "insurance_revenue": insurance_revenue,
                "upgrade_flag": upgrade_flag,
                "monthly_store_goal": STORE_GOALS[store_id],
            })
            transaction_counter += 1

# ---------------------------------------------------------------
# 3. SAVE TO CSV
# ---------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv("store_sales.csv", index=False)

print(f"Done! Created store_sales.csv with {len(df)} transactions.")
print(f"Stores: {df['store_id'].nunique()} | Reps: {df['rep_id'].nunique()} | Days: {df['date'].nunique()}")
print("\nFirst few rows:")
print(df.head().to_string())
