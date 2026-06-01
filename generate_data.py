"""
RetailIQ - Data Generator (Metro by T-Mobile edition)
=====================================================
Creates realistic (SIMULATED) wireless retail transaction data modeled on
Metro by T-Mobile store operations.

Tracks the FIVE core sales targets a real Metro store manages:
  1. Magenta migration  - moving a Metro customer to a T-Mobile postpaid plan
  2. Voice lines         - primary phone activations
  3. Tablet lines        - secondary connected-device adds
  4. Home Internet (HSI) - high-value home internet attach
  5. Insurance (P360)    - Protection 360 recurring add-on

NOTE: Data is fully simulated for portfolio use. No real customer, store,
or employer data is used.

Run with:  python generate_data.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------
# STORES  (anonymized labels, simulated performance)
# ---------------------------------------------------------------
STORES = {
    "Store A": {"daily_traffic": 95, "skill": 1.10},
    "Store B": {"daily_traffic": 130, "skill": 1.00},
    "Store C": {"daily_traffic": 70, "skill": 0.95},
    "Store D": {"daily_traffic": 55, "skill": 0.90},
    "Store E": {"daily_traffic": 80, "skill": 1.05},
    "Store F": {"daily_traffic": 48, "skill": 0.85},
}

# ---------------------------------------------------------------
# REPS  (realistic names; each has a personal skill multiplier)
# ---------------------------------------------------------------
REPS = {
    "Jasmine": {"store": "Store A", "skill": 1.20},
    "Carlos":  {"store": "Store A", "skill": 0.95},
    "Tina":    {"store": "Store B", "skill": 1.05},
    "Marcus":  {"store": "Store B", "skill": 0.80},   # coaching target
    "Priya":   {"store": "Store C", "skill": 1.00},
    "Devon":   {"store": "Store D", "skill": 0.88},
    "Sofia":   {"store": "Store E", "skill": 1.10},
    "Andre":   {"store": "Store F", "skill": 0.85},
}

# Monthly activation (voice-line) goals per store.
STORE_GOALS = {
    "Store A": 340, "Store B": 430, "Store C": 230,
    "Store D": 175, "Store E": 290, "Store F": 150,
}

# Rate plans (T-Mobile / Metro style) with monthly revenue and likelihood.
PLANS = [
    {"plan": "Metro $40",        "revenue": 40, "weight": 0.40, "magenta": False},
    {"plan": "Metro Unlimited",  "revenue": 60, "weight": 0.32, "magenta": False},
    {"plan": "Magenta",          "revenue": 75, "weight": 0.18, "magenta": True},
    {"plan": "Magenta MAX",      "revenue": 90, "weight": 0.10, "magenta": True},
]

DATES = pd.date_range("2025-11-01", "2025-11-30", freq="D")

# ---------------------------------------------------------------
# BUILD TRANSACTIONS
# ---------------------------------------------------------------
rows = []
txn = 1

for date in DATES:
    for store_name, store in STORES.items():
        traffic = int(store["daily_traffic"] * np.random.uniform(0.75, 1.25))
        activations = int(traffic * 0.35 * store["skill"])
        store_reps = [r for r, info in REPS.items() if info["store"] == store_name]

        for _ in range(activations):
            rep = np.random.choice(store_reps)
            skill = REPS[rep]["skill"]

            # ---- Plan / Magenta migration ----
            plan = np.random.choice(
                [p["plan"] for p in PLANS],
                p=[p["weight"] for p in PLANS],
            )
            plan_info = next(p for p in PLANS if p["plan"] == plan)
            plan_revenue = plan_info["revenue"]
            magenta_migration = int(plan_info["magenta"])   # KPI 1

            # ---- Voice line: every activation is at least one voice line ----
            voice_lines = 1                                 # KPI 2
            # ~25% of customers add a second voice line (family).
            if np.random.random() < 0.25 * skill:
                voice_lines += 1

            # ---- Device price (drives accessory + insurance behavior) ----
            device_price = float(np.random.choice([99, 199, 299, 499, 699, 999]))

            # ---- Customer demographics / context ----
            hour = int(np.random.choice(range(10, 21)))
            is_weekend = int(date.weekday() >= 5)
            age_band = np.random.choice(
                ["18-25", "26-40", "41-60", "60+"], p=[0.30, 0.35, 0.25, 0.10]
            )

            # ---- Tablet line attach (KPI 3) ----
            tab_prob = 0.12 * skill + (0.05 if magenta_migration else 0)
            tablet_lines = int(np.random.random() < tab_prob)

            # ---- Home Internet / HSI attach (KPI 4) ----
            # More likely on higher plans and Magenta migrations.
            hsi_prob = 0.10 * skill + (0.08 if magenta_migration else 0)
            home_internet = int(np.random.random() < hsi_prob)
            hsi_revenue = 50.0 if home_internet else 0.0

            # ---- Accessory attach (learnable signal for ML model) ----
            attach_prob = 0.18 * skill
            attach_prob += 0.00045 * device_price
            attach_prob += 0.20 if age_band == "18-25" else 0.0
            attach_prob += 0.10 if age_band == "26-40" else 0.0
            attach_prob -= 0.08 if age_band == "60+" else 0.0
            attach_prob += 0.07 if is_weekend else 0.0
            attach_prob += 0.10 if plan_info["magenta"] else 0.0
            attach_prob = float(np.clip(attach_prob, 0.03, 0.95))
            accessory_attach = int(np.random.random() < attach_prob)
            accessory_revenue = round(np.random.uniform(25, 120), 2) if accessory_attach else 0.0

            # ---- Insurance / Protection 360 (KPI 5) ----
            ins_prob = 0.28 * skill + 0.0002 * device_price
            insurance_attach = int(np.random.random() < float(np.clip(ins_prob, 0.05, 0.9)))
            insurance_revenue = 18.0 if insurance_attach else 0.0   # P360 monthly

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "store_name": store_name,
                "rep_name": rep,
                "transaction_id": f"T_{txn:05d}",
                "daily_traffic": traffic,
                "plan_type": plan,
                "plan_revenue": plan_revenue,
                "magenta_migration": magenta_migration,
                "voice_lines": voice_lines,
                "tablet_lines": tablet_lines,
                "home_internet": home_internet,
                "hsi_revenue": hsi_revenue,
                "device_price": device_price,
                "hour": hour,
                "is_weekend": is_weekend,
                "age_band": age_band,
                "accessory_attach": accessory_attach,
                "accessory_revenue": accessory_revenue,
                "insurance_attach": insurance_attach,
                "insurance_revenue": insurance_revenue,
                "monthly_store_goal": STORE_GOALS[store_name],
            })
            txn += 1

df = pd.DataFrame(rows)
df.to_csv("store_sales.csv", index=False)

print(f"Done! Created store_sales.csv with {len(df)} transactions.")
print(f"Stores: {df['store_name'].nunique()} | Reps: {df['rep_name'].nunique()} | Days: {df['date'].nunique()}")
print("\nCore KPI totals:")
print(f"  Magenta migrations: {df['magenta_migration'].sum()}")
print(f"  Voice lines:        {df['voice_lines'].sum()}")
print(f"  Tablet lines:       {df['tablet_lines'].sum()}")
print(f"  Home Internet:      {df['home_internet'].sum()}")
print(f"  Insurance (P360):   {df['insurance_attach'].sum()}")
