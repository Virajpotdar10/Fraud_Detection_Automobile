import pandas as pd
import numpy as np
import os

os.makedirs("dataset", exist_ok=True)

np.random.seed(42)

n_samples = 500

data = {
    "months_as_customer": np.random.randint(1, 400, n_samples),
    "age": np.random.randint(18, 65, n_samples),
    "policy_state": np.random.choice(["OH", "IL", "IN"], n_samples),
    "policy_deductable": np.random.choice([500, 1000, 2000], n_samples),
    "policy_annual_premium": np.round(np.random.uniform(500, 2500, n_samples), 2),
    "umbrella_limit": np.random.choice([0, 1000000, 2000000, 3000000], n_samples),
    "incident_type": np.random.choice(["Multi-vehicle Collision", "Single Vehicle Collision", "Vehicle Theft", "Parked Car"], n_samples),
    "collision_type": np.random.choice(["Rear Collision", "Side Collision", "Front Collision", "?"], n_samples),
    "incident_severity": np.random.choice(["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"], n_samples),
    "incident_state": np.random.choice(["SC", "VA", "NY", "OH", "WV", "NC", "PA"], n_samples),
    "incident_hour_of_the_day": np.random.randint(0, 24, n_samples),
    "number_of_vehicles_involved": np.random.randint(1, 5, n_samples),
    "bodily_injuries": np.random.randint(0, 3, n_samples),
    "witnesses": np.random.randint(0, 4, n_samples),
    "injury_claim": np.random.randint(0, 30000, n_samples),
    "property_claim": np.random.randint(0, 30000, n_samples),
    "vehicle_claim": np.random.randint(1000, 80000, n_samples),
    "auto_year": np.random.randint(1995, 2023, n_samples),
}

data["total_claim_amount"] = data["injury_claim"] + data["property_claim"] + data["vehicle_claim"]

df = pd.DataFrame(data)

prob_fraud = (df["total_claim_amount"] > 60000).astype(int) + \
             (df["incident_severity"] == "Major Damage").astype(int) + \
             (df["witnesses"] == 0).astype(int)

prob_fraud = prob_fraud / prob_fraud.max()
prob_fraud = prob_fraud * 0.7 + np.random.uniform(0, 0.3, n_samples)

df["fraud_reported"] = (prob_fraud > 0.6).astype(int)

df.to_csv("dataset/insurance_fraud.csv", index=False)
print("Dataset 'dataset/insurance_fraud.csv' generated successfully with 500 samples.")
