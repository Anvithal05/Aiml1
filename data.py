import pandas as pd
import numpy as np
import streamlit as st

def generate_data(n=5000):
    np.random.seed(42)

    data = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "vehicle_age": np.random.randint(0, 15, n),
        "premium": np.random.randint(5000, 50000, n),
        "accidents": np.random.randint(0, 5, n),
        "vehicle_type": np.random.choice(["car", "bike", "truck"], n),
        "region": np.random.choice(["urban", "rural"], n),
    })

    # 🔥 CORE LOGIC
    base_claim = 2 * data["premium"]

    reduction_factor = (
        1
        - (data["vehicle_age"] * 0.04)   # vehicle depreciation
        - (data["accidents"] * 0.08)     # accident penalty
    )

    # Keep factor in valid range
    reduction_factor = reduction_factor.clip(0.3, 1)

    data["claim_amount"] = base_claim * reduction_factor

    # Add small randomness
    data["claim_amount"] += np.random.normal(0, 2000, n)

    # Ensure minimum claim
    data["claim_amount"] = data["claim_amount"].clip(lower=5000)

    return data