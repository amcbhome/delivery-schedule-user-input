import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("TV Delivery Optimizer")

st.markdown("""
**This app minimizes the total delivery cost of transporting televisions from depots to stores.**  
Adjust depot supplies, store capacities, cost per mile, and the depot-to-store distance matrix.
""")

# -----------------------------
# User Inputs
# -----------------------------
st.sidebar.header("Inputs")

# Depot supply
st.sidebar.subheader("Depot Supply")
depot_labels = ["D1", "D2", "D3"]
depot_supply = [st.sidebar.number_input(f"{d} supply", min_value=0, value=val)
                for d, val in zip(depot_labels, [2500, 3100, 1250])]

# Store capacities
st.sidebar.subheader("Store Capacities")
store_labels = ["Store 1", "Store 2", "Store 3"]
store_caps = [st.sidebar.number_input(f"{s} capacity", min_value=0, value=val)
              for s, val in zip(store_labels, [2000, 3000, 2000])]

# Cost per mile
cost_per_mile = st.sidebar.number_input("Cost per mile (£)", min_value=1, value=5)

# Distance matrix input
st.sidebar.subheader("Distances (miles)")
distances = []
for i, d in enumerate(depot_labels):
    row = []
    for j, s in enumerate(store_labels):
        default_vals = [[22, 33, 40],
                        [27, 30, 22],
                        [36, 20, 25]]
        val = st.sidebar.number_input(f"{d} → {s}", min_value=1, value=default_vals[i][j])
        row.append(val)
    distances.append(row)

distances = np.array(distances)

# -----------------------------
# Linear Programming Setup
# -----------------------------
c = (distances * cost_per_mile).flatten()

# Store (capacity) constraints: sum over depots to each store <= capacity
A_store = np.zeros((3, 9))
for j in range(3):
    for i in range(3):
        A_store[j, 3*i + j] = 1
b_store = store_caps

# Depot (supply) constraints: sum to all stores = supply
A_depot = np.zeros((3, 9))
for i in range(3):
    A_depot[i, 3*i:3*i+3] = 1
b_depot = depot_supply

bounds = [(0, None) for _ in range(9)]

# Solve
res = linprog(
    c=c,
    A_ub=A_store, b_ub=b_store,
    A_eq=A_depot, b_eq=b_depot,
    bounds=bounds,
    method="highs"
)

# -----------------------------
# Output
# -----------------------------
st.markdown("## Results")

if res.success:
    total_cost = float(res.fun)
    st.write(f"### 💰 Cost of delivery: £{total_cost:,.0f}")
else:
    st.error("Optimization failed: " + res.message)

