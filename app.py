import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("TV Delivery Optimizer")

st.markdown("""
**Minimize total delivery cost** from depots to stores.  
Set depot supplies, store capacities, **£/mile**, and edit the **3×3 distance matrix**.
""")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")

# Labels
depot_labels = ["D1", "D2", "D3"]
store_labels = ["Store 1", "Store 2", "Store 3"]

# Depot supply
st.sidebar.subheader("Depot Supply")
default_supply = [2500, 3100, 1250]
depot_supply = [
    st.sidebar.number_input(f"{d} supply", min_value=0, value=val, step=50)
    for d, val in zip(depot_labels, default_supply)
]

# Store capacities
st.sidebar.subheader("Store Capacities")
default_caps = [2000, 3000, 2000]
store_caps = [
    st.sidebar.number_input(f"{s} capacity", min_value=0, value=val, step=50)
    for s, val in zip(store_labels, default_caps)
]

# Cost per mile
cost_per_mile = st.sidebar.number_input("Cost per mile (£)", min_value=1, value=5, step=1)

# 3×3 distance matrix editor
st.sidebar.subheader("Distances (miles) — 3×3 Matrix")
default_dist_df = pd.DataFrame(
    [[22, 33, 40],
     [27, 30, 22],
     [36, 20, 25]],
    index=depot_labels, columns=store_labels
)

dist_df = st.sidebar.data_editor(
    default_dist_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=False
)

# Coerce to numeric just in case
dist_df = dist_df.apply(pd.to_numeric, errors="coerce").fillna(0)
distances = dist_df.to_numpy()

# -----------------------------
# Linear program
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

res = linprog(
    c=c,
    A_ub=A_store, b_ub=b_store,
    A_eq=A_depot, b_eq=b_depot,
    bounds=bounds,
    method="highs"
)

# -----------------------------
# Output (minimal)
# -----------------------------
st.markdown("## Results")
if res.success:
    total_cost = float(res.fun)
    st.write(f"### 💰 Cost of delivery: £{total_cost:,.0f}")
else:
    st.error("Optimization failed: " + res.message)
