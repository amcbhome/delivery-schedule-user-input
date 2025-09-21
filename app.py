import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("TV Delivery Optimizer")

st.markdown("""
**Minimize the total delivery cost** from depots to stores.  
Adjust depot supplies, store capacities, £/mile, and the depot–store distance matrix.
""")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")

# Labels
depot_labels = ["D1", "D2", "D3"]
store_labels = ["Store 1", "Store 2", "Store 3"]

# Depot supply table
st.sidebar.subheader("Depot Supply (units)")
default_supply_df = pd.DataFrame({"Supply": [2500, 3100, 1250]}, index=depot_labels)
supply_df = st.sidebar.data_editor(
    default_supply_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=False
)
depot_supply = supply_df["Supply"].to_list()

# Store capacity table
st.sidebar.subheader("Store Capacities (units)")
default_caps_df = pd.DataFrame({"Capacity": [2000, 3000, 2000]}, index=store_labels)
caps_df = st.sidebar.data_editor(
    default_caps_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=False
)
store_caps = caps_df["Capacity"].to_list()

# Cost per mile
cost_per_mile = st.sidebar.number_input("Cost per mile (£)", min_value=1, value=5, step=1)

# Distance matrix
st.sidebar.subheader("Distances (miles)")
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
supply_df = supply_df.apply(pd.to_numeric, errors="coerce").fillna(0)
caps_df = caps_df.apply(pd.to_numeric, errors="coerce").fillna(0)
dist_df = dist_df.apply(pd.to_numeric, errors="coerce").fillna(0)

depot_supply = supply_df["Supply"].to_numpy()
store_caps = caps_df["Capacity"].to_numpy()
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
