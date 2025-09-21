import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("TV Delivery Optimizer")

st.markdown("""
**This app minimizes the total delivery cost of transporting televisions from depots to stores.**  
You can adjust the number of TVs at each depot, store capacity limits, and the delivery cost per mile.
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

# Distances (fixed matrix from the ACCA case)
distances = np.array([
    [22, 33, 40],  # D1 → Stores 1–3
    [27, 30, 22],  # D2 → Stores 1–3
    [36, 20, 25],  # D3 → Stores 1–3
])

# -----------------------------
# Problem Setup (text only)
# -----------------------------
st.markdown("## Problem Setup")

dep_sentence = (
    f"There are {depot_supply[0]:,} TVs to be delivered from {depot_labels[0]}, "
    f"{depot_supply[1]:,} from {depot_labels[1]}, and "
    f"{depot_supply[2]:,} from {depot_labels[2]}."
)
store_sentence = (
    f"Store capacity limits are {store_caps[0]:,} for {store_labels[0]}, "
    f"{store_caps[1]:,} for {store_labels[1]}, and "
    f"{store_caps[2]:,} for {store_labels[2]}."
)

st.write(dep_sentence)
st.write(store_sentence)
st.markdown("<br>", unsafe_allow_html=True)

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
# Results
# -----------------------------
st.markdown("## Results")

def _centered_table_html(df: pd.DataFrame) -> str:
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'center')]},
        {'selector': 'td', 'props': [('text-align', 'center')]}
    ]).to_html()

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🗺️ Distance Matrix (miles)")
    distance_df = pd.DataFrame(distances, index=depot_labels, columns=store_labels)
    st.markdown(_centered_table_html(distance_df), unsafe_allow_html=True)

with col_right:
    st.markdown("### ✅ Optimized Shipment Plan")
    if res.success:
        x = np.round(res.x).astype(int).reshape(3, 3)
        shipment_df = pd.DataFrame(x, index=depot_labels, columns=store_labels)
        st.markdown(_centered_table_html(shipment_df), unsafe_allow_html=True)
    else:
        st.error("Optimization failed: " + res.message)

if res.success:
    total_cost = float(res.fun)
    avg_cost = total_cost / sum(depot_supply) if sum(depot_supply) > 0 else 0
    st.write(f"### 💰 Total Delivery Cost: £{total_cost:,.0f}")
    st.write(f"### 📊 Average Cost per TV: £{avg_cost:,.2f}")
