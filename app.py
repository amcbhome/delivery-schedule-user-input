import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.title("TV Delivery Optimizer")

st.markdown("""
**Minimize the total delivery cost** from depots to stores.  
Use the sidebar to adjust depot supplies, store capacities, cost per mile,  
and the depot–store distance matrix.
""")

# -----------------------------
# Defaults
# -----------------------------
default_supply_df = pd.DataFrame({"Supply": [2500, 3100, 1250]}, index=["D1", "D2", "D3"])
default_caps_df   = pd.DataFrame({"Capacity": [2000, 3000, 2000]}, index=["Store 1", "Store 2", "Store 3"])
default_dist_df   = pd.DataFrame(
    [[22, 33, 40],
     [27, 30, 22],
     [36, 20, 25]],
    index=["D1", "D2", "D3"],
    columns=["Store 1", "Store 2", "Store 3"]
)
default_rate_df   = pd.DataFrame({"£/mile": [5]})

# -----------------------------
# Initialize session state
# -----------------------------
if "supply_df" not in st.session_state:
    st.session_state.supply_df = default_supply_df.copy()
if "caps_df" not in st.session_state:
    st.session_state.caps_df = default_caps_df.copy()
if "dist_df" not in st.session_state:
    st.session_state.dist_df = default_dist_df.copy()
if "rate_df" not in st.session_state:
    st.session_state.rate_df = default_rate_df.copy()

# -----------------------------
# Reset button with toast + rerun (version-safe)
# -----------------------------
if st.sidebar.button("🔄 Reset to Defaults"):
    st.session_state.supply_df = default_supply_df.copy()
    st.session_state.caps_df   = default_caps_df.copy()
    st.session_state.dist_df   = default_dist_df.copy()
    st.session_state.rate_df   = default_rate_df.copy()
    try:
        st.toast("Inputs reset to defaults ✅")
    except Exception:
        pass  # older Streamlit versions may not have st.toast
    if hasattr(st, "rerun"):  # new API
        st.rerun()
    else:                      # fallback for older versions
        st.experimental_rerun()

# -----------------------------
# Sidebar tabs (editable tables)
# -----------------------------
st.sidebar.header("Inputs")
tabs = st.sidebar.tabs(["Depot Supply", "Store Capacities", "Distances", "Cost per Mile"])

with tabs[0]:
    st.session_state.supply_df = st.data_editor(
        st.session_state.supply_df, use_container_width=True, num_rows="fixed"
    )
with tabs[1]:
    st.session_state.caps_df = st.data_editor(
        st.session_state.caps_df, use_container_width=True, num_rows="fixed"
    )
with tabs[2]:
    st.session_state.dist_df = st.data_editor(
        st.session_state.dist_df, use_container_width=True, num_rows="fixed"
    )
with tabs[3]:
    st.session_state.rate_df = st.data_editor(
        st.session_state.rate_df, use_container_width=True, num_rows="fixed"
    )

# Extract numeric arrays for solver
depot_supply  = st.session_state.supply_df["Supply"].to_numpy(dtype=float)
store_caps    = st.session_state.caps_df["Capacity"].to_numpy(dtype=float)
distances     = st.session_state.dist_df.to_numpy(dtype=float)
cost_per_mile = float(st.session_state.rate_df["£/mile"].iloc[0])

# -----------------------------
# Inputs used (read-only summary)
# -----------------------------
st.markdown("## Inputs used")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Depot Supply (units)**")
    st.dataframe(st.session_state.supply_df, use_container_width=True, hide_index=False)
    st.markdown("**Cost per Mile**")
    st.dataframe(st.session_state.rate_df, use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Store Capacities (units)**")
    st.dataframe(st.session_state.caps_df, use_container_width=True, hide_index=False)

st.markdown("**Distances (miles)**")
st.dataframe(st.session_state.dist_df, use_container_width=True, hide_index=False)

# -----------------------------
# Pre-check: supply vs capacity
# -----------------------------
total_supply   = float(depot_supply.sum())
total_capacity = float(store_caps.sum())

if total_supply > total_capacity:
    st.warning(
        f"⚠️ Total supply ({total_supply:,.0f}) exceeds total store capacity ({total_capacity:,.0f}). "
        "Not all TVs can be delivered."
    )
elif total_supply < total_capacity:
    st.info(
        f"ℹ️ Total supply ({total_supply:,.0f}) is less than total store capacity ({total_capacity:,.0f}). "
        "Some capacity will remain unused."
    )
else:
    st.success(
        f"✅ Balanced: total supply ({total_supply:,.0f}) equals total store capacity ({total_capacity:,.0f})."
    )

# -----------------------------
# Linear program
# -----------------------------
# Objective coefficients (£ per unit) for each lane = miles * £/mile
c = (distances * cost_per_mile).flatten()

# Store capacity: sum_i x_{i,j} <= cap_j
A_store = np.zeros((3, 9))
for j in range(3):
    for i in range(3):
        A_store[j, 3*i + j] = 1
b_store = store_caps

# Depot supply: sum_j x_{i,j} = supply_i
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
    st.write(f"### 💰 Cost of delivery: £{float(res.fun):,.0f}")
else:
    st.error("Optimization failed: " + res.message)

