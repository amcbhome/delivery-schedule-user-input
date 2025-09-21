# app.py
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import linprog

st.set_page_config(page_title="TV Delivery Optimizer", layout="wide")
st.title("TV Delivery Optimizer")

st.markdown(
    """
This app solves a 3×3 transportation problem (3 depots → 3 stores) using linear programming to minimise delivery cost.

**Inputs**
- A 3×3 **Distance** matrix (miles)
- A 3×3 **Rate** matrix (£ per mile). Cost per route = Distance × Rate
- A 1×3 row for **Depot Supply** (units available at D1…D3)
- A 1×3 row for **Store Capacity** (units required at S1…S3)

**Output**
- **Cost of delivery:** total minimum cost (in £)
- Optional: Optimised shipment plan (expand to view)

If total supply ≠ total demand, you can tick **Auto‑balance totals** to add a dummy depot/store with zero cost so the model remains solvable.
"""
)

# -------- Helpers ---------

def mk_df(values, index, columns, dtype=float):
    df = pd.DataFrame(values, index=index, columns=columns)
    return df.astype(dtype)

@st.cache_data(show_spinner=False)
def default_inputs():
    # Modest defaults
    dist = mk_df(
        [[10, 20, 30],
         [15, 10, 25],
         [20, 15, 10]],
        index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]
    )
    rate = mk_df(
        [[5, 5, 5],
         [5, 5, 5],
         [5, 5, 5]],
        index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]
    )
    supply = mk_df([[2500, 3100, 1250]], index=["Supply"], columns=["D1", "D2", "D3"])
    demand = mk_df([[2400, 2400, 2050]], index=["Capacity"], columns=["S1", "S2", "S3"])
    return dist, rate, supply, demand

# ---------- Inputs ----------

dist_df, rate_df, supply_row, demand_row = default_inputs()

st.subheader("Inputs")
col_dist, col_rate = st.columns(2)
with col_dist:
    st.markdown("**Distance (miles)** — 3×3")
    dist_df = st.data_editor(
        dist_df, use_container_width=True, num_rows="fixed", key="distance_editor"
    )
with col_rate:
    st.markdown("**Rate (£/mile)** — 3×3")
    rate_df = st.data_editor(
        rate_df, use_container_width=True, num_rows="fixed", key="rate_editor"
    )

col_supply, col_demand = st.columns(2)
with col_supply:
    st.markdown("**Depot Supply (units)** — 1×3")
    supply_row = st.data_editor(
        supply_row, use_container_width=True, num_rows="fixed", key="supply_editor"
    )
with col_demand:
    st.markdown("**Store Capacity (units)** — 1×3")
    demand_row = st.data_editor(
        demand_row, use_container_width=True, num_rows="fixed", key="demand_editor"
    )

auto_balance = st.checkbox("Auto‑balance totals with a dummy node (zero cost)", value=True)

# -------- Build cost matrix --------
try:
    dist = dist_df.to_numpy(dtype=float)
    rate = rate_df.to_numpy(dtype=float)
    cost = dist * rate
except Exception as e:
    st.error(f"Unable to compute cost matrix from Distance × Rate: {e}")
    st.stop()

sup = supply_row.to_numpy(dtype=float).ravel()
dem = demand_row.to_numpy(dtype=float).ravel()

sum_sup, sum_dem = float(sup.sum()), float(dem.sum())

# --------- Transport model (with optional dummy balancing) ----------

def transport_min_cost(cost_mat, supply, demand):
    m, n = cost_mat.shape
    c = cost_mat.reshape(-1)

    # Variables x_ij >= 0
    bounds = [(0, None)] * (m * n)

    A_eq = []
    b_eq = []

    # Supply equalities: for each i, sum_j x_ij = supply_i
    for i in range(m):
        row = np.zeros(m * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])

    # Demand equalities: for each j, sum_i x_ij = demand_j
    for j in range(n):
        row = np.zeros(m * n)
        for i in range(m):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    res = linprog(
        c,
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    return res

# Balance totals if needed
cost_bal = cost.copy()
sup_bal = sup.copy()
dem_bal = dem.copy()
row_labels = ["D1", "D2", "D3"]
col_labels = ["S1", "S2", "S3"]

if not np.isclose(sum_sup, sum_dem):
    if not auto_balance:
        st.warning(
            f"Totals are not equal (Supply = {sum_sup:,.0f}, Demand = {sum_dem:,.0f}). "
            "Either adjust inputs so they match or tick 'Auto‑balance totals'."
        )
        st.stop()
    diff = sum_sup - sum_dem
    if diff > 0:
        # Extra supply → add dummy store
        cost_bal = np.c_[cost_bal, np.zeros((3, 1))]
        dem_bal = np.r_[dem_bal, [diff]]
        col_labels = col_labels + ["DummyStore"]
    else:
        # Extra demand → add dummy depot
        cost_bal = np.r_[cost_bal, [np.zeros(3)]]
        sup_bal = np.r_[sup_bal, [-diff]]
        row_labels = row_labels + ["DummyDepot"]

# ---------- Solve ----------
res = transport_min_cost(cost_bal, sup_bal, dem_bal)

if not res.success:
    st.error("Optimiser failed: " + res.message)
    st.stop()

x = res.x.reshape(len(row_labels), len(col_labels))
plan_df = pd.DataFrame(x, index=row_labels, columns=col_labels)

# Ignore shipments to/from dummy in the cost unless user wants to see them
actual_mask_rows = [not r.startswith("Dummy") for r in row_labels]
actual_mask_cols = [not c.startswith("Dummy") for c in col_labels]
actual_cost = (plan_df.values[np.ix_(actual_mask_rows, actual_mask_cols)] * cost).sum()

# ---------- Output ----------

st.markdown("## Cost of delivery:")
st.metric(label="Minimum total cost (\u00a3)", value=f"{actual_cost:,.2f}")

with st.expander("Show optimised shipment plan (units)"):
    st.dataframe(plan_df.style.format("{:.2f}"), use_container_width=True)

with st.expander("Show cost matrix (\u00a3 per unit)"):
    st.dataframe(pd.DataFrame(cost, index=["D1","D2","D3"], columns=["S1","S2","S3"]).style.format("{:.2f}"), use_container_width=True)

st.caption("Built with SciPy linprog (HiGHS). If totals differ, a zero‑cost dummy node keeps the model feasible.")


# requirements.txt
streamlit>=1.36
numpy>=1.26
pandas>=2.0
scipy>=1.11

# README.md
# TV Delivery Optimizer (3×3 Transport Model)

A Streamlit app that minimises delivery cost from 3 depots to 3 stores using a classic transportation linear program.

## Features
#- Editable **Distance** (miles) and **Rate** (£/mile) 3×3 matrices
#- Editable **Depot Supply** and **Store Capacity** rows (1×3)
#- Cost matrix computed as *Distance × Rate*
#- **Auto‑balance totals** option adds a zero‑cost dummy depot/store if total supply ≠ total demand
#- Shows **Cost of delivery** (total minimum £) with optional plan/cost matrix expanders

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Uses `scipy.optimize.linprog` with the HiGHS solver.
- When auto‑balancing is ON, shipments to the dummy node are displayed but excluded from the reported total cost.


# --- Addendum: requirements.txt (updated) ---
streamlit>=1.36
numpy>=1.26
pandas>=2.0
scipy>=1.11
googlemaps>=4.10

# --- Addendum: README.md (Google Maps helper) ---
## Google Maps driving distance (optional)
- Enable **Distance Matrix API** in your Google Cloud project and create an API key. Restrict the key and enable billing. 
- In the app sidebar, paste your key and enter two UK postcodes; click **Fetch driving distance** to see miles and duration.
- Docs: Google Maps Distance Matrix API overview, getting started, and client libraries.

