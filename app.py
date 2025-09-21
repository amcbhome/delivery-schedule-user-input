# -*- coding: utf-8 -*-
# TV Delivery Optimizer (3x3 transportation model)
# - Streamlit UI for distance, rate, supply, demand
# - Linear program solved with SciPy linprog (HiGHS)
# - Google Maps Distance Matrix helper: single pair and 3x3 matrix from postcodes

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# Optional: Google Maps client (install 'googlemaps' and set key)
try:
    import googlemaps  # noqa: F401
except Exception:
    googlemaps = None

# Page setup
st.set_page_config(page_title="TV Delivery Optimizer", layout="wide")
st.title("TV Delivery Optimizer")

# One-line intro shown in the UI
st.markdown(
    "Solve a 3x3 transportation problem (3 depots -> 3 stores) to minimise delivery cost using linear programming."
)

# Helper to create typed DataFrames
def mk_df(values, index, columns, dtype=float):
    # Create a typed DataFrame with given index/columns
    df = pd.DataFrame(values, index=index, columns=columns)
    return df.astype(dtype)

# Default inputs
@st.cache_data(show_spinner=False)
def default_inputs():
    # Distance defaults (miles)
    dist = mk_df(
        [[10, 20, 30],
         [15, 10, 25],
         [20, 15, 10]],
        index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]
    )
    # Rate defaults (GBP per mile)
    rate = mk_df(
        [[5, 5, 5],
         [5, 5, 5],
         [5, 5, 5]],
        index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]
    )
    # Supply defaults (units)
    supply = mk_df([[2500, 3100, 1250]], index=["Supply"], columns=["D1", "D2", "D3"])
    # Demand defaults (units)
    demand = mk_df([[2400, 2400, 2050]], index=["Capacity"], columns=["S1", "S2", "S3"])
    return dist, rate, supply, demand

# Load defaults
dist_df, rate_df, supply_row, demand_row = default_inputs()

# -----------------------------
# Sidebar: Google Maps helpers
# -----------------------------
with st.sidebar:
    st.header("Driving distance (Google Maps)")
    st.caption("Distance Matrix API. Provide a Maps API key in secrets or below.")

    # Prefer secrets, allow manual entry
    default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Google Maps API key", value=default_key, type="password")

    # --- Single pair helper ---
    st.subheader("Single pair")
    origin_pc = st.text_input("Origin postcode", value="PA3 3BW", key="single_origin")
    dest_pc = st.text_input("Destination postcode", value="CA11 9EU", key="single_dest")
    use_imperial_single = st.toggle("Use miles (imperial)", value=True, key="single_units")
    if st.button("Fetch driving distance", key="btn_single"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' not installed.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                units = "imperial" if use_imperial_single else "metric"
                resp = gmaps.distance_matrix(
                    origins=[origin_pc],
                    destinations=[dest_pc],
                    mode="driving",
                    units=units,
                    region="uk",
                )
                el = resp["rows"][0]["elements"][0]
                if el.get("status") != "OK":
                    st.error(f"API status: {el.get('status')}")
                else:
                    dist_text = el["distance"]["text"]
                    dist_m = el["distance"]["value"]
                    dur_text = el["duration"]["text"]
                    miles = dist_m / 1609.344
                    st.success(f"Driving distance: {dist_text} ({miles:,.1f} mi), duration: {dur_text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

    # --- 3x3 matrix helper ---
    st.subheader("Auto-fill 3x3 matrix from postcodes")
    st.caption("Enter 3 depot postcodes and 3 store postcodes. Fetch fills the Distance 3x3 table (miles).")

    # Depot postcodes
    d1 = st.text_input("Depot D1 postcode", value="PA3 3BW")
    d2 = st.text_input("Depot D2 postcode", value="G2 1DU")
    d3 = st.text_input("Depot D3 postcode", value="KA1 1AA")

    # Store postcodes
    s1 = st.text_input("Store S1 postcode", value="CA11 9EU")
    s2 = st.text_input("Store S2 postcode", value="EH1 1YZ")
    s3 = st.text_input("Store S3 postcode", value="DG1 2BD")

    round_miles = st.checkbox("Round to 1 decimal", value=True)
    use_imperial_matrix = st.checkbox("Use miles (imperial) for API request", value=True)

    if st.button("Fetch 3x3 driving distances", key="btn_matrix"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' not installed.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                units = "imperial" if use_imperial_matrix else "metric"

                # Call Distance Matrix once with 3 origins and 3 destinations
                origins = [d1, d2, d3]
                destinations = [s1, s2, s3]

                with st.spinner("Fetching distances..."):
                    resp = gmaps.distance_matrix(
                        origins=origins,
                        destinations=destinations,
                        mode="driving",
                        units=units,
                        region="uk",
                    )

                # Parse response into a 3x3 miles matrix (value is in meters)
                miles_mat = np.zeros((3, 3), dtype=float)
                for i in range(3):
                    row = resp["rows"][i]["elements"]
                    for j in range(3):
                        el = row[j]
                        if el.get("status") == "OK":
                            meters = el["distance"]["value"]
                            miles = meters / 1609.344
                            miles_mat[i, j] = round(miles, 1) if round_miles else miles
                        else:
                            # Leave as 0 if not OK; you could also raise an error
                            miles_mat[i, j] = 0.0

                # Overwrite the Distance 3x3 editor with fetched miles
                dist_df.loc[:, :] = miles_mat
                st.success("Distance matrix updated from postcodes.")
                st.dataframe(pd.DataFrame(miles_mat, index=["D1","D2","D3"], columns=["S1","S2","S3"]),
                             use_container_width=True)
            except Exception as e:
                st.error(f"Request failed: {e}")

# -----------------------------
# Inputs (editable tables)
# -----------------------------
# Editable Distance (miles) and Rate (GBP/mile) 3x3 matrices
# Editable Depot Supply and Store Capacity rows (1x3)
st.subheader("Inputs")
col_dist, col_rate = st.columns(2)

with col_dist:
    st.markdown("**Distance (miles)** - 3x3")
    dist_df = st.data_editor(
        dist_df, use_container_width=True, num_rows="fixed", key="distance_editor"
    )

with col_rate:
    st.markdown("**Rate (GBP/mile)** - 3x3")
    rate_df = st.data_editor(
        rate_df, use_container_width=True, num_rows="fixed", key="rate_editor"
    )

col_supply, col_demand = st.columns(2)
with col_supply:
    st.markdown("**Depot Supply (units)** - 1x3")
    supply_row = st.data_editor(
        supply_row, use_container_width=True, num_rows="fixed", key="supply_editor"
    )
with col_demand:
    st.markdown("**Store Capacity (units)** - 1x3")
    demand_row = st.data_editor(
        demand_row, use_container_width=True, num_rows="fixed", key="demand_editor"
    )

# Auto-balance option
auto_balance = st.checkbox("Auto-balance totals with a dummy node (zero cost)", value=True)

# -----------------------------
# Build cost matrix and vectors
# -----------------------------
try:
    dist = dist_df.to_numpy(dtype=float)
    rate = rate_df.to_numpy(dtype=float)
    cost = dist * rate  # element-wise cost per unit
except Exception as e:
    st.error(f"Unable to compute cost matrix from Distance * Rate: {e}")
    st.stop()

sup = supply_row.to_numpy(dtype=float).ravel()
dem = demand_row.to_numpy(dtype=float).ravel()
sum_sup, sum_dem = float(sup.sum()), float(dem.sum())

# -----------------------------
# Transportation model
# -----------------------------
def transport_min_cost(cost_mat, supply, demand):
    # Solve min sum(c_ij * x_ij) subject to row/col sums, x_ij >= 0
    m, n = cost_mat.shape
    c = cost_mat.reshape(-1)

    # Bounds x_ij >= 0
    bounds = [(0, None)] * (m * n)

    # Equality constraints A_eq x = b_eq (supplies and demands)
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

    # Solve with HiGHS
    res = linprog(
        c,
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    return res

# Balance totals if needed (add zero-cost dummy row/column)
cost_bal = cost.copy()
sup_bal = sup.copy()
dem_bal = dem.copy()
row_labels = ["D1", "D2", "D3"]
col_labels = ["S1", "S2", "S3"]

if not np.isclose(sum_sup, sum_dem):
    if not auto_balance:
        st.warning(
            f"Totals are not equal (Supply = {sum_sup:,.0f}, Demand = {sum_dem:,.0f}). "
            "Either adjust inputs so they match or tick 'Auto-balance totals'."
        )
        st.stop()
    diff = sum_sup - sum_dem
    if diff > 0:
        # Extra supply -> add dummy store (new column)
        cost_bal = np.c_[cost_bal, np.zeros((3, 1))]
        dem_bal = np.r_[dem_bal, [diff]]
        col_labels = col_labels + ["DummyStore"]
    else:
        # Extra demand -> add dummy depot (new row)
        cost_bal = np.r_[cost_bal, [np.zeros(3)]]
        sup_bal = np.r_[sup_bal, [-diff]]
        row_labels = row_labels + ["DummyDepot"]

# Solve
res = transport_min_cost(cost_bal, sup_bal, dem_bal)

if not res.success:
    st.error("Optimiser failed: " + res.message)
    st.stop()

# Optimised plan
x = res.x.reshape(len(row_labels), len(col_labels))
plan_df = pd.DataFrame(x, index=row_labels, columns=col_labels)

# Calculate total cost ignoring dummy rows/cols
actual_mask_rows = [not r.startswith("Dummy") for r in row_labels]
actual_mask_cols = [not c.startswith("Dummy") for c in col_labels]
actual_cost = (plan_df.values[np.ix_(actual_mask_rows, actual_mask_cols)] * cost).sum()

# Output
st.markdown("## Cost of delivery:")
st.metric(label="Minimum total cost (GBP)", value=f"{actual_cost:,.2f}")

with st.expander("Show optimised shipment plan (units)"):
    st.dataframe(plan_df.style.format("{:.2f}"), use_container_width=True)

with st.expander("Show cost matrix (GBP per unit)"):
    st.dataframe(
        pd.DataFrame(cost, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]).style.format("{:.2f}"),
        use_container_width=True
    )

st.caption("Built with SciPy linprog (HiGHS). Driving distances via Google Maps Distance Matrix API.")
