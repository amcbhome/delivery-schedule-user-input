# -*- coding: utf-8 -*-
# TV Delivery Optimizer (data model + LP) — Sidebar reserved for Maps; main form for inputs
# - Sidebar: Google Maps Distance Matrix helpers (single pair and 3x3 fetch)
# - Main window: "Delivery input data form (use the sidebar to calculate distances)."
# - Linear program solved with SciPy HiGHS
# - All comments use '#', strings are ASCII (GBP instead of pound symbol)

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# Optional: Google Maps client (install 'googlemaps' and set key in secrets or sidebar)
try:
    import googlemaps  # noqa: F401
except Exception:
    googlemaps = None

# Page setup
st.set_page_config(page_title="TV Delivery Optimizer", layout="wide")
st.title("TV Delivery Optimizer")

# ---------------------------------------------------------------------
# Defaults and state
# ---------------------------------------------------------------------
def default_tables():
    # Default distance (miles)
    dist = pd.DataFrame(
        [[10, 20, 30],
         [15, 10, 25],
         [20, 15, 10]],
        index=["D1","D2","D3"], columns=["S1","S2","S3"]
    )
    # Default rate (GBP per mile)
    rate = pd.DataFrame(
        [[5, 5, 5],
         [5, 5, 5],
         [5, 5, 5]],
        index=["D1","D2","D3"], columns=["S1","S2","S3"]
    )
    # Default supply (units)
    supply = pd.DataFrame([[2500, 3100, 1250]], index=["Supply"], columns=["D1","D2","D3"])
    # Default demand (units)
    demand = pd.DataFrame([[2400, 2400, 2050]], index=["Capacity"], columns=["S1","S2","S3"])
    return dist, rate, supply, demand

if "dist_df" not in st.session_state:
    st.session_state.dist_df, st.session_state.rate_df, st.session_state.supply_row, st.session_state.demand_row = default_tables()

# Short references
dist_df = st.session_state.dist_df
rate_df = st.session_state.rate_df
supply_row = st.session_state.supply_row
demand_row = st.session_state.demand_row

# ---------------------------------------------------------------------
# Sidebar — reserved for Google Maps helpers only
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Driving distance (Google Maps)")
    st.caption("Distance Matrix API. Provide a Maps API key in secrets or below.")
    default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Google Maps API key", value=default_key, type="password")

    # Single pair helper
    st.subheader("Single pair")
    origin_pc = st.text_input("Origin postcode", value="PA3 3BW", key="single_origin")
    dest_pc   = st.text_input("Destination postcode", value="CA11 9EU", key="single_dest")
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

    # 3x3 matrix helper
    st.subheader("Auto-fill 3x3 matrix")
    st.caption("Enter 3 depot postcodes and 3 store postcodes. Fetch fills the Distance 3x3 (miles).")
    d1 = st.text_input("Depot D1", value="PA3 3BW")
    d2 = st.text_input("Depot D2", value="G2 1DU")
    d3 = st.text_input("Depot D3", value="KA1 1AA")
    s1 = st.text_input("Store S1", value="CA11 9EU")
    s2 = st.text_input("Store S2", value="EH1 1YZ")
    s3 = st.text_input("Store S3", value="DG1 2BD")
    round_miles = st.checkbox("Round miles to 1dp", value=True)
    if st.button("Fetch 3x3 driving distances", key="btn_matrix"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' not installed.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                resp = gmaps.distance_matrix(
                    origins=[d1,d2,d3],
                    destinations=[s1,s2,s3],
                    mode="driving",
                    units="imperial",
                    region="uk",
                )
                miles = np.zeros((3,3))
                for i in range(3):
                    for j in range(3):
                        el = resp["rows"][i]["elements"][j]
                        if el.get("status") == "OK":
                            meters = el["distance"]["value"]
                            mi = meters / 1609.344
                            miles[i,j] = round(mi,1) if round_miles else mi
                # Update the shared distance table in session state
                st.session_state.dist_df.loc[:,:] = miles
                st.success("Distance matrix updated. Review it in the form and Submit.")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------
# Main window — delivery input data form
# ---------------------------------------------------------------------
st.subheader("Delivery input data form (use the sidebar to calculate distances).")

with st.form("input_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Distance (miles)** - 3x3")
        dist_df = st.data_editor(dist_df, use_container_width=True, num_rows="fixed", key="distance_editor")
    with c2:
        st.markdown("**Rate (GBP/mile)** - 3x6")  # keep 3x3; label typo avoided below
        st.markdown("**Rate (GBP/mile)** - 3x3")
        rate_df = st.data_editor(rate_df, use_container_width=True, num_rows="fixed", key="rate_editor")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Depot Supply (units)** - 1x3")
        supply_row = st.data_editor(supply_row, use_container_width=True, num_rows="fixed", key="supply_editor")
    with c4:
        st.markdown("**Store Capacity (units)** - 1x3")
        demand_row = st.data_editor(demand_row, use_container_width=True, num_rows="fixed", key="demand_editor")

    auto_balance = st.checkbox("Auto-balance totals with a dummy node (zero cost)", value=True)

    # Submit button
    submitted = st.form_submit_button("Submit and optimise")

# Persist any edits from the form back to session state
st.session_state.dist_df = dist_df
st.session_state.rate_df = rate_df
st.session_state.supply_row = supply_row
st.session_state.demand_row = demand_row

# ---------------------------------------------------------------------
# If submitted, solve the LP
# ---------------------------------------------------------------------
if submitted:
    try:
        dist = dist_df.to_numpy(dtype=float)
        rate = rate_df.to_numpy(dtype=float)
        cost = dist * rate
    except Exception as e:
        st.error(f"Unable to compute cost matrix from Distance * Rate: {e}")
        st.stop()

    sup = supply_row.to_numpy(dtype=float).ravel()
    dem = demand_row.to_numpy(dtype=float).ravel()
    sum_sup = float(sup.sum())
    sum_dem = float(dem.sum())

    # Helper to build standard transport model
    def transport_min_cost(cost_mat, supply, demand):
        m, n = cost_mat.shape
        c = cost_mat.reshape(-1)
        bounds = [(0, None)] * (m * n)
        A_eq = []
        b_eq = []
        for i in range(m):
            row = np.zeros(m * n)
            for j in range(n):
                row[i * n + j] = 1
            A_eq.append(row)
            b_eq.append(supply[i])
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
            st.error(
                f"Totals are not equal (Supply = {sum_sup:,.0f}, Demand = {sum_dem:,.0f}). "
                "Enable auto-balance or edit inputs."
            )
            st.stop()
        diff = sum_sup - sum_dem
        if diff > 0:
            # Extra supply -> add dummy store
            cost_bal = np.c_[cost_bal, np.zeros((3, 1))]
            dem_bal = np.r_[dem_bal, [diff]]
            col_labels = col_labels + ["DummyStore"]
        else:
            # Extra demand -> add dummy depot
            cost_bal = np.r_[cost_bal, [np.zeros(3)]]
            sup_bal = np.r_[sup_bal, [-diff]]
            row_labels = row_labels + ["DummyDepot"]

    # Solve
    res = transport_min_cost(cost_bal, sup_bal, dem_bal)
    if not res.success:
        st.error("Optimiser failed: " + res.message)
        st.stop()

    x = res.x.reshape(len(row_labels), len(col_labels))
    plan_df = pd.DataFrame(x, index=row_labels, columns=col_labels)

    # Compute actual cost excluding dummy flows
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
            pd.DataFrame(cost, index=["D1","D2","D3"], columns=["S1","S2","S3"]).style.format("{:.2f}"),
            use_container_width=True
        )

# ---------------------------------------------------------------------
# Optional textual problem setup (mirrors current inputs)
# ---------------------------------------------------------------------
def fmt_row_numbers(arr, labels):
    parts = [f"{lab}: {int(x):,}" for lab, x in zip(labels, arr)]
    return "; ".join(parts)

def matrix_is_uniform(mat):
    return np.allclose(mat, mat[0,0])

try:
    # Use latest values (form editors already persisted)
    dist = st.session_state.dist_df.to_numpy(dtype=float)
    rate = st.session_state.rate_df.to_numpy(dtype=float)
    sup  = st.session_state.supply_row.to_numpy(dtype=float).ravel()
    dem  = st.session_state.demand_row.to_numpy(dtype=float).ravel()

    depot_text = fmt_row_numbers(sup, ["D1","D2","D3"])
    store_text = fmt_row_numbers(dem, ["S1","S2","S3"])
    total_sup  = int(sup.sum())
    total_dem  = int(dem.sum())

    st.subheader("Problem setup (text)")
    st.markdown(
        f"There are {total_sup:,} units available across the depots: {depot_text}. "
        f"The stores require {total_dem:,} units in total: {store_text}."
    )

    if matrix_is_uniform(rate):
        st.markdown(f"Delivery rate per mile is uniform at GBP {rate[0,0]:,.2f}.")
    else:
        st.markdown("Delivery rate per mile varies by route:")
        st.dataframe(
            pd.DataFrame(rate, index=["D1","D2","D3"], columns=["S1","S2","S3"]).style.format("{:.2f}"),
            use_container_width=True
        )
except Exception:
    pass

