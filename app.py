# -*- coding: utf-8 -*-
# Delivery Schedule Optimiser — clean UI with 7 inputs on the main form
# - Sidebar: Google Maps 3x3 distance fetcher (kept, but out of the way)
# - Main page: Data input form with 7 variables (D1..D3 supply, S1..S3 capacity, rate per mile)
# - Requires total supply == total demand (no auto-balance toggle shown)
# - Output: single line "Optimised cost of delivery: £X" (no decimals)
# - Optional XLSX download of the optimised schedule

import io
import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# Optional: Google Maps client (install 'googlemaps'; set key in secrets or sidebar)
try:
    import googlemaps  # noqa: F401
except Exception:
    googlemaps = None

st.set_page_config(page_title="Delivery Schedule Optimiser", layout="wide")
st.title("Delivery Schedule Optimiser")

# --------------------------------------------------------------------------------------
# Session state: keep a 3x3 distance matrix for D1..D3 -> S1..S3 (edited via sidebar)
# --------------------------------------------------------------------------------------
def default_distance():
    return pd.DataFrame(
        [[10, 20, 30],
         [15, 10, 25],
         [20, 15, 10]],
        index=["D1","D2","D3"], columns=["S1","S2","S3"]
    )

if "dist_df" not in st.session_state:
    st.session_state.dist_df = default_distance()

# --------------------------------------------------------------------------------------
# Sidebar reserved for Google Maps Distance Matrix helpers
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Driving distance (Google Maps)")
    st.caption("Use this to calculate the 3×3 distance matrix (miles). The main form uses that matrix.")

    default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Google Maps API key", value=default_key, type="password")

    st.subheader("Auto-fill 3×3 matrix")
    d1 = st.text_input("Depot D1", value="PA3 3BW")
    d2 = st.text_input("Depot D2", value="G2 1DU")
    d3 = st.text_input("Depot D3", value="KA1 1AA")
    s1 = st.text_input("Store S1", value="CA11 9EU")
    s2 = st.text_input("Store S2", value="EH1 1YZ")
    s3 = st.text_input("Store S3", value="DG1 2BD")
    round_miles = st.checkbox("Round to 1 decimal place", value=True)

    if st.button("Fetch distances (3×3)"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' not installed.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                resp = gmaps.distance_matrix(
                    origins=[d1, d2, d3],
                    destinations=[s1, s2, s3],
                    mode="driving",
                    units="imperial",
                    region="uk",
                )
                miles = np.zeros((3, 3), dtype=float)
                for i in range(3):
                    for j in range(3):
                        el = resp["rows"][i]["elements"][j]
                        if el.get("status") == "OK":
                            meters = el["distance"]["value"]
                            mi = meters / 1609.344
                            miles[i, j] = round(mi, 1) if round_miles else mi
                        else:
                            miles[i, j] = 0.0
                st.session_state.dist_df.loc[:, :] = miles
                st.success("Distance matrix updated.")
                st.dataframe(
                    pd.DataFrame(miles, index=["D1","D2","D3"], columns=["S1","S2","S3"]),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Request failed: {e}")

# --------------------------------------------------------------------------------------
# Main form with exactly 7 inputs
# --------------------------------------------------------------------------------------
st.subheader("Data input form")

with st.form("delivery_form", clear_on_submit=False):
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Depot supply (units)**")
        d1_supply = st.number_input("D1", min_value=0, step=50, value=2500)
        d2_supply = st.number_input("D2", min_value=0, step=50, value=3100)
        d3_supply = st.number_input("D3", min_value=0, step=50, value=1250)

    with c2:
        st.markdown("**Store capacity (units)**")
        s1_cap = st.number_input("S1", min_value=0, step=50, value=2400)
        s2_cap = st.number_input("S2", min_value=0, step=50, value=2400)
        s3_cap = st.number_input("S3", min_value=0, step=50, value=2050)

    rate_per_mile = st.number_input("Rate per mile (GBP)", min_value=0.0, step=0.10, value=5.00, format="%.2f")

    want_xlsx = st.checkbox("Offer an XLSX download of the optimised schedule")

    submitted = st.form_submit_button("Submit and optimise")

# --------------------------------------------------------------------------------------
# Optimisation (transportation LP). Requires totals to match.
# --------------------------------------------------------------------------------------
def transport_min_cost(cost_mat, supply, demand):
    m, n = cost_mat.shape
    c = cost_mat.reshape(-1)
    bounds = [(0, None)] * (m * n)

    A_eq = []
    b_eq = []
    for i in range(m):  # supply rows
        row = np.zeros(m * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])
    for j in range(n):  # demand cols
        row = np.zeros(m * n)
        for i in range(m):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    return res

if submitted:
    # Assemble vectors
    supply = np.array([d1_supply, d2_supply, d3_supply], dtype=float)
    demand = np.array([s1_cap, s2_cap, s3_cap], dtype=float)

    if not np.isclose(supply.sum(), demand.sum()):
        st.error(
            f"Total supply ({int(supply.sum()):,}) must equal total demand ({int(demand.sum()):,}). "
            "Adjust the numbers and try again."
        )
        st.stop()

    # Cost matrix = distance (from sidebar state) * single rate
    dist = st.session_state.dist_df.to_numpy(dtype=float)
    cost = dist * float(rate_per_mile)

    res = transport_min_cost(cost, supply, demand)
    if not res.success:
        st.error("Optimiser failed: " + res.message)
        st.stop()

    plan = res.x.reshape(3, 3)
    plan_df = pd.DataFrame(plan, index=["D1","D2","D3"], columns=["S1","S2","S3"])

    total_cost = int(round((plan * cost).sum(), 0))
    st.success(f"Optimised cost of delivery: £{total_cost:,}")

    if want_xlsx:
        # Build an Excel file in-memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            plan_df.to_excel(writer, sheet_name="Optimised Plan")
            pd.DataFrame(cost, index=["D1","D2","D3"], columns=["S1","S2","S3"]).to_excel(
                writer, sheet_name="Cost Matrix (GBP/unit)"
            )
            pd.DataFrame(dist, index=["D1","D2","D3"], columns=["S1","S2","S3"]).to_excel(
                writer, sheet_name="Distance (miles)"
            )
            pd.DataFrame([supply], index=["Supply"], columns=["D1","D2","D3"]).to_excel(
                writer, sheet_name="Supply"
            )
            pd.DataFrame([demand], index=["Capacity"], columns=["S1","S2","S3"]).to_excel(
                writer, sheet_name="Demand"
            )
        st.download_button(
            label="Download XLSX of optimised schedule",
            data=output.getvalue(),
            file_name="delivery_schedule_optimised.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


