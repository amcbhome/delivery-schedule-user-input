# -*- coding: utf-8 -*-
# Delivery Schedule Optimiser — compact UI
# - Sidebar: Google Maps Distance Matrix 3x3 fetch (miles)
# - Main page: single-row form: D1..D3 supply, S1..S3 capacity, Rate per mile (GBP)
# - Totals must match (no auto-balance)
# - Output: single line "Optimised cost of delivery: £X" (no decimals)
# - Optional XLSX download of the optimised schedule
# - All comments use '#'

import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# Optional: Google Maps client (install 'googlemaps'; set key in secrets or sidebar)
try:
    import googlemaps  # noqa: F401
except Exception:
    googlemaps = None

# ---------------------------------------------------------------------
# Page + compact CSS
# ---------------------------------------------------------------------
st.set_page_config(page_title="Delivery Schedule Optimiser", layout="wide")
st.title("Delivery Schedule Optimiser")

# Compact spacing for inputs/buttons
st.markdown(
    """
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
div[data-testid="stNumberInput"] { margin-bottom: .25rem; }
div[data-testid="stNumberInput"] input { padding-top: .25rem; padding-bottom: .25rem; }
div.stButton > button { padding: .4rem .8rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Session state: 3x3 distance matrix (D1..D3 -> S1..S3), edited via sidebar
# ---------------------------------------------------------------------
def default_distance():
    return pd.DataFrame(
        [[10, 20, 30],
         [15, 10, 25],
         [20, 15, 10]],
        index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]
    )

if "dist_df" not in st.session_state:
    st.session_state.dist_df = default_distance()

# ---------------------------------------------------------------------
# Sidebar — Google Maps Distance Matrix helper (fills the distance matrix)
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Driving distance (Google Maps)")
    st.caption("Fetch the 3×3 distance matrix in miles. The main form uses this matrix.")

    default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Google Maps API key", value=default_key, type="password")

    st.subheader("Auto-fill 3×3 matrix from postcodes")
    d1 = st.text_input("Depot D1 postcode", value="PA3 3BW")
    d2 = st.text_input("Depot D2 postcode", value="G2 1DU")
    d3 = st.text_input("Depot D3 postcode", value="KA1 1AA")
    s1 = st.text_input("Store S1 postcode", value="CA11 9EU")
    s2 = st.text_input("Store S2 postcode", value="EH1 1YZ")
    s3 = st.text_input("Store S3 postcode", value="DG1 2BD")
    round_miles = st.checkbox("Round to 1 decimal place", value=True)

    if st.button("Fetch distances (3×3)"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' is not installed.")
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
                    pd.DataFrame(miles, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------
# Main form — single row (6 inputs + rate), labelled with postcodes
# ---------------------------------------------------------------------
# Fallback labels if a postcode is blank
d1_label = (d1.strip() if 'd1' in locals() else "").upper() or "D1"
d2_label = (d2.strip() if 'd2' in locals() else "").upper() or "D2"
d3_label = (d3.strip() if 'd3' in locals() else "").upper() or "D3"
s1_label = (s1.strip() if 's1' in locals() else "").upper() or "S1"
s2_label = (s2.strip() if 's2' in locals() else "").upper() or "S2"
s3_label = (s3.strip() if 's3' in locals() else "").upper() or "S3"

with st.form("delivery_form", clear_on_submit=False):
    cols = st.columns([1, 1, 1, 1, 1, 1, 0.9], gap="small")

    # Depot supplies
    with cols[0]:
        st.caption(d1_label)
        d1_supply = st.number_input("D1", key="d1_supply", min_value=0, step=50,
                                    value=2500, label_visibility="collapsed", format="%d")
    with cols[1]:
        st.caption(d2_label)
        d2_supply = st.number_input("D2", key="d2_supply", min_value=0, step=50,
                                    value=3100, label_visibility="collapsed", format="%d")
    with cols[2]:
        st.caption(d3_label)
        d3_supply = st.number_input("D3", key="d3_supply", min_value=0, step=50,
                                    value=1250, label_visibility="collapsed", format="%d")

    # Store capacities
    with cols[3]:
        st.caption(s1_label)
        s1_cap = st.number_input("S1", key="s1_cap", min_value=0, step=50,
                                 value=2400, label_visibility="collapsed", format="%d")
    with cols[4]:
        st.caption(s2_label)
        s2_cap = st.number_input("S2", key="s2_cap", min_value=0, step=50,
                                 value=2400, label_visibility="collapsed", format="%d")
    with cols[5]:
        st.caption(s3_label)
        s3_cap = st.number_input("S3", key="s3_cap", min_value=0, step=50,
                                 value=2050, label_visibility="collapsed", format="%d")

    # Rate per mile
    with cols[6]:
        st.caption("Rate (GBP/mi)")
        rate_per_mile = st.number_input("Rate", key="rate_per_mile", min_value=0.0,
                                        step=0.10, value=5.00, label_visibility="collapsed", format="%.2f")

    want_xlsx = st.checkbox("Offer an XLSX download of the optimised schedule")
    submitted = st.form_submit_button("Submit and optimise")

# ---------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------
def transport_min_cost(cost_mat, supply, demand):
    m, n = cost_mat.shape
    c = cost_mat.reshape(-1)
    bounds = [(0, None)] * (m * n)

    A_eq = []
    b_eq = []
    # supply rows
    for i in range(m):
        row = np.zeros(m * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])
    # demand cols
    for j in range(n):
        row = np.zeros(m * n)
        for i in range(m):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    res = linprog(c, A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    return res

if submitted:
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
    plan_df = pd.DataFrame(plan, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"])

    total_cost = int(round((plan * cost).sum(), 0))
    st.success(f"Optimised cost of delivery: £{total_cost:,}")

    if want_xlsx:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # optimised plan (units)
            plan_df.to_excel(writer, sheet_name="Optimised Plan")
            # cost per unit matrix (GBP)
            pd.DataFrame(cost, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]).to_excel(
                writer, sheet_name="Cost Matrix (GBP/unit)"
            )
            # distance matrix (miles)
            pd.DataFrame(dist, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]).to_excel(
                writer, sheet_name="Distance (miles)"
            )
            # supply and demand for reference
            pd.DataFrame([supply], index=["Supply"], columns=["D1", "D2", "D3"]).to_excel(
                writer, sheet_name="Supply"
            )
            pd.DataFrame([demand], index=["Capacity"], columns=["S1", "S2", "S3"]).to_excel(
                writer, sheet_name="Demand"
            )
        st.download_button(
            label="Download XLSX of the optimised schedule",
            data=output.getvalue(),
            file_name="delivery_schedule_optimised.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


