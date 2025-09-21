# -*- coding: utf-8 -*-
# Delivery Schedule Optimiser — compact UI with postcode→town labels
# - Store numbers are CAPACITIES (upper bounds). Model ships all depot supply, without exceeding capacities.
# - If total supply > total store capacity, show an error and stop (user must adjust and resubmit).
# - Sidebar: Google Maps 3x3 Distance Matrix (+ Geocoding for labels)
# - Main: 3 depot supplies, 3 store capacities, 1 rate (single compact row)
# - Output: "Optimised cost of delivery: £X" (no decimals)
# - Optional XLSX download (safe sheet names)

import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog

# Optional Google Maps client (Distance Matrix + Geocoding)
try:
    import googlemaps  # noqa: F401
except Exception:
    googlemaps = None

# ------------------------------------------------------------------------------
# Page + compact CSS (ensures the title is fully visible and captions can wrap)
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Delivery Schedule Optimiser", layout="wide")
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; padding-bottom: 1rem; }
.app-title { font-size: 2rem; font-weight: 700; margin: 0 0 1rem 0;
             line-height: 1.2; white-space: normal; overflow-wrap: anywhere; }
div[data-testid="stNumberInput"] { margin-bottom: .25rem; }
div[data-testid="stNumberInput"] input { padding-top: .25rem; padding-bottom: .25rem; }
.small-caption { white-space: normal; font-size: 0.78rem; opacity: .8; margin-bottom: .25rem; }
div.stButton > button, button[kind="primary"] { padding: .4rem .8rem; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="app-title">Delivery Schedule Optimiser</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# State: 3x3 distance matrix; postcode→label cache
# ------------------------------------------------------------------------------
def default_distance():
    return pd.DataFrame(
        [[10, 20, 30],
         [15, 10, 25],
         [20, 15, 10]],
        index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]
    )

if "dist_df" not in st.session_state:
    st.session_state.dist_df = default_distance()

if "pc_labels" not in st.session_state:
    st.session_state.pc_labels = {}  # e.g. {"D1": "PA3 3BW (Paisley)"}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def extract_town_from_components(components):
    prefs = ["postal_town", "locality", "administrative_area_level_2", "administrative_area_level_1"]
    for pref in prefs:
        for comp in components:
            if pref in comp.get("types", []):
                return comp.get("long_name", "")
    return ""

def geocode_postcode(gmaps, postcode):
    try:
        res = gmaps.geocode(postcode, components={"country": "GB"})
        if not res:
            return postcode.upper()
        comps = res[0].get("address_components", [])
        town = extract_town_from_components(comps)
        return f"{postcode.upper()} ({town})" if town else postcode.upper()
    except Exception:
        return postcode.upper()

def label_for(slot_key, raw_pc_default):
    return st.session_state.pc_labels.get(slot_key) or (raw_pc_default.upper() if raw_pc_default else slot_key)

def safe_sheet(name: str) -> str:
    invalid = set(r'[]:*?/\\')
    cleaned = "".join(c for c in name if c not in invalid)
    trimmed = cleaned[:31]
    return trimmed or "Sheet1"

# ------------------------------------------------------------------------------
# Sidebar — Google Maps Distance Matrix + place-name lookup
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("Driving distance (Google Maps)")
    st.caption("Fetch the 3x3 distance matrix in miles. Labels can show the town via Geocoding.")

    default_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Google Maps API key", value=default_key, type="password")

    st.subheader("Postcodes for depots and stores")
    d1_pc = st.text_input("Depot D1 postcode", value="PA3 3BW")
    d2_pc = st.text_input("Depot D2 postcode", value="G2 1DU")
    d3_pc = st.text_input("Depot D3 postcode", value="KA1 1AA")
    s1_pc = st.text_input("Store S1 postcode", value="CA11 9EU")
    s2_pc = st.text_input("Store S2 postcode", value="EH1 1YZ")
    s3_pc = st.text_input("Store S3 postcode", value="DG1 2BD")

    round_miles = st.checkbox("Round to 1 decimal place", value=True)

    if st.button("Fetch distances (3x3)"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' is not installed.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                resp = gmaps.distance_matrix(
                    origins=[d1_pc, d2_pc, d3_pc],
                    destinations=[s1_pc, s2_pc, s3_pc],
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

                # Auto-refresh labels via Geocoding (if enabled)
                try:
                    labels = {}
                    for code, key in [(d1_pc, "D1"), (d2_pc, "D2"), (d3_pc, "D3"),
                                      (s1_pc, "S1"), (s2_pc, "S2"), (s3_pc, "S3")]:
                        labels[key] = geocode_postcode(gmaps, code)
                    st.session_state.pc_labels = labels
                    st.success("Labels updated (postcode + town).")
                except Exception as e:
                    st.warning(f"Distances fetched but place names not updated: {e}")

                st.dataframe(
                    pd.DataFrame(miles, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Request failed: {e}")

    if st.button("Look up place names"):
        if not api_key:
            st.error("Please provide an API key.")
        elif googlemaps is None:
            st.error("Package 'googlemaps' is not installed.")
        else:
            try:
                gmaps = googlemaps.Client(key=api_key)
                labels = {}
                for code, key in [(d1_pc, "D1"), (d2_pc, "D2"), (d3_pc, "D3"),
                                  (s1_pc, "S1"), (s2_pc, "S2"), (s3_pc, "S3")]:
                    labels[key] = geocode_postcode(gmaps, code)
                st.session_state.pc_labels = labels
                st.success("Labels updated (postcode + town).")
            except Exception as e:
                st.error(f"Lookup failed: {e}")

# ------------------------------------------------------------------------------
# Main form — headings + compact inputs with wrapped captions
# ------------------------------------------------------------------------------
d1_label = label_for("D1", d1_pc if "d1_pc" in locals() else "D1")
d2_label = label_for("D2", d2_pc if "d2_pc" in locals() else "D2")
d3_label = label_for("D3", d3_pc if "d3_pc" in locals() else "D3")
s1_label = label_for("S1", s1_pc if "s1_pc" in locals() else "S1")
s2_label = label_for("S2", s2_pc if "s2_pc" in locals() else "S2")
s3_label = label_for("S3", s3_pc if "s3_pc" in locals() else "S3")

with st.form("delivery_form", clear_on_submit=False):
    hdr_left, hdr_right, hdr_rate = st.columns([3, 3, 0.9], gap="small")
    with hdr_left:
        st.markdown("**Quantity at depot**")
    with hdr_right:
        st.markdown("**Capacity at stores**")
    with hdr_rate:
        st.markdown("**Rate (GBP/mi)**")

    left_grp, right_grp, rate_grp = st.columns([3, 3, 0.9], gap="small")

    with left_grp:
        l1, l2, l3 = st.columns(3, gap="small")
        with l1:
            st.markdown(f"<div class='small-caption'>{d1_label}</div>", unsafe_allow_html=True)
            d1_supply = st.number_input("D1", key="d1_supply", min_value=0, step=50,
                                        value=2500, label_visibility="collapsed", format="%d")
        with l2:
            st.markdown(f"<div class='small-caption'>{d2_label}</div>", unsafe_allow_html=True)
            d2_supply = st.number_input("D2", key="d2_supply", min_value=0, step=50,
                                        value=3100, label_visibility="collapsed", format="%d")
        with l3:
            st.markdown(f"<div class='small-caption'>{d3_label}</div>", unsafe_allow_html=True)
            d3_supply = st.number_input("D3", key="d3_supply", min_value=0, step=50,
                                        value=1250, label_visibility="collapsed", format="%d")

    with right_grp:
        r1, r2, r3 = st.columns(3, gap="small")
        with r1:
            st.markdown(f"<div class='small-caption'>{s1_label}</div>", unsafe_allow_html=True)
            s1_cap = st.number_input("S1", key="s1_cap", min_value=0, step=50,
                                     value=2400, label_visibility="collapsed", format="%d")
        with r2:
            st.markdown(f"<div class='small-caption'>{s2_label}</div>", unsafe_allow_html=True)
            s2_cap = st.number_input("S2", key="s2_cap", min_value=0, step=50,
                                     value=2400, label_visibility="collapsed", format="%d")
        with r3:
            st.markdown(f"<div class='small-caption'>{s3_label}</div>", unsafe_allow_html=True)
            s3_cap = st.number_input("S3", key="s3_cap", min_value=0, step=50,
                                     value=2050, label_visibility="collapsed", format="%d")

    with rate_grp:
        st.markdown("<div class='small-caption'>Rate (GBP/mi)</div>", unsafe_allow_html=True)
        rate_per_mile = st.number_input("Rate", key="rate_per_mile", min_value=0.0,
                                        step=0.10, value=5.00, label_visibility="collapsed", format="%.2f")

    want_xlsx = st.checkbox("Offer an XLSX download of the optimised schedule")
    submitted = st.form_submit_button("Submit and optimise")

# ------------------------------------------------------------------------------
# Solver (transportation LP with store capacities as upper bounds)
# ------------------------------------------------------------------------------
def transport_min_cost_with_capacity(cost_mat, supply, capacity):
    """
    Minimise sum(c_ij * x_ij)
    s.t.  For each depot i:  sum_j x_ij = supply_i          (use all supply)
          For each store j:  sum_i x_ij <= capacity_j       (do not exceed capacity)
          x_ij >= 0
    """
    m, n = cost_mat.shape
    c = cost_mat.reshape(-1)
    bounds = [(0, None)] * (m * n)

    A_eq, b_eq = [], []
    A_ub, b_ub = [], []

    # supply equalities
    for i in range(m):
        row = np.zeros(m * n)
        for j in range(n):
            row[i * n + j] = 1.0
        A_eq.append(row)
        b_eq.append(supply[i])

    # store capacity upper bounds
    for j in range(n):
        row = np.zeros(m * n)
        for i in range(m):
            row[i * n + j] = 1.0
        A_ub.append(row)
        b_ub.append(capacity[j])

    res = linprog(
        c,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )
    return res

if submitted:
    supply = np.array([d1_supply, d2_supply, d3_supply], dtype=float)
    capacity = np.array([s1_cap, s2_cap, s3_cap], dtype=float)

    # Correct feasibility rule: total delivered (total supply) must be <= total store capacity
    if supply.sum() > capacity.sum():
        st.error(
            f"Total supply ({int(supply.sum()):,}) exceeds total store capacity "
            f"({int(capacity.sum()):,}). Adjust the numbers and try again."
        )
        st.stop()

    dist = st.session_state.dist_df.to_numpy(dtype=float)
    cost = dist * float(rate_per_mile)

    res = transport_min_cost_with_capacity(cost, supply, capacity)
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
            plan_df.to_excel(writer, sheet_name=safe_sheet("Optimised Plan"))
            pd.DataFrame(cost, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]).to_excel(
                writer, sheet_name=safe_sheet("Cost Matrix (GBP per unit)")
            )
            pd.DataFrame(dist, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]).to_excel(
                writer, sheet_name=safe_sheet("Distance (miles)")
            )
            pd.DataFrame([supply], index=["Supply"], columns=["D1", "D2", "D3"]).to_excel(
                writer, sheet_name=safe_sheet("Supply")
            )
            pd.DataFrame([capacity], index=["Capacity"], columns=["S1", "S2", "S3"]).to_excel(
                writer, sheet_name=safe_sheet("Store Capacity")
            )
        st.download_button(
            label="Download XLSX of the optimised schedule",
            data=output.getvalue(),
            file_name="delivery_schedule_optimised.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

