# -*- coding: utf-8 -*-
# Delivery Schedule Optimiser — compact UI with postcode→town labels
# - Stores are CAPACITIES (upper bounds). Model ships all depot supply, without exceeding capacities.
# - If total supply > total store capacity, show an error and stop.
# - Sidebar: Google Maps 3x3 Distance Matrix (via place IDs) + Geocoding for labels
# - Main: 3 depot supplies, 3 store capacities, 1 rate (single compact row)
# - On submit: show one-line cost and an optimised schedule table (towns, with totals)

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
# Page + compact CSS
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
    # Prefer UK postal_town, then locality, then county/region
    prefs = ["postal_town", "locality", "administrative_area_level_2", "administrative_area_level_1"]
    for pref in prefs:
        for comp in components:
            if pref in comp.get("types", []):
                return comp.get("long_name", "")
    return ""

def geocode_postcode(gmaps, postcode):
    # Return "POSTCODE (Town)" where possible
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
    # Cached "POSTCODE (Town)" if present; else the raw postcode; else slot key
    return st.session_state.pc_labels.get(slot_key) or (raw_pc_default.upper() if raw_pc_default else slot_key)

def display_name(label: str) -> str:
    # Prefer the town inside parentheses; else return the label itself
    if "(" in label and ")" in label:
        inside = label.split("(", 1)[1].split(")", 1)[0].strip()
        if inside:
            return inside
    return label

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

                # Geocode to place IDs for stability
                def pid(pc):
                    r = gmaps.geocode(pc, components={"country": "GB"})
                    return r[0]["place_id"] if r else None

                origin_pids = [pid(x) for x in [d1_pc, d2_pc, d3_pc]]
                dest_pids   = [pid(x) for x in [s1_pc, s2_pc, s3_pc]]

                if any(p is None for p in origin_pids + dest_pids):
                    st.error("One or more postcodes could not be geocoded. Check them and try again.")
                    st.stop()

                resp = gmaps.distance_matrix(
                    origins=[f"place_id:{p}" for p in origin_pids],
                    destinations=[f"place_id:{p}" for p in dest_pids],
                    mode="driving",
                    units="imperial",
                    region="uk",
                )

                # Build miles matrix + a parallel status matrix
                miles = np.full((3, 3), np.nan, dtype=float)
                status = [[""] * 3 for _ in range(3)]
                for i in range(3):
                    for j in range(3):
                        el = resp["rows"][i]["elements"][j]
                        status[i][j] = el.get("status", "")
                        if el.get("status") == "OK":
                            meters = el["distance"]["value"]
                            mi = meters / 1609.344
                            miles[i, j] = round(mi, 1) if round_miles else mi

                st.session_state.dist_df.loc[:, :] = miles
                st.success("Distance matrix updated.")

                # Diagnostics: per-cell status and the distances
                st.caption("Element statuses (non-OK means blank cell):")
                st.dataframe(
                    pd.DataFrame(status, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]),
                    use_container_width=True
                )
                st.dataframe(
                    pd.DataFrame(miles, index=["D1", "D2", "D3"], columns=["S1", "S2", "S3"]),
                    use_container_width=True
                )

                # Refresh postcode→town labels
                try:
                    labels = {}
                    for code, key in [(d1_pc, "D1"), (d2_pc, "D2"), (d3_pc, "D3"),
                                      (s1_pc, "S1"), (s2_pc, "S2"), (s3_pc, "S3")]:
                        labels[key] = geocode_postcode(gmaps, code)
                    st.session_state.pc_labels = labels
                    st.success("Labels updated (postcode + town).")
                except Exception:
                    pass

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

    # Feasibility: total delivered (total supply) must be <= total store capacity
    if supply.sum() > capacity.sum():
        st.error(
            f"Total supply ({int(supply.sum()):,}) exceeds total store capacity "
            f"({int(capacity.sum()):,}). Adjust the numbers and try again."
        )
        st.stop()

    # Compute cost matrix; replace missing distances (NaN) with a huge number to avoid those routes
    dist = st.session_state.dist_df.to_numpy(dtype=float)
    if np.isnan(dist).any():
        st.warning("Some routes had no distance; assigning a very large cost to discourage their use.")
    dist_filled = np.where(np.isnan(dist), 1e9, dist)  # huge miles so they won't be chosen
    cost = dist_filled * float(rate_per_mile)

    res = transport_min_cost_with_capacity(cost, supply, capacity)
    if not res.success:
        st.error("Optimiser failed: " + res.message)
        st.stop()

    plan = res.x.reshape(3, 3)

    # Cost (from unrounded plan)
    total_cost = int(round((plan * cost).sum(), 0))
    st.success(f"Optimised cost of delivery: £{total_cost:,}")

    # Build a human-friendly table with towns and totals
    depot_labels = [display_name(d1_label), display_name(d2_label), display_name(d3_label)]
    store_labels = [display_name(s1_label), display_name(s2_label), display_name(s3_label)]

    schedule = pd.DataFrame(np.rint(plan).astype(int), index=depot_labels, columns=store_labels)
    schedule["Total"] = schedule.sum(axis=1)
    total_row = schedule.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    schedule_with_totals = pd.concat([schedule, total_row], axis=0)

    st.markdown("**Optimised delivery schedule (units)**")
    st.dataframe(schedule_with_totals.style.format("{:,}"), use_container_width=True)

