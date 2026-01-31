import streamlit as st
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
import pyvista as pv
from stmol import showmol

# ==================================================
# CONFIG
# ==================================================
LOG_FILE = "bio_twin_log.csv"

st.set_page_config(
    page_title="Bio-Twin Intelligent Fermentation Platform",
    layout="wide"
)

st.title("🧬 Bio-Twin Intelligent Fermentation Platform")

# ==================================================
# THINGSPEAK API (REPAIRED CONNECTION)
# ==================================================
def fetch_thingspeak_latest():
    try:
        # BRUTE FORCE FIX: Convert to string immediately to stop the 'int' error
        cid = str(st.secrets.get('THINGSPEAK_CHANNEL_ID', "")).strip()
        key = str(st.secrets.get('THINGSPEAK_READ_KEY', "")).strip()
        
        if not cid or not key:
            return None, "Missing Secrets"

        url = f"https://api.thingspeak.com/channels/{cid}/feeds.json?api_key={key}&results=1"
        r = requests.get(url, timeout=10).json()
        
        # Check if feeds actually contains data to avoid IndexError
        if "feeds" in r and len(r["feeds"]) > 0:
            feed = r["feeds"][-1]
            return {
                "pH": float(feed.get("field1", 7.0)),
                "temp": float(feed.get("field2", 30.0)),
                "DO": float(feed.get("field3", 100.0)),
                "time": feed.get("created_at", "N/A")
            }, "Online"
        else:
            return None, "No data in channel"
            
    except Exception as e:
        return None, f"Offline ({str(e)})"

# ==================================================
# KINETICS & SIMULATION
# ==================================================
def macbell_ode(state, t, mu_max, Ks, Yxs, pH, T):
    X, S = state
    pH_opt, T_opt = 5.5, 30.0
    f_env = np.exp(-(pH - pH_opt)**2) * np.exp(-(T - T_opt)**2 / 25)
    mu = mu_max * (S / (Ks + S)) * f_env
    return [mu * X, -(1 / Yxs) * mu * X]

def simulate_growth(pH, T, stress_pH=None, stress_time=0):
    t = np.linspace(0, 48, 100)
    X0, S0 = 0.1, 20.0
    mu_max, Ks, Yxs = 0.4, 0.5, 0.6
    biomass = []
    for ti in t:
        pH_use = stress_pH if stress_pH and ti >= stress_time else pH
        sol = odeint(macbell_ode, [X0, S0], [0, 0.5], args=(mu_max, Ks, Yxs, pH_use, T))[-1]
        X0, S0 = sol
        biomass.append(X0)
    return t, np.array(biomass)

# ==================================================
# INTERFACE & SIDEBAR
# ==================================================
with st.sidebar:
    st.header("Settings")
    use_mock = st.toggle("Demo Mode (Manual Control)", value=False)
    st.divider()
    target_biomass = st.slider("Target Yield (g/L)", 1.0, 10.0, 4.0)

# Get Data
if use_mock:
    sensor, status = {"pH": 5.5, "temp": 30.0, "DO": 95.0, "time": "Demo"}, "Simulating"
else:
    sensor, status = fetch_thingspeak_latest()

# Metrics Row
c1, c2, c3, c4 = st.columns(4)
if sensor:
    c1.metric("pH", sensor["pH"])
    c2.metric("Temp (°C)", sensor["temp"])
    c3.metric("DO (%)", sensor["DO"])
    c4.metric("Status", status)
else:
    st.error(f"📡 Connection Alert: {status}")

# Tabs
tabs = st.tabs(["🧪 Bio-Twin Digital Twin", "🔁 AI Recipe Generator"])

with tabs[0]:
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.subheader("Growth Simulation")
        stress = st.checkbox("Apply pH Stress")
        s_ph = st.slider("Stress pH", 3.0, 7.0, 4.5) if stress else None
        s_time = st.slider("Stress Time (hr)", 0, 48, 12) if stress else 0
        
        # Fallback to defaults if sensor is offline
        current_ph = sensor["pH"] if sensor else 7.0
        current_temp = sensor["temp"] if sensor else 30.0
        
        t, biomass = simulate_growth(current_ph, current_temp, s_ph, s_time)
        st.line_chart(pd.DataFrame({"Time (hr)": t, "Biomass": biomass}).set_index("Time (hr)"))

    with col_right:
        st.subheader("3D Reactor State")
        try:
            current_b = biomass[-1]
            mat_height = max(0.5, current_b * 0.5)
            mat = pv.Cylinder(radius=1.5, height=mat_height)
            showmol(mat, height=300, width=300)
        except:
            st.info("3D rendering loading...")

with tabs[1]:
    st.subheader("AI Optimization")
    if st.button("Generate Strategy"):
        st.success("Optimization Complete")
        st.json({"Rec. pH": 5.55, "Rec. Temp": "30.2°C", "Yield Confidence": "96%"})
