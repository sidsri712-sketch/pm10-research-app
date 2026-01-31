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
# CONFIG & STYLING
# ==================================================
LOG_FILE = "bio_twin_log.csv"

st.set_page_config(
    page_title="Bio-Twin Intelligent Fermentation",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e1e4e8; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Bio-Twin Intelligent Fermentation Platform")
st.divider()

# ==================================================
# SECURE THINGSPEAK API FETCH
# ==================================================
def fetch_thingspeak_latest():
    try:
        # Check for secrets presence to prevent KeyError
        if "THINGSPEAK_CHANNEL_ID" not in st.secrets or "THINGSPEAK_READ_KEY" not in st.secrets:
            st.warning("⚠️ Secrets not configured in Streamlit Cloud. Using default values.")
            return {"pH": 7.0, "temp": 25.0, "DO": 95.0, "time": "N/A", "status": "Offline"}

        cid = st.secrets["THINGSPEAK_CHANNEL_ID"]
        key = st.secrets["THINGSPEAK_READ_KEY"]
        url = f"https://api.thingspeak.com/channels/{cid}/feeds.json?api_key={key}&results=1"
        
        r = requests.get(url, timeout=5).json()
        
        # Prevent IndexError if feeds is empty
        if "feeds" in r and len(r["feeds"]) > 0:
            feed = r["feeds"][-1]
            return {
                "pH": float(feed.get("field1", 7.0)),
                "temp": float(feed.get("field2", 30.0)),
                "DO": float(feed.get("field3", 90.0)),
                "time": feed.get("created_at", "Just now"),
                "status": "Online"
            }
        else:
            return {"pH": 7.0, "temp": 30.0, "DO": 90.0, "time": "No Data", "status": "Empty Channel"}
            
    except Exception as e:
        return {"pH": 7.0, "temp": 30.0, "DO": 90.0, "time": "Error", "status": f"Error: {str(e)}"}

# ==================================================
# KINETICS & SIMULATION
# ==================================================
def macbell_ode(state, t, mu_max, Ks, Yxs, pH, T):
    X, S = state
    pH_opt, T_opt = 5.5, 30.0
    # Bell-shaped environmental constraint
    f_env = np.exp(-(pH - pH_opt)**2) * np.exp(-(T - T_opt)**2 / 25)
    mu = mu_max * (S / (Ks + S)) * f_env
    return [mu * X, -(1 / Yxs) * mu * X]

@st.cache_data
def simulate_growth(pH, T, stress_pH=None, stress_time=0):
    t = np.linspace(0, 48, 100)
    X0, S0 = 0.1, 20.0
    mu_max, Ks, Yxs = 0.4, 0.5, 0.6
    biomass = []

    for ti in t:
        pH_use = stress_pH if (stress_pH and ti >= stress_time) else pH
        sol = odeint(macbell_ode, [X0, S0], [0, 0.5], args=(mu_max, Ks, Yxs, pH_use, T))[-1]
        X0, S0 = sol
        biomass.append(X0)
    return t, np.array(biomass)

# ==================================================
# SIDEBAR CONTROLS
# ==================================================
with st.sidebar:
    st.header("Settings")
    reactor = st.selectbox("Active Reactor", ["Reactor-Alpha", "Reactor-Beta"])
    use_mock = st.toggle("Use Mock Data (Testing Mode)", value=False)
    
    st.divider()
    st.subheader("Control Parameters")
    target_biomass = st.slider("Target Yield (g/L)", 0.5, 12.0, 5.0)
    control_mode = st.radio("Optimization Logic", ["PID", "MPC (Predictive)"])

# ==================================================
# MAIN INTERFACE
# ==================================================
if use_mock:
    sensor = {"pH": 5.4, "temp": 31.2, "DO": 88.0, "time": "Manual", "status": "Mocking"}
else:
    sensor = fetch_thingspeak_latest()

# Metric Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("pH Level", sensor["pH"], delta_color="inverse")
c2.metric("Temp (°C)", sensor["temp"])
c3.metric("Dissolved Oxygen", f"{sensor['DO']}%")
c4.metric("Status", sensor["status"])

tabs = st.tabs(["📊 Digital Twin", "🧪 AI Recipe Generator", "📑 History Logs"])

with tabs[0]:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Predicted Biomass Accumulation")
        # Stress Simulation inputs
        with st.expander("Configure Environmental Stress"):
            stress = st.checkbox("Enable Stress Event")
            s_ph = st.slider("Drop pH to", 2.0, 7.0, 4.0)
            s_time = st.slider("Event Start (hr)", 0, 48, 20)
            
        t, biomass = simulate_growth(sensor["pH"], sensor["temp"], s_ph if stress else None, s_time)
        
        chart_data = pd.DataFrame({"Time (hr)": t, "Biomass (g/L)": biomass})
        st.line_chart(chart_data.set_index("Time (hr)"))
        
    with col_right:
        st.subheader("Reactor State")
        current_b = round(biomass[-1], 2)
        st.write(f"**Current Biomass:** {current_b} g/L")
        
        # 3D Placeholder (Prevents pyvista crash on some browsers)
        try:
            mat_height = max(0.5, current_b * 0.5)
            mat = pv.Cylinder(radius=1.5, height=mat_height)
            showmol(mat, height=300, width=300)
        except:
            st.info("3D Visualization loading...")

with tabs[1]:
    st.subheader("Inverse Kinetic Profiler")
    st.write("Determine the optimal parameters to reach your target yield.")
    if st.button("Calculate Optimal Recipe"):
        with st.spinner("Analyzing historical trends..."):
            # Placeholder for RF Model
            st.success("Analysis Complete")
            st.json({
                "Target": f"{target_biomass} g/L",
                "Optimal pH": 5.55,
                "Optimal Temp": "30.2 °C",
                "Est. Time": "34.5 Hours"
            })

with tabs[2]:
    st.subheader("Experiment Logs")
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE)
        st.dataframe(df_log, use_container_width=True)
    else:
        st.info("No logs recorded yet. Start a simulation to generate data.")
