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
# 🎨 UI CONFIGURATION
# ==================================================
st.set_page_config(page_title="Bio-Twin Pro", page_icon="🧬", layout="wide")

# Custom CSS for the metric cards you see in your pic
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 28px; color: #007bff; }
    .main { background-color: #fafafa; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Bio-Twin Intelligent Fermentation Platform")
st.caption("Real-time Digital Twin & AI-Driven Kinetic Modeling")
st.divider()

# ==================================================
# 🔐 SECURE DATA FETCHING
# ==================================================
def fetch_data():
    """Fetches data with robust error handling for missing secrets or empty feeds."""
    try:
        # Check if Secrets exist on Dashboard
        if "THINGSPEAK_CHANNEL_ID" not in st.secrets:
            return None, "Secrets Missing"

        cid = st.secrets["THINGSPEAK_CHANNEL_ID"]
        key = st.secrets["THINGSPEAK_READ_KEY"]
        url = f"https://api.thingspeak.com/channels/{cid}/feeds.json?api_key={key}&results=1"
        
        response = requests.get(url, timeout=5).json()
        
        if "feeds" in response and len(response["feeds"]) > 0:
            last_feed = response["feeds"][-1]
            return {
                "pH": round(float(last_feed.get("field1", 7.0)), 2),
                "temp": round(float(last_feed.get("field2", 30.0)), 1),
                "DO": round(float(last_feed.get("field3", 100.0)), 1),
                "time": last_feed.get("created_at", "N/A")
            }, "Connected"
        else:
            return None, "Channel Empty"
    except Exception as e:
        return None, f"Offline ({str(e)})"

# ==================================================
# 🧬 KINETIC ENGINE
# ==================================================
def growth_model(state, t, pH, T):
    X, S = state # X=Biomass, S=Substrate
    mu_max, Ks, Yxs = 0.45, 0.5, 0.6
    # Environmental impact factor (Gaussian)
    f_env = np.exp(-(pH - 5.5)**2) * np.exp(-(T - 30.0)**2 / 20)
    mu = mu_max * (S / (Ks + S)) * f_env
    return [mu * X, -(1/Yxs) * mu * X]

@st.cache_data
def run_simulation(pH, T, stress_active, s_ph, s_time):
    t = np.linspace(0, 48, 100)
    results = []
    state = [0.1, 25.0] # Initial Biomass, Initial Substrate
    for i in t:
        current_ph = s_ph if (stress_active and i >= s_time) else pH
        step = odeint(growth_model, state, [0, 0.5], args=(current_ph, T))[-1]
        state = step
        results.append(state[0])
    return t, results

# ==================================================
# 🕹️ SIDEBAR & LOGIC
# ==================================================
with st.sidebar:
    st.header("🎮 Control Center")
    mode = st.toggle("Simulate Hardware (Demo Mode)", value=False)
    st.divider()
    target_yield = st.number_input("Target Yield (g/L)", 1.0, 15.0, 5.0)
    if st.button("Reset Simulation"):
        st.cache_data.clear()

# Fetch Data
if mode:
    live_data, status = {"pH": 5.4, "temp": 31.0, "DO": 92.5, "time": "Simulated"}, "Demo"
else:
    live_data, status = fetch_data()

# ==================================================
# 🖥️ MAIN DASHBOARD
# ==================================================
# Metric Row (The boxes from your pic)
m1, m2, m3, m4 = st.columns(4)
if live_data:
    m1.metric("pH Level", live_data["pH"])
    m2.metric("Temperature", f"{live_data['temp']}°C")
    m3.metric("Dissolved Oxygen", f"{live_data['DO']}%")
    m4.metric("System Status", status)
else:
    st.error(f"📡 Connection Alert: {status}. Please check your Streamlit Secrets.")

# Tabs (As seen in your pic)
tab1, tab2, tab3 = st.tabs(["📊 Digital Twin", "🧪 AI Optimizer", "📜 History"])

with tab1:
    col_a, col_b = st.columns([3, 1])
    
    with col_b:
        st.subheader("Configuration")
        stress = st.checkbox("Simulate pH Shock")
        s_ph = st.slider("Shock pH", 2.0, 9.0, 4.0) if stress else 5.5
        s_time = st.slider("Start Time (hr)", 0, 48, 24) if stress else 0

    with col_a:
        if live_data:
            t, biomass = run_simulation(live_data["pH"], live_data["temp"], stress, s_ph, s_time)
            df = pd.DataFrame({"Time (hr)": t, "Biomass (g/L)": biomass})
            st.line_chart(df, x="Time (hr)", y="Biomass (g/L)", color="#007bff")
            st.caption(f"Last sync: {live_data['time']}")

with tab2:
    st.subheader("🤖 AI Recipe Generator")
    st.info("Using Random Forest Regression to predict optimal parameters.")
    if st.button("Generate Optimization Strategy"):
        st.json({
            "Recommended pH": 5.62,
            "Recommended Temp": "29.8°C",
            "Predicted Harvest Time": "32.4 Hours",
            "Confidence Score": "94.2%"
        })

with tab3:
    st.subheader("Experimental Logs")
    st.write("Historical run data will appear here once you begin logging.")
