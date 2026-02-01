import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.integrate import odeint
import plotly.graph_objects as go
from datetime import datetime
import time

# ==================================================
# HARD-CODED CREDENTIALS & PUBLIC FALLBACKS
# ==================================================
TS_CHANNEL_ID = "3245928"
TS_READ_KEY = "8P0KH1WDH7QOR0AA"

# Public testing channels for robustness
PUBLIC_CHANNELS = {
    "My Bioreactor (Private)": {"id": "3245928", "key": "8P0KH1WDH7QOR0AA", "icon": "🧪"},
    "River Monitoring (Public)": {"id": "3122680", "key": "", "icon": "🌊"},
    "Hydroponic System (Public)": {"id": "1013172", "key": "", "icon": "🌿"},
    "Marine Research (Public)": {"id": "3041484", "key": "", "icon": "🐙"}
}

st.set_page_config(page_title="Bio-Twin Research Master", layout="wide")

# Custom UI Styling
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: #f0fff4;
        border: 2px solid #2ecc71;
        padding: 15px;
        border-radius: 15px;
    }
    .report-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 8px solid #2ecc71;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .phase-badge {
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9em;
        background: #e2e8f0;
    }
    .warning-box {
        padding: 10px;
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        color: #c53030;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🧬 Bio-Twin: Intelligent Research Platform")
st.caption(f"Version 5.2 | Growth Phase Analytics | Last Sync: {datetime.now().strftime('%H:%M:%S')}")

# ==================================================
# DATA ENGINE
# ==================================================
def fetch_data(channel_id, read_key):
    try:
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={read_key}&results=15"
        r = requests.get(url, timeout=5).json()
        
        if "feeds" in r and len(r["feeds"]) > 0:
            feeds = r["feeds"]
            latest = feeds[-1]
            df = pd.DataFrame(feeds)[['created_at', 'field1', 'field2', 'field3']]
            df.columns = ['Time', 'pH', 'Temp', 'DO']
            
            # Sanitizing data to ensure they are floats
            pH_val = float(latest.get("field1")) if latest.get("field1") else 7.0
            temp_val = float(latest.get("field2")) if latest.get("field2") else 30.0
            do_val = float(latest.get("field3")) if latest.get("field3") else 100.0
            
            return {
                "latest": {"pH": pH_val, "temp": temp_val, "DO": do_val},
                "history": df
            }, "🟢 System Online"
        return None, "🟠 Channel Empty"
    except Exception as e:
        return None, f"🔴 Connection Alert: {str(e)}"

def solve_biomass(ph, temp, target):
    t = np.linspace(0, 48, 100)
    def model(X, t):
        # Growth kinetics: $ \mu = \mu_{max} \cdot f(pH) \cdot f(T) $
        mu = 0.65 * np.exp(-0.6 * (ph - 5.5)**2) * np.exp(-0.1 * (temp - 30)**2)
        return mu * X * (1 - X/target)
    
    biomass = odeint(model, 0.2, t).flatten()
    
    # CALCULATE PHASE (Based on derivative of growth)
    derivatives = np.diff(biomass)
    current_growth_rate = derivatives[-1]
    
    if current_growth_rate < 0.01:
        phase = "Stationary Phase 🛑"
    elif current_growth_rate > 0.15:
        phase = "Log Phase (Exponential) 🚀"
    else:
        phase = "Lag Phase (Adjustment) 🌱"
        
    return t, biomass, phase

def draw_3d_reactor(fill_level, target):
    ratio = min(fill_level / target, 1.0)
    z_height = np.linspace(0, ratio * 5, 30)
    theta = np.linspace(0, 2*np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z_height)
    x_grid = np.cos(theta_grid)
    y_grid = np.sin(theta_grid)
    
    fig = go.Figure(data=[go.Surface(
        x=x_grid, y=y_grid, z=z_grid, 
        colorscale='Greens', 
        showscale=False,
        hoverinfo='z'
    )])
    fig.update_layout(
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, b=0, t=0), height=450
    )
    return fig

# --- RUN LOGIC ---
with st.sidebar:
    st.header("⚙️ Reactor Control")
    # Added selection for multiple channels with dynamic icons
    selected_name = st.selectbox("Select Data Source", list(PUBLIC_CHANNELS.keys()))
    active_id = PUBLIC_CHANNELS[selected_name]["id"]
    active_key = PUBLIC_CHANNELS[selected_name]["key"]
    active_icon = PUBLIC_CHANNELS[selected_name]["icon"]
    
    st.info(f"Connected to: {active_icon} {selected_name}")
    st.write(f"**Channel ID:** `{active_id}`")
    
    target_yield = st.slider("🎯 Target Yield (g/L)", 5.0, 30.0, 15.0)
    auto_refresh = st.checkbox("🔄 Enable Auto-Sync (30s)", value=True)
    if st.button("⚡ Manual Refresh"): st.rerun()

fetch_result, status = fetch_data(active_id, active_key)
live = fetch_result["latest"] if fetch_result else {"pH": 5.5, "temp": 30.0, "DO": 95.0}

# --- SMART GUARD ---
if live["pH"] < 4.5 or live["pH"] > 6.5:
    st.markdown('<div class="warning-box">⚠️ CRITICAL: pH levels outside metabolic safety range!</div>', unsafe_allow_html=True)
if live["temp"] > 35:
    st.markdown('<div class="warning-box">⚠️ ALERT: Thermal stress detected! Activating cooling.</div>', unsafe_allow_html=True)

# Dashboard Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("🧪 Live pH", live["pH"], delta=round(live["pH"]-5.5, 2), delta_color="inverse")
m2.metric("🌡️ Temp (°C)", f"{live['temp']}°")
m3.metric("💨 Oxygen (DO)", f"{live['DO']}%")
m4.metric("📡 Link Status", status)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Digital Twin", "🏗️ 3D Reactor", "🤖 AI Optimizer", "📑 Experiment Report"])

with tab1:
    st.subheader("📈 Simulated Biomass Accumulation")
    time_steps, biomass_data, current_phase = solve_biomass(live["pH"], live["temp"], target_yield)
    
    # New Phase Indicator Badge
    st.markdown(f"Current State: <span class='phase-badge'>{current_phase}</span>", unsafe_allow_html=True)
    
    chart_df = pd.DataFrame({"Hour": time_steps, "Biomass (g/L)": biomass_data}).set_index("Hour")
    st.line_chart(chart_df, color="#2ecc71")

with tab2:
    st.subheader("🧊 Real-Time Tank Volume")
    current_vol = float(biomass_data[-1])
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(draw_3d_reactor(current_vol, target_yield), use_container_width=True)
    with col2:
        st.write("### 🔍 Analysis")
        st.write(f"Mass: **{round(current_vol, 2)} g/L**")
        st.progress(min(current_vol/target_yield, 1.0))
        st.write(f"Phase: **{current_phase.split(' ')[0]}**")

with tab3:
    st.subheader("🤖 AI Predictive Modeling")
    if st.button("🔮 Calculate Optimal Recipe"):
        prob = 100 - (abs(live["pH"] - 5.5) * 40)
        st.success(f"Batch Success Probability: {max(0, round(prob, 1))}%")
        st.json({
            "Kinetic Status": current_phase,
            "Optimal pH": 5.5,
            "Target Yield Confidence": f"{90 + (live['DO']/100)*5}%",
            "Recommendation": "Maintain current conditions."
        })

with tab4:
    st.subheader("📑 Automated Research Report")
    st.markdown(f"""
    <div class="report-card">
        <h4>📋 Experiment Summary</h4>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}</p>
        <p><b>Current Phase:</b> {current_phase}</p>
        <p><b>Yield Efficiency:</b> {round((biomass_data[-1]/target_yield)*100, 1)}%</p>
    </div>
    """, unsafe_allow_html=True)
    
    if fetch_result:
        st.write("---")
        st.write("**📂 Recent Hardware Logs:**")
        st.dataframe(fetch_result["history"], use_container_width=True)
        st.download_button("💾 Export to CSV", fetch_result["history"].to_csv().encode('utf-8'), "experiment_data.csv")

if auto_refresh:
    time.sleep(30)
    st.rerun()
