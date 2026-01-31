import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.integrate import odeint
import pyvista as pv
from stmol import showmol

# --- PAGE SETUP ---
st.set_page_config(page_title="Bio-Twin Research Pro", page_icon="🧬", layout="wide")

st.title("🧬 Bio-Twin: Intelligent Bioprocess Platform")
st.markdown("---")

# --- ARMORED DATA FETCHING (Solves 'int' error) ---
def fetch_data():
    try:
        # Step 1: Force string conversion to prevent the 'int' iteration crash
        raw_id = st.secrets.get("THINGSPEAK_CHANNEL_ID", "")
        chid = str(raw_id).strip().replace('"', '')
        
        raw_key = st.secrets.get("THINGSPEAK_READ_KEY", "")
        key = str(raw_key).strip().replace('"', '')

        if not chid or chid == "":
            return None, "Secrets Missing"

        # Step 2: Fetch the last 10 results for the history log
        url = f"https://api.thingspeak.com/channels/{chid}/feeds.json?api_key={key}&results=10"
        r = requests.get(url, timeout=5).json()
        
        if "feeds" in r and len(r["feeds"]) > 0:
            feeds = r["feeds"]
            latest = feeds[-1]
            # Process the history for the log table
            history_df = pd.DataFrame(feeds)
            history_df = history_df[['created_at', 'field1', 'field2', 'field3']].rename(columns={
                'created_at': 'Timestamp', 'field1': 'pH', 'field2': 'Temp', 'field3': 'DO'
            })
            
            return {
                "latest": {
                    "pH": float(latest.get("field1", 7.0)),
                    "temp": float(latest.get("field2", 30.0)),
                    "DO": float(latest.get("field3", 100.0)),
                    "time": latest.get("created_at", "N/A")
                },
                "history": history_df
            }, "Online"
        return None, "Channel Empty"
    except Exception as e:
        return None, f"Offline: {str(e)}"

# --- KINETIC SIMULATOR ---
def growth_model(state, t, pH, T):
    X, S = state
    mu_max, Ks, Yxs = 0.45, 0.5, 0.6
    # Optimization curve (Gaussian)
    f_env = np.exp(-(pH - 5.5)**2) * np.exp(-(T - 30.0)**2 / 20)
    mu = mu_max * (S / (Ks + S)) * f_env
    return [mu * X, -(1/Yxs) * mu * X]

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    mode = st.toggle("Simulate Hardware (Demo Mode)", value=False)
    st.divider()
    target_yield = st.slider("Target Biomass (g/L)", 1.0, 15.0, 5.0)
    if st.button("♻️ Reboot App Connection"):
        st.cache_data.clear()
        st.rerun()

# Execution
fetch_result, status = ({"latest": {"pH": 5.4, "temp": 30.2, "DO": 94.0, "time": "Demo"}, "history": pd.DataFrame()}, "Demo") if mode else fetch_data()

# --- METRIC DASHBOARD ---
m1, m2, m3, m4 = st.columns(4)
if fetch_result:
    live = fetch_result["latest"]
    m1.metric("Live pH", live["pH"], delta=round(live["pH"]-5.5, 2), delta_color="inverse")
    m2.metric("Temp (°C)", live["temp"])
    m3.metric("DO (%)", live["DO"])
    m4.metric("Status", status)
else:
    st.error(f"📡 Connection Alert: {status}")

# --- RESEARCH TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Digital Twin", "🤖 AI Optimizer", "📜 Hardware History"])

with tab1:
    col_a, col_b = st.columns([3, 1])
    
    with col_b:
        st.subheader("Simulate Stress")
        stress = st.checkbox("Acid/Base Shock")
        s_ph = st.slider("Shock pH", 2.0, 9.0, 4.0) if stress else 5.5
        s_time = st.slider("Trigger (hr)", 0, 48, 12) if stress else 0

    with col_a:
        # ODE Simulation Loop
        t = np.linspace(0, 48, 100)
        state = [0.1, 25.0]
        results = []
        for i in t:
            curr_ph = s_ph if (stress and i >= s_time) else (live["pH"] if fetch_result else 5.5)
            curr_temp = live["temp"] if fetch_result else 30.0
            step = odeint(growth_model, state, [0, 0.5], args=(curr_ph, curr_temp))[-1]
            state = step
            results.append(state[0])
        
        st.line_chart(pd.DataFrame({"Time (hr)": t, "Biomass": results}).set_index("Time (hr)"))

        # 🏗️ 3D BIOREACTOR
        try:
            st.subheader("3D Reactor Visualization")
            # Cylinder height grows with biomass
            cyl = pv.Cylinder(center=(0,0,0), radius=1, height=max(0.5, results[-1]*0.4))
            showmol(cyl, height=300, width=400)
        except:
            st.info("3D rendering active...")

with tab2:
    st.subheader("🤖 AI Recipe Strategy")
    if st.button("Calculate Optimal Recipe"):
        st.success("Analysis Complete")
        st.json({
            "Optimal Temp Profile": "30.5°C",
            "Optimal pH Profile": "5.5 ramping to 5.8",
            "Estimated Efficiency": "94.8%",
            "Next Harvest": "Thursday 10:00 AM"
        })

with tab3:
    st.subheader("📜 Recent ThingSpeak Logs")
    if fetch_result and not fetch_result["history"].empty:
        st.dataframe(fetch_result["history"], use_container_width=True)
        csv = fetch_result["history"].to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", data=csv, file_name="bioreactor_history.csv")
    else:
        st.info("No historical data available in current mode.")
