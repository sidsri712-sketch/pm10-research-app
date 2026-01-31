import streamlit as st
import numpy as np
import pandas as pd
import requests
from scipy.integrate import odeint

# --- CONFIG ---
st.set_page_config(page_title="Bio-Twin Pro", page_icon="🧬", layout="wide")
st.title("🧬 Bio-Twin Intelligent Fermentation")

# --- THE CONNECTION FIX ---
def fetch_thingspeak_latest():
    try:
        # We grab the secrets and force them to be text (strings) 
        # This prevents the 'int is not iterable' crash
        cid = str(st.secrets.get("THINGSPEAK_CHANNEL_ID", "")).strip()
        key = str(st.secrets.get("THINGSPEAK_READ_KEY", "")).strip()

        if not cid or not key:
            return None, "Offline (Check Secrets)"

        # Build URL using .format() to ensure no 'int' errors
        url = "https://api.thingspeak.com/channels/{}/feeds.json?api_key={}&results=1".format(cid, key)
        
        r = requests.get(url, timeout=5).json()
        
        if "feeds" in r and len(r["feeds"]) > 0:
            f = r["feeds"][-1]
            return {
                "pH": float(f.get("field1", 7.0)),
                "temp": float(f.get("field2", 30.0)),
                "DO": float(f.get("field3", 95.0)),
                "time": f.get("created_at", "N/A")
            }, "Online"
        return None, "No Data Found"
    except Exception as e:
        return None, f"Error: {str(e)}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    use_mock = st.toggle("Demo Mode", value=False)
    target_yield = st.slider("Target Yield (g/L)", 1.0, 15.0, 5.0)

# Fetch data
sensor, status = ({"pH": 5.5, "temp": 30.0, "DO": 90, "time": "Demo"}, "Simulating") if use_mock else fetch_thingspeak_latest()

# --- TOP METRICS ---
c1, c2, c3, c4 = st.columns(4)
if sensor:
    c1.metric("pH", sensor["pH"])
    c2.metric("Temp (°C)", sensor["temp"])
    c3.metric("DO (%)", sensor["DO"])
    c4.metric("Status", status)
else:
    st.error(f"📡 Connection Alert: {status}")

# --- TABS (Original logic preserved) ---
tab1, tab2, tab3 = st.tabs(["📊 Digital Twin", "🤖 AI Optimizer", "📜 Logs"])

with tab1:
    st.subheader("Predicted Biomass Accumulation")
    t = np.linspace(0, 48, 100)
    growth = (target_yield / (1 + np.exp(-0.2 * (t - 24))))
    st.line_chart(pd.DataFrame({"Time": t, "Biomass": growth}).set_index("Time"))

with tab2:
    st.subheader("AI Recipe Generator")
    if st.button("Generate Optimization"):
        st.json({"Rec. pH": 5.5, "Rec. Temp": "30.2C", "Confidence": "96%"})
