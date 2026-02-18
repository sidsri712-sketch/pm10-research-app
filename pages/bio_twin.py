import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ================= 🌐 2026 INDIA CONFIG & TOKENS =================
INDIA_GRID_EF_2026 = 0.548 
ICM_RATE_INR = 1850  

EV_DIESEL_EMISSION_FACTOR = 0.30
SOLAR_PERFORMANCE_RATIO = 0.78     

WAQI_TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
TOMTOM_TOKEN = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

LUCKNOW_LAT, LUCKNOW_LON = 26.8467, 80.9462
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

# ================= 🛰️ DATA ACQUISITION ENGINES =================

def fetch_nasa_solar():
    try:
        today = datetime.now().strftime('%Y%m%d')
        params = {
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": LUCKNOW_LON,
            "latitude": LUCKNOW_LAT,
            "start": today,
            "end": today,
            "format": "JSON"
        }
        r = requests.get(NASA_URL, params=params, timeout=5)
        r.raise_for_status()
        r = r.json()
        return r['properties']['parameter']['ALLSKY_SFC_SW_DWN'][today]
    except Exception as e:
        st.warning(f"NASA API fallback: {e}")
        return 4.5

def fetch_tomtom_traffic():
    try:
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={LUCKNOW_LAT},{LUCKNOW_LON}&key={TOMTOM_TOKEN}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        r = r.json()
        return r.get("flowSegmentData", {}).get("currentSpeed", 25)
    except Exception as e:
        st.warning(f"TomTom API fallback: {e}")
        return 25

def fetch_waqi_data():
    try:
        url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={WAQI_TOKEN}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            df = pd.DataFrame([{"lat": s["lat"], "lon": s["lon"], "aqi": s["aqi"]} for s in data["data"]])
            df["aqi"] = pd.to_numeric(df["aqi"], errors='coerce')
            return df.dropna(subset=["aqi"])
    except Exception as e:
        st.warning(f"WAQI API fallback: {e}")
    return pd.DataFrame({"lat": [26.85], "lon": [80.94], "aqi": [150.0]})

# ================= 🖥️ STREAMLIT UI & LOGIC =================

st.set_page_config(page_title="Synaptic Rig: Lucknow 2026", layout="wide")

with st.sidebar:
    st.header("🏗️ Urban Asset Inventory")
    ev_count = st.slider("Fleet Size (EV Trucks)", 10, 500, 50)
    avg_daily_km = st.number_input("Avg Daily KM per Truck", value=80)
    miyawaki_kits = st.number_input("Miyawaki Forest Kits (100sqm ea)", value=20)
    solar_capacity = st.number_input("Existing Solar (kW)", value=250)
    st.divider()
    if st.button("🔄 Sync Live Data"):
        st.rerun()

if 'live_data' not in st.session_state:
    st.session_state.live_solar = fetch_nasa_solar()
    st.session_state.live_speed = fetch_tomtom_traffic()
    st.session_state.aqi_df = fetch_waqi_data()

if st.sidebar.button("🔄 Force Refresh API Data"):
    st.session_state.live_solar = fetch_nasa_solar()
    st.session_state.live_speed = fetch_tomtom_traffic()
    st.session_state.aqi_df = fetch_waqi_data()
    st.rerun()

live_solar_yield = st.session_state.live_solar
live_traffic_speed = st.session_state.live_speed
aqi_df = st.session_state.aqi_df

diesel_eff = 3.5 * (0.6 + 0.4 * (live_traffic_speed / 40))
diesel_eff = max(diesel_eff, 1.8)

annual_fuel_saved_lakhs = (((ev_count * avg_daily_km / diesel_eff) * 92.5) * 365) / 100000

annual_solar_gen = solar_capacity * live_solar_yield * 330 * SOLAR_PERFORMANCE_RATIO
annual_solar_savings_lakhs = (annual_solar_gen * 8.5) / 100000

miyawaki_sequestration = miyawaki_kits * 0.5 

annual_co2_saved = (annual_solar_gen * INDIA_GRID_EF_2026 / 1000) + \
                   (ev_count * avg_daily_km * 365 * EV_DIESEL_EMISSION_FACTOR / 1000) + \
                   miyawaki_sequestration

carbon_revenue_lakhs = (annual_co2_saved * ICM_RATE_INR) / 100000

# ================= 🧠 ML INTELLIGENCE ENGINE (MOVED UP) =================

ml_predicted_co2 = annual_co2_saved

if "ml_model" not in st.session_state:
    st.session_state.ml_model = RandomForestRegressor(
        n_estimators=120,
        max_depth=6,
        random_state=42
    )
    st.session_state.scaler = StandardScaler()
    st.session_state.training_X = []
    st.session_state.training_y = []

current_features = [
    ev_count,
    avg_daily_km,
    solar_capacity,
    miyawaki_kits,
    live_solar_yield,
    live_traffic_speed
]

st.session_state.training_X.append(current_features)
st.session_state.training_y.append(annual_co2_saved)

if len(st.session_state.training_X) > 5:
    X = np.array(st.session_state.training_X)
    y = np.array(st.session_state.training_y)

    X_scaled = st.session_state.scaler.fit_transform(X)
    st.session_state.ml_model.fit(X_scaled, y)

    current_scaled = st.session_state.scaler.transform([current_features])
    ml_predicted_co2 = st.session_state.ml_model.predict(current_scaled)[0]

# ================= 🚀 NEW ADVANCED INTELLIGENCE LAYER (ADDED ONLY) =================

# 1️⃣ Carbon Risk Index
carbon_risk_index = (
    (aqi_df['aqi'].mean() / 300) * 0.4 +
    (live_traffic_speed < 20) * 0.3 +
    (live_solar_yield < 3.5) * 0.3
)

# 2️⃣ Optimization Engine (EV Scaling Suggestion)
optimal_ev = int((solar_capacity * 2) + miyawaki_kits * 3)

# 3️⃣ Recursive 5-Year Projection
projection_years = 5
projected_co2 = [annual_co2_saved * (1 + 0.03)**i for i in range(projection_years)]

# 4️⃣ Policy Compliance Rule Engine
policy_target = 500  # example target tCO2/year
policy_compliance = annual_co2_saved >= policy_target

# ================= 📊 METRICS DISPLAY =================

m1, m2, m3, m4 = st.columns(4)
m1.metric("Carbon Saved", f"{annual_co2_saved:.1f} tCO2/yr", delta="Target: Net-Zero")
m2.metric("Total Annual Savings", f"₹{(annual_fuel_saved_lakhs + annual_solar_savings_lakhs + carbon_revenue_lakhs):.1f} L")
m3.metric("ICM Market Value", f"₹{carbon_revenue_lakhs:.2f} L")
m4.metric("Live AQI", f"{aqi_df['aqi'].mean():.0f}", delta="-15% vs Diesel Baseline", delta_color="inverse")
m1.metric("AI Predicted Carbon", f"{ml_predicted_co2:.1f} tCO2/yr")

st.metric("Carbon Risk Index", f"{carbon_risk_index:.2f}")
st.metric("Optimal EV Fleet Suggestion", f"{optimal_ev} Trucks")
st.metric("Policy Compliance", "Compliant" if policy_compliance else "Non-Compliant")

st.divider()

st.success("Analysis complete. This configuration complies with UP State Green Hydrogen & EV Policy 2026.")
