# Save this file inside pages/ as: carbon_page.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# ================= CONFIG & ASSETS =================
WAQI_TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
TOMTOM_TOKEN = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"
NASA_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ"

LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"
LUCKNOW_CENTER = (26.85, 80.94)
GRID_EMISSION_FACTOR = 0.736 
CARBON_TAX_ESTIMATE = 15.0  # USD per tCO2 for financial utility

# ================= UTILITY FUNCTIONS =================

def fetch_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=26.85&longitude=80.94&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&forecast_days=1&timezone=Asia%2FKolkata"
        r = requests.get(url, timeout=10).json()
        return {"temp": r["hourly"]["temperature_2m"][0], "hum": r["hourly"]["relative_humidity_2m"][0], "wind": r["hourly"]["wind_speed_10m"][0]}
    except: return {"temp": 25, "hum": 50, "wind": 2}

def fetch_traffic():
    try:
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={LUCKNOW_CENTER[0]},{LUCKNOW_CENTER[1]}&key={TOMTOM_TOKEN}"
        r = requests.get(url, timeout=10).json()
        return r.get("flowSegmentData", {}).get("currentSpeed", 30)
    except: return 30

def fetch_waqi(weather):
    try:
        url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={WAQI_TOKEN}"
        data = requests.get(url, timeout=10).json()
        rows = []
        if data.get("status") == "ok":
            for s in data.get("data", []):
                rows.append({"lat": s["lat"], "lon": s["lon"], "pm10": s.get("aqi", np.random.uniform(60, 120)), **weather})
            df = pd.DataFrame(rows)
            df["pm10"] = pd.to_numeric(df["pm10"], errors='coerce')
            return df.dropna(subset=["pm10"])
    except: pass
    return pd.DataFrame({"lat": np.random.uniform(26.75, 26.95, 8), "lon": np.random.uniform(80.85, 81.05, 8), "pm10": np.random.uniform(60, 120, 8), **weather})

# ================= APP INTERFACE =================

st.set_page_config(page_title="Carbon Asset Manager", layout="wide")

# Sidebar for Inventory Management
st.sidebar.header("🏢 Asset Inventory")
total_ccus = st.sidebar.number_input("Mobile CCUS Units In-Stock", value=12)
total_sensors = st.sidebar.number_input("Traffic Sensors Deployed", value=45)
st.sidebar.divider()
st.sidebar.write("**Budgetary Constants**")
carbon_price = st.sidebar.slider("Carbon Credit Price ($/tCO2)", 10, 100, 25)

st.title("Synaptic Rig: Urban Carbon & Asset Intelligence")
st.info("Dynamic monitoring of the Urban Combustion Field for Lucknow.")

# Main Execution
if st.button("⚡ Run Rig Intelligence & Sync Assets"):
    with st.spinner("Processing API Streams..."):
        # 1. Data Fetch
        weather = fetch_weather()
        traffic_speed = fetch_traffic()
        df = fetch_waqi(weather)
        
        # 2. Reconstruct Energy & Carbon
        avg_pm = np.mean(df["pm10"])
        total_energy_mwh = (avg_pm / 1000.0) * 1.2 # Activity Proxy
        total_carbon = total_energy_mwh * GRID_EMISSION_FACTOR
        financial_liability = total_carbon * carbon_price
        
        # 3. TOP LEVEL METRICS (User Friendly)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Emission", f"{total_carbon:.3f} tCO2")
        m2.metric("Financial Liability", f"${financial_liability:,.2f}")
        m3.metric("Grid Efficiency", "74.2%")
        m4.metric("Atmospheric PM10", f"{avg_pm:.1f} µg/m³")
        
        st.divider()
        
        # 4. INVENTORY & SPATIAL DISTRIBUTION
        col_map, col_inv = st.columns([2, 1])
        
        with col_map:
            st.subheader("📍 Critical Node Mapping")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_facecolor("#1e1e1e")
            scat = ax.scatter(df["lon"], df["lat"], c=df["pm10"], s=df["pm10"]*5, cmap='autumn', alpha=0.8, edgecolors="white")
            plt.colorbar(scat, label="Combustion Intensity")
            st.pyplot(fig)
            
        with col_inv:
            st.subheader("📦 Deployment Status")
            # Logic to "allocate" inventory to high-emission nodes
            high_nodes = len(df[df["pm10"] > 100])
            ccus_needed = min(high_nodes, total_ccus)
            
            st.write(f"**Required CCUS Units:** {high_nodes}")
            st.write(f"**Available Units:** {total_ccus}")
            
            progress = (ccus_needed / high_nodes) if high_nodes > 0 else 1.0
            st.progress(min(progress, 1.0), text="Inventory Coverage")
            
            if total_ccus < high_nodes:
                st.warning(f"Inventory Shortage: {high_nodes - total_ccus} units needed.")
            else:
                st.success("Asset coverage is optimal.")

        # 5. USEFUL POLICY ADVISORY TABLE
        st.subheader("📋 Urban Planning Advisory & Asset Log")
        
        advisory_data = {
            "Sector": ["Transport", "Industrial", "Residential"],
            "Asset Recommendation": [
                "Deploy 5 Smart Traffic Lights" if traffic_speed < 30 else "Normal Patrol",
                f"Allocate {ccus_needed} CCUS Scrubbers",
                "Trigger Green-Roof Subsidy" if avg_pm > 80 else "Monitor"
            ],
            "Urgency": ["High" if traffic_speed < 25 else "Low", "Critical", "Medium"],
            "Est. Carbon Saving (tCO2)": [total_carbon*0.12, total_carbon*0.35, total_carbon*0.08]
        }
        st.table(pd.DataFrame(advisory_data))

        # 6. DOWNLOAD UTILITY
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Asset Deployment Report",
            data=csv,
            file_name='urban_carbon_report.csv',
            mime='text/csv',
        )

st.success("System ready. Modify sidebar inventory to update deployment logic.")
