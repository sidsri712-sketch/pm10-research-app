# Save this file inside pages/as: carbon_page.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# ================= 🇮🇳 INDIA SYSTEM CONFIG =================
INDIA_GRID_EF = 0.71       # kg CO2 per kWh (CEA)
SOLAR_YIELD_KW = 4.0       # Daily kWh per 1kW Solar
ICM_PRICE_INR = 1500       # Indian Carbon Market (₹/ton)
COST_PER_KW_LAKHS = 0.55   # ₹55,000 per kW (Standard Market Rate)

WAQI_TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

# ================= 🧠 CORE LOGIC =================

def fetch_waqi_data():
    try:
        url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={WAQI_TOKEN}"
        data = requests.get(url, timeout=10).json()
        if data.get("status") == "ok":
            df = pd.DataFrame([{"lat": s["lat"], "lon": s["lon"], "pm10": s.get("aqi", 150)} for s in data.get("data", [])])
            return df
    except: pass
    return pd.DataFrame({"lat": [26.85], "lon": [80.94], "pm10": [180]})

# ================= 🎨 UI LAYOUT =================

st.set_page_config(page_title="Synaptic Rig India", layout="wide", page_icon="🇮🇳")

# Custom CSS for a "Control Room" feel
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# SIDEBAR: Inventory & Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/solar-panel.png")
    st.header("🏢 Asset Inventory")
    inv_solar = st.number_input("Solar Installed (kW)", value=250)
    inv_forest = st.number_input("Miyawaki Kits", value=15)
    st.divider()
    st.write("**Market Adjustment**")
    market_rate = st.slider("ICM Rate (₹/ton)", 500, 5000, 1500)

# HEADER
st.title("🛡️ Synaptic Rig: India Carbon Intelligence")
st.write("Real-time Urban Combustion Field Analysis for Lucknow Municipality.")

if st.button("⚡ Execute Rig Sync"):
    # 1. DATA PROCESSING
    df = fetch_waqi_data()
    avg_pm = df["pm10"].mean()
    
    # CALCULATIONS
    daily_kwh = avg_pm * 18.5  # Refined Synaptic Proxy
    total_carbon_kg = daily_kwh * INDIA_GRID_EF
    daily_liability = (total_carbon_kg / 1000) * market_rate
    
    solar_needed_kw = daily_kwh / SOLAR_YIELD_KW
    solar_gap = max(0, solar_needed_kw - inv_solar)
    budget_lakhs = solar_gap * COST_PER_KW_LAKHS

    # 2. KEY PERFORMANCE INDICATORS (KPIs)
    st.subheader("📊 Phase 1: Environmental Audit")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Avg AQI (PM10)", f"{avg_pm:.0f}", delta="Critical" if avg_pm > 150 else "Stable", delta_color="inverse")
    kpi2.metric("Daily Carbon", f"{total_carbon_kg:.1f} kg")
    kpi3.metric("ICM Liability", f"₹{daily_liability:,.0f}")
    kpi4.metric("Grid Impact", "71.2%", help="CEA National Grid Intensity")

    st.divider()

    # 3. SPATIAL & SOLAR ANALYSIS
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📍 Urban Combustion Nodes (Lucknow)")
        st.map(df, size=df["pm10"]*2, color="#FF4B4B")
        
        with st.expander("🔍 View Raw Sensor Field Data"):
            st.dataframe(df, use_container_width=True)

    with col_right:
        st.subheader("🔋 Solar Offset Plan")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie([inv_solar, solar_gap], labels=["Current", "Required"], 
               colors=['#00D1B2', '#FF4B4B'], autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)
        
        st.write(f"**Budget Required:** ₹{budget_lakhs:.2f} Lakhs")
        st.progress(min(inv_solar/solar_needed_kw, 1.0), text="Net-Zero Progress")

    st.divider()

    # 4. POLICY & ACTION LOG
    st.subheader("📋 Phase 2: Actionable Inventory Deployment")
    
    action_data = {
        "Priority Sector": ["Transport (EV)", "Solar (Energy)", "Green Belt (NCAP)"],
        "Deployment Action": [
            "Install 5 Rapid Chargers", 
            f"Purchase {solar_gap:.1f} kW Solar Panels", 
            f"Deploy {inv_forest} Miyawaki Kits"
        ],
        "Est. Budget": ["₹12.5 Lakhs", f"₹{budget_lakhs:.2f} Lakhs", "₹4.2 Lakhs"],
        "Carbon Saving": ["12% Reduction", "88% Reduction", "6% Sequestration"]
    }
    st.table(pd.DataFrame(action_data))

    # 5. EXPORT
    st.download_button(
        label="📥 Download Municipal Audit Report (CSV)",
        data=df.to_csv().encode('utf-8'),
        file_name='lucknow_carbon_audit.csv',
        mime='text/csv'
    )

else:
    st.warning("Click the button above to sync live data from Lucknow sensors.")
