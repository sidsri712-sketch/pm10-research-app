# Save this file inside pages/ as: carbon_page.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt

# ================= INDIA-SPECIFIC CONFIG =================
INDIA_GRID_EF = 0.71  # kg CO2 per kWh (CEA Standard)
SOLAR_YIELD_KW = 4.0  # Avg daily kWh per 1kW solar in India
COST_PER_KW_INR = 55000  # Avg installation cost (₹55,000 per kW)

WAQI_TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

# ================= DATA FETCH =================

def fetch_waqi():
    try:
        url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={WAQI_TOKEN}"
        data = requests.get(url, timeout=10).json()
        if data.get("status") == "ok":
            df = pd.DataFrame([{"pm10": s.get("aqi", 150)} for s in data.get("data", [])])
            df["pm10"] = pd.to_numeric(df["pm10"], errors='coerce')
            return df["pm10"].mean()
    except: pass
    return 150.0

# ================= APP INTERFACE =================

st.set_page_config(page_title="Lucknow Net-Zero Planner", layout="wide")

st.title("🇮🇳 Lucknow Net-Zero Budget Planner")
st.markdown("### Carbon-to-Solar Conversion & Financial Estimator")

# Sidebar: Inventory & Budget Settings
st.sidebar.header("🏢 Current Assets")
available_solar = st.sidebar.number_input("Solar Already Installed (kW)", value=100)
st.sidebar.divider()
st.sidebar.header("💰 Budget Settings")
solar_rate = st.sidebar.slider("Cost per kW (₹)", 40000, 70000, 55000)

if st.button("📊 Calculate Net-Zero Requirements"):
    with st.spinner("Analyzing Urban Combustion Field..."):
        # 1. Monitoring
        avg_aqi = fetch_waqi()
        
        # Simplified Logic: 1 AQI unit proxy for 15kWh daily urban energy footprint
        daily_kwh = avg_aqi * 15 
        daily_carbon_kg = daily_kwh * INDIA_GRID_EF
        
        # 2. Solar Math
        required_kw = daily_kwh / SOLAR_YIELD_KW
        gap_kw = max(0, required_kw - available_solar)
        total_cost_inr = gap_kw * solar_rate
        cost_in_lakhs = total_cost_inr / 100000

        # DISPLAY RESULTS
        st.subheader("📍 Current Environment & Footprint")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg PM10 (Lucknow)", f"{avg_aqi:.0f} µg/m³")
        c2.metric("Daily Carbon Produced", f"{daily_carbon_kg:.1f} kg")
        c3.metric("Energy Footprint", f"{daily_kwh:.1f} kWh/day")

        st.divider()

        st.subheader("🔋 Solar Inventory & Financial Plan")
        f1, f2, f3 = st.columns(3)
        
        f1.metric("Solar Gap", f"{gap_kw:.1f} kW")
        f2.metric("Investment Required", f"₹{cost_in_lakhs:.2f} Lakhs")
        
        # Financial Health Status
        if gap_kw <= 0:
            f3.success("Status: NET ZERO ACHIEVED")
        else:
            f3.error("Status: CARBON POSITIVE")

        # 3. USEFUL VISUALS
        v1, v2 = st.columns(2)
        with v1:
            st.write("**Solar Asset Gap Analysis**")
            fig, ax = plt.subplots()
            ax.pie([available_solar, gap_kw], labels=["In Stock", "Required"], 
                   autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=140)
            st.pyplot(fig)
        
        with v2:
            st.write("**Quick Decision Log**")
            plan = {
                "Priority": ["Immediate", "Medium-Term", "Long-Term"],
                "Investment": [f"₹{cost_in_lakhs*0.4:.1f} L", f"₹{cost_in_lakhs*0.4:.1f} L", f"₹{cost_in_lakhs*0.2:.1f} L"],
                "Goal": ["Off-set High Intensity Nodes", "Commercial Rooftop Solar", "Residential Subsidies"]
            }
            st.table(pd.DataFrame(plan))

        # Final Summary
        st.info(f"**Executive Summary:** To reach Net-Zero today, Lucknow requires an immediate deployment of **{gap_kw:.1f} kW** of solar capacity, necessitating a budget of **₹{cost_in_lakhs:.2f} Lakhs**.")

st.success("App live. Adjust the 'Cost per kW' in the sidebar to match current vendor quotes.")
