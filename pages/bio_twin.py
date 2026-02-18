import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# ================= 🇮🇳 INDIA SYSTEM CONFIG (2026) =================
GRID_EF_KG_KWH = 0.710  
SOLAR_YIELD_KW = 4.2    
COST_PER_KW_LAKHS = 0.52 

# Financial Factors for 2026
DIESEL_PRICE_INR = 92.5
ELECTRICITY_TARIFF_INR = 8.50
MAINTENANCE_SAVING_EV_PCT = 0.65  
FUEL_EFFICIENCY_DIESEL = 3.5      
EV_ENERGY_USAGE_KWH_KM = 0.8      

# Subsidy Constants
STATE_SOLAR_SUBSIDY_PER_KW = 0.05
EV_PURCHASE_SUBSIDY_UNIT = 2.5    
EV_BASE_COST = 10.5

# ================= 🎨 DASHBOARD UI =================

st.set_page_config(page_title="Lucknow Net-Zero Command", layout="wide", page_icon="🇮🇳")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    div[data-testid="metric-container"] {
        background-color: #ffffff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hourglass-growth.png")
    # Added refresh toggle as requested
    auto_refresh = st.toggle("Enable Live Auto-Refresh", value=True)
    refresh_rate = st.slider("Refresh Interval (min)", 1, 60, 10)
    
    st.divider()
    st.header("🏢 Asset Inventory")
    daily_mwh = st.number_input("Daily Consumption (MWh)", value=12.5)
    existing_solar = st.number_input("Existing Solar (kW)", value=450)
    
    st.subheader("🚗 Fleet Transition")
    fleet_size = st.number_input("Total Diesel Vehicles", value=50)
    target_ev_pct = st.slider("Target EV Transition (%)", 0, 100, 40)
    avg_daily_km = st.number_input("Avg km/vehicle/day", value=80)
    
    st.divider()
    apply_subsidies = st.checkbox("Apply UP Govt Incentives (2026)", value=True)
    icm_rate = st.slider("Carbon Rate (₹/ton)", 800, 3500, 1000)

# ================= 🧠 AUTO-REFRESH WRAPPER =================

@st.fragment(run_every=f"{refresh_rate}m" if auto_refresh else None)
def render_lucknow_dashboard():
    # ALL LOGIC BELOW IS 100% IDENTICAL TO YOUR PREVIOUS VERSION
    required_solar_kw = (daily_mwh * 1000) / SOLAR_YIELD_KW
    solar_gap = max(0, required_solar_kw - existing_solar)
    net_solar_investment = solar_gap * (COST_PER_KW_LAKHS - (STATE_SOLAR_SUBSIDY_PER_KW if apply_subsidies else 0))
    annual_solar_gen = solar_gap * SOLAR_YIELD_KW * 330 
    annual_solar_savings_lakhs = (annual_solar_gen * ELECTRICITY_TARIFF_INR) / 100000

    ev_count = round(fleet_size * (target_ev_pct / 100))
    net_ev_investment = ev_count * (EV_BASE_COST - (EV_PURCHASE_SUBSIDY_UNIT if apply_subsidies else 0))
    daily_diesel_cost = (ev_count * avg_daily_km / FUEL_EFFICIENCY_DIESEL) * DIESEL_PRICE_INR
    daily_ev_cost = (ev_count * avg_daily_km * EV_ENERGY_USAGE_KWH_KM) * ELECTRICITY_TARIFF_INR
    annual_fuel_savings_lakhs = ((daily_diesel_cost - daily_ev_cost) * 365) / 100000

    total_net_investment = net_solar_investment + net_ev_investment
    total_annual_savings = annual_solar_savings_lakhs + annual_fuel_savings_lakhs
    payback_years = total_net_investment / total_annual_savings if total_annual_savings > 0 else 0
    total_carbon_tons = ((daily_mwh * 1000 * GRID_EF_KG_KWH) + (fleet_size * 80 * 0.35)) / 1000

    st.title("🛡️ Synaptic Rig: Lucknow Net-Zero Planner")
    st.write(f"**Last Sync:** {time.strftime('%H:%M:%S')} (Interval: {refresh_rate} min)")

    st.subheader("💰 Phase 1: Investment vs. Returns")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Investment", f"₹{total_net_investment:.1f} L")
    c2.metric("Annual Savings", f"₹{total_annual_savings:.1f} L/yr")
    c3.metric("Payback Period", f"{payback_years:.1f} Years")
    c4.metric("CO2 reduction", f"{total_carbon_tons:.1f} T/Day")

    st.divider()

    col_plot, col_table = st.columns([2, 1])
    with col_plot:
        years = np.arange(0, 11)
        cash_flow = -total_net_investment + (total_annual_savings * years)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(years, cash_flow, marker='o', color='#2ecc71', linewidth=2)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_title("Municipal Breakeven Analysis (Cumulative Savings)")
        st.pyplot(fig)

    with col_table:
        st.write("#### 💸 Savings Breakdown")
        st.table(pd.DataFrame({
            "Source": ["Electricity (Solar)", "Fuel (EV)", "Maintenance"],
            "Annual (Lakhs)": [f"₹{annual_solar_savings_lakhs:.1f} L", f"₹{annual_fuel_savings_lakhs * 0.8:.1f} L", f"₹{annual_fuel_savings_lakhs * 0.2:.1f} L"]
        }))

    st.divider()
    st.subheader("📋 Phase 3: Actionable Municipal Log")
    st.table(pd.DataFrame({
        "Project": ["Solar Phase 1", "EV Fleet Pilot"],
        "Action": [f"Install {solar_gap:.1f}kW", f"Deploy {ev_count} units"],
        "Net Cost": [f"₹{net_solar_investment:.1f} L", f"₹{net_ev_investment:.1f} L"],
        "Priority": ["Immediate", "High"]
    }))

# RUN
render_lucknow_dashboard()
