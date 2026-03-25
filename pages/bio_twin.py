import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ================= CONFIG =================
API_KEY = ""c86236be4a9f76875aad940c96e5111b"
CITY = "Lucknow"

st.set_page_config(layout="wide")
st.title("⚡ Hybrid Renewable Energy BioTwin (SynaptikRig-RF)")

# ================= ONBOARDING =================
with st.expander("📘 Understand This App (Simple Explanation)"):
    st.markdown("""
### 🌍 Hybrid Energy System
Solar + Wind + Battery + Biomass + Grid

X-axis → Time  
Y-axis → Power (kW) / Energy (kWh)

Goal → Maximize renewable usage, minimize cost & carbon
""")

# ================= WEATHER =================
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
    return requests.get(url).json()

data = get_weather()

times, temp, wind, cloud = [], [], [], []

for i in range(8):
    entry = data["list"][i]
    times.append(entry["dt_txt"])
    temp.append(entry["main"]["temp"])
    wind.append(entry["wind"]["speed"])
    cloud.append(entry["clouds"]["all"])

df = pd.DataFrame({
    "Time": times,
    "Temp": temp,
    "Wind": wind,
    "Cloud": cloud
})

# ================= USER INPUT =================
st.sidebar.header("⚙️ System Design")

solar_kw = st.sidebar.slider("☀️ Solar Capacity (kW)", 1, 10, 3)
wind_kw = st.sidebar.slider("🌬 Wind Capacity (kW)", 0, 5, 1)
battery_capacity = st.sidebar.slider("🔋 Battery (kWh)", 5, 25, 10)
battery_level = st.sidebar.slider("Initial Battery (%)", 10, 100, 50)
biomass_power = st.sidebar.slider("🔥 Biomass (kW)", 0.5, 3.0, 1.0)

grid_enabled = st.sidebar.checkbox("⚡ Grid Connection", True)
grid_power = st.sidebar.slider("Grid Limit (kW)", 0.5, 5.0, 2.0)

# ================= LOAD SCENARIOS =================
st.sidebar.header("🏘 Village Load Scenario")

scenario = st.sidebar.selectbox("Select Scenario", [
    "Small Village (10 homes)",
    "Medium Village (25 homes)",
    "Large Village (50 homes)",
    "Custom"
])

if scenario == "Small Village (10 homes)":
    load_base = 2
elif scenario == "Medium Village (25 homes)":
    load_base = 4
elif scenario == "Large Village (50 homes)":
    load_base = 7
else:
    load_base = st.sidebar.slider("Custom Load (kW)", 0.5, 10.0, 3.0)

# ================= IMPROVED MODELS =================

# Solar with irradiance + efficiency
def solar_model(cloud, capacity):
    irradiance = 1000 * (1 - cloud/100)  # W/m2
    efficiency = 0.18
    return max(0, (irradiance * efficiency * capacity) / 1000)

# Wind with power curve
def wind_model(speed, capacity):
    if speed < 3:
        return 0
    elif speed < 12:
        return capacity * (speed/12)**3
    else:
        return capacity

# Synaptic smoothing (UNCHANGED)
def synaptic_memory(series, alpha=0.6):
    smoothed = []
    prev = series[0]
    for val in series:
        new_val = alpha * prev + (1 - alpha) * val
        smoothed.append(new_val)
        prev = new_val
    return smoothed

# RF-OK adjustment (UNCHANGED)
def rf_ok_adjustment(solar, wind):
    return solar * 0.9 + wind * 1.1

# ================= APPLY =================
solar = [solar_model(c, solar_kw) for c in df["Cloud"]]
wind_gen = [wind_model(w, wind_kw) for w in df["Wind"]]

solar = synaptic_memory(solar)
wind_gen = synaptic_memory(wind_gen)

rf_output = [rf_ok_adjustment(s, w) for s, w in zip(solar, wind_gen)]

# ================= REALISTIC LOAD PROFILE =================
load_pattern = [0.6, 0.8, 1.2, 1.5, 1.3, 0.9, 0.7, 0.5]

# ================= SIMULATION =================
battery = battery_level/100 * battery_capacity
battery_series, decision_series = [], []
grid_series, biomass_series = [], []
carbon_emissions = []

battery_eff = 0.9

for i in range(len(df)):
    load = load_base * load_pattern[i]
    generation = rf_output[i]

    grid_use = 0
    biomass_use = 0

    if generation >= load:
        battery = min(battery_capacity, battery + (generation - load) * battery_eff)
        decision = "Renewables"

    else:
        deficit = load - generation

        if battery > deficit:
            battery -= deficit / battery_eff
            decision = "Battery"

        else:
            deficit -= battery
            battery = 0

            if grid_enabled and grid_power > deficit:
                grid_use = deficit
                decision = "Grid"

            else:
                biomass_use = deficit
                decision = "Biomass"

    co2 = (grid_use * 0.82) + (biomass_use * 0.45)
    carbon_emissions.append(co2)

    battery_series.append(battery)
    decision_series.append(decision)
    grid_series.append(grid_use)
    biomass_series.append(biomass_use)

# ================= DATA =================
df["Solar"] = solar
df["Wind"] = wind_gen
df["AI_Output"] = rf_output
df["Battery"] = battery_series
df["Grid"] = grid_series
df["Biomass"] = biomass_series
df["CO2"] = carbon_emissions

# ================= ECONOMICS =================
st.subheader("💰 Economic Analysis")

solar_cost = solar_kw * 60000
wind_cost = wind_kw * 120000
battery_cost = battery_capacity * 15000
biomass_cost = biomass_power * 40000

total_capex = solar_cost + wind_cost + battery_cost + biomass_cost

daily_energy = sum(df["AI_Output"])
annual_energy = daily_energy * 365

lcoe = total_capex / (annual_energy * 10)

grid_cost_per_kwh = 8
annual_savings = annual_energy * grid_cost_per_kwh

payback = total_capex / annual_savings

c1, c2, c3 = st.columns(3)
c1.metric("Total System Cost (₹)", int(total_capex))
c2.metric("Cost per kWh (₹)", round(lcoe,2))
c3.metric("Payback Period (years)", round(payback,2))

# ================= NEW EFFICIENCY METRICS =================
st.subheader("⚡ Efficiency Metrics")

renewable_fraction = (sum(df["Solar"]) + sum(df["Wind"])) / (
    sum(df["Solar"]) + sum(df["Wind"]) + sum(df["Grid"]) + sum(df["Biomass"])
)

st.metric("Renewable Fraction (%)", round(renewable_fraction*100,2))

# ================= LIVE =================
st.subheader("🔴 Live Simulation")

for i in range(len(df)):
    st.markdown(f"### ⏱ {df['Time'][i]}")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Solar", round(df["Solar"][i],2))
    c2.metric("Wind", round(df["Wind"][i],2))
    c3.metric("Battery", round(df["Battery"][i],2))
    c4.metric("Grid", round(df["Grid"][i],2))

    time.sleep(0.3)

# ================= GRAPHS =================
st.subheader("📈 Energy Output")
st.line_chart(df.set_index("Time")[["Solar", "Wind", "AI_Output"]])

st.subheader("🔋 Battery")
st.line_chart(df.set_index("Time")["Battery"])

# ================= ENERGY MIX =================
energy_mix = pd.DataFrame({
    "Source": ["Solar", "Wind", "Grid", "Biomass"],
    "Energy": [
        sum(df["Solar"]),
        sum(df["Wind"]),
        sum(df["Grid"]),
        sum(df["Biomass"])
    ]
})

st.subheader("⚡ Energy Mix")
st.bar_chart(energy_mix.set_index("Source"))

# ================= CARBON =================
st.subheader("🌍 Carbon Footprint")

total_co2 = sum(df["CO2"])
st.metric("CO₂ Emissions (kg)", round(total_co2,2))

# ================= DIAGNOSTICS =================
st.subheader("⚙️ Diagnostics")

st.write(f"Total Energy: {round(sum(df['AI_Output']),2)} kWh")
st.write(f"Efficiency: {round((sum(df['AI_Output'])/(load_base*len(df)))*100,2)} %")
