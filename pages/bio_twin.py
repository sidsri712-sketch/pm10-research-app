import streamlit as st
import requests
import pandas as pd
import numpy as np
import time

# ================= CONFIG =================
API_KEY = "c86236be4a9f76875aad940c96e5111b"
CITY = "Lucknow"

st.set_page_config(layout="wide")
st.title("Hybrid Renewable Energy BioTwin (SynaptikRig-RF)")

# ================= ONBOARDING =================
with st.expander("Understand This App"):
    st.markdown("""
Hybrid Energy System:
Solar + Wind + Battery + Biomass + Grid

X-axis = Time  
Y-axis = Power (kW) or Energy (kWh)

Goal: Maximize renewable usage and minimize cost and carbon
""")

# ================= WEATHER =================
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
    return requests.get(url).json()

data = get_weather()

if "list" not in data:
    st.error("Weather API failed. Check API key or internet.")
    st.stop()

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
st.sidebar.header("System Design")

solar_kw = st.sidebar.slider("Solar Capacity (kW)", 1, 10, 3)
wind_kw = st.sidebar.slider("Wind Capacity (kW)", 0, 5, 1)
battery_capacity = st.sidebar.slider("Battery (kWh)", 5, 25, 10)
battery_level = st.sidebar.slider("Initial Battery (%)", 10, 100, 50)
biomass_power = st.sidebar.slider("Biomass (kW)", 0.5, 3.0, 1.0)

grid_enabled = st.sidebar.checkbox("Grid Connection", True)
grid_power = st.sidebar.slider("Grid Limit (kW)", 0.5, 5.0, 2.0)

# ================= LOAD SCENARIOS =================
st.sidebar.header("Village Load Scenario")

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

# ================= MODELS =================
def solar_model(cloud, capacity):
    irradiance = 1000 * (1 - cloud / 100)
    efficiency = 0.18
    return max(0, (irradiance * efficiency * capacity) / 1000)

def wind_model(speed, capacity):
    if speed < 3:
        return 0
    elif speed < 12:
        return capacity * (speed / 12) ** 3
    else:
        return capacity

def synaptic_memory(series, alpha=0.6):
    smoothed = []
    prev = series[0]
    for val in series:
        new_val = alpha * prev + (1 - alpha) * val
        smoothed.append(new_val)
        prev = new_val
    return smoothed

def rf_ok_adjustment(solar, wind):
    return solar * 0.9 + wind * 1.1

# ================= APPLY =================
solar = [solar_model(c, solar_kw) for c in df["Cloud"]]
wind_gen = [wind_model(w, wind_kw) for w in df["Wind"]]

solar = synaptic_memory(solar)
wind_gen = synaptic_memory(wind_gen)

rf_output = [rf_ok_adjustment(s, w) for s, w in zip(solar, wind_gen)]

# ================= LOAD PROFILE =================
load_pattern = [0.6, 0.8, 1.2, 1.5, 1.3, 0.9, 0.7, 0.5]

# ================= SIMULATION =================
battery = battery_level / 100 * battery_capacity

battery_series = []
decision_series = []
grid_series = []
biomass_series = []
carbon_emissions = []
biogas_diverted = []

battery_eff = 0.9
battery_min = 0.2 * battery_capacity   # 🔒 20% reserve

for i in range(len(df)):
    load = load_base * load_pattern[i]
    generation = rf_output[i]

    grid_use = 0
    biomass_use = 0
    biogas_extra = 0

    # ===== SURPLUS =====
    if generation >= load:
        surplus = generation - load

        available_capacity = battery_capacity - battery
        charge_input = min(surplus, available_capacity / battery_eff)
        battery += charge_input * battery_eff

        export = surplus - charge_input

        if grid_enabled and export > 0:
            grid_use = -min(export, grid_power)
            decision = "Grid Export"
        else:
            decision = "Renewables"

        biogas_extra = biomass_power

    # ===== DEFICIT =====
    else:
        deficit = load - generation

        # 1️⃣ BIOMASS FIRST
        biomass_supply = min(deficit, biomass_power)
        biomass_use = biomass_supply
        deficit -= biomass_supply

        biogas_extra = max(0, biomass_power - biomass_use)

        # 2️⃣ BATTERY (WITH RESERVE)
        if deficit > 0:
            usable_battery = max(0, battery - battery_min)
            possible_supply = usable_battery * battery_eff

            if possible_supply >= deficit:
                battery -= deficit / battery_eff
                deficit = 0
                decision = "Biomass + Battery"
            else:
                battery -= usable_battery
                deficit -= possible_supply
                decision = "Biomass + Battery"

        # 3️⃣ GRID LAST
        if deficit > 0 and grid_enabled:
            grid_use = min(deficit, grid_power)
            deficit -= grid_use
            decision = "Grid Support"

    # ================= CARBON =================
    if grid_use >= 0:
        co2 = (grid_use * 0.82) + (biomass_use * 0.45)
    else:
        co2 = (biomass_use * 0.45)

    battery_series.append(battery)
    decision_series.append(decision)
    grid_series.append(grid_use)
    biomass_series.append(biomass_use)
    carbon_emissions.append(co2)
    biogas_diverted.append(biogas_extra)

# ================= DATA =================
df["Solar"] = solar
df["Wind"] = wind_gen
df["AI_Output"] = rf_output
df["Battery"] = battery_series
df["Grid"] = grid_series
df["Biomass"] = biomass_series
df["CO2"] = carbon_emissions
df["Biogas_Diverted"] = biogas_diverted

# ================= ECONOMICS =================
st.subheader("Economic Analysis")

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

feed_in_tariff = 4
exported_energy = sum([abs(x) for x in df["Grid"] if x < 0])
export_revenue = exported_energy * feed_in_tariff

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total System Cost (Rs)", int(total_capex))
c2.metric("Cost per kWh (Rs)", round(lcoe, 2))
c3.metric("Payback Period (years)", round(payback, 2))
c4.metric("Grid Export Revenue (Rs)", round(export_revenue, 2))

# ================= EFFICIENCY =================
st.subheader("Efficiency Metrics")

total_energy = sum(df["Solar"]) + sum(df["Wind"]) + sum(df["Grid"]) + sum(df["Biomass"])

renewable_fraction = (sum(df["Solar"]) + sum(df["Wind"])) / total_energy if total_energy > 0 else 0
st.metric("Renewable Fraction (%)", round(renewable_fraction * 100, 2))

# ================= LIVE =================
st.subheader("Live Simulation")

for i in range(len(df)):
    st.write("Time:", df["Time"][i])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Solar", round(df["Solar"][i], 2))
    c2.metric("Wind", round(df["Wind"][i], 2))
    c3.metric("Battery", round(df["Battery"][i], 2))
    c4.metric("Grid (+import / -export)", round(df["Grid"][i], 2))

    time.sleep(0.2)

# ================= GRAPHS =================
st.subheader("Energy Output")
st.line_chart(df.set_index("Time")[["Solar", "Wind", "AI_Output"]])

st.subheader("Battery")
st.line_chart(df.set_index("Time")["Battery"])

# ================= ENERGY MIX =================
energy_mix = pd.DataFrame({
    "Source": ["Solar", "Wind", "Grid Import", "Biomass"],
    "Energy": [
        sum(df["Solar"]),
        sum(df["Wind"]),
        sum([x for x in df["Grid"] if x > 0]),
        sum(df["Biomass"])
    ]
})

st.subheader("Energy Mix")
st.bar_chart(energy_mix.set_index("Source"))

# ================= CARBON =================
st.subheader("Carbon Footprint")

total_co2 = sum(df["CO2"])
st.metric("CO2 Emissions (kg)", round(total_co2, 2))

# ================= DIAGNOSTICS =================
st.subheader("Diagnostics")

st.write("Total Energy:", round(sum(df["AI_Output"]), 2), "kWh")
st.write("Efficiency:", round((sum(df["AI_Output"]) / (load_base * len(df))) * 100, 2), "%")
