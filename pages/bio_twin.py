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

# ================= SIMULATION (FIXED ONLY HERE) =================
battery = battery_level / 100 * battery_capacity

battery_series = []
decision_series = []
grid_series = []
biomass_series = []
carbon_emissions = []
biogas_diverted = []

battery_eff = 0.9

for i in range(len(df)):
    load = load_base * load_pattern[i]
    generation = rf_output[i]

    grid_use = 0
    biomass_use = 0
    biogas_extra = 0

    if generation >= load:
        surplus = generation - load

        available_capacity = battery_capacity - battery

        charge_input = min(surplus, available_capacity / battery_eff)
        charge_stored = charge_input * battery_eff
        battery += charge_stored

        export = surplus - charge_input

        if grid_enabled and export > 0:
            export = min(export, grid_power)
            grid_use = -export
            decision = "Grid Export"
        else:
            decision = "Renewables"

        biomass_use = 0
        biogas_extra = biomass_power

    else:
        deficit = load - generation

        if battery > deficit:
            battery -= deficit / battery_eff
            decision = "Battery"
            biogas_extra = biomass_power

        else:
            deficit -= battery
            battery = 0

            if grid_enabled and grid_power > deficit:
                grid_use = deficit
                decision = "Grid"
                biogas_extra = biomass_power

            else:
                biomass_use = deficit
                biogas_extra = max(0, biomass_power - biomass_use)
                decision = "Biomass"

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
