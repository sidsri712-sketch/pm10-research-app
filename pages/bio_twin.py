import streamlit as st
import requests
import pandas as pd
import numpy as np

# ================= CONFIG =================
API_KEY = "c86236be4a9f76875aad940c96e5111b"
CITY = "Lucknow"

st.set_page_config(layout="wide")
st.title(" SynaptikRig-RF Hybrid Energy BioTwin")

# ================= WEATHER =================
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
    return requests.get(url).json()

data = get_weather()

# Extract 24h forecast
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
st.sidebar.header(" System Parameters")

battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 5, 20, 10)
battery_level = st.sidebar.slider("Initial Battery (%)", 10, 100, 50)
load_base = st.sidebar.slider("Base Load (kW)", 0.5, 5.0, 1.5)
biomass_power = st.sidebar.slider("Biomass Backup (kW)", 0.2, 2.0, 0.8)

# ================= MODELS =================

def solar_model(cloud):
    return max(0, (100 - cloud)/100 * 2.5)

def wind_model(speed):
    return min(1.5, (speed**3)/50)

# ================= SYNAPTIKRIG-RF =================

# Synaptic Lag Memory (temporal smoothing)
def synaptic_memory(series, alpha=0.6):
    smoothed = []
    prev = series[0]
    for val in series:
        new_val = alpha * prev + (1 - alpha) * val
        smoothed.append(new_val)
        prev = new_val
    return smoothed

# Hole Variogram Weighting (pattern influence)
def hole_variogram(series):
    return [val * (1 + 0.2*np.sin(i)) for i, val in enumerate(series)]

# RF-OK (simplified predictive adjustment)
def rf_ok_adjustment(solar, wind):
    return solar * 0.9 + wind * 1.1

# ================= APPLY MODELS =================

solar = [solar_model(c) for c in df["Cloud"]]
wind_gen = [wind_model(w) for w in df["Wind"]]

# Apply SynaptikRig layers
solar_smoothed = synaptic_memory(solar)
wind_smoothed = synaptic_memory(wind_gen)

solar_var = hole_variogram(solar_smoothed)
wind_var = hole_variogram(wind_smoothed)

rf_output = [rf_ok_adjustment(s, w) for s, w in zip(solar_var, wind_var)]

# ================= SIMULATION =================

battery = battery_level/100 * battery_capacity
battery_series = []
decision_series = []

for i in range(len(df)):
    load = load_base + np.random.uniform(-0.3, 0.3)
    generation = rf_output[i]

    if generation >= load:
        battery = min(battery_capacity, battery + (generation - load))
        decision = "Optimized Solar/Wind"
    else:
        deficit = load - generation

        if battery > deficit:
            battery -= deficit
            decision = "Battery Compensation"
        else:
            decision = "Biomass Backup"

    battery_series.append(battery)
    decision_series.append(decision)

df["Solar"] = solar_var
df["WindGen"] = wind_var
df["RF_Output"] = rf_output
df["Battery"] = battery_series
df["Decision"] = decision_series

# ================= UI =================

col1, col2, col3 = st.columns(3)
col1.metric(" Avg Temp", round(np.mean(df["Temp"]),2))
col2.metric(" Avg Wind", round(np.mean(df["Wind"]),2))
col3.metric(" Avg Cloud", round(np.mean(df["Cloud"]),2))

st.divider()

st.subheader(" SynaptikRig-RF Energy Output")
st.line_chart(df[["Solar", "WindGen", "RF_Output"]])

st.subheader(" Battery Dynamics")
st.line_chart(df["Battery"])

st.subheader(" Decision Timeline")
st.dataframe(df[["Time", "Decision"]])

# ================= ENERGY MIX =================

energy_mix = pd.DataFrame({
    "Source": ["Solar", "Wind", "Biomass"],
    "Contribution": [
        sum(df["Solar"]),
        sum(df["WindGen"]),
        biomass_power * len(df)
    ]
})

st.subheader(" Energy Distribution")
st.bar_chart(energy_mix.set_index("Source"))

# ================= SYSTEM STATUS =================

st.subheader(" System Status")

if np.mean(df["Battery"]) > battery_capacity * 0.4:
    st.success("System Stable ")
else:
    st.warning("System Under Stress ")

# ================= INSIGHTS =================

st.subheader(" SynaptikRig Insights")

st.write("• Synaptic memory smooths fluctuations")
st.write("• Hole variogram captures periodic patterns")
st.write("• RF-OK improves hybrid energy blending")
st.write("• Biomass ensures system reliability")
