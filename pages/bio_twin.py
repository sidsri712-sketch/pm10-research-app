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

def synaptic_memory(series, alpha=0.6):
    smoothed = []
    prev = series[0]
    for val in series:
        new_val = alpha * prev + (1 - alpha) * val
        smoothed.append(new_val)
        prev = new_val
    return smoothed

def hole_variogram(series):
    return [val * (1 + 0.2*np.sin(i)) for i, val in enumerate(series)]

def rf_ok_adjustment(solar, wind):
    return solar * 0.9 + wind * 1.1

# ================= APPLY MODELS =================

solar = [solar_model(c) for c in df["Cloud"]]
wind_gen = [wind_model(w) for w in df["Wind"]]

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
col1.metric(" Avg Temp (°C)", round(np.mean(df["Temp"]),2))
col2.metric(" Avg Wind (m/s)", round(np.mean(df["Wind"]),2))
col3.metric(" Avg Cloud (%)", round(np.mean(df["Cloud"]),2))

st.divider()

# ================= AXIS EXPLANATION =================

st.markdown("### 📈 Graph Interpretation Guide")
st.info("""
Energy Output Graph:
- X-axis → Time (3-hour interval forecast)
- Y-axis → Power Output (kW)

Battery Graph:
- X-axis → Time
- Y-axis → Stored Energy (kWh)

Energy Distribution:
- X-axis → Energy Source
- Y-axis → Total Contribution (kWh)
""")

# ================= ENERGY OUTPUT =================

st.subheader(" SynaptikRig-RF Energy Output")
st.line_chart(df.set_index("Time")[["Solar", "WindGen", "RF_Output"]])
st.caption("X-axis: Time | Y-axis: Power Output (kW)")

# ================= BATTERY =================

st.subheader(" Battery Dynamics")
st.line_chart(df.set_index("Time")["Battery"])
st.caption("X-axis: Time | Y-axis: Battery Energy (kWh)")

st.markdown("### 🔋 Battery Metrics")
st.write(f"Max Battery: {round(max(df['Battery']),2)} kWh")
st.write(f"Min Battery: {round(min(df['Battery']),2)} kWh")
st.write(f"Final Battery: {round(df['Battery'].iloc[-1],2)} kWh")

# ================= DECISIONS =================

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

# ================= DIAGNOSTICS =================

st.subheader("⚙️ System Diagnostics")

total_gen = sum(df["RF_Output"])
total_load = load_base * len(df)
deficit_events = sum(df["Decision"] == "Biomass Backup")

st.write(f"Total Generation: {round(total_gen,2)} kWh")
st.write(f"Total Load: {round(total_load,2)} kWh")
st.write(f"Biomass Events: {deficit_events}")
st.write(f"Efficiency: {round((total_gen/total_load)*100,2)} %")

# ================= EXPLAINABILITY =================

st.subheader("🧠 Model Explainability")

st.write("""
SynaptikRig-RF Architecture:

• Solar Model → Cloud-based irradiance estimation  
• Wind Model → Cubic power scaling  
• Synaptic Memory → Temporal smoothing  
• Hole Variogram → Pattern modulation  
• RF-OK → Hybrid fusion model  
""")

# ================= WORKING DIAGRAM =================

st.subheader("🔁 Working Model Diagram")

st.markdown("""
```mermaid
flowchart TD
A[Weather API] --> B[Solar Model]
A --> C[Wind Model]
B --> D[Synaptic Memory]
C --> D
D --> E[Variogram Layer]
E --> F[RF-OK Fusion]
F --> G[Energy Output]
G --> H{Load Balance}
H -->|Excess| I[Battery Charge]
H -->|Deficit| J{Battery?}
J -->|Yes| K[Discharge]
J -->|No| L[Biomass]
