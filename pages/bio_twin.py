import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# ================= CONFIG =================
API_KEY = "c86236be4a9f76875aad940c96e5111b"
CITY = "Lucknow"

st.set_page_config(layout="wide")
st.title(" AI-Powered Hybrid Microgrid Dashboard")

# ================= WEATHER =================
def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
    return requests.get(url).json()

data = get_weather()

if "list" not in data:
    st.error("Weather API failed.")
    st.stop()

# ================= DATA =================
times, wind, cloud = [], [], []

for i in range(8):
    entry = data["list"][i]
    times.append(entry["dt_txt"])
    wind.append(entry["wind"]["speed"])
    cloud.append(entry["clouds"]["all"])

df = pd.DataFrame({
    "Time": times,
    "Wind": wind,
    "Cloud": cloud
})

# ================= USER INPUT =================
st.sidebar.header(" System Design")

num_houses = st.sidebar.slider("Number of Houses", 1, 20, 4)
solar_kw = st.sidebar.slider("Solar per House (kW)", 1, 5, 2)
wind_kw = st.sidebar.slider("Wind per House (kW)", 0, 3, 1)
biomass_power = st.sidebar.slider("Biomass per House (kW)", 0.5, 2.0, 1.0)

battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 10, 50, 20)
battery_level = st.sidebar.slider("Initial Battery (%)", 10, 100, 50)

# ================= MODELS =================
def solar_model(cloud, capacity):
    irradiance = 1000 * (1 - cloud / 100)
    return max(0, (irradiance * 0.18 * capacity) / 1000)

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

# ================= GENERATION =================
solar = synaptic_memory([solar_model(c, solar_kw) for c in df["Cloud"]])
wind_gen = synaptic_memory([wind_model(w, wind_kw) for w in df["Wind"]])

# ================= MULTI HOUSE =================
houses = []

for h in range(num_houses):
    house = {"solar": [], "wind": [], "biomass": [], "total": []}

    for i in range(len(df)):
        s = solar[i] * np.random.uniform(0.9, 1.1)
        w = wind_gen[i] * np.random.uniform(0.8, 1.2)
        b = biomass_power

        total = s + w + b

        house["solar"].append(s)
        house["wind"].append(w)
        house["biomass"].append(b)
        house["total"].append(total)

    houses.append(house)

# ================= AGGREGATION =================
df["Solar"] = np.sum([h["solar"] for h in houses], axis=0)
df["Wind"] = np.sum([h["wind"] for h in houses], axis=0)
df["Biomass"] = np.sum([h["biomass"] for h in houses], axis=0)
df["Total_Generation"] = df["Solar"] + df["Wind"] + df["Biomass"]

# ================= LOAD =================
load_base = 1.5 * num_houses
load_pattern = [0.6, 0.8, 1.2, 1.5, 1.3, 0.9, 0.7, 0.5]

# ================= BATTERY =================
battery = battery_level / 100 * battery_capacity
battery_series = []
decision_series = []

battery_min = 0.3 * battery_capacity

# ================= PREDICTION =================
forecast = data["list"]
next_cloud = np.mean([x["clouds"]["all"] for x in forecast])
next_wind = np.mean([x["wind"]["speed"] for x in forecast])

predicted_solar = solar_model(next_cloud, solar_kw)
predicted_wind = wind_model(next_wind, wind_kw)

# ================= SIMULATION =================
for i in range(len(df)):
    load = load_base * load_pattern[i]
    generation = df["Total_Generation"][i]

    decision = "Normal"

    if generation >= load:
        surplus = generation - load
        charge = min(surplus, battery_capacity - battery)
        battery += charge
        decision = "Charging Battery"

    else:
        deficit = load - generation

        if battery > battery_min:
            supply = min(deficit, battery - battery_min)
            battery -= supply
            deficit -= supply
            decision = "Battery Supply"

        if deficit > 0:
            decision = "Biomass Backup"

    battery_series.append(battery)
    decision_series.append(decision)

# ================= DASHBOARD =================
st.markdown("##  Live Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric(" Houses", num_houses)
c2.metric(" Total Energy (kWh)", round(df["Total_Generation"].sum(), 2))
c3.metric(" Battery (kWh)", round(battery, 2))
c4.metric(" Renewable %",
          round((df["Solar"].sum() + df["Wind"].sum()) /
                df["Total_Generation"].sum() * 100, 2))

# ================= STATUS =================
st.markdown("##  System Status")

s1, s2, s3 = st.columns(3)

if battery > 0.7 * battery_capacity:
    s1.success(" Battery Healthy")
elif battery > 0.3 * battery_capacity:
    s1.warning(" Battery Moderate")
else:
    s1.error(" Battery Low")

total_predicted = (predicted_solar + predicted_wind) * num_houses

if total_predicted < load_base:
    s2.error(" Outage Risk Tomorrow")
else:
    s2.success(" Stable Supply")

s3.info(decision_series[-1])

# ================= FORECAST =================
st.markdown("##  Forecast")

f1, f2, f3 = st.columns(3)
f1.metric(" Solar", round(predicted_solar * num_houses, 2))
f2.metric(" Wind", round(predicted_wind * num_houses, 2))
f3.metric(" Load", round(load_base, 2))

# ================= GRAPH (CLEAR) =================
st.markdown("##  Energy Generation (kW vs Time)")

fig, ax = plt.subplots()

ax.plot(df["Time"], df["Solar"], label="Solar")
ax.plot(df["Time"], df["Wind"], label="Wind")
ax.plot(df["Time"], df["Biomass"], label="Biomass")

ax.set_xlabel("Time")
ax.set_ylabel("Power (kW)")
ax.set_title("Energy Generation by Source")

ax.legend()
plt.xticks(rotation=45)

st.pyplot(fig)

# ================= BATTERY GRAPH =================
st.markdown("##  Battery Level (kWh vs Time)")

fig2, ax2 = plt.subplots()

ax2.plot(df["Time"], battery_series)

ax2.set_xlabel("Time")
ax2.set_ylabel("Battery Level (kWh)")
ax2.set_title("Battery Storage Dynamics")

plt.xticks(rotation=45)

st.pyplot(fig2)

# ================= HOUSE =================
st.markdown("##  Per House Contribution")

for i, house in enumerate(houses):
    total = sum(house["total"])

    c1, c2, c3 = st.columns(3)
    c1.metric(f"House {i+1} Solar %",
              round(sum(house["solar"]) / total * 100, 2))
    c2.metric("Wind %",
              round(sum(house["wind"]) / total * 100, 2))
    c3.metric("Biomass %",
              round(sum(house["biomass"]) / total * 100, 2))

# ================= AI DECISION =================
st.markdown("##  AI Decision Engine")

if total_predicted >= load_base:
    st.success(" Renewable Energy Sufficient")
elif battery > 0.5 * battery_capacity:
    st.info(" Use Battery Backup")
elif biomass_power > 0:
    st.warning(" Activate Biomass Backup")
else:
    st.error(" Power Shortage Risk")

# ================= LIVE =================
st.markdown("##  Live Simulation")

progress = st.progress(0)

for i in range(len(df)):
    progress.progress((i+1)/len(df))
    time.sleep(0.2)
# ================= GIS MICROGRID MAP =================

import folium
from streamlit_folium import st_folium

st.markdown("## Campus Microgrid GIS Overlay (Satellite + Building Footprints)")

# ---- REAL CAMPUS CENTER ----
campus_center = [26.8520033, 81.0504370]

# ---- BUILDING LOCATIONS ----
locations = {
    "AB1": [26.852400, 81.050800],
    "AB2": [26.852600, 81.051100],
    "AB3": [26.852800, 81.051400],
    "AB4": [26.853000, 81.051700],
    "AB5": [26.853200, 81.052000],
    "AB6": [26.853400, 81.052300],
    "Library": [26.852700, 81.051000],
    "STP": [26.851400, 81.049500]
}

# ---- CREATE MAP ----
m = folium.Map(
    location=campus_center,
    zoom_start=17,
    tiles=None
)

# ---- SATELLITE TILE ----
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellite",
    overlay=False,
    control=True
).add_to(m)

# ---- CAMPUS BOUNDARY ----
campus_boundary = {
    "type": "Feature",
    "properties": {"name": "Amity Campus"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [81.0490, 26.8510],
            [81.0530, 26.8510],
            [81.0535, 26.8540],
            [81.0495, 26.8542],
            [81.0490, 26.8510]
        ]]
    }
}

folium.GeoJson(
    campus_boundary,
    name="Campus Boundary",
    style_function=lambda x: {
        "fillColor": "none",
        "color": "yellow",
        "weight": 2
    }
).add_to(m)

# ---- BUILDING POLYGON FUNCTION ----
def create_building_polygon(lat, lon, size=0.00015):
    return [
        [lon - size, lat - size],
        [lon + size, lat - size],
        [lon + size, lat + size],
        [lon - size, lat + size],
        [lon - size, lat - size]
    ]

# ---- ADD BUILDINGS ----
for name, coords in locations.items():
    lat = coords[0]
    lon = coords[1]

    polygon = {
        "type": "Feature",
        "properties": {"name": name},
        "geometry": {
            "type": "Polygon",
            "coordinates": [create_building_polygon(lat, lon)]
        }
    }

    folium.GeoJson(
        polygon,
        tooltip=name,
        style_function=lambda x: {
            "fillColor": "green" if x["properties"]["name"] == "STP" else "blue",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.6
        }
    ).add_to(m)

# ---- ADD MARKERS ----
for name, coord in locations.items():
    folium.Marker(
        location=coord,
        popup=name,
        icon=folium.Icon(color="green" if name == "STP" else "blue")
    ).add_to(m)

# ---- ENERGY FLOW CALCULATION ----
flows = []

for i in range(len(houses)):
    house = houses[i]
    generation = sum(house["total"])
    load = load_base

    if i < 6:
        node = f"AB{i+1}"
    else:
        node = "Library"

    if generation > load:
        flows.append((node, "STP", generation - load, "surplus"))
    else:
        flows.append(("STP", node, load - generation, "deficit"))

# ---- DRAW FLOWS ----
for flow in flows:
    src = flow[0]
    dst = flow[1]
    value = flow[2]
    ftype = flow[3]

    if src in locations and dst in locations:
        color = "lime" if ftype == "surplus" else "red"

        folium.PolyLine(
            locations=[locations[src], locations[dst]],
            color=color,
            weight=min(8, 2 + value),
            tooltip=f"{src} → {dst}: {round(value,2)} kW"
        ).add_to(m)

# ---- LAYER CONTROL ----
folium.LayerControl().add_to(m)

# ---- DISPLAY ----
st_folium(m, width=1000, height=650)
