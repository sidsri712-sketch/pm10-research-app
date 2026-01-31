import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from streamlit_autorefresh import st_autorefresh
from streamlit_gsheets import GSheetsConnection # Requirement for Cloud Data
import matplotlib.patches as mpatches
import datetime
import time
import io
import os

# --------------------------------------------------
# CONFIGURATION & DATABASE (GOOGLE SHEETS)
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(
    page_title="Lucknow PM10 Hybrid Spatial Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Connect to Google Sheets instead of Local CSV
conn = st.connection("gsheets", type=GSheetsConnection)

# Auto-refresh every 30 minutes
st_autorefresh(interval=1800000, key="refresh")

# --------------------------------------------------
# WEATHER FETCH
# --------------------------------------------------
def fetch_weather():
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            "latitude=26.85&longitude=80.94&current="
            "temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        r = requests.get(url).json()
        return {
            "temp": r["current"]["temperature_2m"],
            "hum": r["current"]["relative_humidity_2m"],
            "wind": r["current"]["wind_speed_10m"]
        }
    except:
        return {"temp": 25.0, "hum": 50.0, "wind": 5.0}

# --------------------------------------------------
# DATA PIPELINE (UPDATED FOR GOOGLE SHEETS)
# --------------------------------------------------
@st.cache_data(ttl=900)
def fetch_pm10_data():
    weather = fetch_weather()
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    records = []

    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                d = requests.get(
                    f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
                ).json()
                if d.get("status") == "ok" and "pm10" in d["data"].get("iaqi", {}):
                    records.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": d["data"]["iaqi"]["pm10"]["v"],
                        "name": d["data"]["city"]["name"],
                        "temp": weather["temp"],
                        "hum": weather["hum"],
                        "wind": weather["wind"],
                        "timestamp": str(pd.Timestamp.now())
                    })
                time.sleep(0.1)

        df_live = pd.DataFrame(records)

        if not df_live.empty:
            # Load current cloud data
            try:
                df_hist = conn.read()
                df_all = pd.concat([df_hist, df_live], ignore_index=True)
                df_all.drop_duplicates(subset=["lat", "lon", "pm10"], inplace=True)
                # Update cloud sheet
                conn.update(data=df_all)
            except:
                conn.update(data=df_live)

            return df_live.groupby(["lat", "lon"]).agg({
                "pm10": "mean", "name": "first", "temp": "first", "hum": "first", "wind": "first"
            }).reset_index()

        return pd.DataFrame()
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# ALL REMAINING UI AND MODEL LOGIC (PRESERVED)
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")
weather_now = fetch_weather()
c1, c2, c3 = st.columns(3)
c1.metric("🌡 Temperature", f"{weather_now['temp']} °C")
c2.metric("💧 Humidity", f"{weather_now['hum']} %")
c3.metric("💨 Wind Speed", f"{weather_now['wind']} km/h")

# Load for training
try:
    df_history = conn.read()
except:
    df_history = pd.DataFrame()

st.sidebar.header("🛠 Controls")
if not df_history.empty:
    st.sidebar.metric("Historical Samples", len(df_history))

opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.70)
weather_mult = st.sidebar.slider("Weather Amplification (%)", 50, 200, 100) / 100

run_hybrid = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Diagnostics")

st.sidebar.subheader("🎯 Custom Prediction")
custom_lat = st.sidebar.number_input("Latitude", value=26.85, step=0.01)
custom_lon = st.sidebar.number_input("Longitude", value=80.95, step=0.01)
predict_custom = st.sidebar.button("Predict PM10")

st.sidebar.subheader("📅 Historical Filter")
start_date = st.sidebar.date_input("Start", datetime.date.today() - datetime.timedelta(days=7))
end_date = st.sidebar.date_input("End", datetime.date.today())

if run_hybrid or run_diag or predict_custom:
    df_live = fetch_pm10_data()
    if df_live.empty or len(df_live) < 3:
        st.warning("Not enough monitoring stations available.")
        st.stop()

    df_train = df_history.copy()
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
    df_train["hour"] = df_train["timestamp"].dt.hour
    df_train["dayofweek"] = df_train["timestamp"].dt.dayofweek
    df_train["month"] = df_train["timestamp"].dt.month

    features = ["lat", "lon", "hour", "dayofweek", "month", "temp", "hum", "wind"]
    rf = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
    rf.fit(df_train[features], df_train["pm10"])

    now = pd.Timestamp.now()
    df_live["hour"] = now.hour
    df_live["dayofweek"] = now.dayofweek
    df_live["month"] = now.month
    df_live["residuals"] = (df_live["pm10"] - rf.predict(df_live[features])) * weather_mult

    grid_res = 250
    lats = np.linspace(df_live.lat.min()-0.08, df_live.lat.max()+0.08, grid_res)
    lons = np.linspace(df_live.lon.min()-0.08, df_live.lon.max()+0.08, grid_res)

    try:
        OK = OrdinaryKriging(df_live.lon, df_live.lat, df_live["residuals"], variogram_model="gaussian")
        z_res, _ = OK.execute("grid", lons, lats)
    except:
        z_res = np.zeros((grid_res, grid_res))

    lon_g, lat_g = np.meshgrid(lons, lats)
    rf_trend = rf.predict(np.column_stack([
        lat_g.ravel(), lon_g.ravel(),
        np.full(lat_g.size, now.hour),
        np.full(lat_g.size, now.dayofweek),
        np.full(lat_g.size, now.month),
        np.full(lat_g.size, weather_now["temp"]),
        np.full(lat_g.size, weather_now["hum"]),
        np.full(lat_g.size, weather_now["wind"])
    ])).reshape(grid_res, grid_res)

    z_final = gaussian_filter(rf_trend * weather_mult + z_res.T, sigma=2.5)
    z_final[z_final < 0] = 0

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)

    im = ax.imshow(z_final, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="magma", alpha=opacity, interpolation="bilinear", aspect='equal', zorder=2)
    ax.contour(z_final, levels=8, extent=[xmin, xmax, ymin, ymax], colors='white', alpha=0.15, linewidths=0.5, zorder=3)
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)
    ax.scatter(xs, ys, c="white", edgecolors="black", s=80, zorder=4, label="Stations")
    plt.colorbar(im, ax=ax, label="PM10 (µg/m³)")
    ax.legend(loc='upper right')
    ax.set_axis_off()
    st.pyplot(fig)

    if predict_custom:
        c_rf = rf.predict([[custom_lat, custom_lon, now.hour, now.dayofweek, now.month, weather_now["temp"], weather_now["hum"], weather_now["wind"]]])[0]
        try:
            c_res, _ = OK.execute("points", [custom_lon], [custom_lat])
            c_val = max(0, c_rf + c_res[0])
        except:
            c_val = c_rf
        st.metric("🎯 Predicted PM10", f"{c_val:.2f} µg/m³")

st.caption("Data: WAQI API, Open-Meteo & Google Sheets | Model: RFRK (Hybrid)")
