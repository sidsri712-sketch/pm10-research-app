import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from pyproj import Transformer
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from streamlit_autorefresh import st_autorefresh
import matplotlib.patches as mpatches
import datetime
import time
import io
import os

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"
DB_FILE = "lucknow_pm10_history.csv"

# THINGSPEAK CONFIG (New Integration)
TS_CHANNEL_ID = "3245947"
TS_READ_KEY = "KMFZAL0I4BI752II"

st.set_page_config(
    page_title="Lucknow PM10 Hybrid Spatial Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 30 minutes
st_autorefresh(interval=1800000, key="refresh")

# --------------------------------------------------
# THINGSPEAK FETCH (New Feature)
# --------------------------------------------------
def fetch_thingspeak_data():
    try:
        url = f"https://api.thingspeak.com/channels/{TS_CHANNEL_ID}/feeds.json?api_key={TS_READ_KEY}&results=1"
        r = requests.get(url, timeout=5).json()
        if "feeds" in r and len(r["feeds"]) > 0:
            return r["feeds"][-1]
    except:
        return None
    return None

# --------------------------------------------------
# WEATHER FETCH (OPEN-METEO)
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
# DATA PIPELINE
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
                        "timestamp": pd.Timestamp.now()
                    })
                time.sleep(0.1)

        df_live = pd.DataFrame(records)

        if not df_live.empty:
            if os.path.exists(DB_FILE):
                df_hist = pd.read_csv(DB_FILE)
                df_all = pd.concat([df_hist, df_live], ignore_index=True)
                df_all.drop_duplicates(
                    subset=["lat", "lon", "pm10", "timestamp"],
                    inplace=True
                )
                df_all.to_csv(DB_FILE, index=False)
            else:
                df_live.to_csv(DB_FILE, index=False)

            return df_live.groupby(["lat", "lon"]).agg({
                "pm10": "mean",
                "name": "first",
                "temp": "first",
                "hum": "first",
                "wind": "first"
            }).reset_index()

        return pd.DataFrame()

    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# LOOCV DIAGNOSTICS
# --------------------------------------------------
def run_diagnostics(df):
    preds = []
    features = ["lat", "lon", "temp", "hum", "wind"]

    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]

        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(train[features], train["pm10"])
        residuals = train["pm10"] - rf.predict(train[features])

        try:
            ok = OrdinaryKriging(train.lon, train.lat, residuals, variogram_model="gaussian")
            r, _ = ok.execute("points", [test.lon], [test.lat])
            pred = rf.predict([test[features]])[0] + r[0]
        except:
            pred = rf.predict([test[features]])[0]

        preds.append({"Actual": test.pm10, "Predicted": pred})

    res = pd.DataFrame(preds)
    mae = np.mean(np.abs(res["Actual"] - res["Predicted"]))
    return res, mae

# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")

weather_now = fetch_weather()
c1, c2, c3 = st.columns(3)
c1.metric("🌡 Temperature", f"{weather_now['temp']} °C")
c2.metric("💧 Humidity", f"{weather_now['hum']} %")
c3.metric("💨 Wind Speed", f"{weather_now['wind']} km/h")

st.sidebar.caption(f"Last refresh: {time.strftime('%H:%M:%S')}")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

if os.path.exists(DB_FILE):
    df_history = pd.read_csv(DB_FILE)
else:
    df_history = pd.DataFrame()
if not df_history.empty:
    st.sidebar.metric("Historical Samples", len(df_history))
st.sidebar.header("🛠 Controls")

st.sidebar.subheader("📡 Live Sensor Link")
ts_data = fetch_thingspeak_data()
if ts_data:
    st.sidebar.success(f"Live Node PM10: {ts_data.get('field1', 'N/A')} µg/m³")
    st.sidebar.caption(f"Last Sensor Sync: {ts_data.get('created_at')}")
else:
    st.sidebar.info("Waiting for ThingSpeak Sensor...")

opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.75)
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

# >>> ADDED SOURCE ATTRIBUTION <<<
show_sources = st.sidebar.checkbox("🧭 Show PM10 Source Influence")

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_hybrid or run_diag or predict_custom:

    df_live = fetch_pm10_data()
    if df_live.empty or len(df_live) < 3:
        st.warning("Not enough monitoring stations.")
        st.stop()

    df_train = pd.read_csv(DB_FILE)
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
    df_train["hour"] = df_train["timestamp"].dt.hour
    df_train["dayofweek"] = df_train["timestamp"].dt.dayofweek
    df_train["month"] = df_train["timestamp"].dt.month

    features = ["lat", "lon", "hour", "dayofweek", "month", "temp", "hum", "wind"]

    rf = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
    rf.fit(df_train[features], df_train["pm10"])

    # >>> ADDED SOURCE ATTRIBUTION <<<
    def estimate_source_influence(row):
        traffic = 0
        dust = 0
        biomass = 0
        background = 0

        if row["hour"] in [7,8,9,18,19,20,21] and row["dayofweek"] < 5:
            traffic += 0.4

        if row["temp"] > 30 and row["wind"] > 3:
            dust += 0.35

        if row["hour"] >= 20 or row["hour"] <= 6:
            if row["hum"] > 60 and row["month"] in [10,11,12,1]:
                biomass += 0.35

        background = 1.0 - min(traffic + dust + biomass, 1.0)

        return pd.Series({
            "Traffic": max(traffic, 0),
            "Dust": max(dust, 0),
            "Biomass": max(biomass, 0),
            "Background": max(background, 0)
        })

    now = pd.Timestamp.now()
    df_live["hour"] = now.hour
    df_live["dayofweek"] = now.dayofweek
    df_live["month"] = now.month

    # >>> ADDED SOURCE ATTRIBUTION <<<
    df_sources = df_live.apply(estimate_source_influence, axis=1)
    df_live = pd.concat([df_live, df_sources], axis=1)
    df_live["Dominant_Source"] = df_sources.idxmax(axis=1)

    # --------------------------------------------------
    # SOURCE SUMMARY
    # --------------------------------------------------
    if show_sources:
        st.subheader("🧭 PM10 Source Influence (Proxy-Based)")
        source_mean = df_live[["Traffic","Dust","Biomass","Background"]].mean()
        st.bar_chart(source_mean)
        st.dataframe(
            df_live[["name","Traffic","Dust","Biomass","Background","Dominant_Source"]],
            use_container_width=True
        )
        st.metric("Dominant City-wide Source", source_mean.idxmax())
