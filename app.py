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

st.set_page_config(
    page_title="Lucknow PM10 Hybrid Spatial Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 30 minutes
st_autorefresh(interval=1800000, key="refresh")

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

# -------------------------------
# LOAD FULL HISTORICAL DATA
# -------------------------------
if os.path.exists(DB_FILE):
    df_history = pd.read_csv(DB_FILE)
else:
    df_history = pd.DataFrame()
if not df_history.empty:
    st.sidebar.metric("Historical Samples", len(df_history))
st.sidebar.header("🛠 Controls")
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

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_hybrid or run_diag or predict_custom:

    df_live = fetch_pm10_data()
    if df_live.empty or len(df_live) < 3:
        st.warning("Not enough monitoring stations.")
        st.stop()

    if run_diag:
        st.subheader("📊 Model Diagnostics")
        res, mae = run_diagnostics(df_live)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            sns.regplot(data=res, x="Actual", y="Predicted", ax=ax)
            st.pyplot(fig)
            st.metric("MAE", f"{mae:.2f} µg/m³")
        with c2:
            fig, ax = plt.subplots()
            sns.histplot(df_live["pm10"], kde=True, ax=ax)
            st.pyplot(fig)

    # TRAINING DATA
    df_train = pd.read_csv(DB_FILE)
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

    df_live["residuals"] = (
        df_live["pm10"] - rf.predict(df_live[features])
    ) * weather_mult

    # GRID
    grid_res = 250
    lats = np.linspace(df_live.lat.min()-0.06, df_live.lat.max()+0.06, grid_res)
    lons = np.linspace(df_live.lon.min()-0.06, df_live.lon.max()+0.06, grid_res)

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

    z_final = gaussian_filter(rf_trend * weather_mult + z_res.T, sigma=1.5)
    z_final[z_final < 0] = 0

    # --- MAP ---
    st.subheader("📂 Historical PM10 Database")

    if not df_history.empty:
        st.metric("Total Historical Records", len(df_history))

        st.dataframe(
            df_history.sort_values("timestamp", ascending=False),
            use_container_width=True
        )

        st.download_button(
            label="📥 Download Full Historical CSV",
            data=df_history.to_csv(index=False).encode("utf-8"),
            file_name="lucknow_pm10_history.csv",
            mime="text/csv"
       )
    else:
        st.info("No historical data collected yet.")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())

    fig, ax = plt.subplots(figsize=(12, 9))

    # 🔑 THIS IS THE FIX
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Basemap AFTER limits
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)

    # --- MAP --- (Corrected imshow mapping)
    # --- MAP --- 
    # (Updated this specific block to fix the "lines" issue)
    im = ax.imshow(
        z_final,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower", 
        cmap="magma",
        alpha=opacity,
        interpolation="bilinear", # Changed for smoother spatial distribution
        zorder=2,
        aspect='auto'             # Forces the data to fill the map bounds
    )

    # Stations
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)
    ax.scatter(xs, ys, c="white", edgecolors="black", s=70, zorder=3, label="Stations")

    plt.colorbar(im, ax=ax, label="PM10 (µg/m³)")
    ax.legend()
    ax.set_axis_off()

    st.pyplot(fig)

    # CUSTOM POINT
    if predict_custom:
        c_rf = rf.predict([[custom_lat, custom_lon, now.hour, now.dayofweek, now.month,
                             weather_now["temp"], weather_now["hum"], weather_now["wind"]]])[0]
        try:
            c_res, _ = OK.execute("points", [custom_lon], [custom_lat])
            c_val = c_rf + c_res[0]
        except:
            c_val = c_rf

        st.metric("Predicted PM10", f"{c_val:.2f} µg/m³")

    # TREND
    st.subheader("📈 Trend & Forecast")
    df_f = df_train[
        (df_train["timestamp"].dt.date >= start_date) &
        (df_train["timestamp"].dt.date <= end_date)
    ]
    if len(df_f) > 5:
        df_r = df_f.set_index("timestamp").resample("H").mean(numeric_only=True).dropna()
        model = LinearRegression().fit(
            np.arange(len(df_r)).reshape(-1, 1),
            df_r["pm10"].values
        )
        st.line_chart(df_r["pm10"])

st.caption("Data: WAQI + Open-Meteo | Model: Random Forest Residual Kriging")
