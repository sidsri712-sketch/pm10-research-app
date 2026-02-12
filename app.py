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
# PERMANENT DATA STORAGE (GOOGLE SHEETS)
# --------------------------------------------------
GSHEET_READ_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQO7corvhjivltUU1Y1aE4lDH1BmKDSF1O2uDSmSfw6HyNr5RuYz4qXYGCCsNDt3OUqqA7sFHaLqqiO/pub?output=csv"
GOOGLE_SHEET_SEND_URL = "https://script.google.com/macros/s/AKfycbyoy_PD319OgRj9z3j3WR2nrL_FWzLXU15o_a9Edc4ZzEmipvYtBaeCDr1xGdno_O5n/exec"

@st.cache_data(ttl=600)
def load_historical_data():
    """Reads historical records directly from Google Sheets."""
    try:
        df = pd.read_csv(GSHEET_READ_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Connecting to Google Sheets... {e}")
        return pd.DataFrame()
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


# --------------------------------------------------
# WEATHER FETCH (OPEN-METEO)
# --------------------------------------------------
def fetch_weather():
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            "latitude=26.85&longitude=80.94&"
            "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&"
            "forecast_days=2&timezone=Asia%2FKolkata"
        )

        r = requests.get(url).json()

        hourly = r["hourly"]
        df_weather = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly["time"]),
            "temp": hourly["temperature_2m"],
            "hum": hourly["relative_humidity_2m"],
            "wind": hourly["wind_speed_10m"]
        })

        return df_weather

    except Exception as e:
        st.error(f"Weather fetch failed: {e}")
        return pd.DataFrame()

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
                d = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
                if d.get("status") == "ok" and "pm10" in d["data"].get("iaqi", {}):
                    records.append({
                        "lat": s["lat"], "lon": s["lon"],
                        "pm10": d["data"]["iaqi"]["pm10"]["v"],
                        "name": d["data"]["city"]["name"],
                        "temp": weather["temp"], "hum": weather["hum"], "wind": weather["wind"],
                        "timestamp": pd.Timestamp.now()
                    })
                time.sleep(0.1)

        df_live = pd.DataFrame(records)

        if not df_live.empty:
            # Sync to Google Sheets
            import json
            try:
                payload = df_live.copy()
                payload["timestamp"] = payload["timestamp"].astype(str)
                payload_dict = payload.to_dict(orient="records")
                requests.post(GOOGLE_SHEET_SEND_URL, data=json.dumps(payload_dict))
            except Exception as e:
                st.sidebar.warning(f"Cloud sync failed: {e}")
            
            return df_live.groupby(["lat", "lon"]).agg({
                "pm10": "mean", "name": "first", "temp": "first", "hum": "first", "wind": "first"
            }).reset_index()

        return pd.DataFrame()
    except Exception as e:
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
            pred = rf.predict(test[features].values.reshape(1, -1))[0] + r[0]
        except:
            pred = rf.predict([test[features]])[0]

        preds.append({"Actual": test.pm10, "Predicted": pred})
    
    res = pd.DataFrame(preds)
    mae = np.mean(np.abs(res["Actual"] - res["Predicted"]))
    rmse = np.sqrt(np.mean((res["Actual"] - res["Predicted"])**2))
    return res, mae, rmse

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
# --------------------------------------------------
# SIDEBAR DATA LOADING
# --------------------------------------------------
df_history = load_historical_data()

if not df_history.empty:
    st.sidebar.metric("Historical Samples (Cloud)", len(df_history))
else:
    st.sidebar.warning("Syncing Cloud Database...")

# --- LIVE SENSOR LINK (New Sidebar Feature) ---


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
    res, mae, rmse = run_diagnostics(df_live)

    from sklearn.metrics import r2_score
    r2 = r2_score(res["Actual"], res["Predicted"])

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots()
        sns.regplot(data=res, x="Actual", y="Predicted", ax=ax)
        st.pyplot(fig)

        st.metric("MAE", f"{mae:.2f} µg/m³")
        st.metric("RMSE", f"{rmse:.2f} µg/m³")
        st.metric("R²", f"{r2:.3f}")
        with c2:
            fig, ax = plt.subplots()
            sns.histplot(df_live["pm10"], kde=True, ax=ax)
            st.pyplot(fig)

    # TRAINING DATA
    df_train = df_history.copy()
    if df_train.empty:
        st.warning("not enough historical cloud data for training.")
        st.stop()
        
    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
    df_train["hour"] = df_train["timestamp"].dt.hour
    df_train["dayofweek"] = df_train["timestamp"].dt.dayofweek
    df_train["month"] = df_train["timestamp"].dt.month

    features = ["hour", "dayofweek", "month", "temp", "hum", "wind"]

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
    # GRID - Increased buffer to 0.1 to allow radial dispersion
    grid_res = 250
    lats = np.linspace(df_live.lat.min()-0.1, df_live.lat.max()+0.1, grid_res)
    lons = np.linspace(df_live.lon.min()-0.1, df_live.lon.max()+0.1, grid_res)

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

    # Increased sigma for a circular, diffused look
    z_final = gaussian_filter(rf_trend * weather_mult + z_res.T, sigma=3.0)
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

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)

    # Rendering logic changed to 'equal' aspect to prevent broad lines
    im = ax.imshow(
        z_final,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="magma",
        alpha=opacity,
        interpolation="bicubic",
        zorder=2,
        aspect='equal' 
    )

    # Added subtle contour lines to define the heat zones
    ax.contour(
        z_final,
        levels=10, 
        extent=[xmin, xmax, ymin, ymax],
        colors='white',
        alpha=0.2,
        linewidths=0.5,
        zorder=3
    )

    # Stations
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)
    ax.scatter(xs, ys, c="white", edgecolors="black", s=70, zorder=4, label="Stations")

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
# --------------------------------------------------
    # TREND & 24-HOUR FUTURE FORECAST
    # --------------------------------------------------
    st.subheader("📈 Trend & 24-Hour Forecast")
    
    df_f = df_train[
        (df_train["timestamp"].dt.date >= start_date) &
        (df_train["timestamp"].dt.date <= end_date)
    ]

    if len(df_f) > 5:
        # 1. Historical Trend
        df_r = df_f.set_index("timestamp").resample("H").mean(numeric_only=True).dropna()
        
        weather_df = fetch_weather()

future_preds = []

for ft in future_times:
    weather_row = weather_df[weather_df["timestamp"] == ft.floor("H")]

    if not weather_row.empty:
        temp = weather_row["temp"].values[0]
        hum = weather_row["hum"].values[0]
        wind = weather_row["wind"].values[0]
    else:
        # fallback if timestamp mismatch
        temp = weather_df["temp"].iloc[0]
        hum = weather_df["hum"].iloc[0]
        wind = weather_df["wind"].iloc[0]

    pred = rf.predict(np.array([[
        custom_lat, custom_lon, ft.hour, ft.dayofweek, ft.month,
        temp, hum, wind
    ]]))[0]

    future_preds.append({
        "timestamp": ft,
        "pm10": pred,
        "Type": "Forecast"
    })

        df_forecast = pd.DataFrame(future_preds).set_index("timestamp")
        
        # Combine Historical and Forecast for the chart
        df_r["Type"] = "Historical"
        chart_data = pd.concat([df_r[["pm10", "Type"]], df_forecast])

        # Display Chart
        st.line_chart(chart_data["pm10"])
        st.caption("The graph shows historical averages followed by a 24-hour prediction based on time-cycles.")
    else:
        st.info("Collect more historical data to enable forecasting.")
    # --------------------------------------------------
# ACCURACY TRACKER (SIDEBAR ADDITION)
# --------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Intelligence Monitor")

if not df_history.empty and 'pm10' in df_history.columns:
    # Calculate data maturity
    data_richness = len(df_history)
    progress_val = min(data_richness / 1000, 1.0)
    st.sidebar.progress(progress_val, text=f"Data Maturity: {data_richness}/1000")
    
    if data_richness > 50:
        st.sidebar.success("✅ Model is identifying seasonal cycles.")
    elif data_richness > 10:
        st.sidebar.info("📈 Model is gathering local patterns...")
    else:
        st.sidebar.warning("👶 Model 'Brain' is in infant stage.")

    # Visualizing the "Growth" - Simple Trend of captured data points
    if len(df_history) > 5:
        st.sidebar.caption("Data Accumulation Trend")
        growth_data = df_history.groupby(pd.to_datetime(df_history['timestamp']).dt.date).size()
        st.sidebar.line_chart(growth_data)
