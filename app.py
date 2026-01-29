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
import time
import io
import os
from streamlit_autorefresh import st_autorefresh
import datetime 
import matplotlib.patches as mpatches 
from scipy.ndimage import gaussian_filter

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
count = st_autorefresh(interval=1800000, key="fizzbuzz")

# --------------------------------------------------
# NEW: WEATHER DATA FETCHING
# --------------------------------------------------
def fetch_weather():
    """Fetches real-time weather for Lucknow using Open-Meteo (Free)"""
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=26.85&longitude=80.94&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        res = requests.get(url).json()
        return {
            "temp": res['current']['temperature_2m'],
            "hum": res['current']['relative_humidity_2m'],
            "wind": res['current']['wind_speed_10m']
        }
    except:
        return {"temp": 25.0, "hum": 50.0, "wind": 5.0} # Fallback defaults

# --------------------------------------------------
# DATA PIPELINE (HISTORICAL STORAGE)
# --------------------------------------------------
@st.cache_data(ttl=900)
def fetch_pm10_data():
    weather = fetch_weather()
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    stations = []
    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
                if dr.get("status") == "ok" and "pm10" in dr["data"].get("iaqi", {}):
                    stations.append({
                        "lat": s["lat"], "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"],
                        "temp": weather['temp'], "hum": weather['hum'], "wind": weather['wind'],
                        "timestamp": pd.Timestamp.now()
                    })
                time.sleep(0.1)
        df_live = pd.DataFrame(stations)
        if not df_live.empty:
            if os.path.exists(DB_FILE):
                df_history = pd.read_csv(DB_FILE)
                df_combined = pd.concat([df_history, df_live], ignore_index=True)
                df_combined.drop_duplicates(subset=['lat', 'lon', 'pm10', 'timestamp'], inplace=True)
                df_combined.to_csv(DB_FILE, index=False)
            else:
                df_live.to_csv(DB_FILE, index=False)
            return df_live.groupby(['lat', 'lon']).agg({'pm10': 'mean', 'name': 'first', 'temp': 'first', 'hum': 'first', 'wind': 'first'}).reset_index()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# LOOCV + DIAGNOSTICS
# --------------------------------------------------
def run_diagnostics(df):
    results = []
    features = ['lat', 'lon', 'temp', 'hum', 'wind']
    for i in range(len(df)):
        train, test = df.drop(i), df.iloc[i]
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(train[features], train['pm10'])
        residuals = train['pm10'] - rf.predict(train[features])
        try:
            ok = OrdinaryKriging(train.lon, train.lat, residuals, variogram_model="gaussian")
            p_res, _ = ok.execute("points", [test.lon], [test.lat])
            pred = rf.predict([test[features]])[0] + p_res[0]
        except: pred = rf.predict([test[features]])[0]
        results.append({"Actual": test.pm10, "Predicted": pred})
    return pd.DataFrame(results), np.mean(np.abs(pd.DataFrame(results)['Actual'] - pd.DataFrame(results)['Predicted']))

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")
weather_now = fetch_weather()

# Show Weather Metrics in Header
w_col1, w_col2, w_col3 = st.columns(3)
w_col1.metric("Temperature", f"{weather_now['temp']}°C")
w_col2.metric("Humidity", f"{weather_now['hum']}%")
w_col3.metric("Wind Speed", f"{weather_now['wind']} km/h")

st.sidebar.caption(f"Last data refresh: {time.strftime('%H:%M:%S')}")

if os.path.exists(DB_FILE):
    df_hist_export = pd.read_csv(DB_FILE)
    st.sidebar.metric("Historical Samples", len(df_hist_export))
    st.sidebar.download_button(label="📥 Download Historical CSV", data=df_hist_export.to_csv(index=False).encode('utf-8'), file_name='lucknow_pm10_history.csv', mime='text/csv')

st.sidebar.header("🛠 Controls")
opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.75)
weather_mult = st.sidebar.slider("Simulated Weather Factor (%)", 50, 200, 100) / 100
run_hybrid = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Full Diagnostic")

st.sidebar.subheader("Custom Location Prediction")
custom_lat = st.sidebar.number_input("Latitude", value=26.85, step=0.01, format="%.2f")
custom_lon = st.sidebar.number_input("Longitude", value=80.95, step=0.01, format="%.2f")
predict_custom = st.sidebar.button("🎯 Predict Custom PM10")

st.sidebar.subheader("Historical Data Filter")
start_date = st.sidebar.date_input('Start Date', value=datetime.date.today() - datetime.timedelta(days=7))
end_date = st.sidebar.date_input('End Date', value=datetime.date.today())

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_hybrid or run_diag or predict_custom:
    df_live = fetch_pm10_data()
    if df_live.empty or len(df_live) < 3:
        st.warning("Not enough monitoring stations or data for analysis.")
        st.stop()

    if run_diag:
        st.subheader("📊 Model Diagnostics")
        res_df, mae = run_diagnostics(df_live)
        c1, c2 = st.columns(2)
        with c1:
            fig1, ax1 = plt.subplots(); sns.regplot(data=res_df, x="Actual", y="Predicted", ax=ax1, color="teal")
            st.pyplot(fig1); st.metric("Mean Absolute Error", f"{mae:.2f} µg/m³")
        with c2:
            fig2, ax2 = plt.subplots(); sns.histplot(df_live['pm10'], kde=True, ax=ax2, color="orange")
            st.pyplot(fig2)

    # ML TRAINING
    df_train = pd.read_csv(DB_FILE) if os.path.exists(DB_FILE) else df_live.copy()
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    df_train['hour'] = df_train['timestamp'].dt.hour
    df_train['dayofweek'] = df_train['timestamp'].dt.dayofweek
    df_train['month'] = df_train['timestamp'].dt.month

    # Fill missing weather in history if any
    for col in ['temp', 'hum', 'wind']:
        if col not in df_train.columns: df_train[col] = weather_now[col.replace('temp','temp').replace('hum','hum').replace('wind','wind')]

    features = ['lat', 'lon', 'hour', 'dayofweek', 'month', 'temp', 'hum', 'wind']
    rf_final = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
    rf_final.fit(df_train[features], df_train['pm10'])

    # SURFACE GENERATION
    df_live['hour'], df_live['dayofweek'], df_live['month'] = pd.Timestamp.now().hour, pd.Timestamp.now().dayofweek, pd.Timestamp.now().month
    df_live['residuals'] = (df_live['pm10'] - rf_final.predict(df_live[features])) * weather_mult

    grid_res = 200
    lats = np.linspace(df_live.lat.min()-0.06, df_live.lat.max()+0.06, grid_res)
    lons = np.linspace(df_live.lon.min()-0.06, df_live.lon.max()+0.06, grid_res)
    
    try:
        OK = OrdinaryKriging(df_live.lon, df_live.lat, df_live['residuals'], variogram_model="gaussian")
        z_res, _ = OK.execute("grid", lons, lats)
    except: z_res = np.zeros((grid_res, grid_res))

    lon_g, lat_g = np.meshgrid(lons, lats)
    rf_trend = rf_final.predict(np.column_stack([
        lat_g.ravel(), lon_g.ravel(), 
        np.full(lat_g.size, pd.Timestamp.now().hour),
        np.full(lat_g.size, pd.Timestamp.now().dayofweek),
        np.full(lat_g.size, pd.Timestamp.now().month),
        np.full(lat_g.size, weather_now['temp']),
        np.full(lat_g.size, weather_now['hum']),
        np.full(lat_g.size, weather_now['wind'])
    ])).reshape(grid_res, grid_res)

    z_final = gaussian_filter((rf_trend * weather_mult) + z_res.T, sigma=1.5)
    z_final[z_final < 0] = 0

    if predict_custom:
        c_rf = rf_final.predict([[custom_lat, custom_lon, pd.Timestamp.now().hour, pd.Timestamp.now().dayofweek, pd.Timestamp.now().month, weather_now['temp'], weather_now['hum'], weather_now['wind']]])[0]
        try:
            c_res, _ = OK.execute("points", [custom_lon], [custom_lat])
            c_pred = c_rf + c_res[0]
        except: c_pred = c_rf
        st.markdown(f"### Predicted PM10 at ({custom_lat}, {custom_lon}):")
        st.metric("PM10 Value", f"{c_pred:.2f} µg/m³")

    # MAP PLOTTING
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    fig, ax = plt.subplots(figsize=(12, 9))
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)
    im = ax.imshow(z_final, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="magma", alpha=opacity, interpolation="hamming")
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)
    ax.scatter(xs, ys, c="white", edgecolors="black", s=70, label="Stations")
    plt.colorbar(im, label="PM10 (µg/m³)")
    ax.legend(); ax.set_axis_off(); st.pyplot(fig)

    st.subheader("📈 Trend & Forecast")
    if os.path.exists(DB_FILE):
        df_f = df_train[(df_train['timestamp'].dt.date >= start_date) & (df_train['timestamp'].dt.date <= end_date)]
        if len(df_f) > 5:
            df_r = df_f.set_index('timestamp').resample('H').mean(numeric_only=True).dropna()
            model = LinearRegression().fit(np.array(range(len(df_r))).reshape(-1, 1), df_r['pm10'].values)
            st.line_chart(df_r['pm10'])
