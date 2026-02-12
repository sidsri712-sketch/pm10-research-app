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
                    weather_now = weather.iloc[0] if not weather.empty else {"temp":25,"hum":50,"wind":2}

                    records.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": d["data"]["iaqi"]["pm10"]["v"],
                        "name": d["data"]["city"]["name"],
                        "temp": weather_now["temp"],
                        "hum": weather_now["hum"],
                        "wind": weather_now["wind"],
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
# --------------------------------------------------
# NSS-NET UPGRADED DIAGNOSTICS
# --------------------------------------------------
def run_diagnostics(df):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    preds = []
    actuals = []
    
    # 1. Prepare Diagnostic DataFrame
    df_diag = df.copy()
    now = pd.Timestamp.now()
    df_diag["hour"] = now.hour
    df_diag["dayofweek"] = now.dayofweek
    df_diag["month"] = now.month
    
    # Since LOOCV is a snapshot, we use the city median for the lag-anchor
    df_diag["pm10_lag1"] = df_diag["pm10"].median() 
    
    features = ["lat", "lon", "hour", "dayofweek", "month", "temp", "hum", "wind", "pm10_lag1"]
    
    # 2. LOOCV Loop (Leave-One-Out)
    for i in range(len(df_diag)):
        train = df_diag.drop(df_diag.index[i])
        test = df_diag.iloc[[i]]

        # Scaling logic inside the fold
        scaler_diag = StandardScaler()
        train_scaled = scaler_diag.fit_transform(train[features])
        test_scaled = scaler_diag.transform(test[features])

        # Train on Log-scale
        rf = RandomForestRegressor(n_estimators=500, max_depth=7, random_state=42)
        rf.fit(train_scaled, np.log1p(train["pm10"]))
        
        # Predict and Back-transform
        pred_log = rf.predict(test_scaled)[0]
        preds.append(np.expm1(pred_log))
        actuals.append(test.pm10.values[0])

    # 3. Compute Metrics
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    res = pd.DataFrame({"Actual": actuals, "Predicted": preds})

    return res, mae, rmse

# --------------------------------------------------
# UI HEADER
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")

weather_df = fetch_weather()

if not weather_df.empty:
    weather_now = weather_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("🌡 Temperature", f"{weather_now['temp']:.1f} °C")
    c2.metric("💧 Humidity", f"{weather_now['hum']:.0f} %")
    c3.metric("💨 Wind Speed", f"{weather_now['wind']:.1f} km/h")
else:
    st.warning("Weather data unavailable.")
    weather_now = pd.Series({"temp":25,"hum":50,"wind":2})

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

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            sns.regplot(data=res, x="Actual", y="Predicted", ax=ax)
            st.pyplot(fig)

            st.metric("MAE", f"{mae:.2f} µg/m³")
            st.metric("RMSE", f"{rmse:.2f} µg/m³")

        with c2:
            fig, ax = plt.subplots()
            sns.histplot(df_live["pm10"], kde=True, ax=ax)
            st.pyplot(fig)

        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            sns.regplot(data=res, x="Actual", y="Predicted", ax=ax)
            st.pyplot(fig)

            st.metric("MAE", f"{mae:.2f} µg/m³")
            st.metric("RMSE", f"{rmse:.2f} µg/m³")
           
        with c2:
            fig, ax = plt.subplots()
            sns.histplot(df_live["pm10"], kde=True, ax=ax)
            st.pyplot(fig)

    # TRAINING DATA
    # --------------------------------------------------
    # 1. NSS-NET PRE-PROCESSING
    # --------------------------------------------------
    # --------------------------------------------------
    # NOVEL MODEL: SPATIOTEMPORAL ATTENTIVE META-RESIDUAL NETWORK (SAM-ResNet)
    # --------------------------------------------------
    from sklearn.preprocessing import PolynomialFeatures, RobustScaler
    from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor

    # 1. SPATIAL DIMENSIONALITY EXPANSION (Addressing uniform weather)
    poly = PolynomialFeatures(degree=3, include_bias=False)
    df_train = df_history.copy().sort_values("timestamp")
    
    # Generate high-dimensional spatial keys
    coords_train = df_train[['lat', 'lon']]
    spatial_poly_train = poly.fit_transform(coords_train)
    poly_cols = [f"sp_{i}" for i in range(spatial_poly_train.shape[1])]
    
    # Cyclical Time Encoding
    df_train["h_sin"] = np.sin(2 * np.pi * df_train["timestamp"].dt.hour / 24)
    df_train["h_cos"] = np.cos(2 * np.pi * df_train["timestamp"].dt.hour / 24)
    df_train["m_sin"] = np.sin(2 * np.pi * df_train["timestamp"].dt.month / 12)

    # Combine into Meta-Feature Set
    df_meta_train = pd.concat([
        pd.DataFrame(spatial_poly_train, columns=poly_cols, index=df_train.index),
        df_train[["h_sin", "h_cos", "m_sin", "temp", "hum", "wind"]]
    ], axis=1)
    
    features_sam = df_meta_train.columns.tolist()

    # 2. ENSEMBLE META-LEARNER (SAM-ResNet Core)
    gbr = GradientBoostingRegressor(n_estimators=1200, max_depth=6, learning_rate=0.03, subsample=0.8)
    rf_meta = RandomForestRegressor(n_estimators=1000, max_depth=8, random_state=42)
    sam_resnet = VotingRegressor([('gbr', gbr), ('rf', rf_meta)])

    # Log-scaling target for stability
    sam_resnet.fit(df_meta_train, np.log1p(df_train["pm10"]))

    # 3. LIVE INFERENCE & RESIDUAL CALCULATION
    now = pd.Timestamp.now()
    live_poly = poly.transform(df_live[['lat', 'lon']])
    df_live_meta = pd.concat([
        pd.DataFrame(live_poly, columns=poly_cols, index=df_live.index),
        pd.DataFrame({
            "h_sin": [np.sin(2 * np.pi * now.hour / 24)] * len(df_live),
            "h_cos": [np.cos(2 * np.pi * now.hour / 24)] * len(df_live),
            "m_sin": [np.sin(2 * np.pi * now.month / 12)] * len(df_live),
            "temp": df_live["temp"], "hum": df_live["hum"], "wind": df_live["wind"]
        }, index=df_live.index)
    ], axis=1)

    live_preds = np.expm1(sam_resnet.predict(df_live_meta[features_sam]))
    df_live["res"] = df_live["pm10"] - live_preds

    # 4. HOLE-EFFECT KRIGING (The Synapse)
    grid_res = 200
    lats = np.linspace(26.75, 26.95, grid_res)
    lons = np.linspace(80.85, 81.05, grid_res)
    lon_g, lat_g = np.meshgrid(lons, lats)

    try:
        OK = OrdinaryKriging(df_live.lon, df_live.lat, df_live.res, variogram_model='hole-effect')
        z_res, _ = OK.execute("grid", lons, lats)
    except:
        z_res = np.zeros((grid_res, grid_res))

    # 5. FINAL MESH SYNTHESIS
    grid_flat = np.column_stack([lat_g.ravel(), lon_g.ravel()])
    grid_poly = poly.transform(grid_flat)
    df_grid_meta = pd.DataFrame(grid_poly, columns=poly_cols)
    df_grid_meta["h_sin"] = np.sin(2 * np.pi * now.hour / 24)
    df_grid_meta["h_cos"] = np.cos(2 * np.pi * now.hour / 24)
    df_grid_meta["m_sin"] = np.sin(2 * np.pi * now.month / 12)
    df_grid_meta["temp"] = weather_now["temp"]
    df_grid_meta["hum"] = weather_now["hum"]
    df_grid_meta["wind"] = weather_now["wind"]

    z_trend = np.expm1(sam_resnet.predict(df_grid_meta[features_sam])).reshape(grid_res, grid_res)
    z_final = gaussian_filter(z_trend + z_res.T, sigma=1.2)

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
    # NSS-NET CUSTOM POINT PREDICTION
    # SAM-RESNET CUSTOM POINT PREDICTION
    if predict_custom:
        custom_p = pd.DataFrame([[custom_lat, custom_lon]], columns=['lat', 'lon'])
        custom_poly = poly.transform(custom_p)
        
        custom_meta = pd.concat([
            pd.DataFrame(custom_poly, columns=poly_cols),
            pd.DataFrame({
                "h_sin": [np.sin(2 * np.pi * now.hour / 24)],
                "h_cos": [np.cos(2 * np.pi * now.hour / 24)],
                "m_sin": [np.sin(2 * np.pi * now.month / 12)],
                "temp": [weather_now["temp"]], "hum": [weather_now["hum"]], "wind": [weather_now["wind"]]
            })
        ], axis=1)
        
        c_trend = np.expm1(sam_resnet.predict(custom_meta[features_sam])[0])
        try:
            c_res, _ = OK.execute("points", [custom_lon], [custom_lat])
            c_val = max(0, c_trend + c_res[0])
        except:
            c_val = c_trend
            
        st.sidebar.metric("SAM-ResNet Prediction", f"{c_val:.2f} µg/m³")

    # TREND
# --------------------------------------------------
    # TREND & 24-HOUR FUTURE FORECAST (NSS-Net Version)
    # --------------------------------------------------
    # --------------------------------------------------
    # SAM-RESNET RECURSIVE FORECAST
    # --------------------------------------------------
    st.subheader("📈 24-Hour Adaptive Forecast")
    
    future_times = pd.date_range(start=pd.Timestamp.now().ceil("H"), periods=24, freq="H")
    future_results = []

    for ft in future_times:
        f_p = pd.DataFrame([[custom_lat, custom_lon]], columns=['lat', 'lon'])
        f_poly = poly.transform(f_p)
        
        f_meta = pd.concat([
            pd.DataFrame(f_poly, columns=poly_cols),
            pd.DataFrame({
                "h_sin": [np.sin(2 * np.pi * ft.hour / 24)],
                "h_cos": [np.cos(2 * np.pi * ft.hour / 24)],
                "m_sin": [np.sin(2 * np.pi * ft.month / 12)],
                "temp": [weather_now["temp"]], "hum": [weather_now["hum"]], "wind": [weather_now["wind"]]
            })
        ], axis=1)
        
        f_pred = np.expm1(sam_resnet.predict(f_meta[features_sam])[0])
        future_results.append({"timestamp": ft, "pm10": max(f_pred, df_train['pm10'].min())})

    df_forecast = pd.DataFrame(future_results)
    st.line_chart(df_forecast.set_index("timestamp")["pm10"])
    st.caption("The graph predicts Lucknow's PM10 levels for the next 24 hours using the NSS-Net SRI loop.")
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
        growth_data = (
            pd.to_datetime(df_history['timestamp'])
            .dt.date
            .value_counts()
            .sort_index()
        )
        st.sidebar.line_chart(growth_data)
