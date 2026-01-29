# ==================================================
# LUCKNOW PM10 SPATIOTEMPORAL HYBRID MODEL (FINAL)
# RF (Space + Time) + Residual Kriging
# ==================================================

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
import datetime, time, io, os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TOKEN = "YOUR_WAQI_TOKEN"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"
DB_FILE = "lucknow_pm10_history.csv"

st.set_page_config("Lucknow PM10 Hybrid Model", layout="wide")
st_autorefresh(interval=1800000, key="refresh")

# --------------------------------------------------
# DATA FETCH + FEATURE ENGINEERING
# --------------------------------------------------
@st.cache_data(ttl=900)
def fetch_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    rows = []

    r = requests.get(url).json()
    if r.get("status") != "ok":
        return pd.DataFrame()

    for s in r["data"]:
        dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
        if dr.get("status") == "ok" and "pm10" in dr["data"]["iaqi"]:
            rows.append({
                "lat": s["lat"],
                "lon": s["lon"],
                "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                "name": dr["data"]["city"]["name"],
                "timestamp": pd.Timestamp.now()
            })
        time.sleep(0.1)

    df_live = pd.DataFrame(rows)
    if df_live.empty:
        return df_live

    # Time features
    df_live["hour"] = df_live["timestamp"].dt.hour
    df_live["dayofweek"] = df_live["timestamp"].dt.dayofweek
    df_live["month"] = df_live["timestamp"].dt.month

    # Append to DB
    if os.path.exists(DB_FILE):
        df_hist = pd.read_csv(DB_FILE, parse_dates=["timestamp"])
        df_all = pd.concat([df_hist, df_live], ignore_index=True)
    else:
        df_all = df_live.copy()

    # Lag features
    df_all = df_all.sort_values("timestamp")
    df_all["pm10_lag_1"] = df_all.groupby(["lat","lon"])["pm10"].shift(1)
    df_all["pm10_lag_24"] = df_all.groupby(["lat","lon"])["pm10"].shift(24)

    df_all.to_csv(DB_FILE, index=False)
    return df_live

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📍 Lucknow PM10 Spatiotemporal Hybrid Model")

run_model = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Diagnostics")

custom_lat = st.sidebar.number_input("Latitude", 26.85)
custom_lon = st.sidebar.number_input("Longitude", 80.95)
predict_custom = st.sidebar.button("🎯 Predict Custom PM10")

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_model or run_diag or predict_custom:

    df_live = fetch_pm10_data()
    if df_live.empty:
        st.warning("No data available.")
        st.stop()

    df_hist = pd.read_csv(DB_FILE, parse_dates=["timestamp"])

    features = [
        "lat","lon","hour","dayofweek","month",
        "pm10_lag_1","pm10_lag_24"
    ]

    df_train = df_hist.dropna(subset=features)

    rf = RandomForestRegressor(
        n_estimators=800,
        max_depth=6,
        random_state=42
    )
    rf.fit(df_train[features], df_train["pm10"])

    df_live = df_live.merge(
        df_train[["lat","lon","pm10_lag_1","pm10_lag_24"]],
        on=["lat","lon"], how="left"
    ).fillna(0)

    df_live["rf_pred"] = rf.predict(df_live[features])
    df_live["residuals"] = df_live["pm10"] - df_live["rf_pred"]

    # --------------------------------------------------
    # KRIGING
    # --------------------------------------------------
    lats = np.linspace(df_live.lat.min()-0.06, df_live.lat.max()+0.06, 250)
    lons = np.linspace(df_live.lon.min()-0.06, df_live.lon.max()+0.06, 250)

    OK = OrdinaryKriging(
        df_live.lon, df_live.lat, df_live.residuals,
        variogram_model="gaussian",
        variogram_parameters={
            "sill": np.var(df_live.residuals),
            "range": 0.1,
            "nugget": 0.5
        }
    )

    z_res, _ = OK.execute("grid", lons, lats)

    lon_g, lat_g = np.meshgrid(lons, lats)
    time_ref = df_live.iloc[0]

    rf_grid = rf.predict(np.column_stack([
        lat_g.ravel(),
        lon_g.ravel(),
        np.full(lat_g.size, time_ref.hour),
        np.full(lat_g.size, time_ref.dayofweek),
        np.full(lat_g.size, time_ref.month),
        np.full(lat_g.size, time_ref.pm10_lag_1),
        np.full(lat_g.size, time_ref.pm10_lag_24),
    ])).reshape(250,250)

    z_final = gaussian_filter(rf_grid + z_res.T, sigma=1.5)
    z_final[z_final < 0] = 0

    # --------------------------------------------------
    # MAP
    # --------------------------------------------------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)

    fig, ax = plt.subplots(figsize=(12,9))
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)
    im = ax.imshow(
        z_final, extent=[xmin,xmax,ymin,ymax],
        origin="lower", cmap="magma", alpha=0.75
    )
    ax.scatter(xs, ys, c="white", edgecolors="black", s=60)
    plt.colorbar(im, label="PM10 (µg/m³)")
    ax.set_axis_off()
    st.pyplot(fig)

    # --------------------------------------------------
    # CUSTOM PREDICTION
    # --------------------------------------------------
    if predict_custom:
        rf_p = rf.predict([[custom_lat,custom_lon,
                            time_ref.hour,time_ref.dayofweek,
                            time_ref.month,time_ref.pm10_lag_1,
                            time_ref.pm10_lag_24]])[0]
        kr_p,_ = OK.execute("points",[custom_lon],[custom_lat])
        st.metric("Predicted PM10", f"{rf_p+kr_p[0]:.2f} µg/m³")

st.caption("Data: WAQI | Method: Spatiotemporal RF + Residual Kriging")
