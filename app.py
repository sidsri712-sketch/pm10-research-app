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

count = st_autorefresh(interval=1800000, key="fizzbuzz")

# --------------------------------------------------
# DATA PIPELINE
# --------------------------------------------------
@st.cache_data(ttl=900)
def fetch_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    stations = []
    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                dr = requests.get(
                    f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
                ).json()
                if dr.get("status") == "ok" and "pm10" in dr["data"].get("iaqi", {}):
                    stations.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"],
                        "timestamp": pd.Timestamp.now()
                    })
                time.sleep(0.1)

        df_live = pd.DataFrame(stations)

        if not df_live.empty:
            df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
            df_live['hour'] = df_live['timestamp'].dt.hour
            df_live['dayofweek'] = df_live['timestamp'].dt.dayofweek
            df_live['month'] = df_live['timestamp'].dt.month

            if os.path.exists(DB_FILE):
                df_hist = pd.read_csv(DB_FILE, parse_dates=['timestamp'])
                df_all = pd.concat([df_hist, df_live], ignore_index=True)
            else:
                df_all = df_live.copy()

            df_all = df_all.sort_values('timestamp')

            df_all['pm10_lag_1'] = df_all.groupby(['lat','lon'])['pm10'].shift(1)
            df_all['pm10_lag_24'] = df_all.groupby(['lat','lon'])['pm10'].shift(24)

            df_all.to_csv(DB_FILE, index=False)

            return df_live

        return pd.DataFrame()
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# LOOCV
# --------------------------------------------------
def run_diagnostics(df):
    features = ['lat','lon','hour','dayofweek','month','pm10_lag_1','pm10_lag_24']
    df = df.dropna(subset=features)

    results = []
    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]

        rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        rf.fit(train[features], train['pm10'])
        residuals = train['pm10'] - rf.predict(train[features])

        ok = OrdinaryKriging(
            train.lon, train.lat, residuals,
            variogram_model="gaussian",
            variogram_parameters={'sill': np.var(residuals), 'range': 0.1, 'nugget': 0.5},
            verbose=False
        )

        krig_res, _ = ok.execute("points", [test.lon], [test.lat])
        pred = rf.predict(test[features].values.reshape(1,-1))[0] + krig_res[0]
        results.append({"Actual": test.pm10, "Predicted": pred})

    res_df = pd.DataFrame(results)
    mae = np.mean(np.abs(res_df['Actual'] - res_df['Predicted']))
    return res_df, mae

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")

run_hybrid = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Full Diagnostic")

if run_hybrid or run_diag:
    df_live = fetch_pm10_data()
    if df_live.empty:
        st.stop()

    df_hist = pd.read_csv(DB_FILE, parse_dates=['timestamp'])

    features = ['lat','lon','hour','dayofweek','month','pm10_lag_1','pm10_lag_24']
    df_train = df_hist.dropna(subset=features)

    rf_final = RandomForestRegressor(n_estimators=1000, max_depth=6, random_state=42)
    rf_final.fit(df_train[features], df_train['pm10'])

    df_live = df_live.merge(
        df_train[['lat','lon','pm10_lag_1','pm10_lag_24']],
        on=['lat','lon'], how='left'
    ).fillna(0)

    df_live['rf_pred'] = rf_final.predict(df_live[features])
    df_live['residuals'] = df_live['pm10'] - df_live['rf_pred']

    lats = np.linspace(df_live.lat.min()-0.06, df_live.lat.max()+0.06, 250)
    lons = np.linspace(df_live.lon.min()-0.06, df_live.lon.max()+0.06, 250)

    OK = OrdinaryKriging(
        df_live.lon, df_live.lat, df_live.residuals,
        variogram_model="gaussian",
        variogram_parameters={'sill': np.var(df_live.residuals), 'range': 0.1, 'nugget': 0.5}
    )

    z_res, _ = OK.execute("grid", lons, lats)

    lon_g, lat_g = np.meshgrid(lons, lats)
    time_feat = df_live.iloc[0][['hour','dayofweek','month','pm10_lag_1','pm10_lag_24']]

    rf_grid = rf_final.predict(
        np.column_stack([
            lat_g.ravel(),
            lon_g.ravel(),
            np.full(lat_g.size, time_feat['hour']),
            np.full(lat_g.size, time_feat['dayofweek']),
            np.full(lat_g.size, time_feat['month']),
            np.full(lat_g.size, time_feat['pm10_lag_1']),
            np.full(lat_g.size, time_feat['pm10_lag_24'])
        ])
    ).reshape(250,250)

    z_final = gaussian_filter(rf_grid + z_res.T, sigma=1.5)

    st.subheader("🗺 Spatiotemporal PM10 Surface")
    fig, ax = plt.subplots(figsize=(12,9))
    im = ax.imshow(z_final, cmap='magma', origin='lower')
    plt.colorbar(im, label="PM10 (µg/m³)")
    st.pyplot(fig)

st.caption("Data: WAQI API | Method: Spatiotemporal Random Forest + Residual Kriging")
