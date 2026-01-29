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
# DATA PIPELINE (HISTORICAL STORAGE)
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
            if os.path.exists(DB_FILE):
                df_history = pd.read_csv(DB_FILE)
                df_combined = pd.concat([df_history, df_live], ignore_index=True)
                df_combined.drop_duplicates(subset=['lat', 'lon', 'pm10'], inplace=True)
                df_combined.to_csv(DB_FILE, index=False)
            else:
                df_live.to_csv(DB_FILE, index=False)
            
            df_display = df_live.groupby(['lat', 'lon']).agg({
                'pm10': 'mean',
                'name': 'first'
            }).reset_index()
            return df_display
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# LOOCV + DIAGNOSTICS
# --------------------------------------------------
def run_diagnostics(df):
    results = []
    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(train[['lat', 'lon']], train['pm10'])
        residuals = train['pm10'] - rf.predict(train[['lat', 'lon']])
        try:
            ok = OrdinaryKriging(train.lon, train.lat, residuals, variogram_model="gaussian", verbose=False)
            p_res, _ = ok.execute("points", [test.lon], [test.lat])
            pred = rf.predict([[test.lat, test.lon]])[0] + p_res[0]
        except:
            pred = rf.predict([[test.lat, test.lon]])[0]
        results.append({"Actual": test.pm10, "Predicted": pred})
    res_df = pd.DataFrame(results)
    mae = np.mean(np.abs(res_df['Actual'] - res_df['Predicted']))
    return res_df, mae

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")
st.sidebar.caption(f"Last data refresh: {time.strftime('%H:%M:%S')}")

if os.path.exists(DB_FILE):
    df_hist_export = pd.read_csv(DB_FILE)
    st.sidebar.metric("Historical Samples", len(df_hist_export))
    
    csv_data = df_hist_export.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Historical CSV",
        data=csv_data,
        file_name='lucknow_pm10_history.csv',
        mime='text/csv',
    )

st.markdown("""
**Method:** Random Forest for spatial trend + Ordinary Kriging on residuals  
**Purpose:** High-resolution PM10 surface with statistical validation
""")

st.sidebar.header("🛠 Controls")
opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.75)
weather_mult = st.sidebar.slider("Simulated Weather Factor (%)", 50, 200, 100) / 100

run_hybrid = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Full Diagnostic")

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_hybrid or run_diag:
    df_live = fetch_pm10_data()
    if df_live.empty or len(df_live) < 3:
        st.warning("Not enough monitoring stations.")
        st.stop()

    if run_diag:
        st.subheader("📊 Model Diagnostics")
        res_df, mae = run_diagnostics(df_live)
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.regplot(data=res_df, x="Actual", y="Predicted", ax=ax1, color="teal")
            ax1.set_title("Actual vs Predicted PM10")
            st.pyplot(fig1)
            st.metric("Mean Absolute Error", f"{mae:.2f} µg/m³")
        with col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(df_live['pm10'], kde=True, ax=ax2, color="orange")
            ax2.set_title("PM10 Distribution")
            st.pyplot(fig2)

    # ---------- FINAL HYBRID SURFACE ----------
    st.subheader("🗺 High-Resolution PM10 Surface")

    if os.path.exists(DB_FILE):
        df_train = pd.read_csv(DB_FILE)
    else:
        df_train = df_live

    rf_final = RandomForestRegressor(n_estimators=1000, max_depth=1, random_state=42)
    rf_final.fit(df_train[['lat', 'lon']], df_train['pm10'])

    df_live['residuals'] = (df_live['pm10'] - rf_final.predict(df_live[['lat', 'lon']])) * weather_mult

    grid_res = 200 
    lats = np.linspace(df_live.lat.min()-0.06, df_live.lat.max()+0.06, grid_res)
    lons = np.linspace(df_live.lon.min()-0.06, df_live.lon.max()+0.06, grid_res)

    v_range = 0.2 * (2 - weather_mult) 
    
    OK = OrdinaryKriging(
        df_live.lon, df_live.lat, df_live['residuals'],
        variogram_model="gaussian",
        variogram_parameters={'sill': np.var(df_live['residuals']), 'range': v_range, 'nugget': 0.5}
    )
    z_res, _ = OK.execute("grid", lons, lats)

    lon_g, lat_g = np.meshgrid(lons, lats)
    rf_trend = rf_final.predict(np.column_stack([lat_g.ravel(), lon_g.ravel()])).reshape(grid_res, grid_res)

    z_final = (rf_trend * weather_mult) + z_res.T

    # ---------- MAP ----------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)

    im = ax.imshow(
        z_final,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="magma",
        alpha=opacity,
        interpolation="bicubic" 
    )

    ax.scatter(xs, ys, c="white", edgecolors="black", s=70, zorder=3, label="Stations")
    plt.colorbar(im, label="PM10 (µg/m³)")
    ax.set_axis_off()
    st.pyplot(fig)

    # HEALTH SCALE LEGEND
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡 PM10 Health Scale")
    st.sidebar.info("""
    - **0-50**: Good
    - **51-100**: Satisfactory
    - **101-250**: Moderate
    - **251-350**: Poor
    - **351-430**: Very Poor
    - **430+**: Severe
    """)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    st.download_button("💾 Download Map Image", buf.getvalue(), "lucknow_pm10_hybrid.png", "image/png")

    st.subheader("📌 Monitoring Stations")
    st.dataframe(df_live[['name', 'pm10']], use_container_width=True)

    # --- 24-HOUR TREND + FORECAST ---
    if os.path.exists(DB_FILE):
        st.subheader("📈 24-Hour Trend & 3-Hour Forecast")
        df_trend = pd.read_csv(DB_FILE)
        df_trend['timestamp'] = pd.to_datetime(df_trend['timestamp'])
        
        last_24h = pd.Timestamp.now() - pd.Timedelta(hours=24)
        df_filtered = df_trend[df_trend['timestamp'] >= last_24h].copy()
        
        if len(df_filtered) > 5:
            df_resampled = df_filtered.set_index('timestamp').resample('H').mean(numeric_only=True).dropna()
            
            # Predictive Logic: Linear Regression on the last few hours
            X = np.array(range(len(df_resampled))).reshape(-1, 1)
            y = df_resampled['pm10'].values
            model = LinearRegression().fit(X, y)
            
            # Forecast next 3 hours
            future_indices = np.array(range(len(df_resampled), len(df_resampled) + 3)).reshape(-1, 1)
            future_preds = model.predict(future_indices)
            
            # Generate future timestamps
            last_time = df_resampled.index[-1]
            future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 4)]
            
            fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
            ax_trend.plot(df_resampled.index, df_resampled['pm10'], marker='o', color='crimson', label="Observed Trend")
            ax_trend.plot(future_times, future_preds, marker='x', linestyle='--', color='gray', label="3hr Forecast")
            
            ax_trend.set_ylabel("Avg PM10 (µg/m³)")
            ax_trend.legend()
            ax_trend.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig_trend)
        else:
            st.info("Accumulating data for forecast... Please wait for more refresh cycles.")

st.caption("Data: WAQI API | Method: Random Forest Residual Kriging (RFRK)")
