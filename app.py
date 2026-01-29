import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
import time
import io

# --------------------------------------------------
# CONFIGURATION & CONSTANTS
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(
    page_title="Lucknow PM10: Hybrid Spatial Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# DATA PIPELINE
# --------------------------------------------------
@st.cache_data(ttl=900)
def fetch_pm10_data():
    """Fetches real-time PM10 data from WAQI API for Lucknow bounds."""
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    stations = []

    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                # Fetch detailed station feed
                dr = requests.get(
                    f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
                ).json()

                if dr.get("status") == "ok" and "pm10" in dr["data"].get("iaqi", {}):
                    stations.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"]
                    })
                time.sleep(0.1) # Respect API rate limits

        df = pd.DataFrame(stations)
        # Handle duplicate coordinates which break Kriging
        if not df.empty:
            df = df.groupby(['lat', 'lon']).agg({'pm10': 'mean', 'name': 'first'}).reset_index()
        return df

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# MODELING LOGIC (RF-KRIGING HYBRID)
# --------------------------------------------------
def validate_hybrid_model(df):
    """Performs Leave-One-Out Cross-Validation (LOOCV)."""
    errors = []
    progress = st.progress(0)
    n = len(df)

    for i in range(n):
        train = df.drop(i)
        test = df.iloc[i]

        # 1. RF Trend
        rf = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
        rf.fit(train[['lat', 'lon']], train['pm10'])
        
        # 2. Residuals
        train_res = train['pm10'] - rf.predict(train[['lat', 'lon']])

        # 3. Kriging on Residuals
        try:
            ok = OrdinaryKriging(
                train.lon.values, train.lat.values, train_res.values,
                variogram_model="gaussian", verbose=False, enable_plotting=False
            )
            res_pred, _ = ok.execute("points", [test.lon], [test.lat])
            final_pred = rf.predict([[test.lat, test.lon]])[0] + res_pred[0]
        except:
            final_pred = rf.predict([[test.lat, test.lon]])[0]

        errors.append(abs(final_pred - test.pm10))
        progress.progress((i + 1) / n)

    progress.empty()
    return np.mean(errors)

# --------------------------------------------------
# UI & VISUALIZATION
# --------------------------------------------------
st.title("📍 PM10 Spatial Hybrid Analysis (Lucknow)")
st.markdown("""
This model combines **Random Forest** (to capture regional trends) with **Ordinary Kriging** (to interpolate local residuals), providing a high-resolution air quality surface.
""")

st.sidebar.header("🛠 Parameters")
opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.75)
weather_mult = st.sidebar.slider("Simulated Weather Factor (%)", 50, 200, 100) / 100.0

if st.sidebar.button("🚀 Execute Hybrid Model"):
    df = fetch_pm10_data()

    if df.empty or len(df) < 3:
        st.warning("Insufficient station data found in the current bounds.")
    else:
        with st.spinner("Calculating Spatial Residuals..."):
            mae = validate_hybrid_model(df)

            # Fit Final Model
            rf_final = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
            rf_final.fit(df[['lat', 'lon']], df['pm10'])
            
            df['residuals'] = (df['pm10'] - rf_final.predict(df[['lat', 'lon']])) * weather_mult

            # Create Prediction Grid
            grid_res = 100
            lats = np.linspace(df.lat.min() - 0.03, df.lat.max() + 0.03, grid_res)
            lons = np.linspace(df.lon.min() - 0.03, df.lon.max() + 0.03, grid_res)
            lon_g, lat_g = np.meshgrid(lons, lats)

            # Kriging execution
            OK = OrdinaryKriging(df.lon, df.lat, df['residuals'], variogram_model="gaussian")
            z_res, _ = OK.execute("grid", lons, lats)
            
            # RF Trend on Grid
            rf_trend = rf_final.predict(np.column_stack([lat_g.ravel(), lon_g.ravel()])).reshape(grid_res, grid_res)
            
            # Hybrid Surface
            z_final = (rf_trend * weather_mult) + z_res.T

        # --- Plotting ---
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymin = transformer.transform(lons.min(), lats.min())
        xmax, ymax = transformer.transform(lons.max(), lats.max())
        xs, ys = transformer.transform(df.lon.values, df.lat.values)

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
        
        im = ax.imshow(
            z_final, extent=[xmin, xmax, ymin, ymax], 
            origin="lower", cmap="YlOrRd", alpha=opacity, zorder=2
        )
        ax.scatter(xs, ys, c="white", edgecolors="black", s=70, label="Sensors", zorder=3)
        
        plt.colorbar(im, label="Predicted PM10 (µg/m³)")
        ax.set_axis_off()
        ax.set_title(f"Hybrid Spatial Interpolation (MAE: {mae:.2f})")

        # Layout columns
        c1, c2 = st.columns([3, 1])
        with c1:
            st.pyplot(fig)
        with c2:
            st.metric("Mean Prediction", f"{z_final.mean():.1f}")
            st.metric("Model Error (MAE)", f"{mae:.2f}")
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            st.download_button("💾 Download Map", buf.getvalue(), "lucknow_pm10.png", "image/png")
            
            st.write("**Station List**")
            st.dataframe(df[['name', 'pm10']], use_container_width=True)

st.caption("Data source: WAQI API. Method: Random Forest Residual Kriging (RFRK).")
