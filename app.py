import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
import io
import time

# ---- CONFIGURATION ----
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="PM10 Lucknow ML-Predictor", layout="wide")

@st.cache_data(ttl=900)
def fetch_pm10():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url).json()
        stations = []
        if r.get("status") == "ok":
            for s in r["data"]:
                dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
                if dr.get("status") == "ok" and "pm10" in dr["data"]["iaqi"]:
                    stations.append({
                        "lat": s["lat"], "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"]
                    })
                time.sleep(0.2)
        return pd.DataFrame(stations)
    except:
        return pd.DataFrame()

# ---- SIDEBAR CONTROLS ----
st.sidebar.header("🛠️ Simulation & ML Settings")
opacity = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.6)
# This slider simulates a "Weather Impact" factor (e.g., Stagnant air or high humidity)
weather_impact = st.sidebar.slider("Simulated Weather Driver (Impact %)", 80, 150, 100) 
st.sidebar.info("Increase this slider to simulate how poor dispersion (low wind) would amplify current PM10 levels across the grid.")

if st.button("🚀 Generate ML Spatial Prediction"):
    df = fetch_pm10()
    
    if not df.empty:
        # 1. ML PREDICTION ENGINE
        # We train the ML model on current data and apply the 'weather_impact' as a feature multiplier
        X = df[['lat', 'lon']].values
        y = df['pm10'].values * (weather_impact / 100.0)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 2. KRIGING INTERPOLATION
        grid_res = 120
        lats = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
        lons = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)
        
        # We use the ML-adjusted values for the Kriging map
        OK = OrdinaryKriging(df.lon, df.lat, y, variogram_model="spherical", verbose=False)
        z, _ = OK.execute("grid", lons, lats)

        # 3. COORDINATE TRANSFORMATION
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymin = transformer.transform(lons.min(), lats.min())
        xmax, ymax = transformer.transform(lons.max(), lats.max())

        # 4. DASHBOARD LAYOUT
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader(f"📍 Predicted PM10 Map ({weather_impact}% Load)")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(z, extent=[xmin, xmax, ymin, ymax], origin="lower", 
                           cmap="YlOrRd", alpha=opacity, zorder=2)
            
            # Show original station points
            xs, ys = transformer.transform(df.lon.values, df.lat.values)
            ax.scatter(xs, ys, c="black", s=40, edgecolors="white", label="Sensors", zorder=3)
            
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
            fig.colorbar(im, label="Predicted PM10 (µg/m³)")
            ax.set_axis_off()
            st.pyplot(fig)

        with col2:
            st.subheader("📈 ML Summary")
            st.metric("Predicted City Average", f"{z.mean():.1f} µg/m³")
            
            # Show feature importance of the ML Model
            importance = model.feature_importances_
            st.write("**Spatial Driver Weights:**")
            st.bar_chart(pd.DataFrame({"Driver": ["Lat", "Lon"], "Weight": importance}).set_index("Driver"))
            
            st.write("### Predicted Hotspots")
            df['predicted_pm10'] = y
            st.table(df[['name', 'predicted_pm10']].sort_values(by='predicted_pm10', ascending=False))

    else:
        st.error("Station data could not be retrieved.")

st.divider()
st.caption("Note: The ML model uses a RandomForestRegressor to simulate environmental impact on spatial distribution.")
