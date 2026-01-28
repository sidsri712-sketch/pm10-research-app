import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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

def calculate_loocv(df):
    """Scientific Validation: Leave-One-Out Cross-Validation for Kriging."""
    errors = []
    lons, lats, values = df.lon.values, df.lat.values, df.pm10.values
    for i in range(len(df)):
        # Split data: Leave one out
        train_lon = np.delete(lons, i)
        train_lat = np.delete(lats, i)
        train_val = np.delete(values, i)
        test_lon, test_lat, test_val = lons[i], lats[i], values[i]
        
        # Fit model on training subset
        try:
            ok_temp = OrdinaryKriging(train_lon, train_lat, train_val, variogram_model='spherical')
            pred, _ = ok_temp.execute('points', [test_lon], [test_lat])
            errors.append(abs(pred[0] - test_val))
        except:
            continue
    return np.mean(errors) if errors else 0

# ---- SIDEBAR ----
st.sidebar.header("🔬 Research Parameters")
opacity = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.6)
weather_impact = st.sidebar.slider("Simulated Weather Load (%)", 80, 150, 100)

if st.button("🚀 Run Analysis & Validation"):
    df = fetch_pm10()
    
    if not df.empty:
        # 1. SCIENTIFIC VALIDATION (LOOCV)
        mae = calculate_loocv(df)
        
        # 2. ML PREDICTION
        X = df[['lat', 'lon']].values
        y_simulated = df['pm10'].values * (weather_impact / 100.0)
        model = RandomForestRegressor(n_estimators=100).fit(X, y_simulated)

        # 3. SPATIAL INTERPOLATION
        grid_res = 125
        lats_grid = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
        lons_grid = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)
        OK = OrdinaryKriging(df.lon, df.lat, y_simulated, variogram_model="spherical")
        z, _ = OK.execute("grid", lons_grid, lats_grid)

        # 4. PLOTTING
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymin = transformer.transform(lons_grid.min(), lats_grid.min())
        xmax, ymax = transformer.transform(lons_grid.max(), lats_grid.max())

        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Spatial Distribution Map")
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(z, extent=[xmin, xmax, ymin, ymax], origin="lower", cmap="YlOrRd", alpha=opacity)
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
            fig.colorbar(im, label="Predicted PM10 (µg/m³)")
            ax.set_axis_off()
            st.pyplot(fig)

        with col2:
            st.metric("Validation (MAE)", f"{mae:.2f}", help="Mean Absolute Error via Leave-One-Out Cross-Validation")
            st.metric("City Mean", f"{z.mean():.1f} µg/m³")
            st.write("**Model Accuracy:**")
            st.progress(max(0, min(1.0, 1 - (mae/df.pm10.mean()))))
            
            st.write("### Data Export")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV for Paper", csv, "lucknow_pm10_data.csv", "text/csv")
    else:
        st.error("Data fetch failed.")
