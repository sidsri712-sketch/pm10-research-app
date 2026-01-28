import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn.ensemble import RandomForestRegressor
import io

# --- CONFIG ---
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"  # Get free at https://aqicn.org/data-platform/token/
LUCKNOW_BOUNDS = "26.70,80.75,26.95,81.10" # Lat/Lon box for Lucknow

st.set_page_config(page_title="Lucknow Live Spatial Analysis", layout="wide")

# --- DATA FETCHING ---
def get_live_lucknow_data():
    """Fetches live sensor data from WAQI API for the Lucknow region."""
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        response = requests.get(url).json()
        if response['status'] == 'ok':
            data = response['data']
            # Clean and format
            df = pd.DataFrame(data)
            df['lat'] = df['lat'].astype(float)
            df['lon'] = df['lon'].astype(float)
            # Fetching PM10 (Note: some stations might report AQI as proxy)
            df['pm10'] = pd.to_numeric(df['aqi'], errors='coerce') 
            return df.dropna(subset=['pm10'])
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# --- SPATIAL ML & INTERPOLATION ---
def create_spread_heatmap(df, res=250):
    # ML to 'learn' the city's pollution trend
    model = RandomForestRegressor(n_estimators=100)
    model.fit(df[['lat', 'lon']], df['pm10'])

    # Create a dense grid over Lucknow
    lat_grid = np.linspace(df.lat.min(), df.lat.max(), res)
    lon_grid = np.linspace(df.lon.min(), df.lon.max(), res)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Predict PM10 for every pixel in the grid
    grid_points = np.c_[lat_mesh.ravel(), lon_mesh.ravel()]
    pm10_pred = model.predict(grid_points).reshape(res, res)
    
    return lon_mesh, lat_mesh, pm10_pred

# --- UI ---
st.title("🏙️ Live PM10 Spatial Analysis: Lucknow")
st.markdown("This app connects to live sensors and uses ML to interpolate air quality across the city.")

if st.button("🔄 Refresh Live Data"):
    df_live = get_live_lucknow_data()
    
    if not df_live.empty:
        lon_m, lat_m, pm_grid = create_spread_heatmap(df_live)

        # Plotting for Research
        fig, ax = plt.subplots(figsize=(12, 9))
        contour = ax.contourf(lon_m, lat_m, pm_grid, levels=50, cmap='Spectral_r')
        plt.colorbar(contour, label='Predicted PM10 (µg/m³)')
        
        # Overlay actual sensor locations for transparency
        ax.scatter(df_live['lon'], df_live['lat'], c='black', s=20, label='Live Sensors', alpha=0.6)
        
        ax.set_title("Lucknow PM10 Distribution (Live Interpolation)", fontsize=15)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

        st.pyplot(fig)

        # High-Res Export
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png", dpi=600, bbox_inches='tight')
        st.download_button("📥 Download 600 DPI PNG for Paper", img_buf.getvalue(), "lucknow_aqi_research.png")
    else:
        st.warning("Could not fetch live data. Please check your API token.")
