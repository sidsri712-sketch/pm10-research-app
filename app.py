import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
import io

# --- CONFIG ---
# Get a free token at https://aqicn.org/data-platform/token/
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78" 
LUCKNOW_BOUNDS = "26.70,80.75,26.95,81.10" 

st.set_page_config(page_title="Lucknow PM10 Analysis", layout="wide")
st.title("🏙️ Lucknow Live PM10 Research Mapper")

def get_live_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url).json()
        if r['status'] == 'ok':
            df = pd.DataFrame(r['data'])
            df[['lat', 'lon']] = df[['lat', 'lon']].astype(float)
            df['pm10'] = pd.to_numeric(df['aqi'], errors='coerce')
            return df.dropna(subset=['pm10'])
    except: return pd.DataFrame()

# --- PROCESSING ---
if st.button("Generate High-Res Research Map"):
    df = get_live_data()
    
    if not df.empty:
        with st.spinner("Smoothing spatial gradients..."):
            # 1. ML Interpolation
            res = 300 
            lat_range = np.linspace(df.lat.min(), df.lat.max(), res)
            lon_range = np.linspace(df.lon.min(), df.lon.max(), res)
            lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(df[['lat', 'lon']], df['pm10'])
            grid_points = np.c_[lat_mesh.ravel(), lon_mesh.ravel()]
            pm_grid = model.predict(grid_points).reshape(res, res)
            
            # 2. Apply Gaussian Smoothing for the "Heatmap" look
            # This removes the "blocky" squares seen in your previous image
            pm_grid_smooth = gaussian_filter(pm_grid, sigma=5)

            # 3. Plotting with Professional Basemap
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Use extent to align the heatmap with the basemap coordinates
            extent = [lon_range.min(), lon_range.max(), lat_range.min(), lat_range.max()]
            
            im = ax.imshow(pm_grid_smooth, extent=extent, origin='lower', 
                           cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
            
            # Add Lucknow Basemap (Toner or Terrain look good for research)
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
            
            # Overlay sensor locations
            ax.scatter(df.lon, df.lat, c='black', s=30, edgecolors='white', label='Stations')
            
            plt.colorbar(im, label='PM10 Concentration (µg/m³)', shrink=0.7)
            ax.set_title("Lucknow: Spatial PM10 Distribution", fontsize=16, pad=20)
            ax.axis('off')

            st.pyplot(fig)

            # 4. Export
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=600, bbox_inches='tight')
            st.download_button("💾 Download Publication Quality PNG (600 DPI)", 
                               buf.getvalue(), "lucknow_heatmap.png")
    else:
        st.error("Could not fetch live data. Verify your API Token.")
