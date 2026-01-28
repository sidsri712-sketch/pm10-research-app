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
TOKEN = "YOUR_WAQI_API_TOKEN" 
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05" 

st.set_page_config(page_title="Lucknow PM10 Analysis", layout="wide")

def get_live_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url).json()
        if r['status'] == 'ok':
            df = pd.DataFrame(r['data'])
            df[['lat', 'lon']] = df[['lat', 'lon']].astype(float)
            df['pm10'] = pd.to_numeric(df['aqi'], errors='coerce')
            return df.dropna(subset=['pm10'])
    except:
        return pd.DataFrame()

st.title("🏙️ Lucknow PM10 Research Mapper")

if st.button("Generate High-Res Heatmap"):
    df = get_live_data()
    
    if not df.empty:
        with st.spinner("Smoothing spatial gradients..."):
            # 1. Grid & ML Interpolation
            res = 200
            lat_idx = np.linspace(df.lat.min(), df.lat.max(), res)
            lon_idx = np.linspace(df.lon.min(), df.lon.max(), res)
            lon_grid, lat_grid = np.meshgrid(lon_idx, lat_idx)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(df[['lat', 'lon']], df['pm10'])
            pm_pred = model.predict(np.c_[lat_grid.ravel(), lon_grid.ravel()]).reshape(res, res)
            
            # 2. Smooth the data for that "Heatmap" look
            pm_smooth = gaussian_filter(pm_pred, sigma=4)

            # 3. Plotting
            fig, ax = plt.subplots(figsize=(10, 10))
            extent = [df.lon.min(), df.lon.max(), df.lat.min(), df.lat.max()]
            
            # Heat layer
            im = ax.imshow(pm_smooth, extent=extent, origin='lower', 
                           cmap='RdYlGn_r', alpha=0.5, interpolation='bilinear')
            
            # Basemap with automatic CRS detection
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
            
            # Sensor points
            ax.scatter(df.lon, df.lat, c='black', s=25, edgecolors='white', label='Sensors')
            
            plt.colorbar(im, label='PM10 (µg/m³)', shrink=0.5)
            ax.set_axis_off()
            st.pyplot(fig)

            # 4. High-Res Export
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=600, bbox_inches='tight')
            st.download_button("💾 Download 600 DPI PNG", buf.getvalue(), "lucknow_pm10.png")
    else:
        st.error("Live data fetch failed. Check API Token.")
