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
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78" 
# Expanded bounds to ensure enough data points for a "spread" effect
LUCKNOW_BOUNDS = "26.65,80.70,27.00,81.15" 

st.set_page_config(page_title="Lucknow PM10 Analysis", layout="wide")
st.title("🏙️ Lucknow Live PM10 Research Mapper")

def get_live_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url).json()
        if r['status'] == 'ok':
            df = pd.DataFrame(r['data'])
            df[['lat', 'lon']] = df[['lat', 'lon']].astype(float)
            # Use 'aqi' as the proxy for PM10 concentration
            df['pm10'] = pd.to_numeric(df['aqi'], errors='coerce')
            return df.dropna(subset=['pm10'])
    except:
        return pd.DataFrame()

if st.button("Generate High-Res Research Map"):
    df = get_live_data()
    
    if not df.empty and len(df) > 2:
        with st.spinner("Calculating spatial gradients..."):
            # 1. ML Interpolation
            res = 300 
            lat_range = np.linspace(df.lat.min(), df.lat.max(), res)
            lon_range = np.linspace(df.lon.min(), df.lon.max(), res)
            lon_mesh, lat_mesh = np.meshgrid(lon_range, lat_range)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(df[['lat', 'lon']], df['pm10'])
            grid_points = np.c_[lat_mesh.ravel(), lon_mesh.ravel()]
            pm_grid = model.predict(grid_points).reshape(res, res)
            
            # 2. Smooth the heatmap
            pm_grid_smooth = gaussian_filter(pm_grid, sigma=7)

            # 3. Plotting
            fig, ax = plt.subplots(figsize=(15, 15))
            
            extent = [lon_range.min(), lon_range.max(), lat_range.min(), lat_range.max()]
            
            # Create the heat layer
            im = ax.imshow(pm_grid_smooth, extent=extent, origin='lower', 
                           cmap='RdYlGn_r', alpha=0.5, interpolation='bilinear')
            
            # Add Basemap (Positron is best for research papers as it is clean)
            try:
                cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron)
            except Exception as e:
                st.warning("Basemap could not load, showing data only.")
            
            # Add station locations
            ax.scatter(df.lon, df.lat, c='black', s=50, edgecolors='white', linewidth=1, label='Live Sensors')
            
            plt.colorbar(im, label='PM10 Concentration (µg/m³)', shrink=0.5, pad=0.02)
            ax.set_title(f"Lucknow Spatial Analysis: PM10 Distribution\n(Real-time Data via WAQI API)", fontsize=18, pad=20)
            ax.axis('off')

            st.pyplot(fig)

            # 4. High-Res Export
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=600, bbox_inches='tight')
            st.download_button("💾 Download 600 DPI PNG for Research Paper", 
                               buf.getvalue(), "lucknow_pm10_report.png", "image/png")
    else:
        st.error("Not enough live data points found in Lucknow right now. Try expanding the bounds in the code.")

st.info("💡 Tip: Ensure your GitHub has both requirements.txt and packages.txt to prevent installation errors.")
