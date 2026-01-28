import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
import io
import time

TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"  # ← get from https://aqicn.org/data-platform/token/
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Analysis", layout="wide")

@st.cache_data(ttl=900)  # cache 15 min
def get_live_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url, timeout=12).json()
        if r.get('status') != 'ok':
            return None, r.get('data', 'API error')
        
        stations = r['data']
        data = []
        progress = st.progress(0)
        for i, station in enumerate(stations):
            detail_url = f"https://api.waqi.info/feed/@{station['uid']}/?token={TOKEN}"
            try:
                dr = requests.get(detail_url, timeout=8).json()
                if dr.get('status') == 'ok':
                    pm10 = dr['data']['iaqi'].get('pm10', {}).get('v')
                    if pm10 is not None:
                        data.append({
                            'lat': float(station['lat']),
                            'lon': float(station['lon']),
                            'pm10': float(pm10),
                            'name': station.get('station', {}).get('name', '?')
                        })
            except:
                pass
            progress.progress((i+1)/len(stations))
            time.sleep(0.4)  # polite rate limiting
        
        df = pd.DataFrame(data)
        return df.dropna(subset=['pm10']), None
    except Exception as e:
        return None, str(e)

st.title("🏙️ Lucknow PM10 Research Mapper (Real-time)")

if st.button("Generate High-Res PM10 Heatmap"):
    with st.spinner("Fetching live station data + details..."):
        df, error = get_live_pm10_data()
    
    if error:
        st.error(f"Data fetch failed: {error}\nCheck token and internet.")
    elif df.empty:
        st.warning("No stations with valid PM10 data found in this bounding box right now.")
    else:
        st.success(f"Found {len(df)} stations with PM10 readings.")
        
        with st.spinner("Creating ML-based smooth heatmap..."):
            res = 180  # slightly lower to speed up
            lat_idx = np.linspace(df['lat'].min(), df['lat'].max(), res)
            lon_idx = np.linspace(df['lon'].min(), df['lon'].max(), res)
            lon_grid, lat_grid = np.meshgrid(lon_idx, lat_idx)
            
            X = df[['lat', 'lon']].values
            y = df['pm10'].values
            
            model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
            model.fit(X, y)
            
            pm_pred = model.predict(np.c_[lat_grid.ravel(), lon_grid.ravel()]).reshape(res, res)
            pm_smooth = gaussian_filter(pm_pred, sigma=5)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            extent = [lon_idx.min(), lon_idx.max(), lat_idx.min(), lat_idx.max()]
            
            im = ax.imshow(pm_smooth, extent=extent, origin='lower',
                           cmap='OrRd', alpha=0.65, interpolation='bilinear', vmin=0, vmax=250)
            
            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.PositronNoLabels)
            
            ax.scatter(df['lon'], df['lat'], c='black', s=40, edgecolors='white', linewidth=1.2, label='Stations')
            
            cbar = plt.colorbar(im, ax=ax, label='PM10  (µg/m³)', shrink=0.6, pad=0.03)
            cbar.ax.tick_params(labelsize=9)
            
            ax.set_title(f"Lucknow PM10 – {len(df)} stations interpolated", fontsize=13)
            ax.set_axis_off()
            
            st.pyplot(fig)
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=500, bbox_inches='tight')
            buf.seek(0)
            st.download_button(
                label="💾 Download High-Res PNG (500 DPI)",
                data=buf,
                file_name="lucknow_pm10_heatmap.png",
                mime="image/png"
            )
