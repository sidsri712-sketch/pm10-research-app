```python
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

if st.button("Fetch Live PM10 Data"):
    with st.spinner("Fetching live station data + details..."):
        df, error = get_live_pm10_data()
    
    if error:
        st.error(f"Data fetch failed: {error}\nCheck token and internet.")
    elif df.empty:
        st.warning("No stations with valid PM10 data found in this bounding box right now.")
    else:
        st.success(f"Found {len(df)} stations with PM10 readings.")
        st.session_state['df'] = df  # Store data in session state for replotting

# If data is available, show plotting options
if 'df' in st.session_state:
    df = st.session_state['df']
    
    st.subheader("Customize Heatmap")
    opacity = st.slider("Heatmap Opacity", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
    transparency = st.slider("Station Transparency (Alpha)", min_value=0.1, max_value=1.0, value=1.0, step=0.05)  # For scatter points
    cmap_options = ['OrRd', 'YlOrRd', 'hot', 'viridis', 'plasma', 'inferno', 'magma']
    selected_cmap = st.selectbox("Heatmap Color Scheme", cmap_options, index=0)
    show_labels = st.checkbox("Show PM10 Values on Stations", value=True)
    
    if st.button("Generate Customized Heatmap"):
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
            
            # --- Plotting improvements ---
            fig, ax = plt.subplots(figsize=(14, 12))  # slightly larger for better visibility
            extent = [lon_idx.min(), lon_idx.max(), lat_idx.min(), lat_idx.max()]
            
            # Add basemap FIRST (important order: basemap under heatmap)
            # Use ESRI basemap for better detail
            try:
                cx.add_basemap(
                    ax,
                    crs="EPSG:4326",
                    source=cx.providers.Esri.WorldStreetMap,  # ESRI basemap
                    zoom_adjust=0,  # sometimes helps tile fetching
                    reset_extent=False  # critical: prevents zooming out and blanking
                )
            except Exception as e:
                st.warning(f"Basemap fetch issue: {e}. Showing without basemap.")
            
            # Heatmap layer – customizable alpha, colormap, clip extreme values
            im = ax.imshow(
                pm_smooth,
                extent=extent,
                origin='lower',
                cmap=selected_cmap,  # User-selected colormap
                alpha=opacity,  # User-controlled opacity
                interpolation='bilinear',
                vmin=0,
                vmax=250,  # adjust vmax based on Lucknow's typical PM10 (can be dynamic: np.percentile(y, 95))
                zorder=1  # ensure it's below points but above basemap if needed
            )
            
            # Stations on top with real live data points
            ax.scatter(
                df['lon'], df['lat'],
                c='black', s=60, edgecolors='white', linewidth=1.5, alpha=transparency,
                label='Monitoring Stations', zorder=3
            )
            
            # Optional: Label stations with PM10 values
            if show_labels:
                for idx, row in df.iterrows():
                    ax.text(row['lon'], row['lat'], f"{row['pm10']:.0f}", fontsize=8, ha='center', va='bottom', color='blue', zorder=4)
            
            # Colorbar with better placement & label
            cbar = fig.colorbar(
                im, ax=ax, label='PM10 (µg/m³)', shrink=0.6, pad=0.04,
                orientation='vertical'
            )
            cbar.ax.tick_params(labelsize=10)
            
            # Title & cosmetics
            ax.set_title(
                f"Interpolated PM10 Heatmap – Lucknow\n({len(df)} stations • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')})",
                fontsize=14, pad=15
            )
            ax.set_axis_off()
            
            # Force axis limits to match data extent (prevents cropping)
            ax.set_xlim(lon_idx.min(), lon_idx.max())
            ax.set_ylim(lat_idx.min(), lat_idx.max())
            
            # Tight layout to use full figure space
            fig.tight_layout(pad=0.5)
            
            st.pyplot(fig, use_container_width=True)  # better Streamlit integration
            
            # Download (higher DPI for print quality)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=600, bbox_inches='tight', pad_inches=0.1)
            buf.seek(0)
            st.download_button(
                label="💾 Download High-Res PNG (600 DPI)",
                data=buf,
                file_name="lucknow_pm10_heatmap.png",
                mime="image/png"
            )
```
