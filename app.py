import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
import io
import time

# ---- CONFIGURATION ----
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="PM10 Lucknow Analysis", layout="wide")

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

st.title("📊 PM10 Lucknow: Spatial Analysis")

# Sidebar for Controls
st.sidebar.header("Map Settings")
opacity = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.6)
map_style = st.sidebar.selectbox("Base Map", ["Positron", "DarkMatter", "OpenStreetMap"])

if st.button("🚀 Run Analysis"):
    df = fetch_pm10()
    
    if not df.empty:
        # 1. ANALYSIS RESULTS
        st.subheader("📝 Quick Insights")
        avg_pm10 = df['pm10'].mean()
        max_station = df.loc[df['pm10'].idxmax()]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Average PM10", f"{avg_pm10:.1f} µg/m³")
        c2.metric("Hotspot Value", f"{max_station['pm10']} µg/m³", delta="Highest")
        c3.metric("Station Count", len(df))

        # 2. KRIGING & MAPPING
        grid_res = 120
        lats = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
        lons = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)

        OK = OrdinaryKriging(df.lon, df.lat, df.pm10, variogram_model="spherical", verbose=False)
        z, _ = OK.execute("grid", lons, lats)

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymin = transformer.transform(lons.min(), lats.min())
        xmax, ymax = transformer.transform(lons.max(), lats.max())

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(z, extent=[xmin, xmax, ymin, ymax], origin="lower", 
                       cmap="YlOrRd", alpha=opacity, zorder=2)
        
        xs, ys = transformer.transform(df.lon.values, df.lat.values)
        ax.scatter(xs, ys, c="black", s=30, label="Sensors", zorder=3)

        # Map Selection
        sources = {"Positron": cx.providers.CartoDB.Positron, 
                   "DarkMatter": cx.providers.CartoDB.DarkMatter,
                   "OpenStreetMap": cx.providers.OpenStreetMap.Mapnik}
        
        cx.add_basemap(ax, source=sources[map_style], zoom=12)
        fig.colorbar(im, label="PM10")
        ax.set_axis_off()
        st.pyplot(fig)

        # 3. STATISTICAL TABLE
        st.write("### Station Rankings")
        st.dataframe(df[['name', 'pm10']].sort_values(by='pm10', ascending=False), use_container_width=True)
    else:
        st.error("Could not retrieve station data.")
