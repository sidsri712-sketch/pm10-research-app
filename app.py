import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from pykrige.ok import OrdinaryKriging
from datetime import datetime, timedelta
import io
import time

# ---- CONFIGURATION & SECRETS ----
# Pro-tip: In production, use st.secrets["WAQI_TOKEN"]
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

try:
    from meteostat import Point, Hourly
    METEOSTAT_AVAILABLE = True
except ImportError:
    METEOSTAT_AVAILABLE = False

st.set_page_config(page_title="PM10 Spatiotemporal Analysis – Lucknow", layout="wide")

# ---- DATA FETCHING ----
@st.cache_data(ttl=900)
def fetch_pm10():
    """Fetches real-time PM10 data from WAQI stations within Lucknow bounds."""
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url).json()
        stations = []
        if r.get("status") == "ok":
            for s in r["data"]:
                # Fetch detailed feed for each station to get specific PM10 values
                dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
                if dr.get("status") == "ok" and "pm10" in dr["data"]["iaqi"]:
                    stations.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"]
                    })
                time.sleep(0.2) # Reduced sleep for better UX
        return pd.DataFrame(stations)
    except Exception as e:
        st.error(f"Error fetching AQI data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_weather():
    """Fetches recent hourly weather data for Lucknow."""
    if not METEOSTAT_AVAILABLE:
        return None
    try:
        point = Point(26.8467, 80.9462) # Center of Lucknow
        end = datetime.now()
        start = end - timedelta(days=1)
        data = Hourly(point, start, end).fetch()
        return data[["wspd", "wdir", "temp"]].dropna()
    except Exception as e:
        return None

# ---- UI LAYOUT ----
st.title("📊 PM10 Dispersion & Drivers – Lucknow")
st.markdown("This framework performs **Ordinary Kriging** interpolation and explores meteorological drivers.")

if st.button("🚀 Run Spatial Analysis"):
    df = fetch_pm10()
    met = fetch_weather()

    if df.empty:
        st.error("No station data found. Please check your API token or bounds.")
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.success(f"Successfully polled {len(df)} stations in Lucknow.")
            
            # -------- KRIGING INTERPOLATION --------
            # Create a finer grid for a smoother map
            grid_res = 150 
            lats = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
            lons = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)

            OK = OrdinaryKriging(
                df.lon, df.lat, df.pm10,
                variogram_model="spherical",
                verbose=False,
                enable_plotting=False
            )
            z, ss = OK.execute("grid", lons, lats)

            # -------- COORDINATE TRANSFORMATION --------
            # Convert WGS84 (Lat/Lon) to Web Mercator for Contextily basemap
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xmin, ymin = transformer.transform(lons.min(), lats.min())
            xmax, ymax = transformer.transform(lons.max(), lats.max())

            # -------- PLOTTING --------
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(
                z,
                extent=[xmin, xmax, ymin, ymax],
                origin="lower",
                cmap="YlOrRd", # More standard for pollution heatmaps
                alpha=0.7,
                zorder=2
            )

            # Overlay station locations
            xs, ys = transformer.transform(df.lon.values, df.lat.values)
            ax.scatter(xs, ys, c="white", edgecolor="black", s=50, marker='o', label="Stations", zorder=3)

            # Add Basemap
            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
            
            fig.colorbar(im, ax=ax, label="PM10 Concentration (µg/m³)")
            ax.set_title("Kriging Interpolation of PM10 - Lucknow", fontsize=14)
            ax.set_axis_off()
            
            st.pyplot(fig)

            # Download Option
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            st.download_button("💾 Download High-Res Map", buf.getvalue(), "lucknow_pm10.png", "image/png")

        with col2:
            st.subheader("🌬️ Drivers Analysis")
            if met is not None and not met.empty:
                # Logic: We correlate the current spatial mean with current weather 
                # (Ideally, you'd use a time-series here)
                current_weather = met.iloc[-1]
                
                # Simple linear model logic (Placeholder for multi-variate analysis)
                # To make this valid, we use the station variance against current weather
                X = np.column_stack([df.lat, df.lon]) 
                reg = LinearRegression().fit(X, df.pm10)
                
                st.metric("Avg Lucknow Temp", f"{current_weather['temp']} °C")
                st.metric("Wind Speed", f"{current_weather['wspd']} km/h")
                
                st.write("**Spatial Gradient (Regression):**")
                st.write(f"North-South Slope: `{reg.coef_[0]:.2f}`")
                st.write(f"East-West Slope: `{reg.coef_[1]:.2f}`")
                
                # Show raw data table
                st.write("**Station Data:**")
                st.dataframe(df[["name", "pm10"]].sort_values(by="pm10", ascending=False))
            else:
                st.warning("Meteorological data unavailable.")

# Add Footer
st.divider()
st.caption("Data Source: World Air Quality Index (WAQI) & Meteostat. Analysis: Ordinary Kriging Interpolation.")
