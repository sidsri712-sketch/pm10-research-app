import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
from pyproj import Transformer
import io
import time

TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Analysis", layout="wide")

@st.cache_data(ttl=900)
def get_live_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url, timeout=12).json()
        if r.get("status") != "ok":
            return None, r.get("data", "API error")

        stations = r["data"]
        data = []
        progress = st.progress(0)

        for i, station in enumerate(stations):
            detail_url = f"https://api.waqi.info/feed/@{station['uid']}/?token={TOKEN}"
            try:
                dr = requests.get(detail_url, timeout=8).json()
                if dr.get("status") == "ok" and "iaqi" in dr["data"]:
                    pm10 = dr["data"]["iaqi"].get("pm10", {}).get("v")
                    if pm10 is not None:
                        data.append({
                            "lat": float(station["lat"]),
                            "lon": float(station["lon"]),
                            "pm10": float(pm10),
                            "name": station.get("station", {}).get("name", "?")
                        })
            except Exception:
                pass

            progress.progress((i + 1) / len(stations))
            time.sleep(0.7)

        df = pd.DataFrame(data)
        return df.dropna(subset=["pm10"]), None

    except Exception as e:
        return None, str(e)

st.title("🏙️ Lucknow PM10 Research Mapper (Real-time)")

if st.button("Fetch Live PM10 Data"):
    with st.spinner("Fetching live station data + details..."):
        df, error = get_live_pm10_data()

    if error:
        st.error(error)
    elif df.empty:
        st.warning("No valid PM10 data found.")
    else:
        st.success(f"Found {len(df)} stations.")
        st.session_state["df"] = df

if "df" in st.session_state:
    df = st.session_state["df"]

    opacity = st.slider("Heatmap Opacity", 0.2, 1.0, 0.75, 0.05)

    cmap_options = [
        "turbo",
        "Spectral",
        "viridis",
        "plasma",
        "inferno",
        "coolwarm"
    ]
    selected_cmap = st.selectbox("Heatmap Colors", cmap_options)

    if st.button("Generate Heatmap"):
        res = 180

        lat = np.linspace(df.lat.min(), df.lat.max(), res)
        lon = np.linspace(df.lon.min(), df.lon.max(), res)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
        model.fit(df[["lat", "lon"]], df["pm10"])

        pm = model.predict(np.c_[lat_grid.ravel(), lon_grid.ravel()]).reshape(res, res)
        pm = gaussian_filter(pm, sigma=5)

        # ---- CRS TRANSFORMATION (FIX) ----
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xmin, ymin = transformer.transform(lon.min(), lat.min())
        xmax, ymax = transformer.transform(lon.max(), lat.max())

        fig, ax = plt.subplots(figsize=(14, 12))

        im = ax.imshow(
            pm,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap=selected_cmap,
            alpha=opacity,
            vmin=0,
            vmax=np.percentile(df.pm10, 95),
            zorder=2
        )

        # Stations (reprojected)
        xs, ys = transformer.transform(df.lon.values, df.lat.values)
        ax.scatter(xs, ys, c="black", s=60, edgecolors="white", zorder=3)

        cx.add_basemap(
            ax,
            source=cx.providers.Esri.WorldImagery,
            zoom=12
        )

        fig.colorbar(im, ax=ax, label="PM10 (µg/m³)", shrink=0.6)
        ax.set_axis_off()

        st.pyplot(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.savefig(buf, dpi=600, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            "💾 Download PNG",
            buf,
            "lucknow_pm10_esri.png",
            "image/png"
        )
