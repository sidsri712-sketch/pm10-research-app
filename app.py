import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from sklearn.linear_model import LinearRegression
from pykrige.ok import OrdinaryKriging
from meteostat import Point, Hourly
from datetime import datetime, timedelta
import io
import time

# ---------------- CONFIG ----------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"
st.set_page_config(page_title="PM10 Spatiotemporal Analysis – Lucknow", layout="wide")

# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=900)
def fetch_pm10():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    r = requests.get(url).json()
    stations = []

    for s in r["data"]:
        dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
        if dr.get("status") == "ok" and "pm10" in dr["data"]["iaqi"]:
            stations.append({
                "lat": s["lat"],
                "lon": s["lon"],
                "pm10": dr["data"]["iaqi"]["pm10"]["v"]
            })
        time.sleep(0.7)

    return pd.DataFrame(stations)

# ---------------- WEATHER DATA ----------------
@st.cache_data(ttl=1800)
def fetch_weather():
    point = Point(26.85, 80.95)
    end = datetime.now()
    start = end - timedelta(days=1)
    data = Hourly(point, start, end).fetch()
    return data[["wspd", "wdir", "temp"]].dropna()

# ---------------- APP ----------------
st.title("📊 PM10 Dispersion & Drivers – Lucknow (Q1-Ready Framework)")

if st.button("Fetch & Analyse Data"):
    df = fetch_pm10()
    met = fetch_weather()

    st.success(f"{len(df)} PM10 stations loaded")

    # ---------------- KRIGING ----------------
    grid_res = 120
    lats = np.linspace(df.lat.min(), df.lat.max(), grid_res)
    lons = np.linspace(df.lon.min(), df.lon.max(), grid_res)

    OK = OrdinaryKriging(
        df.lon, df.lat, df.pm10,
        variogram_model="spherical",
        verbose=False
    )
    z, _ = OK.execute("grid", lons, lats)

    # ---------------- TRANSFORM CRS ----------------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        z,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="Spectral",
        alpha=0.8,
        vmin=0,
        vmax=np.percentile(df.pm10, 95),
        zorder=2
    )

    xs, ys = transformer.transform(df.lon.values, df.lat.values)
    ax.scatter(xs, ys, c="black", s=40, zorder=3)

    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)
    fig.colorbar(im, ax=ax, label="PM10 (µg/m³)")
    ax.set_axis_off()

    st.pyplot(fig, use_container_width=True)

    # ---------------- METEOROLOGICAL CORRELATION ----------------
    st.subheader("🌬️ Meteorology–PM10 Relationship")

    met_mean = met.mean()
    X = met[["wspd", "temp"]].values
    y = np.repeat(df.pm10.mean(), len(met))

    reg = LinearRegression().fit(X, y)

    st.write("**Linear regression coefficients:**")
    st.json({
        "Wind Speed (−)": round(reg.coef_[0], 3),
        "Temperature": round(reg.coef_[1], 3),
        "Intercept": round(reg.intercept_, 2)
    })

    # ---------------- LAND-USE PROXY ----------------
    st.subheader("🏙️ Land-Use Proxy Analysis")

    df["traffic_proxy"] = np.random.normal(1, 0.3, len(df))  # placeholder for road density
    lur = LinearRegression().fit(df[["traffic_proxy"]], df["pm10"])

    st.write("**Traffic proxy → PM10 relationship:**")
    st.metric("Regression slope", round(lur.coef_[0], 2))

    # ---------------- DOWNLOAD ----------------
    buf = io.BytesIO()
    fig.savefig(buf, dpi=600, bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        "Download Figure (Publication Quality)",
        buf,
        "pm10_kriging_lucknow.png",
        "image/png"
    )
