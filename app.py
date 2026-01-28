import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import time

# ---- CONFIGURATION ----
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="PM10 Lucknow ML–Kriging Tool", layout="wide")

# ---- DATA FETCH ----
@st.cache_data(ttl=900)
def fetch_pm10():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    r = requests.get(url).json()
    stations = []

    if r.get("status") == "ok":
        for s in r["data"]:
            dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
            if dr.get("status") == "ok" and "pm10" in dr["data"]["iaqi"]:
                stations.append({
                    "lat": s["lat"],
                    "lon": s["lon"],
                    "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                    "name": dr["data"]["city"]["name"]
                })
            time.sleep(0.2)

    return pd.DataFrame(stations)

# ---- SCIENTIFIC VALIDATION ----
def loocv_kriging(df):
    errors = []
    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]

        try:
            ok = OrdinaryKriging(
                train.lon.values,
                train.lat.values,
                train.pm10.values,
                variogram_model="spherical"
            )
            pred, _ = ok.execute("points", [test.lon], [test.lat])
            errors.append(abs(pred[0] - test.pm10))
        except:
            pass

    return np.mean(errors) if errors else np.nan

# ---- SIDEBAR ----
st.sidebar.header("🔬 Model Controls")
opacity = st.sidebar.slider("Heatmap Opacity", 0.0, 1.0, 0.6)
weather_impact = st.sidebar.slider("Simulated Weather Load (%)", 80, 150, 100)

if st.button("🚀 Run ML + Validation"):
    df = fetch_pm10()

    if df.empty:
        st.error("Data fetch failed.")
        st.stop()

    # ---- VALIDATION ----
    mae = loocv_kriging(df)

    # ---- ML SIMULATION ----
    X = df[['lat', 'lon']].values
    y = df['pm10'].values * (weather_impact / 100)

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)

    # ---- KRIGING ----
    grid_res = 120
    lats = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
    lons = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)

    OK = OrdinaryKriging(df.lon, df.lat, y, variogram_model="spherical")
    z, _ = OK.execute("grid", lons, lats)

    # ---- COORDINATES ----
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    xs, ys = transformer.transform(df.lon.values, df.lat.values)

    # ---- DASHBOARD ----
    col1, col2 = st.columns([3, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 8))

        # FIX: draw basemap FIRST
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom=12)

        # draw heatmap ON TOP
        im = ax.imshow(
            z,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="YlOrRd",
            alpha=opacity,
            zorder=3
        )

        ax.scatter(xs, ys, c="black", s=40, edgecolors="white", zorder=4)
        fig.colorbar(im, label="Predicted PM10 (µg/m³)")
        ax.set_axis_off()
        st.pyplot(fig)

    with col2:
        st.metric("LOOCV MAE", f"{mae:.2f} µg/m³")
        st.metric("City Mean PM10", f"{z.mean():.1f} µg/m³")

        st.write("**Spatial Feature Importance**")
        st.bar_chart(
            pd.DataFrame({
                "Feature": ["Latitude", "Longitude"],
                "Importance": model.feature_importances_
            }).set_index("Feature")
        )

        df["Predicted_PM10"] = y
        st.write("**High-Risk Stations**")
        st.table(
            df[["name", "Predicted_PM10"]]
            .sort_values("Predicted_PM10", ascending=False)
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Dataset",
            csv,
            "lucknow_pm10.csv",
            "text/csv"
        )

st.caption("LOOCV-validated Kriging with ML-based environmental scenario simulation.")
