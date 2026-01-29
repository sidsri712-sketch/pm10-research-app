import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
import time
import io

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(
    page_title="Lucknow PM10: RF–Kriging Hybrid",
    layout="wide"
)

# --------------------------------------------------
# DATA FETCHING
# --------------------------------------------------
@st.cache_data(ttl=900)
def fetch_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    stations = []

    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                dr = requests.get(
                    f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
                ).json()

                if (
                    dr.get("status") == "ok"
                    and "pm10" in dr["data"].get("iaqi", {})
                ):
                    stations.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"]
                    })
                time.sleep(0.1)

        return pd.DataFrame(stations)

    except Exception as e:
        st.error(f"Connection Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# HYBRID MODEL VALIDATION (LOOCV)
# --------------------------------------------------
def validate_hybrid_model(df):
    errors = []
    progress = st.progress(0)

    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]

        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(train[['lat', 'lon']], train['pm10'])

        train_residuals = train['pm10'] - rf.predict(train[['lat', 'lon']])

        try:
            ok = OrdinaryKriging(
                train.lon.values,
                train.lat.values,
                train_residuals.values,
                variogram_model="gaussian"
            )
            res_pred, _ = ok.execute(
                "points",
                [test.lon],
                [test.lat]
            )

            final_pred = (
                rf.predict([[test.lat, test.lon]])[0] + res_pred[0]
            )

        except:
            # fallback: RF-only prediction
            final_pred = rf.predict([[test.lat, test.lon]])[0]

        errors.append(abs(final_pred - test.pm10))
        progress.progress((i + 1) / len(df))

    progress.empty()
    return np.mean(errors)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📍 Lucknow PM10 Spatial Hybrid Model (RF–Kriging)")

st.sidebar.header("🛠 Model Controls")
opacity = st.sidebar.slider("Heatmap Opacity", 0.1, 1.0, 0.7)
weather_load = st.sidebar.slider(
    "Weather Impact Simulation (%)", 50, 200, 100
) / 100.0

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if st.sidebar.button("🚀 Run Hybrid Analysis"):

    df = fetch_pm10_data()

    if df.empty:
        st.warning("No PM10 data available.")
    else:
        with st.spinner("Running validation & spatial modeling..."):
            mae = validate_hybrid_model(df)

        # -------------------------------
        # FINAL HYBRID MODEL
        # -------------------------------
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42
        )
        rf.fit(df[['lat', 'lon']], df['pm10'])

        df['rf_trend'] = rf.predict(df[['lat', 'lon']])
        df['residuals'] = (df['pm10'] - df['rf_trend']) * weather_load

        grid_res = 100
        lats = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
        lons = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)

        ok = OrdinaryKriging(
            df.lon,
            df.lat,
            df['residuals'],
            variogram_model="gaussian"
        )
        z_res, _ = ok.execute("grid", lons, lats)

        lon_grid, lat_grid = np.meshgrid(lons, lats)
        rf_grid = rf.predict(
            np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        ).reshape(grid_res, grid_res)

        z_final = (rf_grid * weather_load) + z_res.T

        # -------------------------------
        # PLOTTING
        # -------------------------------
        transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3857", always_xy=True
        )

        xmin, ymin = transformer.transform(lons.min(), lats.min())
        xmax, ymax = transformer.transform(lons.max(), lats.max())
        xs, ys = transformer.transform(df.lon.values, df.lat.values)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        cx.add_basemap(
            ax,
            source=cx.providers.CartoDB.Positron,
            zoom=12
        )

        im = ax.imshow(
            z_final,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            cmap="YlOrRd",
            alpha=opacity,
            zorder=2
        )

        ax.scatter(xs, ys, c="black", s=50,
                   edgecolors="white", zorder=3)

        plt.colorbar(im, label="Predicted PM10 (µg/m³)")
        ax.set_axis_off()

        col1, col2 = st.columns([3, 1])

        with col1:
            st.pyplot(fig)

        with col2:
            st.metric("Hybrid MAE (µg/m³)", f"{mae:.2f}")
            st.metric("Mean PM10", f"{z_final.mean():.1f}")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")

            st.download_button(
                "🖼️ Download PM10 Map",
                data=buf.getvalue(),
                file_name="lucknow_pm10_rf_kriging.png",
                mime="image/png"
            )

            st.write("**Monitoring Stations**")
            st.dataframe(df[['name', 'pm10']])

st.caption(
    "Hybrid Random Forest–Kriging framework for real-time PM10 spatial assessment over Lucknow."
)
