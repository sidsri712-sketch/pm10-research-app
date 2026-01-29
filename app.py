import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
import time
import io
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(
    page_title="Lucknow PM10 Hybrid Spatial Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 30 minutes
count = st_autorefresh(interval=1800000, key="fizzbuzz")

# --------------------------------------------------
# DATA PIPELINE (UNCHANGED)
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
                if dr.get("status") == "ok" and "pm10" in dr["data"].get("iaqi", {}):
                    stations.append({
                        "lat": s["lat"],
                        "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"]
                    })
                time.sleep(0.1)
        df = pd.DataFrame(stations)
        if not df.empty:
            df = df.groupby(['lat', 'lon']).agg({
                'pm10': 'mean',
                'name': 'first'
            }).reset_index()
        return df
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# LOOCV + DIAGNOSTICS (UNCHANGED)
# --------------------------------------------------
def run_diagnostics(df):
    results = []
    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(train[['lat', 'lon']], train['pm10'])
        residuals = train['pm10'] - rf.predict(train[['lat', 'lon']])
        try:
            ok = OrdinaryKriging(train.lon, train.lat, residuals, variogram_model="gaussian", verbose=False)
            p_res, _ = ok.execute("points", [test.lon], [test.lat])
            pred = rf.predict([[test.lat, test.lon]])[0] + p_res[0]
        except:
            pred = rf.predict([[test.lat, test.lon]])[0]
        results.append({"Actual": test.pm10, "Predicted": pred})
    res_df = pd.DataFrame(results)
    mae = np.mean(np.abs(res_df['Actual'] - res_df['Predicted']))
    return res_df, mae

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")
st.sidebar.caption(f"Last data refresh: {time.strftime('%H:%M:%S')}")

st.markdown("""
**Method:** Random Forest for spatial trend + Ordinary Kriging on residuals  
**Purpose:** High-resolution PM10 surface with statistical validation
""")

st.sidebar.header("🛠 Controls")
opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.75)
weather_mult = st.sidebar.slider("Simulated Weather Factor (%)", 50, 200, 100) / 100

run_hybrid = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Full Diagnostic")

# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_hybrid or run_diag:
    df = fetch_pm10_data()
    if df.empty or len(df) < 3:
        st.warning("Not enough monitoring stations.")
        st.stop()

    if run_diag:
        st.subheader("📊 Model Diagnostics")
        res_df, mae = run_diagnostics(df)
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.regplot(data=res_df, x="Actual", y="Predicted", ax=ax1, color="teal")
            ax1.set_title("Actual vs Predicted PM10")
            st.pyplot(fig1)
            st.metric("Mean Absolute Error", f"{mae:.2f} µg/m³")
        with col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(df['pm10'], kde=True, ax=ax2, color="orange")
            ax2.set_title("PM10 Distribution")
            st.pyplot(fig2)

    # ---------- FINAL HYBRID SURFACE ----------
    st.subheader("🗺 High-Resolution PM10 Surface")

    # FIX 1: Max depth 1 removes the sharp "cross" splits from the Random Forest
    rf_final = RandomForestRegressor(n_estimators=1000, max_depth=1, random_state=42)
    rf_final.fit(df[['lat', 'lon']], df['pm10'])

    df['residuals'] = (df['pm10'] - rf_final.predict(df[['lat', 'lon']])) * weather_mult

    grid_res = 200 
    lats = np.linspace(df.lat.min()-0.06, df.lat.max()+0.06, grid_res)
    lons = np.linspace(df.lon.min()-0.06, df.lon.max()+0.06, grid_res)

    # FIX 2 & 3: Using a wider range and nugget to "smudge" the influence areas into smooth circles
    v_range = 0.2 * (2 - weather_mult) 
    
    OK = OrdinaryKriging(
        df.lon, df.lat, df['residuals'],
        variogram_model="gaussian",
        variogram_parameters={'sill': np.var(df['residuals']), 'range': v_range, 'nugget': 0.5}
    )
    z_res, _ = OK.execute("grid", lons, lats)

    lon_g, lat_g = np.meshgrid(lons, lats)
    rf_trend = rf_final.predict(np.column_stack([lat_g.ravel(), lon_g.ravel()])).reshape(grid_res, grid_res)

    z_final = (rf_trend * weather_mult) + z_res.T

    # ---------- MAP ----------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    xs, ys = transformer.transform(df.lon.values, df.lat.values)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)

    im = ax.imshow(
        z_final,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="magma",
        alpha=opacity,
        interpolation="bicubic" 
    )

    ax.scatter(xs, ys, c="white", edgecolors="black", s=70, zorder=3, label="Stations")
    plt.colorbar(im, label="PM10 (µg/m³)")
    ax.set_axis_off()
    st.pyplot(fig)

    # PM10 SAFETY LEGEND
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌡 PM10 Health Scale")
    st.sidebar.info("""
    - **0-50**: Good
    - **51-100**: Satisfactory
    - **101-250**: Moderate
    - **251-350**: Poor
    - **351-430**: Very Poor
    - **430+**: Severe
    """)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    st.download_button("💾 Download Map", buf.getvalue(), "lucknow_pm10_hybrid.png", "image/png")

    st.subheader("📌 Monitoring Stations")
    st.dataframe(df[['name', 'pm10']], use_container_width=True)

st.caption("Data: WAQI API | Method: Random Forest Residual Kriging (RFRK)")
