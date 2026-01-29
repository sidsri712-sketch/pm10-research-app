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
from sklearn.linear_model import LinearRegression
import time
import io
import os
from streamlit_autorefresh import st_autorefresh
import datetime # Import the datetime module
import matplotlib.patches as mpatches # Import for custom legend patches

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"
DB_FILE = "lucknow_pm10_history.csv"

st.set_page_config(
    page_title="Lucknow PM10 Hybrid Spatial Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 30 minutes
count = st_autorefresh(interval=1800000, key="fizzbuzz")

# --------------------------------------------------
# DATA PIPELINE (HISTORICAL STORAGE)
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
                        "name": dr["data"]["city"]["name"],
                        "timestamp": pd.Timestamp.now()
                    })
                time.sleep(0.1)

        df_live = pd.DataFrame(stations)

        if not df_live.empty:
            if os.path.exists(DB_FILE):
                df_history = pd.read_csv(DB_FILE)
                df_combined = pd.concat([df_history, df_live], ignore_index=True)
                df_combined.drop_duplicates(subset=['lat', 'lon', 'pm10'], inplace=True)
                df_combined.to_csv(DB_FILE, index=False)
            else:
                df_live.to_csv(DB_FILE, index=False)

            df_display = df_live.groupby(['lat', 'lon']).agg({
                'pm10': 'mean',
                'name': 'first'
            }).reset_index()
            return df_display

        return pd.DataFrame()
    except Exception as e:
        st.error(f"API Error: {e}")
        return pd.DataFrame()

# --------------------------------------------------
# LOOCV + DIAGNOSTICS
# --------------------------------------------------
def run_diagnostics(df):
    results = []
    for i in range(len(df)):
        train = df.drop(i)
        test = df.iloc[i]
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(train[['lat', 'lon']], train['pm10'])
        residuals = train['pm10'] - rf.predict(train[['lat', 'lon']])

        if len(train) > 1:
            lat_min_train, lat_max_train = train['lat'].min(), train['lat'].max()
            lon_min_train, lon_max_train = train['lon'].min(), train['lon'].max()
            approx_max_dist_deg_train = np.sqrt((lat_max_train - lat_min_train)**2 + (lon_max_train - lon_min_train)**2)
            kriging_range_diag = max(0.01, approx_max_dist_deg_train * 0.4)
        else:
            kriging_range_diag = 0.1

        try:
            ok = OrdinaryKriging(
                train.lon, train.lat, residuals, variogram_model="gaussian", verbose=False,
                variogram_parameters={'sill': np.var(residuals), 'range': kriging_range_diag, 'nugget': 0.5}
            )
            p_res, _ = ok.execute("points", [test.lon], [test.lat])
            pred = rf.predict([[test.lat, test.lon]])[0] + p_res[0]
        except Exception as e:
            print(f"Kriging failed in diagnostics: {e}. Falling back to RF prediction.")
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

if os.path.exists(DB_FILE):
    df_hist_export = pd.read_csv(DB_FILE)
    st.sidebar.metric("Historical Samples", len(df_hist_export))

    csv_data = df_hist_export.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Historical CSV",
        data=csv_data,
        file_name='lucknow_pm10_history.csv',
        mime='text/csv',
    )

st.markdown("""
**Method:** Random Forest for spatial trend + Ordinary Kriging on residuals
**Purpose:** High-resolution PM10 surface with statistical validation
""")

st.sidebar.header("🛠 Controls")
opacity = st.sidebar.slider("Layer Transparency", 0.1, 1.0, 0.75)
weather_mult = st.sidebar.slider("Simulated Weather Factor (%)", 50, 200, 100) / 100

run_hybrid = st.sidebar.button("🚀 Run Hybrid Model")
run_diag = st.sidebar.button("📊 Run Full Diagnostic")

# Custom location prediction widgets
st.sidebar.markdown("---")
st.sidebar.subheader("Custom Location Prediction")
custom_lat = st.sidebar.number_input("Latitude", value=26.85, step=0.01, format="%.2f")
custom_lon = st.sidebar.number_input("Longitude", value=80.95, step=0.01, format="%.2f")

predict_custom = st.sidebar.button("🎯 Predict Custom PM10")

# Date input widgets for historical data filtering
st.sidebar.markdown("---")
st.sidebar.subheader("Historical Data Filter")

# Default to today and one week ago
today = datetime.date.today()
one_week_ago = today - datetime.timedelta(days=7)

start_date = st.sidebar.date_input('Start Date', value=one_week_ago)
end_date = st.sidebar.date_input('End Date', value=today)

# Initialize session state for custom prediction trigger
if 'custom_pm10_prediction' not in st.session_state:
    st.session_state.custom_pm10_prediction = None
if 'custom_pm10_lat' not in st.session_state:
    st.session_state.custom_pm10_lat = None
if 'custom_pm10_lon' not in st.session_state:
    st.session_state.custom_pm10_lon = None


# --------------------------------------------------
# RUN MODEL
# --------------------------------------------------
if run_hybrid or run_diag or predict_custom:
    df_live = fetch_pm10_data()
    if df_live.empty or len(df_live) < 3:
        st.warning("Not enough monitoring stations or data for analysis.")
        st.stop()

    if run_diag:
        st.subheader("📊 Model Diagnostics")
        res_df, mae = run_diagnostics(df_live)
        col1, col2 = st.columns(2)
        with col1:
            fig1, ax1 = plt.subplots()
            sns.regplot(data=res_df, x="Actual", y="Predicted", ax=ax1, color="teal")
            ax1.set_title("Actual vs Predicted PM10")
            st.pyplot(fig1)
            st.metric("Mean Absolute Error", f"{mae:.2f} µg/m³")
        with col2:
            fig2, ax2 = plt.subplots()
            sns.histplot(df_live['pm10'], kde=True, ax=ax2, color="orange")
            ax2.set_title("PM10 Distribution")
            st.pyplot(fig2)

    # ---------- FINAL HYBRID SURFACE ----------
    st.subheader("🗺 High-Resolution PM10 Surface")

    if os.path.exists(DB_FILE):
        df_train = pd.read_csv(DB_FILE)
    else:
        df_train = df_live

    rf_final = RandomForestRegressor(n_estimators=1000, max_depth=5, random_state=42)
    rf_final.fit(df_train[['lat', 'lon']], df_train['pm10'])

    df_live['residuals'] = (df_live['pm10'] - rf_final.predict(df_live[['lat', 'lon']])) * weather_mult

    grid_res = 200
    lats = np.linspace(df_live.lat.min()-0.06, df_live.lat.max()+0.06, grid_res)
    lons = np.linspace(df_live.lon.min()-0.06, df_live.lon.max()+0.06, grid_res)

    if len(df_live) > 1:
        lat_min, lat_max = df_live['lat'].min(), df_live['lat'].max()
        lon_min, lon_max = df_live['lon'].min(), df_live['lon'].max()
        max_spatial_extent = np.sqrt((lat_max - lat_min)**2 + (lon_max - lon_min)**2)
        kriging_range_dynamic = max(0.01, max_spatial_extent * 0.5)
    else:
        kriging_range_dynamic = 0.1

    try:
        OK = OrdinaryKriging(
            df_live.lon, df_live.lat, df_live['residuals'],
            variogram_model="gaussian",
            variogram_parameters={'sill': np.var(df_live['residuals']), 'range': kriging_range_dynamic, 'nugget': 0.5}
        )
        z_res, _ = OK.execute("grid", lons, lats)
    except Exception as e:
        st.warning(f"Kriging failed for surface generation: {e}. Displaying RF trend only.")
        z_res = np.zeros((grid_res, grid_res))


    lon_g, lat_g = np.meshgrid(lons, lats)
    rf_trend = rf_final.predict(np.column_stack([lat_g.ravel(), lon_g.ravel()])).reshape(grid_res, grid_res)

    z_final = (rf_trend * weather_mult) + z_res.T

    z_final[z_final < 0] = 0

    # ---------- CUSTOM PREDICTION ----------
    if predict_custom:
        st.session_state.custom_pm10_lat = custom_lat
        st.session_state.custom_pm10_lon = custom_lon

        # Predict RF trend for custom location
        custom_rf_pred = rf_final.predict([[custom_lat, custom_lon]])[0]

        # Predict Kriging residual for custom location
        try:
            custom_res_pred, _ = OK.execute("points", [custom_lon], [custom_lat])
            custom_krig_pred = custom_res_pred[0]
        except Exception as e:
            st.warning(f"Kriging for custom location failed: {e}. Using RF trend only for custom prediction.")
            custom_krig_pred = 0.0 # Fallback to 0 residual

        st.session_state.custom_pm10_prediction = custom_rf_pred + custom_krig_pred

    # Display custom prediction if available
    if st.session_state.custom_pm10_prediction is not None:
        st.markdown(f"### Predicted PM10 at Custom Location ({st.session_state.custom_pm10_lat:.2f}, {st.session_state.custom_pm10_lon:.2f}):")
        st.metric("PM10 Value", f"{st.session_state.custom_pm10_prediction:.2f} µg/m³")

    # ---------- MAP ----------
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin = transformer.transform(lons.min(), lats.min())
    xmax, ymax = transformer.transform(lons.max(), lats.max())
    xs, ys = transformer.transform(df_live.lon.values, df_live.lat.values)

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

    # Add custom location marker if available
    if st.session_state.custom_pm10_prediction is not None:
        custom_x, custom_y = transformer.transform(st.session_state.custom_pm10_lon, st.session_state.custom_pm10_lat)
        ax.scatter(custom_x, custom_y, c="red", marker="X", s=200, zorder=4, label="Custom Location")
        ax.text(custom_x, custom_y, f"{st.session_state.custom_pm10_prediction:.1f}", color='white', fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    ax.legend() # Show legend after adding custom location
    ax.set_axis_off()

    # HEALTH SCALE LEGEND - Defined categories and colors
    health_categories = [
        {'label': '0-50: Good', 'color': '#00e400'},
        {'label': '51-100: Satisfactory', 'color': '#ffff00'},
        {'label': '101-250: Moderate', 'color': '#ff7e00'},
        {'label': '251-350: Poor', 'color': '#ff0000'},
        {'label': '351-430: Very Poor', 'color': '#99004c'},
        {'label': '430+: Severe', 'color': '#7e0023'}
    ]

    # Create legend handles
    legend_patches = []
    for category in health_categories:
        legend_patches.append(mpatches.Patch(color=category['color'], label=category['label']))

    # Add custom legend to the map
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), title="PM10 Health Scale", facecolor='white', framealpha=0.8)

    st.pyplot(fig)


    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, format="png")
    st.download_button("💾 Download Map Image", buf.getvalue(), "lucknow_pm10_hybrid.png", "image/png")

    st.subheader("📌 Monitoring Stations")
    st.dataframe(df_live[['name', 'pm10']], use_container_width=True)

    # --- 24-HOUR TREND + FORECAST ---
    if os.path.exists(DB_FILE):
        st.subheader("📈 24-Hour Trend & 3-Hour Forecast")
        df_trend = pd.read_csv(DB_FILE)
        df_trend['timestamp'] = pd.to_datetime(df_trend['timestamp'])

        # Filter historical data based on selected dates
        df_filtered = df_trend[(df_trend['timestamp'].dt.date >= start_date) & (df_trend['timestamp'].dt.date <= end_date)].copy()

        if len(df_filtered) > 5:
            df_resampled = df_filtered.set_index('timestamp').resample('H').mean(numeric_only=True).dropna()

            # Predictive Logic: Linear Regression on the last few hours
            X = np.array(range(len(df_resampled))).reshape(-1, 1)
            y = df_resampled['pm10'].values
            model = LinearRegression().fit(X, y)

            # Forecast next 3 hours
            future_indices = np.array(range(len(df_resampled), len(df_resampled) + 3)).reshape(-1, 1)
            future_preds = model.predict(future_indices)

            # Generate future timestamps
            last_time = df_resampled.index[-1]
            future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 4)]

            fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
            ax_trend.plot(df_resampled.index, df_resampled['pm10'], marker='o', color='crimson', label="Observed Trend")
            ax_trend.plot(future_times, future_preds, marker='x', linestyle='--', color='gray', label="3hr Forecast")

            ax_trend.set_ylabel("Avg PM10 (µg/m³)")
            ax_trend.legend()
            ax_trend.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig_trend)
        else:
            st.info("Accumulating data for forecast or not enough data for selected date range. Please select a different date range or wait for more refresh cycles.")

st.caption("Data: WAQI API | Method: Random Forest Residual Kriging (RFRK)")
