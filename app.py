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

# --- CONFIGURATION ---
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Hybrid Diagnostic", layout="wide")

# --- DATA FETCHING (Same Functioning) ---
@st.cache_data(ttl=900)
def fetch_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    stations = []
    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                dr = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
                if dr.get("status") == "ok" and "pm10" in dr["data"].get("iaqi", {}):
                    stations.append({
                        "lat": s["lat"], "lon": s["lon"],
                        "pm10": dr["data"]["iaqi"]["pm10"]["v"],
                        "name": dr["data"]["city"]["name"]
                    })
                time.sleep(0.05)
        df = pd.DataFrame(stations)
        if not df.empty:
            df = df.groupby(['lat', 'lon']).mean().reset_index() # Safety against duplicate coords
        return df
    except:
        return pd.DataFrame()

# --- THE RESULTS EXPLAINED ---
def display_explanation():
    with st.expander("📖 What do these results mean?"):
        st.markdown("""
        * **Hybrid MAE:** On average, how many units ($\mu\text{g/m}^3$) the model is 'off'. If it's 15, and PM10 is 150, that's a 10% error—very good for environmental data!
        * **Actual vs Predicted Plot:** The closer the dots are to the diagonal line, the more 'perfect' the model is.
        * **Spatial Heatmap:** We use **Bicubic Interpolation** to make the gradients smooth. Areas of deep purple/red are simulated 'hotspots' based on local sensor residuals.
        """)

# --- MAIN APP ---
st.title("🌪️ Lucknow PM10: Hybrid RFRK Diagnostic")
display_explanation()

if st.sidebar.button("🚀 Run Full Diagnostic"):
    df = fetch_pm10_data()
    
    if df.empty or len(df) < 3:
        st.error("Not enough station data found to run Kriging.")
    else:
        # 1. CROSS VALIDATION (The MAE Logic)
        cv_results = []
        progress = st.progress(0)
        for i in range(len(df)):
            train, test = df.drop(i), df.iloc[i]
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(train[['lat', 'lon']], train['pm10'])
            res = train['pm10'] - rf.predict(train[['lat', 'lon']])
            try:
                ok = OrdinaryKriging(train.lon, train.lat, res, variogram_model="gaussian")
                p_res, _ = ok.execute("points", [test.lon], [test.lat])
                pred = rf.predict([[test.lat, test.lon]])[0] + p_res[0]
            except:
                pred = rf.predict([[test.lat, test.lon]])[0]
            cv_results.append({"Actual": test.pm10, "Predicted": pred})
            progress.progress((i + 1) / len(df))
        
        res_df = pd.DataFrame(cv_results)
        mae = np.mean(np.abs(res_df['Actual'] - res_df['Predicted']))

        # 2. FINAL GRID GENERATION
        rf_final = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf_final.fit(df[['lat', 'lon']], df['pm10'])
        df['res'] = df['pm10'] - rf_final.predict(df[['lat', 'lon']])
        
        grid_res = 120
        lats = np.linspace(df.lat.min() - 0.02, df.lat.max() + 0.02, grid_res)
        lons = np.linspace(df.lon.min() - 0.02, df.lon.max() + 0.02, grid_res)
        lon_g, lat_g = np.meshgrid(lons, lats)
        
        ok_final = OrdinaryKriging(df.lon, df.lat, df['res'], variogram_model="gaussian")
        z_res, _ = ok_final.execute("grid", lons, lats)
        rf_grid = rf_final.predict(np.column_stack([lat_g.ravel(), lon_g.ravel()])).reshape(grid_res, grid_res)
        z_final = rf_grid + z_res.T

        # 3. VISUALIZATION
        col_main, col_side = st.columns([2, 1])
        
        with col_main:
            st.subheader("High-Resolution Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xmin, ymin = transformer.transform(lons.min(), lats.min())
            xmax, ymax = transformer.transform(lons.max(), lats.max())
            
            # Using 'magma' for high-contrast 'appealing' visuals
            im = ax.imshow(z_final, extent=[xmin, xmax, ymin, ymax], 
                           origin="lower", cmap="magma", alpha=0.7, interpolation='bicubic')
            cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=12)
            
            xs, ys = transformer.transform(df.lon.values, df.lat.values)
            ax.scatter(xs, ys, c="cyan", edgecolors="white", s=80, label="Sensors", zorder=5)
            plt.colorbar(im, label="PM10 Concentration")
            ax.set_axis_off()
            st.pyplot(fig)

        with col_side:
            st.metric("Hybrid MAE", f"{mae:.2f}")
            st.metric("Avg Lucknow PM10", f"{z_final.mean():.1f}")
            
            st.subheader("Accuracy Check")
            fig_acc, ax_acc = plt.subplots()
            sns.scatterplot(data=res_df, x="Actual", y="Predicted", color="orange", ax=ax_acc)
            # Perfect prediction line
            line_max = max(res_df.max())
            ax_acc.plot([0, line_max], [0, line_max], 'k--', alpha=0.5)
            st.pyplot(fig_acc)
            
            st.write("**Station Raw Data**")
            st.dataframe(df[['name', 'pm10']], height=200)

st.caption("Integrated RFRK Model: Capturing Regional Trends + Local Residuals.")
