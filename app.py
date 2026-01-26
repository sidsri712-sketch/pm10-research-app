import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# --- SAFETY CHECK: Dependencies ---
try:
    from rapidfuzz import process, fuzz
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError as e:
    st.error(f"**Missing Library:** {e.name}. Please ensure your `requirements.txt` is updated.")
    st.stop()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="PM10 Spatial ML Tool", layout="wide")

st.title("PM10 Spatial Analysis & Machine Learning Tool")

st.info("""
**Welcome! To get started:**
1. Upload your **CSV** with columns: `location_id`, `traffic`, `pm10`, and `hour_type`.
2. Upload your **Shapefile** (.shp, .shx, .dbf). 
3. The tool will automatically try to match your locations and run the model.
""")

# --------------------------------------------------
# SIDEBAR / UPLOAD
# --------------------------------------------------
st.sidebar.header("Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set", type=["shp", "shx", "dbf"], accept_multiple_files=True)

if csv_file and shp_files:
    try:
        # 1. LOAD DATA
        df = pd.read_csv(csv_file)
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 2. SMART COLUMN MATCHING
        # Check CSV
        if 'location_id' not in df.columns:
            st.error(f"CSV is missing 'location_id'. Found: {list(df.columns)}")
            st.stop()
        
        # Check Shapefile - Auto-rename 'id' to 'location_id' if found
        if 'location_id' not in gdf.columns:
            if 'id' in gdf.columns:
                gdf = gdf.rename(columns={'id': 'location_id'})
                st.toast("Auto-detected 'id' column in shapefile.")
            else:
                st.error(f"Shapefile needs a 'location_id' or 'id' column. Found: {list(gdf.columns)}")
                st.stop()

        # 3. CLEANING & JOINING
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        # Mapping hour_type
        hour_map = {"offpeak": 0, "peak": 1}
        df["hour_type_num"] = df["hour_type"].astype(str).str.strip().str.lower().map(hour_map)
        
        # Fuzzy Match IDs
        csv_ids = df["location_id"].unique()
        shp_ids = gdf["location_id"].unique()
        id_map = {cid: process.extractOne(str(cid), [str(x) for x in shp_ids])[0] for cid in csv_ids}
        
        df["matched_id"] = df["location_id"].map(id_map)
        data = gdf.merge(df.dropna(subset=["pm10"]), left_on="location_id", right_on="matched_id")

        if data.empty:
            st.warning("No matches found between datasets.")
            st.stop()

        # 4. ML ENGINE
        data["x"] = data.geometry.centroid.x
        data["y"] = data.geometry.centroid.y
        data["spatial_block"] = pd.qcut(data["x"], q=min(5, data["x"].nunique()), labels=False, duplicates="drop")

        X = data[["traffic", "hour_type_num"]]
        y = data["pm10"]
        
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        data["Predicted_PM10"] = model.predict(X)

        # 5. USER FRIENDLY DISPLAY
        st.success("✅ Analysis Complete!")
        
        tab1, tab2, tab3 = st.tabs(["🗺️ Spatial Map", "📊 Explainability", "📋 Data View"])
        
        with tab1:
            st.subheader("PM10 Prediction Map")
            fig, ax = plt.subplots(figsize=(10, 6))
            data.plot(column="Predicted_PM10", cmap="YlOrRd", legend=True, ax=ax, edgecolor="black")
            ax.set_axis_off()
            st.pyplot(fig)

        with tab2:
            st.subheader("Feature Importance (SHAP)")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            fig_shap, _ = plt.subplots()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig_shap)

        with tab3:
            st.dataframe(data[["location_id", "traffic", "hour_type", "pm10", "Predicted_PM10"]].head(20))

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.write("---")
    st.write("👈 **Please upload your files in the sidebar to begin.**")
