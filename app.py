import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

# Try-except for rapidfuzz to provide a helpful UI error if requirements.txt is missed
try:
    from rapidfuzz import process, fuzz
except ImportError:
    st.error("Missing dependency: 'rapidfuzz'. Please add it to your requirements.txt.")
    st.stop()

import shap

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="PM10 Spatial ML Tool",
    layout="wide"
)

st.title("PM10 Spatial Analysis & Machine Learning Tool")

st.markdown("""
This tool integrates **environmental monitoring data**, **GIS shapefiles**,  
**machine learning**, **spatial cross-validation**, and **explainable AI (SHAP)** to generate **publication-ready PM10 predictions**.
""")

# --------------------------------------------------
# FUZZY MATCH FUNCTION
# --------------------------------------------------
def fuzzy_match_ids(csv_ids, shp_ids, threshold=85):
    mapping = {}
    for cid in csv_ids:
        # Returns (match_string, score, index)
        result = process.extractOne(
            str(cid), 
            [str(x) for x in shp_ids], 
            scorer=fuzz.token_sort_ratio
        )
        if result and result[1] >= threshold:
            mapping[cid] = result[0]
    return mapping

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
st.sidebar.header("Data Upload")

csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
shp_files = st.sidebar.file_uploader(
    "Upload Shapefile Set (.shp, .shx, .dbf)", 
    type=["shp", "shx", "dbf"], 
    accept_multiple_files=True
)

if csv_file and shp_files:
    try:
        # --- LOAD CSV ---
        df = pd.read_csv(csv_file)
        
        # Standardize ID column
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
        
        df["hour_type_num"] = df["hour_type"].astype(str).str.strip().str.lower().map({
            "offpeak": 0, "peak": 1
        })
        
        df = df.dropna(subset=["traffic", "hour_type_num", "pm10"])

        # --- LOAD SHAPEFILE ---
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()

        # --- FUZZY MATCHING ---
        csv_ids = df["location_id"].unique().tolist()
        shp_ids = gdf["location_id"].unique().tolist()
        id_map = fuzzy_match_ids(csv_ids, shp_ids)

        df["matched_id"] = df["location_id"].map(id_map)
        df = df.dropna(subset=["matched_id"])

        data = gdf.merge(df, left_on="location_id", right_on="matched_id", how="inner")

        if data.empty:
            st.error("No matching locations found. Check your Location IDs.")
            st.stop()

        # --- SPATIAL BLOCKING ---
        data["x"] = data.geometry.centroid.x
        data["y"] = data.geometry.centroid.y
        data["spatial_block"] = pd.qcut(data["x"], q=5, labels=False, duplicates="drop")

        # --- MODEL + SPATIAL CV ---
        X = data[["traffic", "hour_type_num"]]
        y = data["pm10"]
        groups = data["spatial_block"]

        gkf = GroupKFold(n_splits=min(5, len(groups.unique())))
        cv_r2, cv_rmse = [], []

        for train_idx, test_idx in gkf.split(X, y, groups):
            model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            cv_r2.append(r2_score(y.iloc[test_idx], preds))
            cv_rmse.append(mean_squared_error(y.iloc[test_idx], preds, squared=False))

        model.fit(X, y)
        data["Predicted_PM10"] = model.predict(X)

        # --- METRICS & VISUALS ---
        st.success("Model trained successfully")
        m1, m2, m3 = st.columns(3)
        m1.metric("Spatial CV Mean R²", f"{np.mean(cv_r2):.3f}")
        m2.metric("Spatial CV RMSE", f"{np.mean(cv_rmse):.2f}")
        m3.metric("Valid Locations", len(data))

        # Map
        st.subheader("Spatial Prediction Map")
        fig_map, ax = plt.subplots(figsize=(10, 8))
        data.plot(column="Predicted_PM10", cmap="YlOrRd", legend=True, ax=ax, edgecolor="black")
        ax.set_axis_off()
        st.pyplot(fig_map)

        # SHAP
        st.subheader("Model Explainability (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig_shap, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_shap)

    except Exception as e:
        st.error(f"Analysis failed: {e}")
else:
    st.info("Upload both CSV and Shapefile to start analysis.")
