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
    st.error(f"Missing dependency: {e.name}. Did you add it to requirements.txt?")
    st.stop()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="PM10 Spatial ML Tool", layout="wide")

st.title("PM10 Spatial Analysis & Machine Learning Tool")

st.markdown("""
This tool integrates **environmental monitoring data**, **GIS shapefiles**,  
**machine learning**, and **SHAP explainability** for PM10 predictions.
""")

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def fuzzy_match_ids(csv_ids, shp_ids, threshold=85):
    mapping = {}
    for cid in csv_ids:
        # returns (match, score, index)
        match_data = process.extractOne(str(cid), [str(x) for x in shp_ids], scorer=fuzz.token_sort_ratio)
        if match_data and match_data[1] >= threshold:
            mapping[cid] = match_data[0]
    return mapping

# --------------------------------------------------
# SIDEBAR / UPLOAD
# --------------------------------------------------
st.sidebar.header("Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

if csv_file and shp_files:
    try:
        # Load & Clean CSV
        df = pd.read_csv(csv_file)
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
        
        # Map hour type
        hour_map = {"offpeak": 0, "peak": 1}
        df["hour_type_num"] = df["hour_type"].astype(str).str.strip().str.lower().map(hour_map)
        df = df.dropna(subset=["traffic", "hour_type_num", "pm10"])

        # Load Shapefile via Temp Directory
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()

        # Fuzzy Join
        id_map = fuzzy_match_ids(df["location_id"].unique(), gdf["location_id"].unique())
        df["matched_id"] = df["location_id"].map(id_map)
        data = gdf.merge(df.dropna(subset=["matched_id"]), left_on="location_id", right_on="matched_id")

        if data.empty:
            st.warning("No matches found between CSV and Shapefile IDs. Check your threshold or ID naming.")
            st.stop()

        # ML Pre-processing
        data["x"] = data.geometry.centroid.x
        data["y"] = data.geometry.centroid.y
        data["spatial_block"] = pd.qcut(data["x"], q=min(5, data["x"].nunique()), labels=False, duplicates="drop")

        X = data[["traffic", "hour_type_num"]]
        y = data["pm10"]
        
        # Spatial CV
        gkf = GroupKFold(n_splits=min(5, data["spatial_block"].nunique()))
        cv_r2 = []
        
        for train_idx, test_idx in gkf.split(X, y, groups=data["spatial_block"]):
            model_cv = RandomForestRegressor(n_estimators=100, random_state=42)
            model_cv.fit(X.iloc[train_idx], y.iloc[train_idx])
            cv_r2.append(r2_score(y.iloc[test_idx], model_cv.predict(X.iloc[test_idx])))

        # Final Model
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        data["Predicted_PM10"] = model.predict(X)

        # UI Results
        st.success("Analysis Complete")
        col1, col2 = st.columns(2)
        col1.metric("Mean Spatial R²", f"{np.mean(cv_r2):.2f}")
        col2.metric("Total Data Points", len(data))

        # Spatial Map
        st.subheader("PM10 Prediction Map")
        fig, ax = plt.subplots()
        data.plot(column="Predicted_PM10", cmap="viridis", legend=True, ax=ax)
        ax.set_axis_off()
        st.pyplot(fig)

        # SHAP Plot
        st.subheader("Feature Importance (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        fig_shap, ax_shap = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig_shap)

    except Exception as e:
        st.error(f"Critical Error: {e}")
else:
    st.info("Please upload your data files to begin.")
