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
    st.error(f"**Missing Library:** {e.name}. Please ensure your `requirements.txt` includes: streamlit, pandas, geopandas, scikit-learn, rapidfuzz, shap, and pyproj.")
    st.stop()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="PM10 Spatial ML Tool", layout="wide")

st.title("PM10 Spatial Analysis & Machine Learning Tool")

st.info("""
**How to use this tool:**
1. **Upload a CSV** containing `location_id`, `traffic`, `pm10`, and `hour_type`.
2. **Upload Shapefile components** (.shp, .shx, .dbf) containing a matching `location_id` column.
3. The tool will automatically align the data and run the Spatial ML model.
""")

# --------------------------------------------------
# SIDEBAR / UPLOAD
# --------------------------------------------------
st.sidebar.header("Step 1: Upload Data")
csv_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if csv_file and shp_files:
    try:
        # 1. LOAD CSV
        df = pd.read_csv(csv_file)
        
        # 2. LOAD SHAPEFILE
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 3. USER-FRIENDLY COLUMN CHECK
        csv_cols = df.columns.tolist()
        gdf_cols = gdf.columns.tolist()

        if 'location_id' not in csv_cols or 'location_id' not in gdf_cols:
            st.error("### ❌ Column Error")
            st.write("Both files must have a column named exactly **'location_id'**.")
            col1, col2 = st.columns(2)
            col1.write(f"**Your CSV Columns:** {csv_cols}")
            col2.write(f"**Your Shapefile Columns:** {gdf_cols}")
            st.stop()

        # 4. DATA CLEANING
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
        
        # Map hour type
        hour_map = {"offpeak": 0, "peak": 1}
        df["hour_type_num"] = df["hour_type"].astype(str).str.strip().str.lower().map(hour_map)
        df = df.dropna(subset=["traffic", "hour_type_num", "pm10"])

        # 5. FUZZY JOINING
        with st.status("Aligning spatial data...", expanded=False) as status:
            csv_ids = df["location_id"].unique()
            shp_ids = gdf["location_id"].unique()
            
            mapping = {}
            for cid in csv_ids:
                match = process.extractOne(str(cid), [str(x) for x in shp_ids], scorer=fuzz.token_sort_ratio)
                if match and match[1] >= 85:
                    mapping[cid] = match[0]
            
            df["matched_id"] = df["location_id"].map(mapping)
            data = gdf.merge(df.dropna(subset=["matched_id"]), left_on="location_id", right_on="matched_id")
            status.update(label="Data alignment complete!", state="complete")

        if data.empty:
            st.warning("No matches found. Check if the location IDs in your CSV resemble those in your Shapefile.")
            st.stop()

        # 6. ML PROCESSING
        data["x"] = data.geometry.centroid.x
        data["y"] = data.geometry.centroid.y
        data["spatial_block"] = pd.qcut(data["x"], q=min(5, data["x"].nunique()), labels=False, duplicates="drop")

        X = data[["traffic", "hour_type_num"]]
        y = data["pm10"]
        
        # Spatial CV
        gkf = GroupKFold(n_splits=min(5, data["spatial_block"].nunique()))
        cv_r2 = [r2_score(y.iloc[te], RandomForestRegressor(n_estimators=100).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])) 
                 for tr, te in gkf.split(X, y, groups=data["spatial_block"])]

        # Final Model
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        data["Predicted_PM10"] = model.predict(X)

        # 7. RESULTS DISPLAY
        st.success("🎉 Analysis Successful!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Spatial R² (Accuracy)", f"{np.mean(cv_r2):.2f}")
        m2.metric("Locations Analyzed", len(data))
        m3.metric("Avg Predicted PM10", f"{data['Predicted_PM10'].mean():.1f}")

        tab1, tab2, tab3 = st.tabs(["🗺️ Prediction Map", "📊 Feature Importance", "📑 Raw Results"])

        with tab1:
            fig, ax = plt.subplots(figsize=(10, 6))
            data.plot(column="Predicted_PM10", cmap="YlOrRd", legend=True, ax=ax, edgecolor="black", linewidth=0.5)
            ax.set_axis_off()
            st.pyplot(fig)

        with tab2:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            fig_shap, _ = plt.subplots()
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig_shap)

        with tab3:
            st.dataframe(data[["location_id", "traffic", "hour_type", "pm10", "Predicted_PM10"]])

    except Exception as e:
        st.error(f"### ⚠️ An error occurred during analysis")
        st.info(f"Details: {e}")
        st.write("Please ensure your CSV has these columns: `location_id`, `traffic`, `pm10`, `hour_type`.")
else:
    st.write("---")
    st.write("👈 **Please upload your files in the sidebar to get started.**")
