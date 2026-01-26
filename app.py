import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import io

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

st.markdown("""
### 🔬 Publication-Ready ML Pipeline
This tool automates the alignment of monitoring data with GIS geometries, performs spatial cross-validation, 
and generates explainable AI (SHAP) insights.
""")

# --------------------------------------------------
# SIDEBAR: DATA UPLOAD
# --------------------------------------------------
st.sidebar.header("📁 Step 1: Upload Data")
csv_file = st.sidebar.file_uploader("Upload Monitoring CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
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

        # 2. SMART ID RESOLUTION (Fixes the Index/Column Errors)
        # Find the ID column in CSV
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id']), df.columns[0])
        # Find the ID column in Shapefile
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid']), gdf.columns[0])

        # Rename to a standard for internal join
        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf = gdf.rename(columns={shp_id_col: 'location_id'})

        # Clean strings
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        # Numeric checks
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
        df["hour_type_num"] = df["hour_type"].astype(str).str.strip().str.lower().map({"offpeak": 0, "peak": 1})
        df = df.dropna(subset=["traffic", "hour_type_num", "pm10"])

        # 3. FUZZY JOINING
        with st.spinner("🔄 Aligning locations..."):
            csv_ids = df["location_id"].unique()
            shp_ids = gdf["location_id"].unique()
            id_map = {cid: process.extractOne(str(cid), [str(x) for x in shp_ids], scorer=fuzz.token_sort_ratio)[0] for cid in csv_ids}
            df["matched_id"] = df["location_id"].map(id_map)
            data = gdf.merge(df.dropna(subset=["matched_id"]), left_on="location_id", right_on="matched_id")

        if data.empty:
            st.error("❌ No matches found between CSV and Shapefile. Please check your IDs.")
            st.stop()

        # 4. SPATIAL ML ENGINE
        data["x"] = data.geometry.centroid.x
        data["y"] = data.geometry.centroid.y
        # Create 5 spatial blocks for cross-validation
        data["spatial_block"] = pd.qcut(data["x"], q=min(5, data["x"].nunique()), labels=False, duplicates="drop")

        X = data[["traffic", "hour_type_num"]]
        y = data["pm10"]
        
        # Spatial CV
        gkf = GroupKFold(n_splits=min(5, data["spatial_block"].nunique()))
        cv_scores = []
        for tr, te in gkf.split(X, y, groups=data["spatial_block"]):
            m = RandomForestRegressor(n_estimators=100, random_state=42).fit(X.iloc[tr], y.iloc[tr])
            cv_scores.append(r2_score(y.iloc[te], m.predict(X.iloc[te])))

        # Final Model
        model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
        data["Predicted_PM10"] = model.predict(X)

        # 5. RESULTS DASHBOARD
        st.success(f"✅ Success! Analysis completed for {len(data)} locations.")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Spatial R² (CV Accuracy)", f"{np.mean(cv_scores):.2f}")
        col2.metric("Mean PM10 Predicted", f"{data['Predicted_PM10'].mean():.1f} µg/m³")
        col3.metric("Spatial Blocks", data["spatial_block"].nunique())

        tabs = st.tabs(["🗺️ Spatial Prediction Map", "📊 Model Explainability (SHAP)", "📥 Export Results"])

        with tabs[0]:
            st.subheader("PM10 Spatial Distribution")
            fig_map, ax = plt.subplots(figsize=(12, 8))
            data.plot(column="Predicted_PM10", cmap="YlOrRd", legend=True, 
                      legend_kwds={'label': "PM10 Concentration (µg/m³)"},
                      ax=ax, edgecolor="black", linewidth=0.5)
            ax.set_axis_off()
            st.pyplot(fig_map)

        with tabs[1]:
            
            st.subheader("What drives the predictions?")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            fig_shap, _ = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig_shap)

        with tabs[2]:
            st.subheader("Download Files")
            
            # Export CSV
            csv_data = data.drop(columns='geometry').to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Results (CSV)", csv_data, "pm10_results.csv", "text/csv")
            
            # Export Map Image
            buf = io.BytesIO()
            fig_map.savefig(buf, format="png", dpi=300)
            st.download_button("🖼️ Download Map (PNG)", buf.getvalue(), "pm10_map.png", "image/png")

    except Exception as e:
        st.error(f"### ⚠️ An Error Occurred")
        st.info(f"Technical Details: {e}")
else:
    st.write("---")
    st.write("👈 **Please upload your files in the sidebar to begin.**")
