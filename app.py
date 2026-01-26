import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import io
import contextily as cx # For the basemap

# --- SAFETY CHECK: Dependencies ---
try:
    from rapidfuzz import process, fuzz
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import r2_score
except ImportError as e:
    st.error(f"**Missing Library:** {e.name}. Ensure 'contextily' and 'pyproj' are in requirements.txt.")
    st.stop()

st.set_page_config(page_title="PM10 Spatial ML Tool", layout="wide")
st.title("PM10 Spatial Analysis & Machine Learning Tool")

# --------------------------------------------------
# SIDEBAR: DATA UPLOAD
# --------------------------------------------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("Upload Monitoring CSV", type=["csv"])
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

        # 2. SMART ID RESOLUTION 
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid']), gdf.columns[0])

        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf = gdf.rename(columns={shp_id_col: 'location_id'})

        # Standardize strings
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        # Numeric checks
        df["traffic"] = pd.to_numeric(df["traffic"], errors="coerce")
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
        df["hour_type_num"] = df["hour_type"].astype(str).str.strip().str.lower().map({"offpeak": 0, "peak": 1})
        df = df.dropna(subset=["traffic", "hour_type_num", "pm10"])

        # 3. FUZZY JOINING
        csv_ids = df["location_id"].unique()
        shp_ids = gdf["location_id"].unique()
        id_map = {cid: process.extractOne(str(cid), [str(x) for x in shp_ids], scorer=fuzz.token_sort_ratio)[0] for cid in csv_ids}
        df["matched_id"] = df["location_id"].map(id_map)
        data = gdf.merge(df.dropna(subset=["matched_id"]), left_on="location_id", right_on="matched_id")

        # 4. ML ENGINE
        data["x"] = data.geometry.centroid.x
        data["y"] = data.geometry.centroid.y
        X = data[["traffic", "hour_type_num"]]
        y = data["pm10"]
        
        model = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
        # Apply prediction to the FULL shapefile grid to fill the map
        full_X = gdf.merge(df[['traffic', 'hour_type_num', 'matched_id']], left_on='location_id', right_on='matched_id', how='left')
        full_X = full_X.fillna(full_X.mean(numeric_only=True)) # Handle grid points without direct data
        gdf["Predicted_PM10"] = model.predict(full_X[["traffic", "hour_type_num"]])

        # 5. RESULTS DASHBOARD
        st.success(f"✅ Analysis complete for {len(data)} stations across the grid.")
        
        tabs = st.tabs(["🗺️ Spatial Map", "📊 Analysis", "📥 Export"])

        with tabs[0]:
            st.subheader("PM10 Spatial Distribution with Basemap")
            
            # Ensure CRS is Web Mercator for the basemap
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True) # Assume Lat/Lon if missing
            gdf_web = gdf.to_crs(epsg=3857)

            fig_map, ax = plt.subplots(figsize=(12, 10))
            
            # Plot the points - smaller size (s=5) and no edges (lw=0) makes it look like a smooth heatmap
            gdf_web.plot(column="Predicted_PM10", cmap="YlOrRd", legend=True, 
                         legend_kwds={'label': "PM10 (µg/m³)", 'shrink': 0.6},
                         ax=ax, markersize=10, alpha=0.7, edgecolor='none')
            
            # Add the basemap (OpenStreetMap)
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik)
            
            ax.set_axis_off()
            st.pyplot(fig_map)

        with tabs[1]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            fig_shap, _ = plt.subplots(figsize=(10, 5))
            shap.summary_plot(shap_values, X, show=False)
            st.pyplot(fig_shap)

        with tabs[2]:
            buf = io.BytesIO()
            fig_map.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("🖼️ Download High-Res Map (PNG)", buf.getvalue(), "pm10_spatial_map.png", "image/png")

    except Exception as e:
        st.error(f"### ⚠️ Error: {e}")
else:
    st.info("👈 Upload your files in the sidebar to generate the spatial map.")
