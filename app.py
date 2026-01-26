import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from scipy.interpolate import griddata
import io
import tempfile
import os

# --- SAFETY CHECK ---
try:
    from rapidfuzz import process, fuzz
except ImportError:
    st.error("Please ensure 'rapidfuzz', 'contextily', and 'scipy' are in your requirements.txt")
    st.stop()

st.set_page_config(page_title="PM10 Spatial Analysis", layout="wide")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📁 Step 1: Data Upload")
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

        # 2. SMART ID MATCHING
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid']), gdf.columns[0])
        
        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf = gdf.rename(columns={shp_id_col: 'location_id'})
        
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        # 3. FUZZY JOIN
        with st.spinner("Processing spatial alignment..."):
            id_map = {cid: process.extractOne(str(cid), [str(x) for x in gdf["location_id"]])[0] for cid in df["location_id"].unique()}
            df["matched_id"] = df["location_id"].map(id_map)
            data = gdf.merge(df.dropna(subset=["pm10"]), left_on="location_id", right_on="matched_id")

        # 4. IDW CALCULATION
        known_x = data.geometry.centroid.x.values
        known_y = data.geometry.centroid.y.values
        known_z = data["pm10"].values
        grid_x = gdf.geometry.centroid.x.values
        grid_y = gdf.geometry.centroid.y.values

        grid_z = griddata((known_x, known_y), known_z, (grid_x, grid_y), method='linear')
        nan_mask = np.isnan(grid_z)
        grid_z[nan_mask] = griddata((known_x, known_y), known_z, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        gdf["IDW_PM10"] = grid_z

        # 5. JOURNAL-STYLE MAPPING
        st.subheader("Final Spatial Prediction Map")
        
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plotting with the vibrant 'RdYlGn_r' scale
        gdf_web.plot(
            column="IDW_PM10", 
            cmap="RdYlGn_r", 
            legend=True, 
            legend_kwds={'label': "PM10 (µg/m³)", 'shrink': 0.5},
            ax=ax, 
            markersize=35, # Increased size for fuller coverage
            alpha=0.6,     # Transparency to see the satellite map below
            edgecolor='none'
        )
        
        # VISUALLY APPEALING BASEMAP: Esri World Imagery (Satellite)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. EXPORT
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("🖼️ Download Map (PNG)", buf.getvalue(), "pm10_map.png", "image/png")
        with col2:
            st.success("Analysis complete. Use the PNG for reports and the CSV for raw data.")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your files in the sidebar to generate the high-resolution map.")
