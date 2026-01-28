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

# --- DEPENDENCY CHECK ---
try:
    from rapidfuzz import process, fuzz
except ImportError:
    st.error("Missing libraries! Please run: pip install rapidfuzz contextily scipy geopandas")
    st.stop()

st.set_page_config(page_title="PM10 Solid Map Tool", layout="wide")

st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload Air Quality CSV", type=["csv"])
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

        # 2. COLUMN CLEANING (Fixes "int has no len" and "merge on int64" errors)
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        if not pm10_col:
            st.error("Could not find a 'PM10' column in your CSV.")
            st.stop()

        # Force all IDs to strings to prevent crash
        df[csv_id_col] = df[csv_id_col].astype(str).str.strip()
        gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.strip()

        # 3. ALIGN DATA
        with st.spinner("Aligning sensor data..."):
            shp_ids = gdf[shp_id_col].unique().tolist()
            id_map = {str(cid): process.extractOne(str(cid), shp_ids, scorer=fuzz.WRatio)[0] for cid in df[csv_id_col].unique()}
            df['matched_id'] = df[csv_id_col].map(id_map)
            merged = gdf.merge(df.dropna(subset=[pm10_col]), left_on=shp_id_col, right_on='matched_id')

        # 4. CREATE SOLID COLOR SURFACE (The "No Dots" fix)
        with st.spinner("Creating solid exposure surface..."):
            # Get bounds of your study area
            xmin, ymin, xmax, ymax = gdf.total_bounds
            
            # Create a high-density 300x300 grid (Increase 300 for more detail)
            grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
            
            # Known points and values
            points = np.array([(p.x, p.y) for p in merged.geometry])
            values = merged[pm10_col].values

            # Interpolate to create a solid sheet of color
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            
            # Use 'nearest' to fill in the edges where linear interpolation fails
            nan_mask = np.isnan(grid_z)
            if nan_mask.any():
                grid_z[nan_mask] = griddata(points, values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')

        # 5. VISUALIZATION
        st.subheader("📍 Continuous PM10 Surface Map")
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the solid surface as an image overlay
        # We use imshow instead of scatter to get rid of dots
        im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), 
                       origin='lower', cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
        
        # Add Satellite map
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        # Add colorbar
        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. DOWNLOAD
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Solid Map (PNG)", buf.getvalue(), "solid_pm10_map.png")

    except Exception as e:
        st.error(f"Processing Error: {e}")
else:
    st.info("Please upload your files to generate the solid map.")
