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
    st.error("Missing libraries! Please run: pip install rapidfuzz scipy contextily")
    st.stop()

st.set_page_config(page_title="PM10 Heatmap Tool", layout="wide")

st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

if csv_file and shp_files:
    try:
        # 1. LOAD CSV DATA
        df = pd.read_csv(csv_file)
        
        # 2. LOAD SHAPEFILE DATA
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 3. DATA ALIGNMENT (Fixing the ID mismatch)
        # Based on your files: CSV uses 'location_id', SHP uses 'id'
        csv_id = 'location_id'
        shp_id = 'id'
        pm10_col = 'pm10'

        # Force both to string to prevent merge failure
        df[csv_id] = df[csv_id].astype(str).str.strip()
        gdf[shp_id] = gdf[shp_id].astype(str).str.strip()

        # Merge files
        merged = gdf.merge(df, left_on=shp_id, right_on=csv_id)

        if merged.empty:
            st.error("Merge failed: No IDs matched between CSV and Shapefile.")
            st.stop()

        # 4. INTERPOLATION (Turning points into a solid heatmap)
        
        
        # Set boundaries and grid resolution (300x300 pixels)
        xmin, ymin, xmax, ymax = gdf.total_bounds
        grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
        
        # Extract points and PM10 values
        points = np.array([(geom.x, geom.y) for geom in merged.geometry])
        values = merged[pm10_col].values

        # Generate smooth surface
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        
        # Fill edges where linear interpolation can't reach
        grid_z_fill = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

        # 5. VISUALIZATION
        st.subheader("📍 Continuous PM10 Surface Map")
        
        # Ensure CRS is set (Defaults to WGS84 if missing)
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the solid surface as an image
        im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), 
                       origin='lower', cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
        
        # Add high-resolution satellite imagery underneath
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. EXPORT
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Heatmap", buf.getvalue(), "pm10_heatmap.png")

    except Exception as e:
        st.error(f"Execution Error: {str(e)}")
else:
    st.info("Upload your CSV and Shapefile set to generate the heatmap.")
