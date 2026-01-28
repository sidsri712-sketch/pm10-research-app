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
    st.error("Missing library! Please run: pip install rapidfuzz")
    st.stop()

st.set_page_config(page_title="PM10 Solid Map Tool", layout="wide")

st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload Air Quality CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

if csv_file and shp_files:
    try:
        # 1. LOAD DATA
        # amity_campus_reformatted_data.csv contains the PM10 values
        df = pd.read_csv(csv_file)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            
            # tramity.shp contains the sensor point locations
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 2. TYPE SAFETY & MATCHING (Fixes the "int64 vs object" error)
        # Identifying columns based on your files
        csv_id_col = 'location_id' 
        shp_id_col = 'id'
        pm10_col = 'pm10'

        # Force both ID columns to be strings to ensure they match
        df[csv_id_col] = df[csv_id_col].astype(str).str.strip()
        gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.strip()

        # 3. DATA MERGE
        # This joins the PM10 values to the geographic points
        merged = gdf.merge(df, left_on=shp_id_col, right_on=csv_id_col)
        
        if merged.empty:
            st.warning("No matches found between CSV and Shapefile IDs. Check ID columns.")
            st.stop()
        else:
            st.success(f"Successfully matched {len(merged)} sensor locations.")

        # 4. CREATE SOLID COLOR SURFACE (Interpolation)
        st.subheader("🗺️ Continuous PM10 Distribution")
        
        # Get bounds of the sensor grid
        xmin, ymin, xmax, ymax = gdf.total_bounds
        # Create a dense grid (300x300 pixels) to "paint" the color
        grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
        
        # Extract points and values for the heatmap
        points = np.array([(p.x, p.y) for p in merged.geometry])
        values = merged[pm10_col].values

        # Interpolate: 'linear' for smooth transitions, 'nearest' to fill edges
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        grid_z_fill = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

        # 5. VISUALIZATION
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Use imshow to draw the solid color surface
        im = ax.imshow(
            grid_z.T, 
            extent=(xmin, xmax, ymin, ymax), 
            origin='lower', 
            cmap='RdYlGn_r', 
            alpha=0.6, 
            interpolation='bilinear'
        )
        
        # Add high-resolution satellite imagery
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. DOWNLOAD
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Solid Map", buf.getvalue(), "pm10_heatmap.png")

    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
else:
    st.info("Upload your CSV and Shapefile components to generate the solid map.")
