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
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

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

        # 2. MATCHING & COLUMN CLEANING
        # Force everything to string to prevent 'int64' vs 'object' merge errors
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        if not pm10_col:
            st.error("Could not find a 'PM10' column in your CSV.")
            st.stop()

        # 3. FUZZY JOIN WITH TYPE SAFETY
        with st.spinner("Aligning datasets..."):
            unique_shp_ids = gdf[shp_id_col].astype(str).unique().tolist()
            id_map = {}
            for cid in df[csv_id_col].dropna().unique():
                # Force cid to string to prevent 'int' has no len() error
                match = process.extractOne(str(cid), unique_shp_ids, scorer=fuzz.WRatio)
                if match and match[1] > 60:
                    id_map[str(cid)] = match[0]

            df['temp_match_id'] = df[csv_id_col].astype(str).map(id_map)
            merged = gdf.merge(df.dropna(subset=[pm10_col]), left_on=shp_id_col, right_on='temp_match_id')

        # 4. CREATE SOLID INTERPOLATED SURFACE
        st.subheader("🗺️ Continuous PM10 Distribution (Solid Map)")
        
        # Define the grid bounds
        xmin, ymin, xmax, ymax = gdf.total_bounds
        # 300j creates a high-resolution 300x300 pixel grid
        grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
        
        # Points and values for interpolation
        points = np.array([(p.x, p.y) for p in merged.geometry.centroid])
        values = merged[pm10_col].values

        # Linear interpolation for the main surface
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        
        # Nearest neighbor interpolation to fill the gaps/edges
        grid_z_fill = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

        # 5. VISUALIZATION
        if gdf.crs is None: 
            gdf.set_crs(epsg=4326, inplace=True)
            
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the solid surface as an image (imshow)
        im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), 
                       origin='lower', cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
        
        # Overlay original boundaries for context
        gdf.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5, alpha=0.4)
        
        # Add the satellite basemap
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. EXPORT
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Solid Map (PNG)", buf.getvalue(), "pm10_solid_map.png")

    except Exception as e:
        # This catches any remaining errors and displays them clearly
        st.error(f"Processing Error: {str(e)}")
else:
    st.info("Upload your CSV and Shapefile components to generate the solid map.")
