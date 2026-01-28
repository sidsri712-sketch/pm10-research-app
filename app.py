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

st.set_page_config(page_title="PM10 Solid Map Analysis", layout="wide")

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

        # 2. MATCHING & COLUMN CLEANING
        # Fix: Ensure all IDs are strings to avoid "int has no len()" error
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        if not pm10_col:
            st.error("Could not find a 'PM10' column in your CSV.")
            st.stop()

        # 3. FUZZY JOIN
        with st.spinner("Processing spatial alignment..."):
            unique_shp_ids = gdf[shp_id_col].astype(str).unique().tolist()
            id_map = {}
            for cid in df[csv_id_col].dropna().unique():
                # Force cid to string to prevent TypeError
                match = process.extractOne(str(cid), unique_shp_ids, scorer=fuzz.WRatio)
                if match and match[1] > 60:
                    id_map[str(cid)] = match[0]

            df['temp_id'] = df[csv_id_col].astype(str)
            merged = gdf.merge(df.dropna(subset=[pm10_col]), left_on=shp_id_col, right_on='temp_id')

        # 4. CREATE SOLID INTERPOLATED SURFACE (No more dots!)
        st.subheader("📍 Continuous PM10 Distribution")
        
        # We create a dense grid over the entire area
        xmin, ymin, xmax, ymax = gdf.total_bounds
        # Increase these numbers for higher resolution (e.g., 200x200)
        grid_x, grid_y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        
        # Get known points and values
        points = np.array([(p.x, p.y) for p in merged.geometry.centroid])
        values = merged[pm10_col].values

        # Generate the solid color surface
        grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
        
        # Fill edges with nearest neighbor to ensure a "solid" look
        grid_z_near = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = grid_z_near[np.isnan(grid_z)]

        # 5. VISUALIZATION
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        # Convert grid extent for basemap compatibility
        # We plot using imshow for a true heatmap/solid look
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the solid surface
        im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), 
                       origin='lower', cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
        
        # Overlay the district boundaries for context
        gdf.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5, alpha=0.3)
        
        # Add Satellite Basemap
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        plt.colorbar(im, label="PM10 (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. EXPORT
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Solid Map (PNG)", buf.getvalue(), "solid_pm10_map.png")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your files to generate the solid distribution map.")
