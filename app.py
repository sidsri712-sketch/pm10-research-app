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
    st.error("Missing libraries! Run: pip install rapidfuzz contextily scipy geopandas")
    st.stop()

st.set_page_config(page_title="PM10 Solid Map Tool", layout="wide")

st.sidebar.header("📁 Data Upload")
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

        # 2. TYPE SAFETY (Fixes the "int64 and object" merge error)
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        # Convert IDs to strings to ensure they match correctly
        df[csv_id_col] = df[csv_id_col].astype(str).str.strip()
        gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.strip()

        # 3. MERGE & INTERPOLATE
        with st.spinner("Creating solid exposure surface..."):
            # Align data
            shp_ids = gdf[shp_id_col].unique().tolist()
            id_map = {str(cid): process.extractOne(str(cid), shp_ids, scorer=fuzz.WRatio)[0] for cid in df[csv_id_col].unique()}
            df['matched_id'] = df[csv_id_col].map(id_map)
            merged = gdf.merge(df.dropna(subset=[pm10_col]), left_on=shp_id_col, right_on='matched_id')

            # Create the Solid Grid
            xmin, ymin, xmax, ymax = gdf.total_bounds
            # 300j creates a high-res 300x300 color sheet
            grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
            
            # Use the points to calculate the solid color between them
            points = np.array([(p.x, p.y) for p in merged.geometry])
            values = merged[pm10_col].values

            # Interpolate (IDW style)
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            # Fill the edges so the map is completely solid
            grid_z_fill = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

        # 4. VISUALIZATION
        st.subheader("🗺️ Continuous PM10 Surface Map")
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the solid surface (imshow paints the areas between points)
        im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), 
                       origin='lower', cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
        
        # Add high-res satellite imagery underneath
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 5. EXPORT
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Solid Map", buf.getvalue(), "solid_pm10_map.png")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your CSV and Shapefile set to see the solid distribution.")
