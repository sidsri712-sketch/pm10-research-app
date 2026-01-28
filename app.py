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

# --------------------------------------------------
# SIDEBAR: DATA INPUT
# --------------------------------------------------
st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload Air Quality CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------
if csv_file and shp_files:
    try:
        # 1. LOAD DATA
        df = pd.read_csv(csv_file)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            
            shp_list = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
            if not shp_list:
                st.error("No .shp file found. Please upload the complete set.")
                st.stop()
            
            gdf = gpd.read_file(os.path.join(tmpdir, shp_list[0]))

        # 2. DATA TYPE SAFETY (Prevents "int has no len" and merge errors)
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        if not pm10_col:
            st.error("Could not find a 'PM10' column in your CSV.")
            st.stop()

        # FORCE ALL IDs TO STRINGS IMMEDIATELY
        df[csv_id_col] = df[csv_id_col].astype(str).str.strip()
        gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.strip()

        # 3. FUZZY JOIN
        with st.spinner("Aligning geographic boundaries..."):
            shp_ids = gdf[shp_id_col].unique().tolist()
            id_map = {}
            for cid in df[csv_id_col].unique():
                # Ensure input is string to avoid TypeErrors
                match = process.extractOne(str(cid), shp_ids, scorer=fuzz.WRatio)
                if match and match[1] > 60:
                    id_map[cid] = match[0]

            df["matched_id"] = df[csv_id_col].map(id_map)
            merged_data = gdf.merge(df.dropna(subset=[pm10_col]), left_on=shp_id_col, right_on="matched_id")

        # 4. CREATE SOLID INTERPOLATED SURFACE (Heatmap Logic)
        with st.spinner("Generating continuous surface map..."):
            # Define the boundary for our solid color area
            xmin, ymin, xmax, ymax = gdf.total_bounds
            # Increase grid resolution for a sharp, high-res look
            grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]
            
            # Points where we have sensor data (centroids)
            known_coords = np.array([(p.x, p.y) for p in merged_data.geometry.centroid])
            known_values = merged_data[pm10_col].values

            # Interpolate to create a solid surface instead of dots
            grid_z = griddata(known_coords, known_values, (grid_x, grid_y), method='linear')
            
            # Fill outer edges with nearest sensor data for a completely solid look
            nan_mask = np.isnan(grid_z)
            if nan_mask.any():
                grid_z[nan_mask] = griddata(known_coords, known_values, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')

        # 5. VISUALIZATION (SOLID RENDERING)
        st.subheader("🗺️ Continuous PM10 Exposure Map")
        
        if gdf.crs is None: 
            gdf.set_crs(epsg=4326, inplace=True)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the solid color surface (imshow) - NOT a scatter plot
        im = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), 
                       origin='lower', cmap='RdYlGn_r', alpha=0.6, interpolation='bilinear')
        
        # Overlay original boundaries for research context
        gdf.plot(ax=ax, color='none', edgecolor='white', linewidth=0.5, alpha=0.3)
        
        # Add high-resolution satellite imagery
        cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
        
        plt.colorbar(im, label="PM10 (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. DOWNLOADS
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("🖼️ Download Solid Map (PNG)", buf.getvalue(), "solid_pm10_map.png", "image/png")
        with c2:
            st.success("Solid distribution analysis complete. Map ready for download.")

    except Exception as e:
        # This catches all processing errors and displays them safely
        st.error(f"Execution Error: {e}")

else:
    st.info("Upload your CSV and Shapefile set to generate the solid distribution map.")
