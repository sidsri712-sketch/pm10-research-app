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

st.set_page_config(page_title="PM10 Spatial Analysis Tool", layout="wide")

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
            
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 2. SMART ID COLUMN IDENTIFICATION
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        
        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf_clean = gdf.rename(columns={shp_id_col: 'location_id'})
        
        df["location_id"] = df["location_id"].astype(str).str.strip()
        gdf_clean["location_id"] = gdf_clean["location_id"].astype(str).str.strip()
        
        # 3. FUZZY JOIN
        with st.spinner("Aligning datasets and interpolating..."):
            unique_shp_ids = gdf_clean["location_id"].unique().tolist()
            id_map = {}
            for cid in df["location_id"].unique():
                match = process.extractOne(str(cid), unique_shp_ids, scorer=fuzz.WRatio)
                if match and match[1] > 70: # Relaxed to 70 for better matching
                    id_map[cid] = match[0]

            df["matched_id"] = df["location_id"].map(id_map)
            
            pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)
            if not pm10_col:
                st.error("Could not find a 'PM10' column in your CSV.")
                st.stop()
                
            merged_data = gdf_clean.merge(df.dropna(subset=[pm10_col]), left_on="location_id", right_on="matched_id")

        # 4. SPATIAL INTERPOLATION
        # Use Centroids for calculation
        known_points = merged_data.geometry.centroid
        known_x = known_points.x.values
        known_y = known_points.y.values
        known_z = merged_data[pm10_col].values

        all_centroids = gdf_clean.geometry.centroid
        grid_x = all_centroids.x.values
        grid_y = all_centroids.y.values

        grid_z = griddata((known_x, known_y), known_z, (grid_x, grid_y), method='linear')
        
        # Fill NaNs using nearest neighbor for edge polygons
        nan_mask = np.isnan(grid_z)
        if nan_mask.any():
            grid_z[nan_mask] = griddata((known_x, known_y), known_z, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        
        gdf_clean["Predicted_PM10"] = grid_z

        # 5. VISUALIZATION
        st.subheader("📍 PM10 Spatial Distribution Map")
        
        if gdf_clean.crs is None: 
            gdf_clean.set_crs(epsg=4326, inplace=True)
        
        gdf_web = gdf_clean.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=(10, 10))
        
        gdf_web.plot(
            column="Predicted_PM10", 
            cmap="RdYlGn_r", 
            legend=True, 
            legend_kwds={'label': "PM10 (µg/m³)", 'orientation': "horizontal", 'pad': 0.02},
            ax=ax, 
            alpha=0.6, 
            edgecolor='white',
            linewidth=0.3
        )
        
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        ax.set_axis_off()
        st.pyplot(fig)

        # 6. EXPORT
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("🖼️ Download Map (PNG)", buf.getvalue(), "pm10_map.png", "image/png")
        with c2:
            csv_out = gdf_clean[['location_id', 'Predicted_PM10']].to_csv(index=False).encode('utf-8')
            st.download_button("📊 Download CSV Data", csv_out, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Error during processing: {e}")

else:
    st.info("Please upload both a CSV and a Shapefile set to generate the map.")
