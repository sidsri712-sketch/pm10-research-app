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

st.set_page_config(page_title="PM10 Regional Choropleth", layout="wide")

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

        # 2. IDENTIFY COLUMNS & CLEAN TYPES
        # This prevents the "int has no len()" error by forcing everything to strings
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        if not pm10_col:
            st.error("No 'PM10' column found in CSV.")
            st.stop()

        df[csv_id_col] = df[csv_id_col].astype(str).str.strip()
        gdf[shp_id_col] = gdf[shp_id_col].astype(str).str.strip()

        # 3. FUZZY MATCHING
        with st.spinner("Aligning geographic boundaries..."):
            shp_ids = gdf[shp_id_col].unique().tolist()
            id_map = {}
            for cid in df[csv_id_col].unique():
                # ExtractOne requires strings; forcing str(cid) here
                match = process.extractOne(str(cid), shp_ids, scorer=fuzz.WRatio)
                if match and match[1] > 60:
                    id_map[cid] = match[0]

            df["matched_id"] = df[csv_id_col].map(id_map)
            merged_data = gdf.merge(df.dropna(subset=[pm10_col]), left_on=shp_id_col, right_on="matched_id")

        # 4. CHOROPLETH INTERPOLATION
        with st.spinner("Calculating regional exposure..."):
            # Known values at sensor centroids
            known_coords = np.array([(p.x, p.y) for p in merged_data.geometry.centroid])
            known_values = merged_data[pm10_col].values

            # Targets: centroids of EVERY polygon in the shapefile
            target_coords = np.array([(p.x, p.y) for p in gdf.geometry.centroid])

            # Interpolate for a smooth surface
            grid_z = griddata(known_coords, known_values, target_coords, method='linear')
            
            # Use nearest neighbor to fill polygons outside the sensor hull
            nan_mask = np.isnan(grid_z)
            if nan_mask.any():
                grid_z[nan_mask] = griddata(known_coords, known_values, target_coords[nan_mask], method='nearest')
            
            gdf["Predicted_PM10"] = grid_z

        # 5. FINAL VISUALIZATION (SOLID POLYGONS)
        st.subheader("🗺️ Regional PM10 Choropleth Map")
        
        if gdf.crs is None: 
            gdf.set_crs(epsg=4326, inplace=True)
        
        gdf_web = gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # This renders the shapes as solid colors based on the data
        gdf_web.plot(
            column="Predicted_PM10", 
            cmap="RdYlGn_r", 
            legend=True, 
            legend_kwds={'label': "PM10 (µg/m³)", 'orientation': "horizontal", 'pad': 0.02},
            ax=ax, 
            alpha=0.6,
            edgecolor='white', # Draws clean district lines
            linewidth=0.5
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
            csv_out = gdf[[shp_id_col, 'Predicted_PM10']].to_csv(index=False).encode('utf-8')
            st.download_button("📊 Download Data (CSV)", csv_out, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.info("Upload your CSV and Shapefile set to generate the choropleth.")
