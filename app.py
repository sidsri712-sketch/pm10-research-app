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

st.set_page_config(page_title="PM10 Choropleth Tool", layout="wide")

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

        # 2. MATCHING & CLEANING
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf_clean = gdf.rename(columns={shp_id_col: 'location_id'})

        # 3. INTERPOLATION (CHOROPLETH LOGIC)
        with st.spinner("Generating Choropleth..."):
            # Fuzzy match to get data onto geometries
            unique_shp_ids = gdf_clean["location_id"].unique().tolist()
            id_map = {cid: process.extractOne(str(cid), unique_shp_ids, scorer=fuzz.WRatio)[0] for cid in df["location_id"].unique()}
            df["matched_id"] = df["location_id"].map(id_map)
            
            merged_data = gdf_clean.merge(df.dropna(subset=[pm10_col]), left_on="location_id", right_on="matched_id")

            # Interpolate values for EVERY polygon in the shapefile
            known_coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in merged_data.geometry])
            known_values = merged_data[pm10_col].values
            
            target_coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf_clean.geometry])

            # Fill the 'Predicted_PM10' column for ALL polygons
            predictions = griddata(known_coords, known_values, target_coords, method='linear')
            
            # Fill edges with nearest neighbor
            nan_mask = np.isnan(predictions)
            if nan_mask.any():
                predictions[nan_mask] = griddata(known_coords, known_values, target_coords[nan_mask], method='nearest')
            
            gdf_clean["Predicted_PM10"] = predictions

        # 4. THE PLOT (STRICT CHOROPLETH)
        st.subheader("🗺️ Regional PM10 Exposure (Choropleth)")
        
        if gdf_clean.crs is None: gdf_clean.set_crs(epsg=4326, inplace=True)
        gdf_web = gdf_clean.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 12))
        
        # KEY CHANGE: We plot ONLY the dataframe 'gdf_web' 
        # and use the 'column' argument to fill the shapes.
        gdf_web.plot(
            column="Predicted_PM10", 
            cmap="RdYlGn_r", 
            legend=True, 
            ax=ax, 
            alpha=0.7, 
            edgecolor='black', # Thin borders to distinguish districts
            linewidth=0.5,
            legend_kwds={'label': "PM10 (µg/m³)", 'orientation': "horizontal", 'pad': 0.01}
        )
        
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        ax.set_axis_off()
        st.pyplot(fig)

        # 5. DOWNLOAD
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Choropleth Map", buf.getvalue(), "pm10_choropleth.png")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Awaiting file uploads...")
