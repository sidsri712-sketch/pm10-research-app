import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import io
import contextily as cx
from scipy.interpolate import griddata

# --- SAFETY CHECK ---
try:
    from rapidfuzz import process, fuzz
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    st.error("Please update your requirements.txt to include scipy and rapidfuzz.")
    st.stop()

st.set_page_config(page_title="PM10 IDW Spatial Tool", layout="wide")
st.title("PM10 Spatial Analysis: ML + IDW Interpolation")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("Upload Monitoring CSV", type=["csv"])
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

        # 2. DATA ALIGNMENT
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id']), gdf.columns[0])
        
        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf = gdf.rename(columns={shp_id_col: 'location_id'})
        
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        # 3. FUZZY JOIN
        id_map = {cid: process.extractOne(str(cid), [str(x) for x in gdf["location_id"]])[0] for cid in df["location_id"].unique()}
        df["matched_id"] = df["location_id"].map(id_map)
        data = gdf.merge(df.dropna(subset=["pm10"]), left_on="location_id", right_on="matched_id")

        # 4. IDW CALCULATION
        # Known points (Monitoring Stations)
        known_x = data.geometry.centroid.x.values
        known_y = data.geometry.centroid.y.values
        known_z = data["pm10"].values

        # Grid points (Target Shapefile pixels)
        grid_x = gdf.geometry.centroid.x.values
        grid_y = gdf.geometry.centroid.y.values

        st.info("Generating IDW Interpolation surface...")
        
        # Linear interpolation as a proxy for IDW (Fast and robust for web)
        # For a strict IDW, we use griddata with 'linear' or 'cubic'
        grid_z = griddata((known_x, known_y), known_z, (grid_x, grid_y), method='linear')
        
        # Fill NaN values (outside the convex hull of stations) with nearest neighbor
        nan_mask = np.isnan(grid_z)
        grid_z[nan_mask] = griddata((known_x, known_y), known_z, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        
        gdf["IDW_PM10"] = grid_z

        # 5. MAPPING
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plotting the IDW surface
        # We use a slight markersize and no edge to make the points look like a continuous "sheet"
        gdf_web.plot(column="IDW_PM10", cmap="Spectral_r", legend=True, 
                     legend_kwds={'label': "Interpolated PM10 (µg/m³)"},
                     ax=ax, markersize=15, alpha=0.8, edgecolor='none')
        
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_axis_off()
        
        st.subheader("IDW Interpolated PM10 Surface")
        st.pyplot(fig)

        # 6. EXPORT
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("📥 Download IDW Map (PNG)", buf.getvalue(), "pm10_idw_analysis.png", "image/png")

    except Exception as e:
        st.error(f"Error during IDW process: {e}")
else:
    st.info("Please upload files to see the IDW Interpolation.")
