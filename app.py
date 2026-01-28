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
csv_file = st.sidebar.file_uploader("Upload Air Quality CSV (must have 'pm10' and 'id' columns)", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

st.sidebar.info("Note: The CSV should contain a column for PM10 values and an ID column that matches your Shapefile boundaries.")

# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------
if csv_file and shp_files:
    try:
        # 1. LOAD DATA
        df = pd.read_csv(csv_file)
        
        # Handle Shapefile upload via temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            
            # Find the .shp file in the uploaded set
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 2. SMART ID COLUMN IDENTIFICATION
        # Look for common ID names if not explicitly defined
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id', 'gid']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid', 'name']), gdf.columns[0])
        
        # Standardize for merging
        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf_clean = gdf.rename(columns={shp_id_col: 'location_id'})
        
        df["location_id"] = df["location_id"].astype(str).str.strip()
        gdf_clean["location_id"] = gdf_clean["location_id"].astype(str).str.strip()
        
        # 3. FUZZY JOIN (Linking CSV data to Shapefile polygons)
        with st.spinner("Aligning datasets and interpolating..."):
            # Create a mapping dictionary: {CSV_ID: Best_SHP_ID}
            unique_shp_ids = gdf_clean["location_id"].unique().tolist()
            id_map = {}
            for cid in df["location_id"].unique():
                # Only map if similarity is > 80% to avoid wrong assignments
                match = process.extractOne(str(cid), unique_shp_ids, scorer=fuzz.WRatio)
                if match and match[1] > 80:
                    id_map[cid] = match[0]

            df["matched_id"] = df["location_id"].map(id_map)
            
            # Filter for rows that actually have PM10 data
            pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)
            if not pm10_col:
                st.error("Could not find a 'PM10' column in your CSV.")
                st.stop()
                
            merged_data = gdf_clean.merge(df.dropna(subset=[pm10_col]), left_on="location_id", right_on="matched_id")

        # 4. SPATIAL INTERPOLATION (The "Science" Bit)
        # We take known sensor points (centroids of matched polygons)
        known_points = merged_data.geometry.centroid
        known_x = known_points.x.values
        known_y = known_points.y.values
        known_z = merged_data[pm10_col].values

        # We predict values for ALL polygons in the shapefile
        all_centroids = gdf_clean.geometry.centroid
        grid_x = all_centroids.x.values
        grid_y = all_centroids.y.values
