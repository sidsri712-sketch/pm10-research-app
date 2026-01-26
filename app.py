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
except ImportError:
    st.error("Missing dependencies. Please ensure your requirements.txt is updated.")
    st.stop()

st.set_page_config(page_title="PM10 IDW Spatial Tool", layout="wide")
st.title("PM10 Spatial Analysis: IDW Interpolation Surface")

st.info("""
**Publication Mode:** This tool generates a continuous IDW surface. 
It takes your specific monitoring points and interpolates the values across the entire grid provided in your shapefile.
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("Upload Monitoring CSV", type=["csv"])
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

        # 2. SMART ID RESOLUTION
        csv_id_col = next((c for c in df.columns if c.lower() in ['location_id', 'id', 'station_id']), df.columns[0])
        shp_id_col = next((c for c in gdf.columns if c.lower() in ['location_id', 'id', 'gid']), gdf.columns[0])
        
        df = df.rename(columns={csv_id_col: 'location_id'})
        gdf = gdf.rename(columns={shp_id_col: 'location_id'})
        
        df["location_id"] = df["location_id"].astype(str).str.strip().str.lower()
        gdf["location_id"] = gdf["location_id"].astype(str).str.strip().str.lower()
        
        # 3. FUZZY JOIN
        with st.status("Analyzing spatial relationships...", expanded=False):
            id_map = {cid: process.extractOne(str(cid), [str(x) for x in gdf["location_id"]])[0] for cid in df["location_id"].unique()}
            df["matched_id"] = df["location_id"].map(id_map)
            data = gdf.merge(df.dropna(subset=["pm10"]), left_on="location_id", right_on="matched_id")

        # 4. IDW INTERPOLATION LOGIC
        # Extract known points
        known_x = data.geometry.centroid.x.values
        known_y = data.geometry.centroid.y.values
        known_z = data["pm10"].values

        # Extract target grid (the "pixels" from your shapefile)
        grid_x = gdf.geometry.centroid.x.values
        grid_y = gdf.geometry.centroid.y.values

        # Perform Interpolation
        # method='linear' creates a smooth surface between points
        grid_z = griddata((known_x, known_y), known_z, (grid_x, grid_y), method='linear')
        
        # Handle edges (points outside the station hull) with nearest neighbor
        nan_mask = np.isnan(grid_z)
        grid_z[nan_mask] = griddata((known_x, known_y), known_z, (grid_x[nan_mask], grid_y[nan_mask]), method='nearest')
        
        gdf["IDW_PM10"] = grid_z

        # 5. MAPPING
        st.subheader("IDW Interpolated PM10 Map")
        
        if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
        gdf_web = gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plotting the IDW surface 
        # Using markersize=20 and no edge makes the grid look like a continuous color sheet
        gdf_web.plot(column="IDW_PM10", cmap="RdYlGn_r", legend=True, 
                     legend_kwds={'label': "Interpolated PM10 (µg/m³)", 'shrink': 0.5},
                     ax=ax, markersize=25, alpha=0.7, edgecolor='none')
        
        # Add high-quality basemap
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
        ax.set_axis_off()
        
        st.pyplot(fig)

        # 6. DOWNLOAD SECTION
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("🖼️ Download High-Res Map (PNG)", buf.getvalue(), "pm10_idw_map.png", "image/png")
        
        with col2:
            export_csv = gdf.drop(columns='geometry').to_csv(index=False).encode('utf-8')
            st.download_button("📑 Download Grid Data (CSV)", export_csv, "pm10_grid_analysis.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong during the IDW analysis.")
        st.exception(e)
else:
    st.write("---")
    st.write("👈 **Upload your files in the sidebar to generate the IDW map.**")
