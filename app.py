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

st.set_page_config(page_title="Final PM10 Heatmap", layout="wide")

st.title("📍 Amity Campus PM10 Solid Heatmap")

# Uploaders
csv_file = st.sidebar.file_uploader("1. Upload amity_campus_reformatted_data.csv", type=["csv"])
shp_files = st.sidebar.file_uploader("2. Upload tramity Shapefile (.shp, .shx, .dbf)", type=["shp", "shx", "dbf"], accept_multiple_files=True)

if csv_file and shp_files:
    try:
        # 1. Load Data
        df = pd.read_csv(csv_file)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)

        # 2. Match IDs (Force String for Success)
        df['location_id'] = df['location_id'].astype(str).str.strip()
        gdf['id'] = gdf['id'].astype(str).str.strip()

        # 3. Merge
        merged = gdf.merge(df, left_on='id', right_on='location_id')

        if merged.empty:
            st.error("Error: Could not match IDs. Ensure CSV 'location_id' matches Shapefile 'id'.")
        else:
            # 4. Create the Continuous Color Surface (Interpolation)
            # We create a grid over the area
            bounds = merged.total_bounds # [xmin, ymin, xmax, ymax]
            grid_x, grid_y = np.mgrid[bounds[0]:bounds[2]:300j, bounds[1]:bounds[3]:300j]
            
            # Get points coordinates and PM10 values
            points = np.array([(geom.x, geom.y) for geom in merged.geometry])
            values = merged['pm10'].values

            # Fill the gaps between points
            grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
            
            # Fill the edges (nearest neighbor)
            grid_z_fill = griddata(points, values, (grid_x, grid_y), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

            # 5. Plotting
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Paint the solid color
            im = ax.imshow(
                grid_z.T, 
                extent=(bounds[0], bounds[2], bounds[1], bounds[3]), 
                origin='lower', 
                cmap='RdYlGn_r', 
                alpha=0.7, 
                interpolation='bilinear'
            )
            
            # Add Satellite Background
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)
            cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.Esri.WorldImagery)
            
            plt.colorbar(im, label='PM10 Concentration', ax=ax)
            ax.set_title("Amity Campus PM10 Distribution")
            ax.set_axis_off()
            
            st.pyplot(fig)
            
            # Download link
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=300)
            st.download_button("Download High-Res Map", buf.getvalue(), "pm10_heatmap.png")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload both the CSV and the Shapefile set to begin.")
