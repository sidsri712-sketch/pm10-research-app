import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import io
import tempfile
import os

st.set_page_config(page_title="PM10 Solid Map Tool", layout="wide")

st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile", type=["shp", "shx", "dbf"], accept_multiple_files=True)

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

        # 2. MATCH DATA
        pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)
        # Assuming we join on the first column or matching IDs
        data = gdf.copy()
        # Simple merge for demonstration - adjust 'on' to your ID column
        # data = gdf.merge(df, left_on='id', right_on='id') 

        # 3. CONVERT POINTS TO SOLID AREAS (VORONOI)
        # This is the "Magic" that removes the dots
        coords = np.array([(p.x, p.y) for p in gdf.geometry])
        
        # Create a bounding box based on your data
        boundary = gdf.total_bounds
        points_for_voronoi = np.append(coords, [
            [boundary[0]-1, boundary[1]-1], [boundary[0]-1, boundary[3]+1],
            [boundary[2]+1, boundary[1]-1], [boundary[2]+1, boundary[3]+1]
        ], axis=0)
        
        vor = Voronoi(points_for_voronoi)
        
        lines = []
        for region in vor.regions:
            if not -1 in region and len(region) > 0:
                polygon = Polygon([vor.vertices[i] for i in region])
                lines.append(polygon)
        
        # Create a new GeoDataFrame of Polygons
        voronoi_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)
        
        # Spatial join to attach PM10 values to these new regions
        # We find which original point is inside which new Voronoi cell
        result_gdf = gpd.sjoin(voronoi_gdf, gdf, how="inner", predicate="intersects")

        # 4. PLOT SOLID CHOROPLETH
        st.subheader("🗺️ Solid Area PM10 Distribution")
        
        if result_gdf.crs is None: result_gdf.set_crs(epsg=4326, inplace=True)
        result_web = result_gdf.to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 12))
        
        # PLOT THE POLYGONS, NOT THE POINTS
        result_web.plot(
            column=pm10_col, 
            cmap="RdYlGn_r", 
            legend=True, 
            ax=ax, 
            alpha=0.5, 
            edgecolor='none' # Remove lines for a smooth look
        )
        
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
        ax.set_axis_off()
        st.pyplot(fig)

        # 5. DOWNLOAD
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("🖼️ Download Solid Map", buf.getvalue(), "solid_map.png")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload files to see the change.")
