import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box
import io
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PM10 Choropleth Tool", layout="wide")

st.title("🗺️ PM10 Point-Grid Choropleth")
st.markdown("Turning your sensor point grid into a solid choropleth map.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Input")
csv_file = st.sidebar.file_uploader("1. Upload CSV (location_id, pm10)", type=["csv"])
shp_files = st.sidebar.file_uploader(
    "2. Upload Shapefile Points", 
    type=["shp", "shx", "dbf", "prj"], 
    accept_multiple_files=True
)
cell_size = st.sidebar.slider("Cell Size adjustment", 1.0, 50.0, 10.0)
map_alpha = st.sidebar.slider("Color Opacity", 0.0, 1.0, 0.7)

# ---------------- MAIN ----------------
if csv_file and shp_files:
    try:
        # ---------- LOAD DATA ----------
        df = pd.read_csv(csv_file)
        df["location_id"] = df["location_id"].astype(str).str.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)
        
        gdf["id"] = gdf["id"].astype(str).str.strip()
        gdf = gdf.to_crs(epsg=3857) # Project for accurate square tiling

        # Merge
        merged = gdf.merge(df, left_on="id", right_on="location_id")
        
        if merged.empty:
            st.error("Merge failed. IDs in CSV do not match IDs in Shapefile.")
            st.stop()

        # ---------- POINT TO SQUARE CHOROPLETH ----------
        # Instead of complex Voronoi, we create a square 'buffer' around each point 
        # to create a grid-like choropleth effect.
        half_size = cell_size / 2
        
        def make_square(point):
            return box(point.x - half_size, point.y - half_size, 
                       point.x + half_size, point.y + half_size)

        merged["geometry"] = merged.geometry.apply(make_square)

        # ---------- PLOTTING ----------
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Plot the squares
        merged.plot(
            column="pm10",
            cmap="RdYlGn_r",
            alpha=map_alpha,
            edgecolor=None, # Set to 'black' if you want a grid outline
            ax=ax,
            legend=True,
            legend_kwds={'label': "PM10 Concentration (µg/m³)", 'shrink': 0.5},
            zorder=2
        )
        
        # Add Satellite Basemap
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zorder=1)
        
        ax.set_axis_off()
        
        # Focus the map on the data area
        xmin, ymin, xmax, ymax = merged.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        st.pyplot(fig)

        # ---------- EXPORT ----------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button("💾 Download Map", buf.getvalue(), "pm10_choropleth.png")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your Point Shapefile and CSV to see the choropleth.")
