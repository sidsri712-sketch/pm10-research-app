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

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PM10 Point-Choropleth", layout="wide")

st.title("🗺️ Voronoi Choropleth Map")
st.markdown("Your shapefile contains points. This tool generates regions around those points to create a Choropleth map.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Input")
csv_file = st.sidebar.file_uploader("1. Upload CSV (location_id, pm10)", type=["csv"])
shp_files = st.sidebar.file_uploader("2. Upload Shapefile Points", type=["shp", "shx", "dbf", "prj"], accept_multiple_files=True)
map_alpha = st.sidebar.slider("Color Opacity", 0.0, 1.0, 0.6)

# ---------------- HELPER FUNCTION ----------------
def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions into finite polygons."""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for i, reg_num in enumerate(vor.point_region):
        region = vor.regions[reg_num]
        if all(v >= 0 for v in region):
            new_regions.append(region)
            continue
        
        # Infinite region logic
        ridges = all_ridges[i]
        new_region = [v for v in region if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0: v1, v2 = v2, v1
            if v1 >= 0: continue
            t = vor.points[p2] - vor.points[i]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[i, p2]].mean(axis=0)
            far_point = vor.points[i] + (np.sign(np.dot(midpoint - center, n)) * n) * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        new_regions.append(new_region)
    return new_regions, np.array(new_vertices)

# ---------------- MAIN ----------------
if csv_file and shp_files:
    try:
        # Load CSV and Shapefile (same as previous steps)
        df = pd.read_csv(csv_file)
        df["location_id"] = df["location_id"].astype(str).str.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            for f in shp_files:
                with open(os.path.join(tmpdir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
            gdf = gpd.read_file(shp_path)
        
        gdf["id"] = gdf["id"].astype(str).str.strip()
        gdf = gdf.to_crs(epsg=3857) # Project to Web Mercator

        # Merge
        merged = gdf.merge(df, left_on="id", right_on="location_id")
        
        # ---------- VORONOI GENERATION ----------
        coords = np.array([(geom.x, geom.y) for geom in merged.geometry])
        vor = Voronoi(coords)
        regions, vertices = voronoi_finite_polygons_2d(vor)

        # Create polygons from Voronoi regions
        polygons = []
        for reg in regions:
            poly = Polygon(vertices[reg])
            polygons.append(poly)
        
        # Create a new GeoDataFrame with these polygons
        v_gdf = gpd.GeoDataFrame(merged.drop(columns='geometry'), geometry=polygons, crs=merged.crs)

        # ---------- PLOTTING ----------
        fig, ax = plt.subplots(figsize=(10, 10))
        
        v_gdf.plot(
            column="pm10",
            cmap="RdYlGn_r",
            alpha=map_alpha,
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
            legend=True,
            zorder=2
        )
        
        # Clip to original data bounds to avoid infinite-looking edges
        xmin, ymin, xmax, ymax = merged.total_bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zorder=1)
        ax.set_axis_off()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your Point Shapefile and CSV to create the Voronoi Choropleth.")
