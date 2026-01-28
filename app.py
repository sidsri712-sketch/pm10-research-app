import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from scipy.interpolate import griddata
from rasterio import features
from affine import Affine
import io
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PM10 Heatmap Tool", layout="wide")

st.title("🌍 Environmental PM10 Interpolation Map")
st.markdown("""
Upload your **sensor data (CSV)** and your **boundary/site data (Shapefile)** to generate a continuous air quality surface.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV (location_id, pm10)", type=["csv"])
shp_files = st.sidebar.file_uploader(
    "Upload Shapefile Set (.shp, .shx, .dbf, .prj)",
    type=["shp", "shx", "dbf", "prj"],
    accept_multiple_files=True
)

# ---------------- MAIN ----------------
if csv_file and shp_files:
    try:
        with st.spinner("Processing spatial data..."):
            # ---------- LOAD CSV ----------
            df = pd.read_csv(csv_file)
            csv_id, pm10_col = "location_id", "pm10"

            if csv_id not in df.columns or pm10_col not in df.columns:
                st.error(f"CSV must contain '{csv_id}' and '{pm10_col}' columns.")
                st.stop()

            df[csv_id] = df[csv_id].astype(str).str.strip()

            # ---------- LOAD SHAPEFILE ----------
            with tempfile.TemporaryDirectory() as tmpdir:
                for f in shp_files:
                    with open(os.path.join(tmpdir, f.name), "wb") as out:
                        out.write(f.getbuffer())
                
                shp_path = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
                gdf = gpd.read_file(shp_path)

            shp_id = "id"
            if shp_id not in gdf.columns:
                st.error("Shapefile must contain an 'id' column.")
                st.stop()

            gdf[shp_id] = gdf[shp_id].astype(str).str.strip()

            # ---------- MERGE ----------
            merged = gdf.merge(df, left_on=shp_id, right_on=csv_id)
            if merged.empty:
                st.error("Merge failed: No matching IDs between CSV and Shapefile.")
                st.stop()

            # Set CRS if missing and project to Web Mercator
            if merged.crs is None:
                merged = merged.set_crs(epsg=4326)
            merged = merged.to_crs(epsg=3857)

            # Store original geometry for masking before calculating centroids
            study_area_poly = merged.unary_union 
            merged_points = merged.copy()
            merged_points["geometry"] = merged.geometry.centroid

            # ---------- INTERPOLATION ----------
            points = np.array([(geom.x, geom.y) for geom in merged_points.geometry])
            values = merged_points[pm10_col].values

            xmin, ymin, xmax, ymax = merged_points.total_bounds
            grid_res = 300
            grid_x, grid_y = np.mgrid[xmin:xmax:complex(grid_res), ymin:ymax:complex(grid_res)]

            # Perform Interpolation
            grid_z = griddata(points, values, (grid_x, grid_y), method="linear")
            grid_z_fill = griddata(points, values, (grid_x, grid_y), method="nearest")
            grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

            # ---------- MASKING ----------
            # Calculate transform for rasterio masking
            res_x = (xmax - xmin) / grid_res
            res_y = (ymax - ymin) / grid_res
            transform = Affine.translation(xmin, ymin) * Affine.scale(res_x, res_y)
            
            mask = features.geometry_mask([study_area_poly], out_shape=grid_z.shape, transform=transform, invert=True)
            grid_z[~mask.T] = np.nan # Masking outside the polygons

        # ---------- PLOTTING ----------
        st.subheader("📍 Continuous PM10 Surface Map")
        fig, ax = plt.subplots(figsize=(12, 12))

        im = ax.imshow(
            grid_z.T,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="RdYlGn_r",
            alpha=0.7,
            zorder=2
        )

        # Plot the points for reference
        merged_points.plot(ax=ax, color='black', markersize=10, alpha=0.5, zorder=3)

        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zorder=1)

        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.5)
        ax.set_axis_off()
        st.pyplot(fig)

        # ---------- EXPORT ----------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button("🖼️ Download Heatmap", buf.getvalue(), "pm10_heatmap.png")

    except Exception as e:
        st.error(f"Execution Error: {e}")
else:
    st.info("Waiting for CSV and Shapefile upload...")
