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

# ---------------- CONFIG ----------------
st.set_page_config(page_title="PM10 Heatmap Tool", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Step 1: Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
shp_files = st.sidebar.file_uploader(
    "Upload Shapefile Set (.shp, .shx, .dbf)",
    type=["shp", "shx", "dbf"],
    accept_multiple_files=True
)

# ---------------- MAIN ----------------
if csv_file and shp_files:
    try:
        # ---------- LOAD CSV ----------
        df = pd.read_csv(csv_file)

        csv_id = "location_id"
        pm10_col = "pm10"

        if csv_id not in df.columns or pm10_col not in df.columns:
            st.error("CSV must contain 'location_id' and 'pm10' columns.")
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
            st.error("Merge failed: No matching IDs.")
            st.stop()

        # ---------- GEOMETRY FIX ----------
        if merged.geometry.geom_type.isin(["Polygon", "MultiPolygon", "LineString"]).any():
            merged["geometry"] = merged.geometry.centroid

        # ---------- CRS FIX ----------
        if merged.crs is None:
            merged = merged.set_crs(epsg=4326)

        # Reproject to Web Mercator for basemap
        merged = merged.to_crs(epsg=3857)

        # ---------- INTERPOLATION ----------
        points = np.array([(geom.x, geom.y) for geom in merged.geometry])
        values = merged[pm10_col].values

        if len(points) < 3:
            st.error("At least 3 locations are required for interpolation.")
            st.stop()

        xmin, ymin, xmax, ymax = merged.total_bounds
        grid_x, grid_y = np.mgrid[xmin:xmax:300j, ymin:ymax:300j]

        grid_z = griddata(points, values, (grid_x, grid_y), method="linear")
        grid_z_fill = griddata(points, values, (grid_x, grid_y), method="nearest")
        grid_z[np.isnan(grid_z)] = grid_z_fill[np.isnan(grid_z)]

        # ---------- PLOT ----------
        st.subheader("📍 Continuous PM10 Surface Map")

        fig, ax = plt.subplots(figsize=(12, 12))

        im = ax.imshow(
            grid_z.T,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="RdYlGn_r",
            alpha=0.65
        )

        cx.add_basemap(
            ax,
            source=cx.providers.Esri.WorldImagery,
            crs=merged.crs
        )

        plt.colorbar(im, label="PM10 Concentration (µg/m³)", ax=ax, shrink=0.6)
        ax.set_axis_off()

        st.pyplot(fig)

        # ---------- EXPORT ----------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "🖼️ Download Heatmap",
            buf.getvalue(),
            "pm10_heatmap.png"
        )

    except Exception as e:
        st.error(f"Execution Error: {e}")

else:
    st.info("Upload both CSV and Shapefile set to generate the PM10 heatmap.")
