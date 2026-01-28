import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor
import io
import tempfile
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart PM10 Mapper", layout="wide")

st.title("🌍 Smart PM10 Mapping Tool (ML Powered)")

with st.expander("📖 Instructions & Formatting"):
    st.markdown("""
    - **CSV:** Must have `location_id` and `pm10`.
    - **Shapefile:** Must have an `id` column (case-insensitive).
    - **Empty Cells:** AI will automatically fill any empty `pm10` rows in your CSV.
    """)

# ---------------- PROCESSING ----------------
@st.cache_data
def load_and_clean_gdf(shp_path_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        for f_name, f_content in shp_path_data:
            with open(os.path.join(tmpdir, f_name), "wb") as out:
                out.write(f_content)
        shp_file = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
        gdf = gpd.read_file(shp_file)
        
        # Resolve 'id' column naming issues (Case-insensitive check)
        possible_id_cols = [c for c in gdf.columns if c.lower() == 'id']
        if possible_id_cols:
            gdf = gdf.rename(columns={possible_id_cols[0]: 'id'})
        else:
            st.error(f"❌ No 'id' column found in Shapefile. Available columns: {list(gdf.columns)}")
            st.stop()
            
        return gdf.to_crs(epsg=3857)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set", type=["shp", "shx", "dbf", "prj"], accept_multiple_files=True)

cell_size = st.sidebar.slider("Grid Square Size", 5, 100, 25)
map_alpha = st.sidebar.slider("Map Transparency", 0.0, 1.0, 0.7)

# ---------------- MAIN ----------------
if csv_file and shp_files:
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        df["location_id"] = df["location_id"].astype(str).str.strip()

        # Load GDF
        shp_data_blobs = [(f.name, f.getvalue()) for f in shp_files]
        gdf = load_and_clean_gdf(shp_data_blobs)
        gdf["id"] = gdf["id"].astype(str).str.strip()

        # Merge
        merged = gdf.merge(df, left_on="id", right_on="location_id")
        
        if merged.empty:
            st.error("❌ IDs do not match. Check that CSV 'location_id' matches Shapefile 'id'.")
            st.stop()

        # ML Gap Filling
        data_present = merged[merged["pm10"].notna()].copy()
        data_missing = merged[merged["pm10"].isna()].copy()
        merged['status'] = 'Original'

        if not data_missing.empty and len(data_present) >= 3:
            X_train = np.array([(g.x, g.y) for g in data_present.geometry])
            y_train = data_present["pm10"].values
            X_predict = np.array([(g.x, g.y) for g in data_missing.geometry])

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            merged.loc[merged["pm10"].isna(), "pm10"] = model.predict(X_predict)
            merged.loc[merged["pm10"].isna() == False, "status"] = np.where(merged.loc[merged["pm10"].isna() == False, "location_id"].isin(data_missing["location_id"]), "AI Predicted", "Original")

        # Point to Grid logic
        half = cell_size / 2
        merged["geometry"] = merged.geometry.apply(lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half))

        # Output
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(10, 10))
            merged.plot(column="pm10", cmap="RdYlGn_r", alpha=map_alpha, ax=ax, legend=True)
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery)
            ax.set_axis_off()
            st.pyplot(fig)
        
        with col2:
            st.write("### Data Table")
            st.dataframe(merged[['location_id', 'pm10', 'status']])

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("Upload data to start.")
