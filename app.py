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
    - **CSV:** Should contain `location_id` and `pm10`.
    - **Shapefile:** Must contain an `id` column.
    - **AI Feature:** If you leave `pm10` cells empty in the CSV, the AI will predict them!
    """)

# ---------------- HELPERS ----------------
def fuzzy_column_match(df, target_name):
    """Finds a column in a dataframe ignoring case and extra spaces."""
    cols = {c.lower().strip(): c for c in df.columns}
    target = target_name.lower().strip()
    if target in cols:
        return cols[target]
    return None

@st.cache_data
def load_gdf(shp_path_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        for f_name, f_content in shp_path_data:
            with open(os.path.join(tmpdir, f_name), "wb") as out:
                out.write(f_content)
        shp_file = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
        gdf = gpd.read_file(shp_file)
        
        # Resolve 'id' column naming
        id_col = fuzzy_column_match(gdf, 'id')
        if id_col:
            gdf = gdf.rename(columns={id_col: 'id'})
        else:
            st.error(f"❌ Shapefile is missing an 'id' column. Found: {list(gdf.columns)}")
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
        # Load and Clean CSV
        df = pd.read_csv(csv_file)
        
        loc_col = fuzzy_column_match(df, 'location_id')
        pm_col = fuzzy_column_match(df, 'pm10')
        
        if not loc_col or not pm_col:
            st.error(f"❌ CSV columns missing. Required: 'location_id' and 'pm10'. Found: {list(df.columns)}")
            st.stop()
            
        df = df.rename(columns={loc_col: 'location_id', pm_col: 'pm10'})
        df["location_id"] = df["location_id"].astype(str).str.strip()

        # Load GDF
        shp_data_blobs = [(f.name, f.getvalue()) for f in shp_files]
        gdf = load_gdf(shp_data_blobs)
        gdf["id"] = gdf["id"].astype(str).str.strip()

        # Merge
        merged = gdf.merge(df, left_on="id", right_on="location_id")
        if merged.empty:
            st.error("❌ IDs do not match between files.")
            st.stop()

        # AI Gap Filling (Using modern np.ptp logic internally)
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
            merged.loc[merged["location_id"].isin(data_missing["location_id"]), "status"] = "AI Predicted"

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
        st.error(f"⚠️ Process Error: {e}")
else:
    st.info("Please upload your files to begin.")
