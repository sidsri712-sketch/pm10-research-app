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

# ---------------- CONFIG & CACHING ----------------
st.set_page_config(page_title="Smart PM10 Mapper", layout="wide")

# This helps the app load faster after "sleeping" by remembering processed data
@st.cache_data
def process_spatial_data(csv_df, shp_path_list):
    with tempfile.TemporaryDirectory() as tmpdir:
        for f_name, f_content in shp_path_list:
            with open(os.path.join(tmpdir, f_name), "wb") as out:
                out.write(f_content)
        shp_file = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")][0]
        gdf = gpd.read_file(shp_file)
    return gdf

# ---------------- UI HELPERS ----------------
st.title("🌍 Smart PM10 Mapping Tool (ML Powered)")
st.info("💡 **Tip:** If the app was 'asleep', it may take 30 seconds to boot up. Once open, it will run fast!")

with st.expander("📖 Instructions: How to prepare your files"):
    st.markdown("""
    ### 1. CSV Data Format
    Your CSV must have exactly two columns:
    - **location_id**: The name/ID of your sensor.
    - **pm10**: The value. **Leave empty (NaN) or blank** if you want the AI to predict it.
    
    ### 2. Shapefile Data
    Upload the set of files (.shp, .shx, .dbf, .prj). 
    - The **'id'** attribute in your shapefile must match the **'location_id'** in the CSV.
    """)

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
shp_files = st.sidebar.file_uploader("Upload Shapefile Set", type=["shp", "shx", "dbf", "prj"], accept_multiple_files=True)

st.sidebar.header("⚙️ Map Settings")
cell_size = st.sidebar.slider("Cell Size (Square Width)", 5, 100, 20, help="Adjust this if you see gaps between your grid points.")
map_alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.7)

# ---------------- MAIN LOGIC ----------------
if csv_file and shp_files:
    try:
        # Load CSV
        df = pd.read_csv(csv_file)
        df["location_id"] = df["location_id"].astype(str).str.strip()

        # Load Shapefile using cache-friendly method
        shp_data = [(f.name, f.getvalue()) for f in shp_files]
        gdf = process_spatial_data(df, shp_data)
        gdf["id"] = gdf["id"].astype(str).str.strip()
        gdf = gdf.to_crs(epsg=3857) # Project to Web Mercator

        # Merge
        merged = gdf.merge(df, left_on="id", right_on="location_id")
        
        if merged.empty:
            st.error("❌ Merge failed. The 'location_id' in your CSV does not match the 'id' in your Shapefile.")
            st.stop()

        # 🤖 MACHINE LEARNING GAP FILLING
        # Identify rows with and without data
        data_present = merged[merged["pm10"].notna()].copy()
        data_missing = merged[merged["pm10"].isna()].copy()
        
        merged['status'] = 'Original' # Track which values are real vs predicted

        if not data_missing.empty:
            if len(data_present) < 5:
                st.warning("⚠️ Need at least 5 filled sensors to use Machine Learning accurately. Showing current data only.")
            else:
                with st.spinner("🤖 AI is learning spatial patterns to fill gaps..."):
                    # Feature Extraction: Use X and Y coordinates to predict PM10
                    X_train = np.array([(g.x, g.y) for g in data_present.geometry])
                    y_train = data_present["pm10"].values
                    X_predict = np.array([(g.x, g.y) for g in data_missing.geometry])

                    # Random Forest Model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Fill missing values
                    predictions = model.predict(X_predict)
                    merged.loc[merged["pm10"].isna(), "pm10"] = predictions
                    merged.loc[merged["location_id"].isin(data_missing["location_id"]), "status"] = "AI Predicted"
                    st.success(f"✅ AI successfully filled {len(data_missing)} gaps!")

        # ---------- CREATE CHOROPLETH GRID ----------
        # Transform points into squares for a "Choropleth" look
        half = cell_size / 2
        merged["geometry"] = merged.geometry.apply(lambda p: box(p.x - half, p.y - half, p.x + half, p.y + half))

        # ---------- DISPLAY ----------
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📍 PM10 Choropleth Map")
            fig, ax = plt.subplots(figsize=(10, 10))
            
            merged.plot(
                column="pm10", 
                cmap="RdYlGn_r", 
                alpha=map_alpha, 
                ax=ax, 
                legend=True, 
                legend_kwds={'label': "PM10 (µg/m³)", 'orientation': "horizontal"},
                zorder=2
            )
            
            cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zorder=1)
            ax.set_axis_off()
            
            # Zoom to bounds
            xmin, ymin, xmax, ymax = merged.total_bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Data Preview")
            st.write("Check the 'status' column to see what the AI predicted.")
            # Displaying the table with a color highlight for predictions
            st.dataframe(merged[['location_id', 'pm10', 'status']].style.apply(
                lambda x: ['background-color: #fce4ec' if v == 'AI Predicted' else '' for v in x], 
                subset=['status']
            ))

        # ---------- DOWNLOAD ----------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button("🖼️ Download Map Image", buf.getvalue(), "pm10_map.png")

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("👋 Welcome! Please upload your CSV and Shapefile in the sidebar to generate your map.")
