import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box, Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import io
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Smart PM10 Mapper", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌍 Smart PM10 Mapping Tool (ML Powered)")
st.markdown("---")

# ---------------- INSTRUCTIONS ----------------
with st.expander("📖 Instructions", expanded=False):
    st.markdown("""
    **CSV Requirements:**
    - `location_id` (matches shapefile `id`)
    - `pm10` (numeric PM10 values)
    
    **Shapefile Requirements:**
    - Must contain `id` column (case-insensitive)
    - Supports `.shp`, `.shx`, `.dbf`, `.prj` files
    
    **Features:**
    - ✅ Auto-fills missing PM10 values with ML
    - ✅ Converts points to customizable grid
    - ✅ Interactive map with basemap
    - ✅ Data validation & error handling
    """)

# ---------------- PROCESSING FUNCTIONS ----------------
@st.cache_data
def load_and_clean_gdf(shp_path_data):
    """Load and clean shapefile data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        for f_name, f_content in shp_path_data:
            with open(os.path.join(tmpdir, f_name), "wb") as out:
                out.write(f_content)
        
        shp_file = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) 
                   if f.endswith(".shp")][0]
        gdf = gpd.read_file(shp_file)
        
        # Handle ID column (case-insensitive)
        id_cols = [c for c in gdf.columns if c.lower() == 'id']
        if not id_cols:
            st.error(f"❌ No 'id' column found. Available: {list(gdf.columns)}")
            st.stop()
        
        gdf = gdf.rename(columns={id_cols[0]: 'id'})
        gdf["id"] = gdf["id"].astype(str).str.strip()
        
        return gdf.to_crs(epsg=3857)

def validate_csv(df):
    """Validate CSV structure"""
    required_cols = ['location_id', 'pm10']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns: {missing_cols}")
        return False
    
    df["location_id"] = df["location_id"].astype(str).str.strip()
    return True

def ml_fill_gaps(merged):
    """ML-based gap filling for missing PM10 values"""
    data_present = merged[merged["pm10"].notna()].copy()
    data_missing = merged[merged["pm10"].isna()].copy()
    
    if data_missing.empty or len(data_present) < 3:
        return merged
    
    # Extract coordinates
    X_train = np.column_stack([data_present.geometry.x, data_present.geometry.y])
    y_train = data_present["pm10"].values
    
    X_predict = np.column_stack([data_missing.geometry.x, data_missing.geometry.y])
    
    # Train ML model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict missing values
    predictions = model.predict(X_predict)
    merged.loc[merged["pm10"].isna(), "pm10"] = predictions
    
    # Track prediction status
    merged['status'] = 'Original'
    merged.loc[merged.index.isin(data_missing.index), 'status'] = 'AI Predicted'
    
    return merged

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("📊 CSV File", type=["csv"], help="location_id + pm10 columns")
shp_files = st.sidebar.file_uploader(
    "🗺️ Shapefile Set", 
    type=["shp", "shx", "dbf", "prj"], 
    accept_multiple_files=True,
    help="Upload all related files (.shp, .shx, .dbf, .prj)"
)

st.sidebar.header("🎛️ Map Settings")
cell_size = st.sidebar.slider("Grid Size (km)", 1, 100, 25, help="Size of each grid square")
map_alpha = st.sidebar.slider("Map Transparency", 0.1, 1.0, 0.7, 0.1)
color_scheme = st.sidebar.selectbox("Color Scheme", ["RdYlGn_r", "Reds", "viridis", "plasma"])

st.sidebar.markdown("---")
st.sidebar.caption("💡 ML auto-fills missing PM10 values")

# ---------------- MAIN APP ----------------
if csv_file and shp_files:
    with st.spinner("🔄 Processing your data..."):
        try:
            # Load and validate data
            df = pd.read_csv(csv_file)
            if not validate_csv(df):
                st.stop()
            
            shp_data_blobs = [(f.name, f.getvalue()) for f in shp_files]
            gdf = load_and_clean_gdf(shp_data_blobs)
            
            # Merge data
            merged = gdf.merge(df, left_on="id", right_on="location_id", how="left")
            
            if merged["pm10"].isna().all():
                st.error("❌ No matching IDs found between CSV and shapefile.")
                st.info("Check that `location_id` (CSV) matches `id` (shapefile)")
                st.stop()
            
            # ML gap filling
            merged = ml_fill_gaps(merged)
            
            # Convert to grid
            half_size = cell_size * 1000  # Convert km to meters
            merged["grid_geometry"] = merged.geometry.apply(
                lambda p: box(p.x - half_size, p.y - half_size, p.x + half_size, p.y + half_size)
            )
            merged = merged.set_geometry("grid_geometry")
            
            # Stats
            total_points = len(merged)
            predicted = len(merged[merged['status'] == 'AI Predicted'])
            avg_pm10 = merged["pm10"].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Points", total_points)
            col2.metric("AI Predictions", predicted, delta=f"{predicted/total_points:.1%}")
            col3.metric("Avg PM10", f"{avg_pm10:.1f} µg/m³")
            col4.metric("Data Coverage", f"{1-merged['pm10'].isna().mean():.1%}")
            
            # Visualization
            col_left, col_right = st.columns([3, 1])
            
            with col_left:
                st.subheader("🗺️ PM10 Heatmap")
                fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
                merged.plot(
                    column="pm10", 
                    cmap=color_scheme, 
                    alpha=map_alpha, 
                    ax=ax, 
                    legend=True,
                    legend_kwds={'shrink': 0.8, 'label': "PM10 (µg/m³)"},
                    missing_kwds={'color': 'lightgrey', 'alpha': 0.3, 'label': 'No Data'}
                )
                cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, zoom=10, alpha=0.8)
                ax.set_axis_off()
                ax.set_title("PM10 Distribution Map", fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig)
            
            with col_right:
                st.subheader("📊 Data Summary")
                st.dataframe(
                    merged[['id', 'location_id', 'pm10', 'status']].round(2),
                    use_container_width=True,
                    height=400
                )
                
                st.subheader("PM10 Statistics")
                st.json({
                    "Min": f"{merged['pm10'].min():.1f}",
                    "Max": f"{merged['pm10'].max():.1f}",
                    "Mean": f"{merged['pm10'].mean():.1f}",
                    "Std": f"{merged['pm10'].std():.1f}"
                })
            
            # Download options
            st.markdown("---")
            csv_download = merged[['id', 'location_id', 'pm10', 'status']].round(2)
            st.download_button(
                "💾 Download Results CSV",
                csv_download.to_csv(index=False),
                f"pm10_results_{cell_size}km.csv",
                "text/csv"
            )
            
        except Exception as e:
            st.error(f"⚠️ Processing error: {str(e)}")
            st.info("💡 Common fixes: Check file formats, column names, or try smaller files")

else:
    st.info("👆 **Please upload both CSV and Shapefile data to get started**")
    st.markdown("### Example Workflow:")
    st.markdown("""
    1. **CSV**: `location_id,pm10` (e.g., "001,25.3")
    2. **Shapefile**: Monitoring station locations with matching `id`
    3. **Result**: ML-enhanced heatmap + gap-filled data
    """)
