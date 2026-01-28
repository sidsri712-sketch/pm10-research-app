import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import box, Point
from sklearn.ensemble import RandomForestRegressor
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
    - `lat` / `latitude`: Numeric latitude
    - `lon` / `longitude`: Numeric longitude
    - `pm10`: Numeric PM10 values (missing values are okay!)
    
    **Features:**
    - ✅ **No Shapefile needed**: Uses Lat/Lon from CSV
    - ✅ **ML Gap Filling**: Predicts PM10 for rows with missing values based on location
    - ✅ **Auto-Gridding**: Converts points to square heatmap cells
    """)

# ---------------- PROCESSING FUNCTIONS ----------------
def validate_and_geo_csv(df):
    """Ensure CSV has lat/lon and convert to GeoDataFrame"""
    cols = df.columns.str.lower()
    lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'long']), None)
    pm10_col = next((c for c in df.columns if c.lower() == 'pm10'), None)

    if not lat_col or not lon_col or not pm10_col:
        st.error(f"❌ Missing required columns. Found: {list(df.columns)}")
        st.info("Ensure your CSV has: 'lat', 'lon', and 'pm10'")
        return None
    
    # Drop rows where coordinates are missing (cannot map those)
    df = df.dropna(subset=[lat_col, lon_col])
    
    # Convert to GeoDataFrame (WGS84 initially)
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    # Convert to Web Mercator for accurate km-based buffering and basemaps
    return gdf.to_crs(epsg=3857), pm10_col

def ml_fill_gaps(gdf, pm10_col):
    """ML-based gap filling for missing PM10 values"""
    data_present = gdf[gdf[pm10_col].notna()].copy()
    data_missing = gdf[gdf[pm10_col].isna()].copy()
    
    if data_missing.empty or len(data_present) < 3:
        gdf['status'] = 'Original'
        return gdf
    
    X_train = np.column_stack([data_present.geometry.x, data_present.geometry.y])
    y_train = data_present[pm10_col].values
    X_predict = np.column_stack([data_missing.geometry.x, data_missing.geometry.y])
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_predict)
    gdf.loc[gdf[pm10_col].isna(), pm10_col] = predictions
    
    gdf['status'] = 'Original'
    gdf.loc[gdf.index.isin(data_missing.index), 'status'] = 'AI Predicted'
    
    return gdf

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Upload")
csv_file = st.sidebar.file_uploader("📊 CSV File", type=["csv"])

st.sidebar.header("🎛️ Map Settings")
cell_size = st.sidebar.slider("Grid Size (km)", 1, 50, 5)
map_alpha = st.sidebar.slider("Map Transparency", 0.1, 1.0, 0.7)
color_scheme = st.sidebar.selectbox("Color Scheme", ["RdYlGn_r", "viridis", "YlOrRd"])

# ---------------- MAIN APP ----------------
if csv_file:
    try:
        raw_df = pd.read_csv(csv_file)
        result = validate_and_geo_csv(raw_df)
        
        if result:
            gdf, pm10_col = result
            
            # 1. Fill Gaps
            gdf = ml_fill_gaps(gdf, pm10_col)
            
            # 2. Create Grid Geometry
            half_size = (cell_size * 1000) / 2
            gdf["grid_geometry"] = gdf.geometry.apply(
                lambda p: box(p.x - half_size, p.y - half_size, p.x + half_size, p.y + half_size)
            )
            gdf = gdf.set_geometry("grid_geometry")
            
            # 3. Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Points", len(gdf))
            col2.metric("Avg PM10", f"{gdf[pm10_col].mean():.1f}")
            col3.metric("AI Predicted", len(gdf[gdf['status']=='AI Predicted']))
            
            # 4. Visualization
            c_left, c_right = st.columns([3, 1])
            
            with c_left:
                fig, ax = plt.subplots(figsize=(10, 8))
                gdf.plot(
                    column=pm10_col, 
                    cmap=color_scheme, 
                    alpha=map_alpha, 
                    ax=ax, 
                    legend=True,
                    legend_kwds={'label': "PM10 Concentration"}
                )
                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
                ax.set_axis_off()
                st.pyplot(fig)
                
            with c_right:
                st.write("### Preview")
                st.dataframe(gdf[[pm10_col, 'status']].head(10), use_container_width=True)
                
                # Download
                csv_out = gdf.drop(columns='grid_geometry').to_csv(index=False)
                st.download_button("💾 Download Results", csv_out, "pm10_output.csv", "text/csv")
                
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("💡 Please upload a CSV with 'lat', 'lon', and 'pm10' columns.")
