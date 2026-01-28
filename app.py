import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Pro PM10 Heatmapper", layout="wide")
st.title("🌍 Smart PM10 Smooth Heatmap")

# ---------------- ML GAP FILLING ----------------
def ml_fill_gaps(df, lat_col, lon_col, pm10_col):
    """Predicts missing PM10 values to ensure a continuous heatmap surface."""
    data_present = df[df[pm10_col].notna()].copy()
    data_missing = df[df[pm10_col].isna()].copy()
    
    if data_missing.empty or len(data_present) < 5:
        return df
    
    # Train on available coordinates to predict 'unknown' spots
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(data_present[[lat_col, lon_col]], data_present[pm10_col])
    
    df.loc[df[pm10_col].isna(), pm10_col] = model.predict(data_missing[[lat_col, lon_col]])
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data & Style")
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("🔥 Heatmap Settings")
# These controls allow you to fine-tune the 'smoothness' of the map
radius = st.sidebar.slider("Heat Radius", 10, 100, 35, help="Size of the 'glow' around points")
blur = st.sidebar.slider("Blur Amount", 1, 50, 15, help="Smoothness of the transition")
min_opacity = st.sidebar.slider("Transparency", 0.0, 1.0, 0.4)

# ---------------- MAIN APP ----------------
if csv_file:
    df = pd.read_csv(csv_file)
    
    # Identify columns (case-insensitive)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

    if lat_col and lon_col and pm10_col:
        # Step 1: Fill missing PM10 gaps using ML
        df = ml_fill_gaps(df, lat_col, lon_col, pm10_col)
        
        # Step 2: Create Map. 'CartoDB dark_matter' makes colors pop.
        m = folium.Map(
            location=[df[lat_col].mean(), df[lon_col].mean()], 
            zoom_start=12, 
            tiles="CartoDB dark_matter" 
        )

        # Step 3: Prepare Heatmap data [[lat, lon, weight]]
        heat_data = df[[lat_col, lon_col, pm10_col]].values.tolist()
        
        # Add smooth HeatMap layer
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)

        # Render the map
        st_folium(m, width="100%", height=600)
        
        st.success("Heatmap generated! Use the sidebar to adjust the 'smoothness'.")
    else:
        st.error("CSV must contain 'latitude', 'longitude', and 'pm10' columns.")
else:
    st.info("👆 Please upload a CSV to get started.")
