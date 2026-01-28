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
st.title("🌿 Professional PM10 Heatmap")

# ---------------- ML GAP FILLING ----------------
def ml_fill_gaps(df, lat_col, lon_col, pm10_col):
    data_present = df[df[pm10_col].notna()].copy()
    data_missing = df[df[pm10_col].isna()].copy()
    
    if data_missing.empty or len(data_present) < 5:
        return df
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(data_present[[lat_col, lon_col]], data_present[pm10_col])
    
    df.loc[df[pm10_col].isna(), pm10_col] = model.predict(data_missing[[lat_col, lon_col]])
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.header("🎨 Heatmap Controls")
radius = st.sidebar.slider("Heat Radius", 10, 80, 40, help="How far the color 'glows'")
blur = st.sidebar.slider("Blur", 1, 50, 20)
min_opacity = st.sidebar.slider("Transparency", 0.0, 1.0, 0.4)

csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ---------------- MAIN APP ----------------
if csv_file:
    df = pd.read_csv(csv_file)
    
    # Simple column detection
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

    if lat_col and lon_col and pm10_col:
        # Fill missing values with ML
        df = ml_fill_gaps(df, lat_col, lon_col, pm10_col)
        
        # Create Folium Map - "CartoDB dark_matter" makes the heat colors pop!
        m = folium.Map(location=[df[lat_col].mean(), df[lon_col].mean()], 
                       zoom_start=11, 
                       tiles="CartoDB dark_matter")

        # Prepare heat data
        heat_data = df[[lat_col, lon_col, pm10_col]].values.tolist()
        
        # Add the HeatMap layer (Smooth gradients)
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)

        st_folium(m, width="100%", height=600)
        
        st.success("✅ Heatmap generated! Use the sidebar to adjust the 'glow' effect.")
    else:
        st.error("CSV must contain columns for Latitude, Longitude, and PM10.")
else:
    st.info("Waiting for CSV upload...")
