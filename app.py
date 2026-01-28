import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import warnings

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Pro PM10 Heatmapper", layout="wide")

st.title("🌿 Professional PM10 Heatmap (ML Powered)")
st.markdown("---")

# ---------------- PROCESSING FUNCTIONS ----------------
def ml_fill_gaps(df, lat_col, lon_col, pm10_col):
    """Predicts missing PM10 values using Random Forest"""
    data_present = df[df[pm10_col].notna()].copy()
    data_missing = df[df[pm10_col].isna()].copy()
    
    if data_missing.empty or len(data_present) < 5:
        df['status'] = 'Original'
        return df
    
    # Train on coordinates
    X_train = data_present[[lat_col, lon_col]]
    y_train = data_present[pm10_col]
    X_predict = data_missing[[lat_col, lon_col]]
    
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    
    df.loc[df[pm10_col].isna(), pm10_col] = model.predict(X_predict)
    df['status'] = 'Original'
    df.loc[df.index.isin(data_missing.index), 'status'] = 'AI Predicted'
    
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.header("📁 Data Source")
csv_file = st.sidebar.file_uploader("Upload Air Quality CSV", type=["csv"])

st.sidebar.header("🔥 Heatmap Settings")
radius = st.sidebar.slider("Heat Intensity Radius", 10, 50, 25)
blur = st.sidebar.slider("Blur Amount", 5, 30, 15)
min_opacity = st.sidebar.slider("Min Opacity", 0.0, 1.0, 0.5)

# ---------------- MAIN APP ----------------
if csv_file:
    df = pd.read_csv(csv_file)
    
    # Auto-detect columns
    lat_col = next((c for c in df.columns if c.lower() in ['lat', 'latitude']), None)
    lon_col = next((c for c in df.columns if c.lower() in ['lon', 'longitude', 'long']), None)
    pm10_col = next((c for c in df.columns if c.lower() == 'pm10'), None)

    if lat_col and lon_col and pm10_col:
        # 1. ML Prediction & Gap Filling
        df = ml_fill_gaps(df, lat_col, lon_col, pm10_col)
        
        # 2. Stats Dashboard
        c1, c2, c3 = st.columns(3)
        c1.metric("Coverage", f"{len(df)} Points")
        c2.metric("Mean PM10", f"{df[pm10_col].mean():.1f} µg/m³")
        c3.metric("AI Predictions", len(df[df['status'] == 'AI Predicted']))

        # 3. Interactive Heatmap
        st.subheader("🗺️ Interactive Air Quality Heatmap")
        
        # Center map on data average
        m = folium.Map(
            location=[df[lat_col].mean(), df[lon_col].mean()], 
            zoom_start=12, 
            tiles="CartoDB positron" # Clean, modern basemap
        )

        # Prepare data for HeatMap plugin: [[lat, lon, weight], ...]
        heat_data = df[[lat_col, lon_col, pm10_col]].values.tolist()
        
        HeatMap(
            heat_data,
            radius=radius,
            blur=blur,
            min_opacity=min_opacity,
            gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}
        ).add_to(m)

        # Render Folium Map
        st_folium(m, width=1200, height=600)

        # 4. Data Preview & Export
        with st.expander("View Processed Data"):
            st.dataframe(df.style.background_gradient(subset=[pm10_col], cmap='YlOrRd'))
            st.download_button("💾 Download Full Results", df.to_csv(index=False), "air_quality_results.csv")
    else:
        st.error("CSV must contain Lat, Lon, and PM10 columns.")
else:
    st.info("Please upload your CSV to generate the heatmap.")
