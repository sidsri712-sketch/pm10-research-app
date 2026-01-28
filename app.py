import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
import time

# --- CONFIGURATION ---
# Replace with your actual token from https://aqicn.org/data-platform/token/
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Live Mapper", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_stdio=True)

@st.cache_data(ttl=900)
def get_live_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url, timeout=12).json()
        if r.get('status') != 'ok':
            return None, "API access error. Check your token."
        
        stations = r['data']
        processed_data = []
        
        progress_text = "Scanning Lucknow air quality stations..."
        bar = st.progress(0, text=progress_text)
        
        for i, s in enumerate(stations):
            detail_url = f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
            try:
                dr = requests.get(detail_url, timeout=5).json()
                if dr.get('status') == 'ok':
                    iaqi = dr['data'].get('iaqi', {})
                    pm10 = iaqi.get('pm10', {}).get('v')
                    if pm10 is not None:
                        processed_data.append({
                            'lat': float(s['lat']),
                            'lon': float(s['lon']),
                            'pm10': float(pm10),
                            'name': s.get('station', {}).get('name', 'Unknown')
                        })
            except:
                continue
            bar.progress((i + 1) / len(stations))
            time.sleep(0.1) # Polite API usage
        
        bar.empty()
        return pd.DataFrame(processed_data), None
    except Exception as e:
        return None, str(e)

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Map Customization")
st.sidebar.markdown("---")

map_style = st.sidebar.selectbox(
    "Base Map Theme", 
    ["carto-darkmatter", "carto-positron", "open-street-map", "stamen-terrain"],
    help="Dark mode makes pollution colors pop!"
)

color_theme = st.sidebar.selectbox(
    "Pollution Palette", 
    ["RdYlGn_r", "Turbo", "Viridis", "Plasma", "Inferno"],
    index=0
)

h_opacity = st.sidebar.slider("Heatmap Transparency", 0.1, 1.0, 0.7)
radius_val = st.sidebar.slider("Blur Radius", 10, 50, 25)

st.sidebar.markdown("---")
st.sidebar.info("This tool uses Random Forest Machine Learning to estimate PM10 levels between sensors.")

# --- MAIN INTERFACE ---
st.title("🏙️ Lucknow PM10 Research Mapper")
st.write(f"Showing real-time data for the Lucknow Bounding Box: `{LUCKNOW_BOUNDS}`")

if st.button("🚀 Fetch Live Station Data"):
    df, err = get_live_pm10_data()
    if err:
        st.error(err)
    elif df is None or df.empty:
        st.warning("No active PM10 sensors found in this area right now.")
    else:
        st.session_state['df'] = df
        st.success(f"Successfully synced with {len(df)} stations.")

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # 1. Machine Learning Interpolation
    # Create a grid for the background heatmap
    grid_res = 100 
    lat_range = np.linspace(df['lat'].min() - 0.02, df['lat'].max() + 0.02, grid_res)
    lon_range = np.linspace(df['lon'].min() - 0.02, df['lon'].max() + 0.02, grid_res)
    ln, lt = np.meshgrid(lon_range, lat_range)
    
    # Train RF model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[['lat', 'lon']], df['pm10'])
    
    # Predict over grid
    grid_points = np.c_[lt.ravel(), ln.ravel()]
    preds = model.predict(grid_points).reshape(grid_res, grid_res)
    preds = gaussian_filter(preds, sigma=1.5) # Soften the edges

    # Prepare data for Plotly Density Mapbox
    grid_df = pd.DataFrame({
        'lat': lt.ravel(),
        'lon': ln.ravel(),
        'pm10': preds.ravel()
    })

    # 2. Build Interactive Plotly Map
    fig = px.density_mapbox(
        grid_df, lat='lat', lon='lon', z='pm10',
        radius=radius_val,
        center=dict(lat=26.8467, lon=80.9462),
        zoom=11,
        mapbox_style=map_style,
        color_continuous_scale=color_theme,
        opacity=h_opacity,
        range_color=[df['pm10'].min() * 0.8, df['pm10'].max() * 1.1]
    )

    # Add the actual monitoring stations as clickable dots
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(size=12, color='white', opacity=0.9),
        text=df['pm10'].astype(int).astype(str),
        textposition="top center",
        hoverinfo='text',
        hovertext=df['name'] + "<br>Exact PM10: " + df['pm10'].astype(str),
        name="Stations"
    ))

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=700,
        title=f"Interpolated PM10 Map (Updated: {pd.Timestamp.now().strftime('%H:%M')})"
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg PM10", f"{df['pm10'].mean():.1f} µg/m³")
    with col2:
        st.metric("Max PM10", f"{df['pm10'].max():.1f} µg/m³", delta_color="inverse")
    with col3:
        st.metric("Stations", len(df))

    # Data Table
    with st.expander("View Raw Station Data"):
        st.dataframe(df.sort_values(by='pm10', ascending=False), use_container_width=True)
