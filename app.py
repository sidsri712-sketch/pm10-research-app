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
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Research Mapper", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=900)
def get_live_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url, timeout=12).json()
        if r.get('status') != 'ok': return None, "API error."
        stations = r['data']
        processed_data = []
        bar = st.progress(0, text="Fetching station details...")
        for i, s in enumerate(stations):
            detail_url = f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
            try:
                dr = requests.get(detail_url, timeout=5).json()
                if dr.get('status') == 'ok':
                    val = dr['data'].get('iaqi', {}).get('pm10', {}).get('v')
                    if val is not None:
                        processed_data.append({
                            'lat': float(s['lat']), 'lon': float(s['lon']),
                            'pm10': float(val), 'name': s.get('station', {}).get('name', 'Unknown')
                        })
            except: continue
            bar.progress((i + 1) / len(stations))
            time.sleep(0.1)
        bar.empty()
        return pd.DataFrame(processed_data), None
    except Exception as e: return None, str(e)

# --- SIDEBAR ---
st.sidebar.title("🛠️ Settings")
map_style = st.sidebar.selectbox("Base Map", ["carto-darkmatter", "carto-positron", "open-street-map"])
# RdYlGn_r is best for Air Quality (Green=Good, Red=Bad)
color_theme = st.sidebar.selectbox("Palette", ["RdYlGn_r", "Turbo", "Viridis", "YlOrRd"])
h_opacity = st.sidebar.slider("Opacity", 0.1, 1.0, 0.5)
radius_val = st.sidebar.slider("Blur Radius", 5, 50, 20)

st.title("🏙️ Lucknow PM10 Analysis")

if st.button("🚀 Sync Live Data"):
    df, err = get_live_pm10_data()
    if err: st.error(err)
    elif df is not None and not df.empty: st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # ML INTERPOLATION
    # Lower resolution (35x35) prevents the "All One Color" overlap in Plotly
    grid_res = 35 
    lat_range = np.linspace(df['lat'].min() - 0.01, df['lat'].max() + 0.01, grid_res)
    lon_range = np.linspace(df['lon'].min() - 0.01, df['lon'].max() + 0.01, grid_res)
    ln, lt = np.meshgrid(lon_range, lat_range)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[['lat', 'lon']], df['pm10'])
    preds = model.predict(np.c_[lt.ravel(), ln.ravel()]).reshape(grid_res, grid_res)
    preds = gaussian_filter(preds, sigma=1.0) # Sharpens the gradients

    grid_df = pd.DataFrame({'lat': lt.ravel(), 'lon': ln.ravel(), 'pm10': preds.ravel()})

    # BUILD THE HEATMAP
    fig = px.density_mapbox(
        grid_df, lat='lat', lon='lon', z='pm10',
        radius=radius_val, 
        center=dict(lat=26.8467, lon=80.9462),
        zoom=11,
        mapbox_style=map_style,
        color_continuous_scale=color_theme,
        opacity=h_opacity,
        # THIS FORCES THE GRADIENT:
        range_color=[df['pm10'].min(), df['pm10'].max()] 
    )

    # Station markers on top
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'], lon=df['lon'], mode='markers+text',
        marker=go.scattermapbox.Marker(size=12, color='white', opacity=0.8),
        text=df['pm10'].astype(int).astype(str),
        textposition="top center",
        hovertext=df['name'],
        showlegend=False
    ))

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, height=700)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Min PM10 (Green)", f"{df['pm10'].min():.0f}")
    c2.metric("Max PM10 (Red)", f"{df['pm10'].max():.0f}")
    c3.metric("Avg Level", f"{df['pm10'].mean():.1f}")
