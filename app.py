import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
import time

# --- Configuration ---
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Pro Mapper", layout="wide")

@st.cache_data(ttl=900)
def get_live_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url, timeout=12).json()
        if r.get('status') != 'ok': return None, "API Error"
        
        stations = r['data']
        processed_data = []
        
        bar = st.progress(0, text="Contacting Lucknow monitoring stations...")
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
                            'name': s.get('station', {}).get('name', 'Unknown Station')
                        })
            except: continue
            bar.progress((i + 1) / len(stations))
            time.sleep(0.2)
        
        return pd.DataFrame(processed_data), None
    except Exception as e:
        return None, str(e)

# --- UI Sidebar ---
st.sidebar.header("🎨 Visual Settings")
map_style = st.sidebar.selectbox("Base Map Style", 
    ["carto-positron", "carto-darkmatter", "stamen-terrain", "open-street-map"])

color_theme = st.sidebar.selectbox("Heatmap Palette", 
    ["Viridis", "Plasma", "Inferno", "Turbo", "Hot", "RdYlGn_r"])

heatmap_opacity = st.sidebar.slider("Heatmap Opacity", 0.1, 1.0, 0.6)
point_size = st.sidebar.slider("Station Dot Size", 5, 20, 12)

st.title("🏙️ Lucknow PM10 Interactive Analysis")
st.markdown("Real-time air quality interpolation using Machine Learning.")

if st.button("🔄 Fetch & Analyze Live Data"):
    df, err = get_live_pm10_data()
    if err: st.error(err)
    else: st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # ML Interpolation for the "Smooth" look
    res = 100 
    lat_range = np.linspace(df['lat'].min() - 0.01, df['lat'].max() + 0.01, res)
    lon_range = np.linspace(df['lon'].min() - 0.01, df['lon'].max() + 0.01, res)
    ln, lt = np.meshgrid(lon_range, lat_range)
    
    model = RandomForestRegressor(n_estimators=50)
    model.fit(df[['lat', 'lon']], df['pm10'])
    grid_points = np.c_[lt.ravel(), ln.ravel()]
    preds = model.predict(grid_points).reshape(res, res)
    preds = gaussian_filter(preds, sigma=2) # Smooth out the RF "blocks"

    # Flatten grid for Plotly
    grid_df = pd.DataFrame({
        'lat': lt.ravel(),
        'lon': ln.ravel(),
        'pm10': preds.ravel()
    })

    # --- Create Figure ---
    fig = px.density_mapbox(
        grid_df, lat='lat', lon='lon', z='pm10',
        radius=25, # Adjusts how much the colors bleed
        center=dict(lat=26.8467, lon=80.9462),
        zoom=11,
        mapbox_style=map_style,
        color_continuous_scale=color_theme,
        opacity=heatmap_opacity,
        range_color=[df['pm10'].min(), df['pm10'].max()]
    )

    # Add the actual station dots on top
    fig.add_trace(go.Scattermapbox(
        lat=df['lat'],
        lon=df['lon'],
        mode='markers+text',
        marker=go.scattermapbox.Marker(
            size=point_size,
            color='white',
            opacity=1.0
        ),
        text=df['pm10'].astype(int).astype(str),
        textposition="top center",
        hoverinfo='text',
        hovertext=df['name'] + "<br>PM10: " + df['pm10'].astype(str)
    ))

    fig.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        height=700,
        coloraxis_colorbar=dict(title="PM10 (µg/m³)")
    )

    st.plotly_chart(fig, use_container_width=True)
    
    # Quick Statistics
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg PM10", f"{df['pm10'].mean():.1f}")
    c2.metric("Highest Reading", f"{df['pm10'].max():.1f}")
    c3.metric("Stations Active", len(df))
