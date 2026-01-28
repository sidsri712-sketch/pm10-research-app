import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import io
import time

# --- CONFIGURATION ---
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

st.set_page_config(page_title="Lucknow PM10 Research Hub", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stButton>button { width: 100%; background-color: #007bff; color: white; border-radius: 8px; font-weight: bold; }
    .reportview-container { background: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=900)
def get_live_pm10_data():
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    try:
        r = requests.get(url, timeout=12).json()
        if r.get('status') != 'ok': return None, "API access error."
        stations = r['data']
        processed_data = []
        bar = st.progress(0, text="Fetching scientific data from Lucknow stations...")
        for i, s in enumerate(stations):
            detail_url = f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}"
            try:
                dr = requests.get(detail_url, timeout=5).json()
                if dr.get('status') == 'ok':
                    val = dr['data'].get('iaqi', {}).get('pm10', {}).get('v')
                    if val is not None:
                        processed_data.append({
                            'lat': float(s['lat']), 'lon': float(s['lon']),
                            'pm10': float(val), 'name': s.get('station', {}).get('name', 'Station')
                        })
            except: continue
            bar.progress((i + 1) / len(stations))
            time.sleep(0.1)
        bar.empty()
        return pd.DataFrame(processed_data), None
    except Exception as e: return None, str(e)

st.title("🏙️ Lucknow PM10 Spatial Distribution Analysis")
st.info("This tool generates publication-quality heatmaps using Radial Basis Function (Rbf) Interpolation.")

if st.button("🔄 Sync Current Lucknow Data"):
    df, err = get_live_pm10_data()
    if err: st.error(err)
    elif df is not None and not df.empty: st.session_state['df'] = df

if 'df' in st.session_state:
    df = st.session_state['df']
    
    # --- SCIENTIFIC INTERPOLATION ---
    # 1. Setup Grid
    grid_x, grid_y = np.mgrid[df.lon.min():df.lon.max():200j, df.lat.min():df.lat.max():200j]
    
    # 2. Rbf Interpolation (Multiquadric) - Creates clear red/green gradients
    rbf = Rbf(df.lon, df.lat, df.pm10, function='multiquadric', smooth=0.1)
    z = rbf(grid_x, grid_y)

    # --- MATPLOTLIB PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # Plot the interpolated surface
    # 'RdYlGn_r' ensures High PM10 = Red, Low PM10 = Green
    im = ax.imshow(z.T, extent=(df.lon.min(), df.lon.max(), df.lat.min(), df.lat.max()), 
                   origin='lower', cmap='RdYlGn_r', aspect='auto', 
                   vmin=df.pm10.min(), vmax=df.pm10.max())
    
    # Add actual station points for reference
    ax.scatter(df.lon, df.lat, c='white', edgecolors='black', s=50, label='Stations')
    
    # Annotate points with values
    for i, txt in enumerate(df.pm10):
        ax.annotate(int(txt), (df.lon.iloc[i], df.lat.iloc[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    # Formatting for Research Paper
    plt.colorbar(im, ax=ax, label='PM10 Concentration (µg/m³)')
    ax.set_title("Lucknow PM10 Spatial Interpolation Map", fontsize=14, pad=20)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Display in Streamlit
    st.pyplot(fig)

    # --- PNG EXPORT FOR RESEARCH PAPER ---
    fn = "Lucknow_PM10_Heatmap.png"
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight') # 300 DPI is required for papers

    st.download_button(
        label="📥 Download PNG for Research Paper (High Res)",
        data=img.getvalue(),
        file_name=fn,
        mime="image/png"
    )

    # Statistics Table
    st.subheader("Station Data Summary")
    st.dataframe(df[['name', 'pm10']].sort_values(by='pm10', ascending=False), use_container_width=True)
