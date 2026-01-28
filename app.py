import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st_stats
from sklearn.ensemble import RandomForestRegressor
import io

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Research Heatmap Export", layout="wide")
st.title("🧪 Spatial Analysis: High-Res Heatmap Export")

# ---------------- CORE FUNCTIONS ----------------
def generate_high_res_heatmap(df, lat_col, lon_col, pm10_col, resolution=200, sigma=2):
    """Creates a continuous, spread-out heatmap surface."""
    # 1. Fill missing values with ML so the surface is complete
    data_present = df[df[pm10_col].notna()]
    data_missing = df[df[pm10_col].isna()]
    
    if not data_missing.empty:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(data_present[[lat_col, lon_col]], data_present[pm10_col])
        df.loc[df[pm10_col].isna(), pm10_col] = model.predict(data_missing[[lat_col, lon_col]])

    # 2. Create a dense grid across the entire extent
    x = df[lon_col].values
    y = df[lat_col].values
    z = df[pm10_col].values

    xi = np.linspace(x.min(), x.max(), resolution)
    yi = np.linspace(y.min(), y.max(), resolution)
    xi, yi = np.meshgrid(xi, yi)

    # 3. Gaussian Interpolation (spreads the data across the image)
    from scipy.interpolate import griddata
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    # Fill edges where linear interpolation fails
    zi_nearest = griddata((x, y), z, (xi, yi), method='nearest')
    zi = np.where(np.isnan(zi), zi_nearest, zi)

    return xi, yi, zi

# ---------------- UI & RENDER ----------------
csv_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
dpi_value = st.sidebar.select_slider("Export Resolution (DPI)", options=[100, 300, 600], value=300)
cmap_choice = st.sidebar.selectbox("Color Theme", ["RdYlGn_r", "magma", "viridis", "Spectral_r"])

if csv_file:
    df = pd.read_csv(csv_file)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    pm10_col = next((c for c in df.columns if 'pm10' in c.lower()), None)

    if lat_col and lon_col and pm10_col:
        xi, yi, zi = generate_high_res_heatmap(df, lat_col, lon_col, pm10_col)

        # Matplotlib Plotting for PNG Export
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # The 'shading' and 'interpolation' here make it look smooth for research
        im = ax.pcolormesh(xi, yi, zi, cmap=cmap_choice, shading='auto', antialiased=True)
        
        # Formatting for paper
        plt.colorbar(im, label='PM10 Concentration (µg/m³)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f"Spatial Distribution of PM10", fontsize=14)
        
        st.pyplot(fig)

        # Buffer for Download
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi_value, bbox_inches='tight')
        st.download_button(
            label=f"💾 Download High-Res PNG ({dpi_value} DPI)",
            data=buf.getvalue(),
            file_name="pm10_spatial_analysis.png",
            mime="image/png"
        )
else:
    st.info("Upload your CSV. The app will automatically interpolate values to create a full-spread heatmap.")
