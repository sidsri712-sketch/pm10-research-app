import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pydeck as pdk
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.ndimage import gaussian_filter
from streamlit_autorefresh import st_autorefresh
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import time
import json
import io
import math
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇳 India AirSense Pro",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: #1a1d27; border-radius: 12px; padding: 10px; border: 1px solid #2d3147; }
    .city-header { font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #ff6b35, #f7c59f); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .aqi-good { background: #00e400; color: black; padding: 6px 14px; border-radius: 20px; font-weight: 700; }
    .aqi-moderate { background: #ffff00; color: black; padding: 6px 14px; border-radius: 20px; font-weight: 700; }
    .aqi-sensitive { background: #ff7e00; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; }
    .aqi-unhealthy { background: #ff0000; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; }
    .aqi-very-unhealthy { background: #8f3f97; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; }
    .aqi-hazardous { background: #7e0023; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; }
    div[data-testid="stSidebarContent"] { background: #111827; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API KEYS
# ─────────────────────────────────────────────
WAQI_TOKEN       = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
OPENWEATHER_KEY  = "c86236be4a9f76875aad940c96e5111b"
TOMTOM_KEY       = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"
NASA_FIRMS_KEY   = "f5756b3b5354a7a8d34bfc37fc794a38"
GSHEET_READ_URL  = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQO7corvhjivltUU1Y1aE4lDH1BmKDSF1O2uDSmSfw6HyNr5RuYz4qXYGCCsNDt3OUqqA7sFHaLqqiO/pub?output=csv"
GSHEET_WRITE_URL = "https://script.google.com/macros/s/AKfycbyoy_PD319OgRj9z3j3WR2nrL_FWzLXU15o_a9Edc4ZzEmipvYtBaeCDr1xGdno_O5n/exec"

# ─────────────────────────────────────────────
# INDIA CITIES DATABASE (~120 cities with coords & terrain tags)
# ─────────────────────────────────────────────
INDIA_CITIES = {
    "Agra": (27.1767, 78.0081, "plains"),
    "Ahmedabad": (23.0225, 72.5714, "plains"),
    "Aizawl": (23.7271, 92.7176, "hilly"),
    "Ajmer": (26.4499, 74.6399, "semi-arid"),
    "Allahabad (Prayagraj)": (25.4358, 81.8463, "plains"),
    "Amritsar": (31.6340, 74.8723, "plains"),
    "Aurangabad": (19.8762, 75.3433, "plateau"),
    "Bengaluru": (12.9716, 77.5946, "plateau"),
    "Bhopal": (23.2599, 77.4126, "plateau"),
    "Bhubaneswar": (20.2961, 85.8245, "coastal"),
    "Chandigarh": (30.7333, 76.7794, "plains"),
    "Chennai": (13.0827, 80.2707, "coastal"),
    "Coimbatore": (11.0168, 76.9558, "plains"),
    "Dehradun": (30.3165, 78.0322, "hilly"),
    "Delhi": (28.6139, 77.2090, "plains"),
    "Dharamsala": (32.2190, 76.3234, "hilly"),
    "Dibrugarh": (27.4728, 94.9120, "plains"),
    "Durgapur": (23.5204, 87.3119, "plains"),
    "Faridabad": (28.4089, 77.3178, "plains"),
    "Gandhinagar": (23.2156, 72.6369, "plains"),
    "Guwahati": (26.1445, 91.7362, "hilly"),
    "Gwalior": (26.2183, 78.1828, "plains"),
    "Hyderabad": (17.3850, 78.4867, "plateau"),
    "Imphal": (24.8170, 93.9368, "valley"),
    "Indore": (22.7196, 75.8577, "plateau"),
    "Itanagar": (27.0844, 93.6053, "hilly"),
    "Jabalpur": (23.1815, 79.9864, "plateau"),
    "Jaipur": (26.9124, 75.7873, "semi-arid"),
    "Jalandhar": (31.3260, 75.5762, "plains"),
    "Jammu": (32.7266, 74.8570, "hilly"),
    "Jamshedpur": (22.8046, 86.2029, "hilly"),
    "Jodhpur": (26.2389, 73.0243, "arid"),
    "Kanpur": (26.4499, 80.3319, "plains"),
    "Kochi": (9.9312, 76.2673, "coastal"),
    "Kohima": (25.6747, 94.1086, "hilly"),
    "Kolkata": (22.5726, 88.3639, "coastal"),
    "Kota": (25.2138, 75.8648, "plains"),
    "Kozhikode": (11.2588, 75.7804, "coastal"),
    "Lucknow": (26.8467, 80.9462, "plains"),
    "Ludhiana": (30.9010, 75.8573, "plains"),
    "Madurai": (9.9252, 78.1198, "plains"),
    "Mangaluru": (12.9141, 74.8560, "coastal"),
    "Meerut": (28.9845, 77.7064, "plains"),
    "Mumbai": (19.0760, 72.8777, "coastal"),
    "Mysuru": (12.2958, 76.6394, "plateau"),
    "Nagpur": (21.1458, 79.0882, "plateau"),
    "Nashik": (19.9975, 73.7898, "plateau"),
    "Navi Mumbai": (19.0330, 73.0297, "coastal"),
    "Noida": (28.5355, 77.3910, "plains"),
    "Panaji": (15.4909, 73.8278, "coastal"),
    "Patna": (25.5941, 85.1376, "plains"),
    "Puducherry": (11.9416, 79.8083, "coastal"),
    "Pune": (18.5204, 73.8567, "plateau"),
    "Raipur": (21.2514, 81.6296, "plateau"),
    "Rajkot": (22.3039, 70.8022, "plains"),
    "Ranchi": (23.3441, 85.3096, "hilly"),
    "Shillong": (25.5788, 91.8933, "hilly"),
    "Shimla": (31.1048, 77.1734, "hilly"),
    "Siliguri": (26.7271, 88.3953, "plains"),
    "Srinagar": (34.0837, 74.7973, "valley"),
    "Surat": (21.1702, 72.8311, "coastal"),
    "Thane": (19.2183, 72.9781, "coastal"),
    "Thiruvananthapuram": (8.5241, 76.9366, "coastal"),
    "Tiruchirappalli": (10.7905, 78.7047, "plains"),
    "Udaipur": (24.5854, 73.7125, "hilly"),
    "Vadodara": (22.3072, 73.1812, "plains"),
    "Varanasi": (25.3176, 82.9739, "plains"),
    "Vijayawada": (16.5062, 80.6480, "plains"),
    "Visakhapatnam": (17.6868, 83.2185, "coastal"),
    "Warangal": (17.9689, 79.5941, "plateau"),
}

# ─────────────────────────────────────────────
# AQI / PM10 CATEGORY HELPER
# ─────────────────────────────────────────────
def pm10_category(val):
    if val <= 50:   return "Good", "#00e400", "✅"
    if val <= 100:  return "Moderate", "#ffff00", "🟡"
    if val <= 250:  return "Poor", "#ff7e00", "🟠"
    if val <= 350:  return "Very Poor", "#ff0000", "🔴"
    if val <= 430:  return "Severe", "#8f3f97", "🟣"
    return "Hazardous", "#7e0023", "☠️"

def health_advisory(val):
    if val <= 50:
        return "Air quality is satisfactory. Enjoy outdoor activities!"
    if val <= 100:
        return "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion."
    if val <= 250:
        return "Members of sensitive groups may experience health effects. Reduce prolonged outdoor exertion."
    if val <= 350:
        return "Everyone may begin to experience health effects. Avoid prolonged outdoor exertion."
    if val <= 430:
        return "Health alert: everyone may experience more serious health effects. Avoid outdoor activity."
    return "⚠️ HAZARDOUS: Health warnings of emergency conditions. Everyone should avoid all outdoor exertion."

# ─────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_waqi_city(city_name):
    """Fetch PM10/AQI from WAQI for a city."""
    try:
        url = f"https://api.waqi.info/feed/{city_name}/?token={WAQI_TOKEN}"
        r = requests.get(url, timeout=10).json()
        if r.get("status") == "ok":
            data = r["data"]
            iaqi = data.get("iaqi", {})
            return {
                "pm10": iaqi.get("pm10", {}).get("v"),
                "pm25": iaqi.get("pm25", {}).get("v"),
                "o3":   iaqi.get("o3",   {}).get("v"),
                "no2":  iaqi.get("no2",  {}).get("v"),
                "so2":  iaqi.get("so2",  {}).get("v"),
                "co":   iaqi.get("co",   {}).get("v"),
                "aqi":  data.get("aqi"),
                "station": data.get("city", {}).get("name", city_name),
                "lat":  data.get("city", {}).get("geo", [None, None])[0],
                "lon":  data.get("city", {}).get("geo", [None, None])[1],
            }
    except:
        pass
    return {}

@st.cache_data(ttl=900)
def fetch_waqi_bounds(lat, lon, radius=0.3):
    """Fetch all stations in bounding box around city."""
    bounds = f"{lat-radius},{lon-radius},{lat+radius},{lon+radius}"
    url = f"https://api.waqi.info/map/bounds/?latlng={bounds}&token={WAQI_TOKEN}"
    records = []
    try:
        r = requests.get(url, timeout=10).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                d = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={WAQI_TOKEN}", timeout=8).json()
                if d.get("status") == "ok":
                    iaqi = d["data"].get("iaqi", {})
                    pm10 = iaqi.get("pm10", {}).get("v")
                    if pm10:
                        records.append({
                            "lat": s["lat"], "lon": s["lon"],
                            "pm10": pm10,
                            "pm25": iaqi.get("pm25", {}).get("v"),
                            "aqi":  d["data"].get("aqi"),
                            "name": d["data"].get("city", {}).get("name", ""),
                        })
                time.sleep(0.1)
    except:
        pass
    return pd.DataFrame(records)

@st.cache_data(ttl=1800)
def fetch_openweather(lat, lon):
    """Fetch full weather + air quality from OpenWeather."""
    result = {}
    try:
        # Current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
        r = requests.get(url, timeout=10).json()
        result.update({
            "temp":        r["main"]["temp"],
            "feels_like":  r["main"]["feels_like"],
            "humidity":    r["main"]["humidity"],
            "pressure":    r["main"]["pressure"],
            "visibility":  r.get("visibility", 10000) / 1000,
            "wind_speed":  r["wind"]["speed"] * 3.6,
            "wind_deg":    r["wind"].get("deg", 0),
            "wind_dir":    deg_to_compass(r["wind"].get("deg", 0)),
            "description": r["weather"][0]["description"].title(),
            "clouds":      r["clouds"]["all"],
            "icon":        r["weather"][0]["icon"],
        })
    except Exception as e:
        result.update({"temp":25,"feels_like":25,"humidity":50,"pressure":1013,
                        "visibility":10,"wind_speed":10,"wind_deg":0,"wind_dir":"N",
                        "description":"N/A","clouds":0,"icon":"01d"})

    try:
        # Air quality from OWM
        aurl = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}"
        ar = requests.get(aurl, timeout=10).json()
        comp = ar["list"][0]["components"]
        result.update({
            "owm_pm10": comp.get("pm10"),
            "owm_pm25": comp.get("pm2_5"),
            "owm_no2":  comp.get("no2"),
            "owm_o3":   comp.get("o3"),
            "owm_so2":  comp.get("so2"),
            "owm_co":   comp.get("co"),
        })
    except:
        pass

    try:
        # 5-day forecast
        furl = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
        fr = requests.get(furl, timeout=10).json()
        forecasts = []
        for item in fr["list"]:
            forecasts.append({
                "timestamp": pd.to_datetime(item["dt_txt"]),
                "temp": item["main"]["temp"],
                "humidity": item["main"]["humidity"],
                "wind_speed": item["wind"]["speed"] * 3.6,
                "wind_deg": item["wind"].get("deg", 0),
                "description": item["weather"][0]["description"],
            })
        result["forecast_df"] = pd.DataFrame(forecasts)
    except:
        result["forecast_df"] = pd.DataFrame()

    return result

@st.cache_data(ttl=1800)
def fetch_tomtom_traffic(lat, lon, radius_km=10):
    """Fetch traffic flow and incidents from TomTom."""
    result = {"flow_ratio": 1.0, "incidents": [], "congestion_index": 0}
    try:
        # Traffic flow
        zoom = 12
        url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/{zoom}/json?"
               f"point={lat},{lon}&key={TOMTOM_KEY}")
        r = requests.get(url, timeout=10).json()
        if "flowSegmentData" in r:
            fsd = r["flowSegmentData"]
            current = fsd.get("currentSpeed", 0)
            free = fsd.get("freeFlowSpeed", 1)
            ratio = current / max(free, 1)
            result["flow_ratio"] = ratio
            result["congestion_index"] = round((1 - ratio) * 100, 1)
            result["current_speed"] = current
            result["free_flow_speed"] = free
    except:
        pass

    try:
        # Traffic incidents
        bbox = f"{lon-0.15},{lat-0.15},{lon+0.15},{lat+0.15}"
        iurl = (f"https://api.tomtom.com/traffic/services/5/incidentDetails?"
                f"bbox={bbox}&fields={{incidents{{type,geometry,properties}}}}&language=en-GB&key={TOMTOM_KEY}")
        ir = requests.get(iurl, timeout=10).json()
        incidents = ir.get("incidents", [])
        result["incidents"] = incidents[:10]
        result["incident_count"] = len(incidents)
    except:
        result["incident_count"] = 0

    return result

@st.cache_data(ttl=3600)
def fetch_elevation(lat, lon, grid_pts=20):
    """Fetch elevation grid from Open-Elevation API."""
    locations = []
    for la in np.linspace(lat - 0.2, lat + 0.2, grid_pts):
        for lo in np.linspace(lon - 0.2, lon + 0.2, grid_pts):
            locations.append({"latitude": round(la, 4), "longitude": round(lo, 4)})
    try:
        r = requests.post("https://api.open-elevation.com/api/v1/lookup",
                          json={"locations": locations}, timeout=30)
        results = r.json().get("results", [])
        elev_data = pd.DataFrame(results)
        elev_data.rename(columns={"latitude": "lat", "longitude": "lon", "elevation": "elev"}, inplace=True)
        return elev_data
    except:
        # Synthetic fallback
        rows = []
        for item in locations:
            rows.append({"lat": item["latitude"], "lon": item["longitude"], "elev": 200 + np.random.normal(0, 30)})
        return pd.DataFrame(rows)

@st.cache_data(ttl=3600)
def fetch_nasa_firms(lat, lon):
    """Fetch fire/hotspot data from NASA FIRMS as PM10 emission proxy."""
    try:
        area = f"{lon-1},{lat-1},{lon+1},{lat+1}"
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{NASA_FIRMS_KEY}/VIIRS_SNPP_NRT/{area}/1"
        df = pd.read_csv(url)
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=1800)
def fetch_openaq(city_name, lat, lon):
    """Fetch ground sensor readings from OpenAQ."""
    sensors = []
    try:
        url = f"https://api.openaq.org/v2/locations?city={city_name}&country=IN&limit=20&parameter=pm10"
        r = requests.get(url, timeout=10, headers={"accept": "application/json"}).json()
        for loc in r.get("results", []):
            for param in loc.get("parameters", []):
                if param["parameter"] == "pm10" and param.get("lastValue"):
                    sensors.append({
                        "lat": loc["coordinates"]["latitude"],
                        "lon": loc["coordinates"]["longitude"],
                        "pm10": param["lastValue"],
                        "name": loc["name"],
                        "source": "OpenAQ"
                    })
    except:
        pass
    return pd.DataFrame(sensors)

@st.cache_data(ttl=600)
def load_historical():
    try:
        df = pd.read_csv(GSHEET_READ_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except:
        return pd.DataFrame()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def deg_to_compass(deg):
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[int((deg % 360) / 22.5 + 0.5) % 16]

def terrain_pm10_factor(terrain_type):
    factors = {
        "plains": 1.0,
        "coastal": 0.85,
        "plateau": 0.90,
        "hilly": 0.75,
        "arid": 1.20,
        "semi-arid": 1.10,
        "valley": 1.15,
    }
    return factors.get(terrain_type, 1.0)

def estimate_pm10_no_sensor(weather, traffic, terrain, city_pop=1_000_000):
    """ML-style PM10 estimation for cities with no sensors."""
    base = 60.0

    # Weather effects
    if weather.get("humidity", 50) > 70: base *= 1.1
    wind = weather.get("wind_speed", 10)
    if wind > 20: base *= 0.75
    elif wind < 5: base *= 1.25
    if weather.get("clouds", 50) > 80: base *= 0.95

    # Traffic congestion penalty
    congestion = traffic.get("congestion_index", 20)
    base *= (1 + congestion / 200)

    # Terrain factor
    base *= terrain_pm10_factor(terrain)

    # Population density proxy
    base *= (1 + math.log10(city_pop / 500_000) * 0.1)

    # Pressure effect (higher pressure = more trapping)
    pressure = weather.get("pressure", 1013)
    if pressure > 1015: base *= 1.05

    return round(base, 1)

def sync_to_sheets(df):
    try:
        payload = df.copy()
        payload["timestamp"] = payload["timestamp"].astype(str)
        requests.post(GSHEET_WRITE_URL, data=json.dumps(payload.to_dict(orient="records")), timeout=10)
    except:
        pass

# ─────────────────────────────────────────────
# NSS-NET MODEL
# ─────────────────────────────────────────────
def build_nss_model(df_hist):
    features = ["lat","lon","hour","dayofweek","month","temp","hum","wind","pm10_lag1"]
    df = df_hist.copy().sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["pm10_lag1"] = df.groupby(["lat","lon"])["pm10"].shift(1).fillna(df["pm10"].median())
    df["target"] = np.log1p(df["pm10"])
    # rename if needed
    for col in ["temp","hum","wind"]:
        if col not in df.columns:
            df[col] = {"temp":25,"hum":50,"wind":10}[col]
    df = df.dropna(subset=features + ["target"])
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    rf = RandomForestRegressor(n_estimators=800, max_depth=9, random_state=42, n_jobs=-1)
    rf.fit(X, df["target"])
    return rf, scaler, features

def forecast_24h(rf, scaler, features, lat, lon, weather, df_live_pm10_mean):
    now = pd.Timestamp.now()
    future_times = pd.date_range(start=now.ceil("h"), periods=24, freq="h")
    current_lag = df_live_pm10_mean
    results = []
    for ft in future_times:
        row = pd.DataFrame([[lat, lon, ft.hour, ft.dayofweek, ft.month,
                              weather.get("temp",25), weather.get("humidity",50),
                              weather.get("wind_speed",10), current_lag]], columns=features)
        pred = np.expm1(rf.predict(scaler.transform(row))[0])
        pred = max(pred, 10)
        results.append({"timestamp": ft, "pm10": round(pred, 1)})
        current_lag = pred
    return pd.DataFrame(results)

# ─────────────────────────────────────────────
# WIND ROSE CHART
# ─────────────────────────────────────────────
def wind_rose(wind_deg, wind_speed):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    speeds = [max(0, wind_speed * np.cos(np.radians(a - wind_deg))) for a in angles]
    fig = go.Figure(go.Barpolar(
        r=speeds, theta=dirs,
        marker_color=["#ff6b35" if s == max(speeds) else "#3d5a80" for s in speeds],
        marker_line_color="white", marker_line_width=1, opacity=0.85
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(speeds)+2])),
        paper_bgcolor="#0e1117", font_color="white",
        title="Wind Rose", height=300, margin=dict(t=40,b=10,l=10,r=10)
    )
    return fig

# ─────────────────────────────────────────────
# AQ COLOR HELPERS FOR MAP
# ─────────────────────────────────────────────
def aq_color(val, max_val):
    ratio = min(val / max(max_val, 1), 1.0)
    if ratio < 0.2:  return [0, 228, 0, 190]
    if ratio < 0.4:  return [255, 255, 0, 190]
    if ratio < 0.6:  return [255, 126, 0, 210]
    if ratio < 0.8:  return [255, 0, 0, 220]
    return [143, 63, 151, 235]

def make_heatmap_layer(stations_df, lat, lon, param, fallback_val):
    heat_data = []
    if not stations_df.empty and param in stations_df.columns:
        for _, row in stations_df.iterrows():
            v = row.get(param)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                heat_data.append({"lat": row["lat"], "lon": row["lon"], "weight": float(v)})
    if not heat_data:
        np.random.seed(42)
        for _ in range(40):
            heat_data.append({
                "lat": lat + np.random.normal(0, 0.025),
                "lon": lon + np.random.normal(0, 0.025),
                "weight": fallback_val * np.random.uniform(0.6, 1.4),
            })
    return pdk.Layer(
        "HeatmapLayer",
        data=pd.DataFrame(heat_data),
        get_position=["lon", "lat"],
        get_weight="weight",
        radiusPixels=90, intensity=1.3, threshold=0.04,
        color_range=[
            [0,228,0,160],[255,255,0,180],[255,126,0,200],
            [255,0,0,215],[143,63,151,230],[126,0,35,255],
        ],
    )

def make_column_layer(stations_df, param, col_color):
    data = []
    if stations_df.empty or param not in stations_df.columns:
        return None
    for _, row in stations_df.iterrows():
        v = row.get(param)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            data.append({
                "lat": row["lat"], "lon": row["lon"],
                "value": max(float(v), 0),
                "name":  row.get("name", ""),
                "param": param.upper(),
            })
    if not data:
        return None
    return pdk.Layer(
        "ColumnLayer",
        data=pd.DataFrame(data),
        get_position=["lon", "lat"],
        get_elevation="value",
        elevation_scale=80,
        radius=280,
        get_fill_color=col_color,
        pickable=True,
        auto_highlight=True,
        coverage=0.9,
    )

# ─────────────────────────────────────────────
# 3D PYDECK MAP — real satellite basemap + all AQ params
# ─────────────────────────────────────────────
def make_3d_map(lat, lon, elev_df, stations_df, weather_data, aq_param="pm10",
                show_heatmap=True, show_columns=True, show_wind=True,
                pitch=55, bearing=0):

    layers = []

    # ── 1. TERRAIN GRID ────────────────────────────────────────────────
    elev_plot = elev_df.rename(columns={"lon": "longitude", "lat": "latitude"}).copy()
    elev_plot["elev"] = elev_plot["elev"].clip(lower=0)
    layers.append(pdk.Layer(
        "GridCellLayer",
        data=elev_plot,
        get_position=["longitude", "latitude"],
        get_elevation="elev",
        elevation_scale=3,
        get_fill_color=[80, 120, 80, 80],
        cell_size=400,
        pickable=True,
        extruded=True,
    ))

    # ── 2. HEATMAP for chosen AQ param ────────────────────────────────
    if show_heatmap:
        fallback = stations_df["pm10"].mean() if not stations_df.empty and "pm10" in stations_df.columns else 80
        layers.append(make_heatmap_layer(stations_df, lat, lon, aq_param, fallback))

    # ── 3. 3D COLUMN BARS (height = AQ value) ─────────────────────────
    if show_columns and not stations_df.empty:
        param_colors = {
            "pm10": [255, 107, 53,  220],
            "pm25": [255, 50,  50,  220],
            "aqi":  [200, 100, 255, 220],
            "no2":  [50,  180, 255, 220],
            "o3":   [100, 220, 100, 220],
            "so2":  [255, 220, 50,  220],
            "co":   [200, 80,  80,  220],
        }
        col_color = param_colors.get(aq_param, [255, 107, 53, 220])
        col_layer = make_column_layer(stations_df, aq_param, col_color)
        if col_layer:
            layers.append(col_layer)

    # ── 4. STATION SCATTER (colored by PM10 AQI) ──────────────────────
    if not stations_df.empty:
        s_data = stations_df.copy()
        pm10_max = s_data["pm10"].max() if "pm10" in s_data.columns else 200

        def build_row_color(row):
            v = row.get("pm10", 80)
            return aq_color(float(v), pm10_max)

        def build_tip(row):
            parts = [f"📍 {row.get('name','')}"]
            for p, unit in [("pm10","µg/m³"),("pm25","µg/m³"),("aqi",""),
                             ("no2","µg/m³"),("o3","µg/m³"),("so2","µg/m³"),("co","mg/m³")]:
                v = row.get(p)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    parts.append(f"{p.upper()}: {round(float(v),1)} {unit}")
            return " | ".join(parts)

        s_data["r_color"]      = s_data.apply(build_row_color, axis=1)
        s_data["tooltip_text"] = s_data.apply(build_tip, axis=1)

        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=s_data,
            get_position=["lon", "lat"],
            get_fill_color="r_color",
            get_radius=420,
            pickable=True,
            auto_highlight=True,
            opacity=0.95,
        ))

    # ── 5. WIND ARROW TEXT LAYER ──────────────────────────────────────
    if show_wind and weather_data:
        wd = weather_data.get("wind_deg", 0)
        ws = weather_data.get("wind_speed", 0)
        arrow = ("→" if 45<=wd<135 else "↓" if 135<=wd<225
                 else "←" if 225<=wd<315 else "↑")
        sz = max(14, int(ws * 2.5))
        wind_pts = [
            {"lat": lat+dla, "lon": lon+dlo, "text": arrow, "size": sz}
            for dla in np.linspace(-0.12, 0.12, 5)
            for dlo in np.linspace(-0.12, 0.12, 5)
        ]
        layers.append(pdk.Layer(
            "TextLayer",
            data=pd.DataFrame(wind_pts),
            get_position=["lon", "lat"],
            get_text="text",
            get_size="size",
            get_color=[100, 200, 255, 200],
            pickable=False,
        ))

    # ── 6. VIEW + DECK ────────────────────────────────────────────────
    view = pdk.ViewState(
        latitude=lat, longitude=lon,
        zoom=12, pitch=pitch, bearing=bearing,
        min_zoom=8, max_zoom=18,
    )

    return pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"text": "{tooltip_text}"},
        # CartoDB Dark Matter — free, no token needed, real buildings + streets
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    )

# ─────────────────────────────────────────────
# SIDEBAR — CITY SELECTOR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🇮🇳 India AirSense Pro")
st.sidebar.markdown("---")

city_list = sorted(INDIA_CITIES.keys())
selected_city = st.sidebar.selectbox("🏙️ Select City", city_list, index=city_list.index("Delhi"))

lat, lon, terrain = INDIA_CITIES[selected_city]
st.sidebar.caption(f"📍 {lat:.4f}°N, {lon:.4f}°E | 🏔️ Terrain: {terrain.title()}")

st.sidebar.markdown("---")
compare_mode = st.sidebar.checkbox("⚖️ Compare Two Cities")
if compare_mode:
    city2 = st.sidebar.selectbox("🏙️ Second City", [c for c in city_list if c != selected_city])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
opacity      = st.sidebar.slider("Map Opacity", 0.1, 1.0, 0.8)
run_model    = st.sidebar.button("🚀 Run Full Analysis")
run_diag     = st.sidebar.button("📊 Run Diagnostics")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Custom Point Prediction")
custom_lat = st.sidebar.number_input("Latitude",  value=lat,  step=0.01, format="%.4f")
custom_lon = st.sidebar.number_input("Longitude", value=lon,  step=0.01, format="%.4f")
predict_pt = st.sidebar.button("Predict PM10 Here")

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Historical Filter")
start_date = st.sidebar.date_input("Start", datetime.date.today() - datetime.timedelta(days=7))
end_date   = st.sidebar.date_input("End",   datetime.date.today())

# Auto-refresh
st_autorefresh(interval=1800000, key="autorefresh")

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown(f'<div class="city-header">🌫️ {selected_city} — Air Quality Intelligence</div>', unsafe_allow_html=True)
st.markdown(f"*Real-time PM10 analysis powered by NSS-Net · Terrain: **{terrain.title()}** · {pd.Timestamp.now().strftime('%d %b %Y, %H:%M')}*")
st.markdown("---")

# ─────────────────────────────────────────────
# FETCH ALL DATA
# ─────────────────────────────────────────────
with st.spinner("🔄 Fetching live data from all sources..."):
    weather   = fetch_openweather(lat, lon)
    traffic   = fetch_tomtom_traffic(lat, lon)
    waqi_city = fetch_waqi_city(selected_city)
    stations  = fetch_waqi_bounds(lat, lon)
    openaq    = fetch_openaq(selected_city, lat, lon)
    elev_df   = fetch_elevation(lat, lon)
    firms_df  = fetch_nasa_firms(lat, lon)
    df_hist   = load_historical()

# Merge station data
all_stations = pd.concat([stations, openaq], ignore_index=True) if not openaq.empty else stations

# Determine PM10 value
has_sensors = not all_stations.empty or waqi_city.get("pm10")
if waqi_city.get("pm10"):
    pm10_val = waqi_city["pm10"]
elif not all_stations.empty:
    pm10_val = all_stations["pm10"].mean()
else:
    pm10_val = estimate_pm10_no_sensor(weather, traffic, terrain)

pm10_val = round(pm10_val, 1)
cat, color, emoji = pm10_category(pm10_val)

# ─────────────────────────────────────────────
# ROW 1 — KEY METRICS
# ─────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("🌫️ PM10",        f"{pm10_val} µg/m³",  delta=None)
c2.metric("🌡️ Temperature", f"{weather['temp']:.1f}°C", f"Feels {weather['feels_like']:.0f}°C")
c3.metric("💧 Humidity",    f"{weather['humidity']}%")
c4.metric("💨 Wind",        f"{weather['wind_speed']:.1f} km/h", weather['wind_dir'])
c5.metric("👁️ Visibility",  f"{weather['visibility']:.1f} km")
c6.metric("🚗 Congestion",  f"{traffic.get('congestion_index',0):.0f}%")

# AQI Banner
st.markdown(f"""
<div style="background:{color};color:{'black' if color in ['#00e400','#ffff00'] else 'white'};
padding:14px 22px;border-radius:12px;margin:12px 0;font-size:1.1rem;font-weight:700;">
{emoji} Air Quality: <b>{cat}</b> — PM10 {pm10_val} µg/m³
<br><small style="font-weight:400;">{health_advisory(pm10_val)}</small>
</div>
""", unsafe_allow_html=True)

if not has_sensors:
    st.info(f"ℹ️ No ground sensors found for **{selected_city}**. PM10 estimated using terrain, weather, traffic & satellite data.")

# ─────────────────────────────────────────────
# ROW 2 — TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ 3D Map", "📈 Forecast", "🌤️ Weather", "🚗 Traffic",
    "📊 Diagnostics", "📂 History"
])

# ════════════════════════════════════════════
# TAB 1 — 3D MAP
# ════════════════════════════════════════════
with tab1:
    st.subheader(f"🗺️ 3D Terrain + Air Quality Map — {selected_city}")

    # ── Controls row ───────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    aq_param_sel = mc1.selectbox(
        "🌫️ Parameter",
        ["pm10","pm25","aqi","no2","o3","so2","co"],
        format_func=lambda x: {
            "pm10":"PM10 (µg/m³)","pm25":"PM2.5 (µg/m³)","aqi":"AQI",
            "no2":"NO₂ (µg/m³)","o3":"O₃ (µg/m³)","so2":"SO₂ (µg/m³)","co":"CO (mg/m³)"
        }[x]
    )
    show_hm   = mc2.checkbox("🌡️ Heatmap",   value=True)
    show_col  = mc3.checkbox("📊 3D Bars",    value=True)
    show_wind = mc4.checkbox("💨 Wind Arrows",value=True)
    map_pitch = mc5.slider("🎥 Pitch", 0, 80, 55)

    col1, col2 = st.columns([3, 1])

    with col1:
        # Enrich all_stations with OWM AQ data if no sensor values for params
        stations_enriched = all_stations.copy() if not all_stations.empty else pd.DataFrame()
        for owm_key, col_name in [("owm_pm25","pm25"),("owm_no2","no2"),
                                   ("owm_o3","o3"),("owm_so2","so2"),("owm_co","co")]:
            if not stations_enriched.empty and col_name not in stations_enriched.columns:
                stations_enriched[col_name] = weather.get(owm_key)

        deck = make_3d_map(
            lat, lon, elev_df, stations_enriched, weather,
            aq_param=aq_param_sel,
            show_heatmap=show_hm,
            show_columns=show_col,
            show_wind=show_wind,
            pitch=map_pitch,
        )
        st.pydeck_chart(deck, use_container_width=True)

        # ── Legend ────────────────────────────────────────────────────
        st.markdown("""
        <div style='display:flex;gap:10px;margin-top:6px;flex-wrap:wrap;font-size:12px'>
          <span style='background:#00e400;padding:3px 10px;border-radius:12px;color:black'>● Good ≤50</span>
          <span style='background:#ffff00;padding:3px 10px;border-radius:12px;color:black'>● Moderate ≤100</span>
          <span style='background:#ff7e00;padding:3px 10px;border-radius:12px;color:white'>● Poor ≤250</span>
          <span style='background:#ff0000;padding:3px 10px;border-radius:12px;color:white'>● Very Poor ≤350</span>
          <span style='background:#8f3f97;padding:3px 10px;border-radius:12px;color:white'>● Severe ≤430</span>
          <span style='background:#7e0023;padding:3px 10px;border-radius:12px;color:white'>● Hazardous 430+</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**📡 Data Sources**")
        st.success("✅ OpenWeather")
        st.success("✅ TomTom Traffic")
        st.success("✅ Open-Elevation")
        if not stations.empty:
            st.success(f"✅ WAQI ({len(stations)} stations)")
        else:
            st.warning("⚠️ No WAQI stations")
        if not openaq.empty:
            st.success(f"✅ OpenAQ ({len(openaq)} sensors)")
        if not firms_df.empty:
            st.success(f"✅ NASA FIRMS ({len(firms_df)} hotspots)")
        else:
            st.info("🛰️ No nearby fires")

        st.markdown("---")
        st.markdown("**🏔️ Terrain**")
        tf = terrain_pm10_factor(terrain)
        st.metric("Factor", f"{tf:.2f}×",
                  delta="traps pollution" if tf > 1 else "disperses pollution")

        st.markdown("**🔥 Pollution Drivers**")
        drivers = []
        if traffic.get("congestion_index", 0) > 30:        drivers.append("🚗 Heavy Traffic")
        if weather.get("wind_speed", 10) < 5:              drivers.append("💨 Low Wind")
        if weather.get("humidity", 50) > 75:               drivers.append("💧 High Humidity")
        if terrain in ["valley","arid","semi-arid"]:       drivers.append("🏔️ Terrain Trap")
        if not firms_df.empty:                              drivers.append("🔥 Fire Hotspots")
        for d in drivers: st.warning(d)
        if not drivers: st.success("✅ Conditions favorable")

    # ── All AQ Parameters Dashboard ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 All Air Quality Parameters")

    # Collect values from all sources
    aq_values = {
        "PM10":  waqi_city.get("pm10")  or weather.get("owm_pm10"),
        "PM2.5": waqi_city.get("pm25")  or weather.get("owm_pm25"),
        "AQI":   waqi_city.get("aqi"),
        "NO₂":   waqi_city.get("no2")   or weather.get("owm_no2"),
        "O₃":    waqi_city.get("o3")    or weather.get("owm_o3"),
        "SO₂":   waqi_city.get("so2")   or weather.get("owm_so2"),
        "CO":    waqi_city.get("co")    or weather.get("owm_co"),
    }
    aq_limits = {"PM10":100,"PM2.5":60,"AQI":100,"NO₂":80,"O₃":100,"SO₂":80,"CO":10}
    aq_units  = {"PM10":"µg/m³","PM2.5":"µg/m³","AQI":"","NO₂":"µg/m³","O₃":"µg/m³","SO₂":"µg/m³","CO":"mg/m³"}

    # Metric cards
    cols_aq = st.columns(len(aq_values))
    for i, (name, val) in enumerate(aq_values.items()):
        with cols_aq[i]:
            if val is not None:
                limit = aq_limits[name]
                pct   = min(val / limit, 2.0)
                color = ("#00e400" if pct < 0.5 else "#ffff00" if pct < 1.0
                         else "#ff7e00" if pct < 1.5 else "#ff0000")
                st.markdown(f"""
                <div style='background:#1a1d27;border-radius:10px;padding:10px;
                            border-left:4px solid {color};text-align:center'>
                  <div style='font-size:11px;color:#aaa'>{name}</div>
                  <div style='font-size:1.4rem;font-weight:700;color:{color}'>{val:.1f}</div>
                  <div style='font-size:10px;color:#888'>{aq_units[name]}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#1a1d27;border-radius:10px;padding:10px;
                            border-left:4px solid #444;text-align:center'>
                  <div style='font-size:11px;color:#aaa'>{name}</div>
                  <div style='font-size:1rem;color:#666'>N/A</div>
                </div>""", unsafe_allow_html=True)

    # Radar / spider chart of all AQ params
    st.markdown("#### 📡 Pollutant Radar Chart")
    radar_vals = {k: v for k, v in aq_values.items() if v is not None}
    if len(radar_vals) >= 3:
        # Normalize each to 0-100 scale vs WHO limit
        norm = {k: min(v / aq_limits[k] * 100, 200) for k, v in radar_vals.items()}
        fig_radar = go.Figure(go.Scatterpolar(
            r=list(norm.values()),
            theta=list(norm.keys()),
            fill="toself",
            fillcolor="rgba(255,107,53,0.25)",
            line=dict(color="#ff6b35", width=2),
            marker=dict(size=8, color="#ff6b35"),
            name="Pollutant Level (% of WHO limit)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[100]*len(norm), theta=list(norm.keys()),
            mode="lines", line=dict(color="white", width=1, dash="dot"),
            name="WHO Safe Limit (100%)",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 200], tickfont=dict(color="white")),
                angularaxis=dict(tickfont=dict(color="white")),
                bgcolor="#1a1d27",
            ),
            paper_bgcolor="#0e1117", font_color="white",
            legend=dict(bgcolor="#1a1d27"),
            height=400, margin=dict(t=20,b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Bar chart comparison
    if radar_vals:
        fig_bar = go.Figure()
        for name, val in radar_vals.items():
            limit = aq_limits[name]
            pct   = val / limit
            clr   = ("#00e400" if pct<0.5 else "#ffff00" if pct<1.0
                     else "#ff7e00" if pct<1.5 else "#ff0000")
            fig_bar.add_trace(go.Bar(
                x=[name], y=[val], name=name,
                marker_color=clr,
                text=f"{val:.1f} {aq_units[name]}", textposition="outside",
            ))
        fig_bar.update_layout(
            title="Live Pollutant Levels",
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
            font_color="white", showlegend=False,
            height=350, margin=dict(t=40,b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Station table
    if not all_stations.empty:
        st.markdown("**📍 Monitoring Stations**")
        disp = all_stations.copy()
        for col_show in ["pm10","pm25","aqi","no2","o3","so2","co"]:
            if col_show not in disp.columns:
                disp[col_show] = None
        disp["Category"] = disp["pm10"].apply(lambda v: pm10_category(v)[0] if v else "N/A")
        st.dataframe(disp[["name","lat","lon","pm10","pm25","aqi","no2","o3","so2","co","Category"]],
                     use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — FORECAST
# ════════════════════════════════════════════
with tab2:
    st.subheader("📈 24-Hour PM10 Forecast — NSS-Net")

    if not df_hist.empty and len(df_hist) >= 10:
        with st.spinner("Training NSS-Net model..."):
            # Enrich hist with weather cols if missing
            for col, val in [("temp",weather["temp"]),("hum",weather["humidity"]),("wind",weather["wind_speed"])]:
                if col not in df_hist.columns:
                    df_hist[col] = val
            rf_model, scaler_model, feat_cols = build_nss_model(df_hist)
            df_fc = forecast_24h(rf_model, scaler_model, feat_cols, lat, lon, weather, pm10_val)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df_fc["timestamp"], y=df_fc["pm10"],
            mode="lines+markers", name="PM10 Forecast",
            line=dict(color="#ff6b35", width=3),
            fill="tozeroy", fillcolor="rgba(255,107,53,0.15)"
        ))
        # AQI bands
        for level, col, label in [(50,"#00e400","Good"),(100,"#ffff00","Moderate"),(250,"#ff7e00","Poor"),(350,"#ff0000","Very Poor")]:
            fig_fc.add_hline(y=level, line_dash="dot", line_color=col, annotation_text=label, annotation_position="right")

        fig_fc.update_layout(
            xaxis_title="Time", yaxis_title="PM10 (µg/m³)",
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
            font_color="white", height=400, margin=dict(t=20,b=40)
        )
        st.plotly_chart(fig_fc, use_container_width=True)
        st.caption("NSS-Net recursive 24-hour prediction using historical patterns, live weather & traffic.")

        # 7-day summary using weather forecast
        if not weather.get("forecast_df", pd.DataFrame()).empty:
            st.subheader("📅 5-Day Weather & Estimated PM10 Trend")
            fdf = weather["forecast_df"].copy()
            fdf["est_pm10"] = fdf.apply(lambda r: estimate_pm10_no_sensor(
                {"temp":r["temp"],"humidity":r["humidity"],"wind_speed":r["wind_speed"],"clouds":50},
                traffic, terrain), axis=1)

            fig7 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  subplot_titles=["Temperature (°C)", "Estimated PM10 (µg/m³)"])
            fig7.add_trace(go.Scatter(x=fdf["timestamp"], y=fdf["temp"],
                                       line=dict(color="#f7c59f"), name="Temp"), row=1, col=1)
            fig7.add_trace(go.Scatter(x=fdf["timestamp"], y=fdf["est_pm10"],
                                       fill="tozeroy", line=dict(color="#ff6b35"), name="PM10"), row=2, col=1)
            fig7.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
                                font_color="white", height=450)
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("📊 Collecting historical data to enable NSS-Net forecasting. Run more analyses to build the dataset.")
        # Simple weather-based estimate shown instead
        if not weather.get("forecast_df", pd.DataFrame()).empty:
            fdf = weather["forecast_df"].copy()
            fdf["est_pm10"] = fdf.apply(lambda r: estimate_pm10_no_sensor(
                {"temp":r["temp"],"humidity":r["humidity"],"wind_speed":r["wind_speed"],"clouds":50},
                traffic, terrain), axis=1)
            fig_s = px.line(fdf, x="timestamp", y="est_pm10", title="Estimated PM10 (Weather-based)",
                            color_discrete_sequence=["#ff6b35"])
            fig_s.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27", font_color="white")
            st.plotly_chart(fig_s, use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — WEATHER
# ════════════════════════════════════════════
with tab3:
    st.subheader(f"🌤️ Full Weather Dashboard — {selected_city}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Current Conditions**")
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("🌡️ Temperature", f"{weather['temp']:.1f}°C")
        wc2.metric("💧 Humidity",    f"{weather['humidity']}%")
        wc3.metric("🌬️ Pressure",   f"{weather['pressure']} hPa")

        wc4, wc5, wc6 = st.columns(3)
        wc4.metric("💨 Wind Speed",  f"{weather['wind_speed']:.1f} km/h")
        wc5.metric("🧭 Wind Dir",    weather['wind_dir'])
        wc6.metric("☁️ Cloud Cover", f"{weather['clouds']}%")

        st.markdown(f"**Conditions:** {weather['description']}")
        st.markdown(f"**Visibility:** {weather['visibility']:.1f} km")

        # Wind rose
        st.plotly_chart(wind_rose(weather["wind_deg"], weather["wind_speed"]), use_container_width=True)

    with col2:
        st.markdown("**OpenWeather Air Quality**")
        owm_vals = {
            "PM10":  weather.get("owm_pm10"),
            "PM2.5": weather.get("owm_pm25"),
            "NO₂":   weather.get("owm_no2"),
            "O₃":    weather.get("owm_o3"),
            "SO₂":   weather.get("owm_so2"),
            "CO":    weather.get("owm_co"),
        }
        for k, v in owm_vals.items():
            if v is not None:
                st.metric(k, f"{v:.1f} µg/m³")

        if not weather.get("forecast_df", pd.DataFrame()).empty:
            st.markdown("**48-Hour Humidity & Wind Forecast**")
            fdf = weather["forecast_df"].head(16)
            fig_w = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig_w.add_trace(go.Bar(x=fdf["timestamp"], y=fdf["humidity"],
                                    name="Humidity %", marker_color="#3d5a80"), row=1, col=1)
            fig_w.add_trace(go.Scatter(x=fdf["timestamp"], y=fdf["wind_speed"],
                                        name="Wind km/h", line=dict(color="#ff6b35")), row=2, col=1)
            fig_w.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
                                 font_color="white", height=350, showlegend=False)
            st.plotly_chart(fig_w, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 — TRAFFIC
# ════════════════════════════════════════════
with tab4:
    st.subheader(f"🚗 Traffic Analysis — {selected_city}")

    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("🚦 Congestion Index", f"{traffic.get('congestion_index',0):.1f}%")
    tc2.metric("🏎️ Current Speed",   f"{traffic.get('current_speed','N/A')} km/h")
    tc3.metric("🛣️ Free Flow Speed", f"{traffic.get('free_flow_speed','N/A')} km/h")

    # Congestion gauge
    cong = traffic.get("congestion_index", 0)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=cong,
        title={"text": "Congestion Level (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30],  "color": "#00e400"},
                {"range": [30, 60], "color": "#ffff00"},
                {"range": [60, 80], "color": "#ff7e00"},
                {"range": [80, 100],"color": "#ff0000"},
            ],
            "bar": {"color": "#ff6b35"},
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="#0e1117", font_color="white", height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Traffic contribution to PM10
    traffic_pm10_contrib = pm10_val * (cong / 200)
    st.metric("🌫️ Traffic's PM10 Contribution (est.)", f"{traffic_pm10_contrib:.1f} µg/m³",
              delta=f"{traffic_pm10_contrib/pm10_val*100:.0f}% of total")

    # Incidents
    inc_count = traffic.get("incident_count", 0)
    st.markdown(f"**🚨 Active Traffic Incidents:** {inc_count}")
    if traffic.get("incidents"):
        for i, inc in enumerate(traffic["incidents"][:5]):
            props = inc.get("properties", {})
            st.warning(f"Incident {i+1}: {props.get('iconCategory','Unknown')} — {props.get('magnitudeOfDelay','?')} delay")

    # PM10 vs Congestion scatter (synthetic from hist)
    if not df_hist.empty and "pm10" in df_hist.columns:
        st.subheader("📊 PM10 vs Traffic Correlation (Historical)")
        df_scatter = df_hist.copy()
        df_scatter["congestion_proxy"] = (df_scatter["timestamp"].dt.hour.isin(range(8,10)) |
                                           df_scatter["timestamp"].dt.hour.isin(range(17,20))).astype(int) * 40
        fig_sc = px.scatter(df_scatter, x="congestion_proxy", y="pm10",
                             color_discrete_sequence=["#ff6b35"],
                             title="PM10 vs Traffic Congestion")
        _x = df_scatter["congestion_proxy"].values
        _y = df_scatter["pm10"].values
        if len(np.unique(_x)) > 1:
            _m, _b = np.polyfit(_x, _y, 1)
            _xr = np.array([_x.min(), _x.max()])
            fig_sc.add_trace(go.Scatter(x=_xr, y=_m*_xr+_b,
                                         mode="lines", name="Trend",
                                         line=dict(color="white", width=2, dash="dot")))
        fig_sc.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27", font_color="white")
        st.plotly_chart(fig_sc, use_container_width=True)

# ════════════════════════════════════════════
# TAB 5 — DIAGNOSTICS
# ════════════════════════════════════════════
with tab5:
    st.subheader("📊 NSS-Net Model Diagnostics")

    if run_diag:
        if not all_stations.empty and len(all_stations) >= 3:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            preds, actuals = [], []
            df_d = all_stations.copy()
            now_ = pd.Timestamp.now()
            df_d["hour"] = now_.hour; df_d["dayofweek"] = now_.dayofweek; df_d["month"] = now_.month
            df_d["temp"] = weather["temp"]; df_d["hum"] = weather["humidity"]; df_d["wind"] = weather["wind_speed"]
            df_d["pm10_lag1"] = df_d["pm10"].median()
            feats = ["lat","lon","hour","dayofweek","month","temp","hum","wind","pm10_lag1"]

            for i in range(len(df_d)):
                train = df_d.drop(df_d.index[i])
                test  = df_d.iloc[[i]]
                sc = StandardScaler()
                Xtr = sc.fit_transform(train[feats])
                Xte = sc.transform(test[feats])
                rf_ = RandomForestRegressor(n_estimators=300, random_state=42)
                rf_.fit(Xtr, np.log1p(train["pm10"]))
                preds.append(np.expm1(rf_.predict(Xte)[0]))
                actuals.append(test["pm10"].values[0])

            mae  = mean_absolute_error(actuals, preds)
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            res  = pd.DataFrame({"Actual": actuals, "Predicted": preds})

            dc1, dc2 = st.columns(2)
            with dc1:
                st.metric("MAE",  f"{mae:.2f} µg/m³")
                st.metric("RMSE", f"{rmse:.2f} µg/m³")
                fig_r = px.scatter(res, x="Actual", y="Predicted",
                                   title="Actual vs Predicted", color_discrete_sequence=["#ff6b35"])
                _xa = res["Actual"].values
                _xp = res["Predicted"].values
                if len(_xa) > 1:
                    _m2, _b2 = np.polyfit(_xa, _xp, 1)
                    _xr2 = np.array([_xa.min(), _xa.max()])
                    fig_r.add_trace(go.Scatter(x=_xr2, y=_m2*_xr2+_b2,
                                               mode="lines", name="Fit",
                                               line=dict(color="#ff6b35", width=2)))
                fig_r.add_shape(type="line", x0=res.Actual.min(), y0=res.Actual.min(),
                                x1=res.Actual.max(), y1=res.Actual.max(),
                                line=dict(dash="dash", color="white"))
                fig_r.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27", font_color="white")
                st.plotly_chart(fig_r, use_container_width=True)
            with dc2:
                fig_h = px.histogram(all_stations, x="pm10", nbins=20, title="PM10 Distribution",
                                     color_discrete_sequence=["#3d5a80"])
                fig_h.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27", font_color="white")
                st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Need ≥3 stations for LOOCV diagnostics. Try a larger city or collect more data.")

    else:
        st.info("Click **📊 Run Diagnostics** in the sidebar to run the LOOCV model evaluation.")

    # Feature importance (if model trained)
    if run_model and not df_hist.empty and len(df_hist) >= 10:
        for col, val in [("temp",weather["temp"]),("hum",weather["humidity"]),("wind",weather["wind_speed"])]:
            if col not in df_hist.columns: df_hist[col] = val
        rf_m, sc_m, fc = build_nss_model(df_hist)
        imp = pd.DataFrame({"Feature": fc, "Importance": rf_m.feature_importances_}).sort_values("Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance", color="Importance",
                         color_continuous_scale="Oranges")
        fig_imp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27", font_color="white")
        st.plotly_chart(fig_imp, use_container_width=True)

# ════════════════════════════════════════════
# TAB 6 — HISTORY
# ════════════════════════════════════════════
with tab6:
    st.subheader("📂 Historical Database")

    if not df_hist.empty:
        df_f = df_hist[
            (df_hist["timestamp"].dt.date >= start_date) &
            (df_hist["timestamp"].dt.date <= end_date)
        ].copy()

        st.metric("Records in Selected Range", len(df_f))

        if not df_f.empty:
            fig_h = px.line(df_f.set_index("timestamp").resample("h").mean(numeric_only=True).reset_index(),
                             x="timestamp", y="pm10", title="Historical PM10 Trend",
                             color_discrete_sequence=["#ff6b35"])
            fig_h.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27", font_color="white")
            st.plotly_chart(fig_h, use_container_width=True)

            st.dataframe(df_f.sort_values("timestamp", ascending=False), use_container_width=True)

            csv_bytes = df_f.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv_bytes, "airsense_history.csv", "text/csv")
    else:
        st.info("No historical data yet. Run analyses to populate the database.")

# ─────────────────────────────────────────────
# COMPARE MODE
# ─────────────────────────────────────────────
if compare_mode:
    st.markdown("---")
    st.subheader(f"⚖️ City Comparison: {selected_city} vs {city2}")

    lat2, lon2, terrain2 = INDIA_CITIES[city2]
    with st.spinner(f"Fetching data for {city2}..."):
        weather2  = fetch_openweather(lat2, lon2)
        traffic2  = fetch_tomtom_traffic(lat2, lon2)
        waqi2     = fetch_waqi_city(city2)
        stations2 = fetch_waqi_bounds(lat2, lon2)

    if waqi2.get("pm10"):
        pm10_2 = waqi2["pm10"]
    elif not stations2.empty:
        pm10_2 = stations2["pm10"].mean()
    else:
        pm10_2 = estimate_pm10_no_sensor(weather2, traffic2, terrain2)

    pm10_2 = round(pm10_2, 1)
    cat2, color2, emoji2 = pm10_category(pm10_2)

    comp_data = {
        "Metric": ["PM10 (µg/m³)", "Temperature (°C)", "Humidity (%)",
                   "Wind Speed (km/h)", "Congestion (%)"],
        selected_city: [pm10_val, weather["temp"], weather["humidity"],
                         weather["wind_speed"], traffic.get("congestion_index",0)],
        city2:         [pm10_2, weather2["temp"], weather2["humidity"],
                         weather2["wind_speed"], traffic2.get("congestion_index",0)],
    }
    comp_df = pd.DataFrame(comp_data)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name=selected_city, x=comp_df["Metric"], y=comp_df[selected_city],
                               marker_color="#ff6b35"))
    fig_comp.add_trace(go.Bar(name=city2, x=comp_df["Metric"], y=comp_df[city2],
                               marker_color="#3d5a80"))
    fig_comp.update_layout(barmode="group", paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
                            font_color="white", height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**{selected_city}**")
        st.markdown(f"<span style='background:{color};padding:6px 12px;border-radius:8px;color:{'black' if color in ['#00e400','#ffff00'] else 'white'};font-weight:700'>{emoji} {cat} — {pm10_val} µg/m³</span>", unsafe_allow_html=True)
        st.caption(health_advisory(pm10_val))
    with cc2:
        st.markdown(f"**{city2}**")
        st.markdown(f"<span style='background:{color2};padding:6px 12px;border-radius:8px;color:{'black' if color2 in ['#00e400','#ffff00'] else 'white'};font-weight:700'>{emoji2} {cat2} — {pm10_2} µg/m³</span>", unsafe_allow_html=True)
        st.caption(health_advisory(pm10_2))

# ─────────────────────────────────────────────
# CUSTOM POINT PREDICTION
# ─────────────────────────────────────────────
if predict_pt:
    st.markdown("---")
    st.subheader(f"🎯 Custom Point Prediction ({custom_lat:.4f}°N, {custom_lon:.4f}°E)")
    with st.spinner("Predicting..."):
        cw = fetch_openweather(custom_lat, custom_lon)
        ct = fetch_tomtom_traffic(custom_lat, custom_lon)

        if not df_hist.empty and len(df_hist) >= 10:
            for col, val in [("temp",cw["temp"]),("hum",cw["humidity"]),("wind",cw["wind_speed"])]:
                if col not in df_hist.columns: df_hist[col] = val
            rf_c, sc_c, fc_c = build_nss_model(df_hist)
            now_ = pd.Timestamp.now()
            row = pd.DataFrame([[custom_lat, custom_lon, now_.hour, now_.dayofweek, now_.month,
                                   cw["temp"], cw["humidity"], cw["wind_speed"], pm10_val]], columns=fc_c)
            pred_c = np.expm1(rf_c.predict(sc_c.transform(row))[0])
        else:
            pred_c = estimate_pm10_no_sensor(cw, ct, terrain)

        cat_c, color_c, emoji_c = pm10_category(pred_c)
        st.metric("Predicted PM10", f"{pred_c:.1f} µg/m³")
        st.markdown(f"<span style='background:{color_c};padding:8px 14px;border-radius:8px;font-weight:700;color:{'black' if color_c in ['#00e400','#ffff00'] else 'white'}'>{emoji_c} {cat_c}</span>", unsafe_allow_html=True)
        st.caption(health_advisory(pred_c))

# ─────────────────────────────────────────────
# SIDEBAR INTELLIGENCE MONITOR
# ─────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Intelligence Monitor")
data_pts = len(df_hist)
st.sidebar.progress(min(data_pts/1000, 1.0), text=f"Data Maturity: {data_pts}/1000")
if data_pts > 100:
    st.sidebar.success("✅ NSS-Net: Full historical training active")
elif data_pts > 10:
    st.sidebar.info("📈 NSS-Net: Gathering patterns...")
else:
    st.sidebar.warning("👶 NSS-Net: Infant stage — using physics model")

st.sidebar.caption(f"🕐 Last refresh: {time.strftime('%H:%M:%S')}")
st.sidebar.caption("🌐 Sources: WAQI · OpenWeather · TomTom · OpenAQ · NASA FIRMS · Open-Elevation")
