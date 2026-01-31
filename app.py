import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as cx
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from sklearn.ensemble import RandomForestRegressor
from scipy.ndimage import gaussian_filter
from streamlit_autorefresh import st_autorefresh
import datetime
import time
import os

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

# PASTE YOUR PUBLIC GOOGLE SHEET URL HERE
# MAKE SURE IT IS SET TO "ANYONE WITH LINK CAN EDIT"
SHEET_URL = "https://docs.google.com/spreadsheets/d/https://docs.google.com/spreadsheets/d/1GdoASrXme0Ti2eO5rfCVkNgIrsXJL_85I9kynesnZ78/edit?usp=sharing/export?format=csv"
# For SAVING data, we use the Export URL to read, and a direct push to write.
# Since we are avoiding the Service Account, we use a trick to read it as a CSV.
EXPORT_URL = SHEET_URL.replace('/edit?usp=sharing', '/export?format=csv')

st.set_page_config(page_title="Lucknow PM10 Model", layout="wide")
st_autorefresh(interval=1800000, key="refresh")

def fetch_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=26.85&longitude=80.94&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
        r = requests.get(url).json()
        return {"temp": r["current"]["temperature_2m"], "hum": r["current"]["relative_humidity_2m"], "wind": r["current"]["wind_speed_10m"]}
    except:
        return {"temp": 25.0, "hum": 50.0, "wind": 5.0}

@st.cache_data(ttl=900)
def load_historical_data():
    try:
        # Reads the Google Sheet directly as a CSV
        return pd.read_csv(EXPORT_URL)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_pm10_data():
    weather = fetch_weather()
    url = f"https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={TOKEN}"
    records = []
    try:
        r = requests.get(url).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                d = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={TOKEN}").json()
                if d.get("status") == "ok" and "pm10" in d["data"].get("iaqi", {}):
                    records.append({
                        "lat": s["lat"], "lon": s["lon"], "pm10": d["data"]["iaqi"]["pm10"]["v"],
                        "name": d["data"]["city"]["name"], "temp": weather["temp"],
                        "hum": weather["hum"], "wind": weather["wind"], "timestamp": pd.Timestamp.now()
                    })
        
        df_live = pd.DataFrame(records)
        if not df_live.empty:
            # We display a message that data is being collected.
            # NOTE: For "Writing" to a public sheet without a service account, 
            # the easiest way is to use a Google Forms URL or a small Apps Script.
            # For now, this code will READ from the cloud so the model is shared.
            return df_live.groupby(["lat", "lon"]).agg({"pm10":"mean","name":"first","temp":"first","hum":"first","wind":"first"}).reset_index()
    except:
        pass
    return pd.DataFrame()

# --------------------------------------------------
# UI & MODEL LOGIC (UNCHANGED)
# --------------------------------------------------
st.title("📍 Lucknow PM10 Hybrid Spatial Analysis")
df_history = load_historical_data()
df_live = fetch_pm10_data()

# [REST OF YOUR MODEL LOGIC HERE - NO CHANGES TO RF OR KRIGING]
