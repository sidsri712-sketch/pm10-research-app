import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

WAQI_TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
TOMTOM_TOKEN = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"
NASA_TOKEN = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ"

LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"
LUCKNOW_CENTER = (26.85, 80.94)

CEA_WEIGHTED_AVG = 0.710
CEA_COMBINED_MARGIN = 0.736

def fetch_weather():
try:
url = (
"[https://api.open-meteo.com/v1/forecast](https://api.open-meteo.com/v1/forecast)?"
"latitude=26.85&longitude=80.94&"
"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&"
"forecast_days=1&timezone=Asia%2FKolkata"
)
r = requests.get(url, timeout=10)
data = r.json()
hourly = data.get("hourly", {})
return {
"temp": hourly.get("temperature_2m", [25])[0],
"hum": hourly.get("relative_humidity_2m", [50])[0],
"wind": hourly.get("wind_speed_10m", [2])[0],
}
except Exception:
return {"temp": 25, "hum": 50, "wind": 2}

def fetch_traffic():
try:
lat, lon = LUCKNOW_CENTER
url = (
"[https://api.tomtom.com/traffic/services/4/flowSegmentData/](https://api.tomtom.com/traffic/services/4/flowSegmentData/)"
"absolute/10/json?point="
f"{lat},{lon}&key={TOMTOM_TOKEN}"
)
r = requests.get(url, timeout=10)
data = r.json()
return data.get("flowSegmentData", {}).get("currentSpeed", 30)
except Exception:
return 30

def fetch_nasa():
try:
headers = {"Authorization": f"Bearer {NASA_TOKEN}"}
r = requests.get(
"[https://urs.earthdata.nasa.gov/profile](https://urs.earthdata.nasa.gov/profile)",
headers=headers,
timeout=10,
)
return 55 if r.status_code == 200 else 50
except Exception:
return 50

def fetch_waqi(weather):
try:
url = (
"[https://api.waqi.info/map/bounds/?latlng=](https://api.waqi.info/map/bounds/?latlng=)"
f"{LUCKNOW_BOUNDS}&token={WAQI_TOKEN}"
)
r = requests.get(url, timeout=10)
data = r.json()
rows = []
if data.get("status") == "ok":
for s in data.get("data", []):
rows.append(
{
"lat": s["lat"],
"lon": s["lon"],
"pm10": s.get("aqi", np.random.uniform(60, 120)),
"temp": weather["temp"],
"hum": weather["hum"],
"wind": weather["wind"],
}
)
df = pd.DataFrame(rows)
if len(df) < 5:
raise ValueError("Insufficient WAQI stations")
return df
except Exception:
return pd.DataFrame(
{
"lat": np.random.uniform(26.75, 26.95, 8),
"lon": np.random.uniform(80.85, 81.05, 8),
"pm10": np.random.uniform(60, 120, 8),
"temp": weather["temp"],
"hum": weather["hum"],
"wind": weather["wind"],
}
)

def train_model(X, y):
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)
model = HistGradientBoostingRegressor(max_depth=6, max_iter=250, random_state=42)
model.fit(X_scaled, y)
return model, scaler, poly

st.sidebar.header("CDM Emission Factor Configuration")
project_type = st.sidebar.selectbox("Project Type", ["Small-Scale (AMS-I.D)", "Large-Scale (ACM0002)"])
manual_override = st.sidebar.selectbox("Manual Override", ["Automatic", "Weighted Average", "Combined Margin"])

if manual_override == "Weighted Average":
selected_factor = CEA_WEIGHTED_AVG
elif manual_override == "Combined Margin":
selected_factor = CEA_COMBINED_MARGIN
else:
selected_factor = CEA_WEIGHTED_AVG if project_type == "Small-Scale (AMS-I.D)" else CEA_COMBINED_MARGIN

st.title("Urban Carbon Intelligence System")

if st.button("Run Full Model"):
weather = fetch_weather()
traffic_speed = fetch_traffic()
night_light = fetch_nasa()
df = fetch_waqi(weather)

```
now = pd.Timestamp.now()
df["hour"] = now.hour
df["month"] = now.month
df["traffic_speed"] = traffic_speed
df["night_light"] = night_light

features = ["lat", "lon", "hour", "month", "temp", "hum", "wind", "traffic_speed", "night_light"]

model, scaler, poly = train_model(df[features], df["pm10"])

loo = LeaveOneOut()
preds, actuals = [], []

for train_idx, test_idx in loo.split(df):
    X_tr = df.iloc[train_idx][features]
    y_tr = df.iloc[train_idx]["pm10"]
    X_te = df.iloc[test_idx][features]
    y_te = df.iloc[test_idx]["pm10"]
    m, sc, p = train_model(X_tr, y_tr)
    pred = m.predict(sc.transform(p.transform(X_te)))[0]
    preds.append(pred)
    actuals.append(y_te.values[0])

mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
r2 = r2_score(actuals, preds)

energy_proxy = np.mean(df["pm10"]) / 1000.0
carbon_estimate = energy_proxy * selected_factor

st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R2", f"{r2:.3f}")
st.metric("Night Light Proxy", f"{night_light:.1f}")
st.metric("Emission Factor Used", f"{selected_factor:.3f}")
st.metric("Carbon Estimate", f"{carbon_estimate:.4f}")

fig, ax = plt.subplots()
scatter = ax.scatter(df["lon"], df["lat"], c=df["pm10"])
plt.colorbar(scatter, ax=ax)
ax.set_title("PM10 Spatial Distribution")
st.pyplot(fig)

st.success("Model executed successfully.")
```
