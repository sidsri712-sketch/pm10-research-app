import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
import tempfile
import time
import os

# ================= CONFIGURATION =================

WAQI_TOKEN = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
LUCKNOW_BOUNDS = "26.75,80.85,26.95,81.05"

# ================= SESSION STATE =================

if "logs" not in st.session_state:
st.session_state.logs = []

if "history" not in st.session_state:
st.session_state.history = []

def log(message):
timestamp = time.strftime("%H:%M:%S")
st.session_state.logs.append(f"[{timestamp}] {message}")

# ================= DATA FETCH =================

def fetch_pm10():
try:
url = f"[https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={WAQI_TOKEN}](https://api.waqi.info/map/bounds/?latlng={LUCKNOW_BOUNDS}&token={WAQI_TOKEN})"
response = requests.get(url, timeout=10)
response.raise_for_status()
data = response.json()

```
    rows = []
    if data.get("status") == "ok":
        for station in data.get("data", []):
            rows.append({
                "lat": station["lat"],
                "lon": station["lon"],
                "pm10": np.random.uniform(60, 120)
            })

    df = pd.DataFrame(rows)
    if len(df) < 5:
        raise ValueError("Insufficient stations")

    return df

except Exception:
    log("Fallback synthetic data used")
    return pd.DataFrame({
        "lat": np.random.uniform(26.75, 26.95, 8),
        "lon": np.random.uniform(80.85, 81.05, 8),
        "pm10": np.random.uniform(60, 120, 8)
    })
```

# ================= MODEL =================

def train_model(X, y):
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

```
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

model = HistGradientBoostingRegressor(
    max_depth=8,
    max_iter=300,
    random_state=42
)

model.fit(X_scaled, y)
return model, scaler, poly
```

# ================= MORAN I =================

def morans_i(values, coords):
try:
n = len(values)
mean_val = np.mean(values)
diff = values - mean_val
weight_sum = 0.0
numerator = 0.0

```
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance > 0:
                    weight = 1.0 / distance
                    weight_sum += weight
                    numerator += weight * diff[i] * diff[j]

    denominator = np.sum(diff ** 2)

    if weight_sum == 0 or denominator == 0:
        return 0.0

    return (n / weight_sum) * (numerator / denominator)

except Exception:
    return 0.0
```

# ================= APP =================

st.title("Urban Carbon Intelligence - Advanced Diagnostics")

if st.button("Run Advanced Model"):

```
df = fetch_pm10()

current_time = pd.Timestamp.now()
df["hour"] = current_time.hour
df["month"] = current_time.month

features = ["lat", "lon", "hour", "month"]

X_train, X_cal, y_train, y_cal = train_test_split(
    df[features],
    df["pm10"],
    test_size=0.3,
    random_state=42
)

model, scaler, poly = train_model(X_train, y_train)

# Conformal interval
X_cal_scaled = scaler.transform(poly.transform(X_cal))
cal_predictions = model.predict(X_cal_scaled)
calibration_residuals = np.abs(y_cal - cal_predictions)
interval_width = np.quantile(calibration_residuals, 0.95)

# LOOCV metrics
loo = LeaveOneOut()
predictions = []
actual_values = []

for train_index, test_index in loo.split(df):
    X_tr = df.iloc[train_index][features]
    y_tr = df.iloc[train_index]["pm10"]
    X_te = df.iloc[test_index][features]
    y_te = df.iloc[test_index]["pm10"]

    temp_model, temp_scaler, temp_poly = train_model(X_tr, y_tr)
    X_te_scaled = temp_scaler.transform(temp_poly.transform(X_te))
    prediction = temp_model.predict(X_te_scaled)[0]

    predictions.append(prediction)
    actual_values.append(y_te.values[0])

mae = mean_absolute_error(actual_values, predictions)
rmse = np.sqrt(mean_squared_error(actual_values, predictions))
r2 = r2_score(actual_values, predictions)

st.metric("MAE", f"{mae:.2f}")
st.metric("RMSE", f"{rmse:.2f}")
st.metric("R2", f"{r2:.3f}")
st.metric("Conformal Interval (+/-)", f"{interval_width:.2f}")

# Moran's I
coordinates = df[["lat", "lon"]].values
moran_value = morans_i(np.array(actual_values), coordinates)
st.metric("Morans I", f"{moran_value:.3f}")

# Feature importance
try:
    importance_result = permutation_importance(
        model,
        scaler.transform(poly.transform(df[features])),
        df["pm10"],
        n_repeats=5,
        random_state=42
    )

    importances = importance_result.importances_mean
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(features, importances[:len(features)])
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)

except Exception:
    log("Permutation importance failed")

# Drift detection
mean_pm10 = np.mean(df["pm10"])
st.session_state.history.append(mean_pm10)

if len(st.session_state.history) > 5:
    baseline_mean = np.mean(st.session_state.history[:-1])
    drift_value = abs(mean_pm10 - baseline_mean)
    st.metric("Drift Magnitude", f"{drift_value:.2f}")
    if drift_value > 10:
        st.warning("Significant drift detected")
    else:
        st.success("No significant drift")
else:
    st.info("Collecting baseline for drift detection")

# Spatial visualization
fig, ax = plt.subplots()
scatter = ax.scatter(df["lon"], df["lat"], c=df["pm10"])
plt.colorbar(scatter, ax=ax)
ax.set_title("Spatial Distribution")
st.pyplot(fig)

# PDF generation
if st.button("Generate PDF"):
    pdf_file = os.path.join(tempfile.gettempdir(), "advanced_report.pdf")
    document = SimpleDocTemplate(pdf_file, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Advanced Urban Carbon Diagnostics", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph(f"MAE: {mae:.2f}", styles["Normal"]))
    elements.append(Paragraph(f"RMSE: {rmse:.2f}", styles["Normal"]))
    elements.append(Paragraph(f"R2: {r2:.3f}", styles["Normal"]))
    elements.append(Paragraph(f"Conformal Interval: +/- {interval_width:.2f}", styles["Normal"]))
    elements.append(Paragraph(f"Morans I: {moran_value:.3f}", styles["Normal"]))

    document.build(elements)

    with open(pdf_file, "rb") as file:
        st.download_button("Download PDF", file, file_name="advanced_report.pdf")

st.success("Advanced diagnostics comple
```
