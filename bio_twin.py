import streamlit as st
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime
from scipy.integrate import odeint
from sklearn.ensemble import RandomForestRegressor
import pyvista as pv
from stmol import showmol

# ==================================================
# CONFIG
# ==================================================
LOG_FILE = "bio_twin_log.csv"

st.set_page_config(
    page_title="Bio-Twin Intelligent Fermentation Platform",
    layout="wide"
)

st.title("🧬 Bio-Twin Intelligent Fermentation Platform")

# ==================================================
# THINGSPEAK API
# ==================================================
def fetch_thingspeak_latest():
    url = (
        f"https://api.thingspeak.com/channels/"
        f"{st.secrets['THINGSPEAK_CHANNEL_ID']}/feeds.json"
        f"?api_key={st.secrets['THINGSPEAK_READ_KEY']}&results=1"
    )
    r = requests.get(url, timeout=10).json()
    feed = r["feeds"][-1]

    return {
        "pH": float(feed["field1"]),
        "temp": float(feed["field2"]),
        "DO": float(feed["field3"]),
        "time": feed["created_at"]
    }

# ==================================================
# MACBELL KINETICS
# ==================================================
def macbell_ode(state, t, mu_max, Ks, Yxs, pH, T):
    X, S = state
    pH_opt, T_opt = 5.5, 30.0

    f_env = np.exp(-(pH - pH_opt)**2) * np.exp(-(T - T_opt)**2 / 25)
    mu = mu_max * (S / (Ks + S)) * f_env

    return [mu * X, -(1 / Yxs) * mu * X]


def simulate_growth(pH, T, stress_pH=None, stress_time=0):
    t = np.linspace(0, 48, 240)
    X0, S0 = 0.1, 20.0
    mu_max, Ks, Yxs = 0.4, 0.5, 0.6

    biomass = []

    for ti in t:
        pH_use = stress_pH if stress_pH and ti >= stress_time else pH
        sol = odeint(
            macbell_ode,
            [X0, S0],
            [0, 0.2],
            args=(mu_max, Ks, Yxs, pH_use, T)
        )[-1]

        X0, S0 = sol
        biomass.append(X0)

    return t, np.array(biomass)

# ==================================================
# 3D BIOREACTOR
# ==================================================
def render_bioreactor(biomass):
    mat_height = max(0.3, biomass * 4)
    mat = pv.Cylinder(radius=2.2, height=mat_height)
    mat.translate([0, 0, -5 + mat_height / 2])
    showmol(mat, height=420, width=420)

# ==================================================
# PID CONTROLLER
# ==================================================
class PIDController:
    def __init__(self, Kp=1.2, Ki=0.1, Kd=0.05):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, measured, dt=1.0):
        error = setpoint - measured
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

# ==================================================
# MPC CONTROLLER (LIGHTWEIGHT)
# ==================================================
def mpc_temperature_control(current_biomass, target_biomass, temp):
    error = target_biomass - current_biomass
    adjustment = np.clip(error * 0.6, -2, 2)
    return np.clip(temp + adjustment, 25, 35)

# ==================================================
# ALARMS
# ==================================================
def check_alarms(pH, DO, growth_rate):
    alarms = []
    if pH < 4.0:
        alarms.append("⚠ pH CRASH detected")
    if DO < 20:
        alarms.append("⚠ Oxygen limitation")
    if growth_rate < 0.001:
        alarms.append("⚠ Growth stagnation")
    return alarms

# ==================================================
# DATA LOGGER
# ==================================================
def log_state(pH, temp, DO, biomass, reactor):
    row = {
        "timestamp": datetime.utcnow(),
        "reactor": reactor,
        "pH": pH,
        "temp": temp,
        "DO": DO,
        "biomass": biomass
    }
    df = pd.DataFrame([row])

    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

# ==================================================
# INVERSE KINETIC MODEL
# ==================================================
def inverse_predict(target_biomass):
    if not os.path.exists(LOG_FILE):
        return None

    df = pd.read_csv(LOG_FILE)
    if len(df) < 20:
        return None

    X = df[["biomass"]]
    y = df[["pH", "temp", "DO"]]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    pred = model.predict([[target_biomass]])[0]

    return {
        "Recommended pH": round(pred[0], 2),
        "Recommended Temperature (°C)": round(pred[1], 2),
        "Recommended DO (%)": round(pred[2], 2)
    }

# ==================================================
# APP STATE
# ==================================================
reactor = st.sidebar.selectbox(
    "Select Reactor",
    ["Reactor-A", "Reactor-B", "Reactor-C"]
)

if "pid" not in st.session_state:
    st.session_state.pid = PIDController()

# ==================================================
# LIVE SENSOR DATA
# ==================================================
sensor = fetch_thingspeak_latest()

c1, c2, c3 = st.columns(3)
c1.metric("pH", sensor["pH"])
c2.metric("Temperature (°C)", sensor["temp"])
c3.metric("DO (%)", sensor["DO"])
st.caption(f"Last update: {sensor['time']}")

# ==================================================
# TABS
# ==================================================
tabs = st.tabs(["🧪 Bio-Twin Digital Twin", "🔁 Inverse Kinetic Profiler"])

# ==================================================
# BIO-TWIN TAB
# ==================================================
with tabs[0]:
    st.subheader("Live Digital Twin Simulation")

    stress = st.checkbox("Apply pH Stress")
    stress_pH = st.slider("Stress pH", 3.0, 7.0, 4.5)
    stress_time = st.slider("Stress Time (hr)", 0, 48, 12)

    target_biomass = st.slider("Target Biomass", 1.0, 10.0, 4.0)
    control_mode = st.selectbox("Control Mode", ["PID", "MPC"])

    t, biomass = simulate_growth(
        sensor["pH"],
        sensor["temp"],
        stress_pH if stress else None,
        stress_time
    )

    current = biomass[-1]
    prev = biomass[-5] if len(biomass) > 5 else current
    growth_rate = current - prev

    if control_mode == "PID":
        adjusted_temp = sensor["temp"] + st.session_state.pid.compute(
            target_biomass, current
        ) * 0.4
    else:
        adjusted_temp = mpc_temperature_control(
            current, target_biomass, sensor["temp"]
        )

    st.metric("Adjusted Reactor Temperature (°C)", round(adjusted_temp, 2))
    st.line_chart(biomass)
    render_bioreactor(current)

    log_state(
        sensor["pH"],
        sensor["temp"],
        sensor["DO"],
        current,
        reactor
    )

    alarms = check_alarms(sensor["pH"], sensor["DO"], growth_rate)
    if alarms:
        for a in alarms:
            st.error(a)
    else:
        st.success("System operating normally")

# ==================================================
# INVERSE MODEL TAB
# ==================================================
with tabs[1]:
    st.subheader("Inverse Kinetic Profiler (AI Recipe Generator)")

    desired = st.slider("Desired Final Biomass", 0.5, 10.0, 3.0)

    if st.button("Generate Optimal Recipe"):
        recipe = inverse_predict(desired)
        if recipe:
            st.json(recipe)
        else:
            st.warning("Not enough logged data yet. Run more experiments.")
