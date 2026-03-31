import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import streamlit as st
import requests
from datetime import datetime
import sqlite3

# ==========================================
# WEATHER API
# ==========================================
API_KEY = "c86236be4a9f76875aad940c96e5111b"

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Lucknow&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    return data['main']['humidity'], data['main']['temp']

# ==========================================
# DATABASE
# ==========================================
conn = sqlite3.connect("biotwin.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    user TEXT,
    time TEXT,
    temp REAL,
    pressure REAL,
    naoh REAL,
    humidity REAL,
    stir REAL,
    mode TEXT,
    viscosity REAL,
    moisture REAL
)
""")
conn.commit()

def log_data(data):
    cursor.execute("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?,?)", data)
    conn.commit()

# ==========================================
# PHYSICS (NEW ADDITION)
# ==========================================
R = 8.314  # J/mol·K
A = 1.2e3  # pre-exponential factor
E = 42000  # activation energy (J/mol)

def reaction_rate(temp, naoh, ds):
    T = temp + 273.15
    k = A * np.exp(-E/(R*T))
    return k * naoh * (1 - ds)

def energy_balance(temp, stir, humidity):
    return temp + 0.02*stir - 0.01*humidity

# ==========================================
# DATA GENERATION (MODIFIED WITH PHYSICS)
# ==========================================
seq_len = 10

def generate_data(n=1000):
    X, y = [], []
    for _ in range(n):
        temp = np.random.uniform(60,80)
        press = np.random.uniform(180,250)
        naoh = np.random.uniform(0.38,0.44)
        hum = np.random.uniform(40,70)
        stir = np.random.uniform(200,400)
        mode = np.random.choice([0,1])

        seq, ds = [], 0.05
        for t in range(seq_len):

            temp_eff = energy_balance(temp, stir, hum)

            rate = reaction_rate(temp_eff, naoh, ds)

            if mode==1:
                ds += rate * (t/seq_len)
            else:
                ds += rate

            ds = min(ds, 1.0)

            seq.append([temp_eff,press,naoh,hum,stir,mode,ds])

        viscosity = 0.06*temp + 15*ds + 10*naoh
        moisture = 0.1*hum - 0.02*temp + 3.5 - 2*ds

        X.append(seq)
        y.append([viscosity, moisture])

    return np.array(X), np.array(y)

# ==========================================
# MODEL
# ==========================================
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1)
        )
    def forward(self,x):
        w = torch.softmax(self.net(x),dim=1)
        return torch.sum(w*x,dim=1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(7,64,batch_first=True)
        self.attn = Attention()
        self.fc = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Softplus()
        )
    def forward(self,x):
        out,_ = self.lstm(x)
        ctx = self.attn(out)
        return self.fc(ctx)

# ==========================================
# TRAIN MODEL
# ==========================================
@st.cache_resource
def train_model():
    X, y = generate_data()

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1,7)).reshape(X.shape)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_t = torch.FloatTensor(X_scaled)
    y_t = torch.FloatTensor(y_scaled)

    model = Model()
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    for _ in range(80):
        pred = model(X_t)
        loss = loss_fn(pred,y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = train_model()

# ==========================================
# SIMULATION (PHYSICS APPLIED)
# ==========================================
def simulate(temp,press,naoh,hum,stir,mode):
    seq, ds = [],0.05
    for t in range(seq_len):

        temp_eff = energy_balance(temp, stir, hum)
        rate = reaction_rate(temp_eff, naoh, ds)

        if mode==1:
            ds += rate * (t/seq_len)
        else:
            ds += rate

        ds = min(ds,1.0)

        seq.append([temp_eff,press,naoh,hum,stir,mode,ds])

    seq = scaler_X.transform(np.array(seq)).reshape(1,seq_len,-1)
    pred = model(torch.FloatTensor(seq)).detach().numpy()
    return scaler_y.inverse_transform(pred)[0]

# ==========================================
# GENETIC OPT (UNCHANGED)
# ==========================================
def genetic_opt(hum):
    pop = [np.random.uniform([60,180,0.38,200,0],
                             [80,250,0.44,400,1]) for _ in range(20)]

    for _ in range(10):
        scores = []
        for ind in pop:
            temp,press,naoh,stir,mode = ind
            mode = int(round(mode))
            v,m = simulate(temp,press,naoh,hum,stir,mode)
            scores.append(v - 2*m)

        pop = [pop[i] for i in np.argsort(scores)[-10:]]

        for _ in range(10):
            idx = np.random.choice(len(pop),2,replace=False)
            p1,p2 = pop[idx[0]],pop[idx[1]]

            child = (p1+p2)/2 + np.random.normal(0,0.02,5)

            child = np.clip(child,
                            [60,180,0.38,200,0],
                            [80,250,0.44,400,1])

            pop.append(child)

    best = pop[np.argmax(scores)]
    temp,press,naoh,stir,mode = best
    mode = int(round(mode))
    v,m = simulate(temp,press,naoh,hum,stir,mode)

    return temp,press,naoh,stir,mode,v,m

# ==========================================
# STREAMLIT UI (UNCHANGED)
# ==========================================
st.set_page_config(layout="wide")
st.title("🧪 BioTwin Industrial Digital Twin")

if "user" not in st.session_state:
    user = st.text_input("Enter Username")
    if st.button("Login"):
        st.session_state.user = user

if "user" in st.session_state:

    st.success(f"Logged in as {st.session_state.user}")

    hum,temp_out = get_weather()

    col1, col2 = st.columns(2)
    col1.metric("Ambient Temp (°C)", temp_out)
    col2.metric("Humidity (%)", hum)

    st.divider()

    temp = st.slider("Temperature (°C)",60,80,70)
    press = st.slider("Pressure (kPa)",180,250,210)
    naoh = st.slider("NaOH Ratio",0.38,0.44,0.40)
    stir = st.slider("Stir Speed (RPM)",200,400,300)
    mode = st.selectbox("Mode",["Bulk","Stepwise"])
    mode_val = 1 if mode=="Stepwise" else 0

    if st.button("Run Prediction"):
        v,m = simulate(temp,press,naoh,hum,stir,mode_val)

        st.success(f"Viscosity: {v:.2f} Pa·s")
        st.success(f"Moisture: {m:.2f} %")

        log_data((st.session_state.user,str(datetime.now()),
                  temp,press,naoh,hum,stir,mode,v,m))

    if st.button("Optimize Process"):
        best = genetic_opt(hum)

        temp,press,naoh,stir,mode,v,m = best
        mode_text = "Stepwise" if mode==1 else "Bulk"

        st.subheader("Optimal Settings")

        st.write(f"Temperature: {temp:.2f} °C")
        st.write(f"Pressure: {press:.2f} kPa")
        st.write(f"NaOH: {naoh:.3f}")
        st.write(f"Stir: {stir:.2f} RPM")
        st.write(f"Mode: {mode_text}")

        st.subheader("Expected Output")
        st.write(f"Viscosity: {v:.2f} Pa·s")
        st.write(f"Moisture: {m:.2f} %")

    if st.checkbox("Show Logs"):
        import pandas as pd
        df = pd.read_sql("SELECT * FROM logs", conn)
        st.dataframe(df)
