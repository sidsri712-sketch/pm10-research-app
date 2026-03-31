import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
import pandas as pd
import streamlit as st
import sqlite3
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ==========================================
#  WEATHER API
# ==========================================
API_KEY = "c86236be4a9f76875aad940c96e5111b"

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Lucknow&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    return data['main']['humidity'], data['main']['temp']

# ==========================================
#  DATABASE SETUP
# ==========================================
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
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
    cursor.execute("INSERT INTO logs VALUES (?,?,?,?,?,?,?,?,?)", data)
    conn.commit()

# ==========================================
#  DATA GENERATOR (for training)
# ==========================================
def generate_data(n=1000, seq_len=10):
    X, y = [], []
    for _ in range(n):
        temp = np.random.uniform(60,80)
        press = np.random.uniform(180,250)
        naoh = np.random.uniform(0.38,0.44)
        hum = np.random.uniform(40,70)
        stir = np.random.uniform(200,400)
        mode = np.random.choice([0,1])

        seq, ds = [], 0
        for t in range(seq_len):
            if mode==1:
                ds += naoh*0.12*(t/seq_len)
            else:
                ds = naoh*1.5 if t>3 else naoh*0.4*t

            seq.append([temp,press,naoh,hum,stir,mode,ds])

        viscosity = 0.06*temp + 12*naoh + 2.5*ds
        moisture = 0.1*hum - 0.02*temp + 3.5

        X.append(seq)
        y.append([viscosity, moisture])

    return np.array(X), np.array(y)

# ==========================================
#  SCALING + TRAIN MODEL
# ==========================================
seq_len = 10
X, y = generate_data()

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X.reshape(-1,7)).reshape(X.shape)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

X_t = torch.FloatTensor(X_scaled)
y_t = torch.FloatTensor(y_scaled)

# ==========================================
#  MODEL
# ==========================================
class Attention(nn.Module):
    def _init_(self):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1)
        )
    def forward(self,x):
        w = torch.softmax(self.net(x),dim=1)
        return torch.sum(w*x,dim=1), w

class Model(nn.Module):
    def _init_(self):
        super()._init_()
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
        ctx,_ = self.attn(out)
        return self.fc(ctx)

model = Model()
opt = torch.optim.Adam(model.parameters(),lr=0.005)
loss_fn = nn.MSELoss()

# quick training
for _ in range(80):
    pred = model(X_t)
    loss = loss_fn(pred,y_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

# ==========================================
#  SIMULATION
# ==========================================
def simulate(temp,press,naoh,hum,stir,mode):
    seq, ds = [],0
    for t in range(seq_len):
        if mode==1:
            ds += naoh*0.12*(t/seq_len)
        else:
            ds = naoh*1.5 if t>3 else naoh*0.4*t
        seq.append([temp,press,naoh,hum,stir,mode,ds])

    seq = scaler_X.transform(np.array(seq)).reshape(1,seq_len,-1)
    pred = model(torch.FloatTensor(seq)).detach().numpy()
    return scaler_y.inverse_transform(pred)[0]

# ==========================================
#  GENETIC OPTIMIZATION
# ==========================================
def genetic_opt():
    hum,_ = get_weather()
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
            p1,p2 = np.random.choice(pop,2)
            child = (p1+p2)/2 + np.random.normal(0,0.02,5)
            pop.append(child)

    best = pop[np.argmax(scores)]
    temp,press,naoh,stir,mode = best
    mode = int(round(mode))
    v,m = simulate(temp,press,naoh,hum,stir,mode)

    return temp,press,naoh,stir,mode,v,m

# ==========================================
#  STREAMLIT UI
# ==========================================
st.title(" BioTwin SaaS - Smart HPMC Digital Twin")

hum,temp_out = get_weather()
st.info(f" Humidity: {hum}% | Temp: {temp_out}°C")

# Inputs
temp = st.slider("Temperature",60,80,70)
press = st.slider("Pressure",180,250,210)
naoh = st.slider("NaOH Ratio",0.38,0.44,0.40)
stir = st.slider("Stir Speed",200,400,300)
mode = st.selectbox("Mode",["Bulk","Stepwise"])
mode_val = 1 if mode=="Stepwise" else 0

# Prediction
if st.button("Predict"):
    v,m = simulate(temp,press,naoh,hum,stir,mode_val)

    st.success(f"Viscosity: {v:.2f}")
    st.success(f"Moisture: {m:.2f}")

    log_data((datetime.now(),temp,press,naoh,hum,stir,mode,v,m))

# Optimization
if st.button("Optimize Process"):
    best = genetic_opt()
    st.write(" Optimal Settings:", best)

# View logs
if st.checkbox("Show Logs"):
    df = pd.read_sql("SELECT * FROM logs", conn)
    st.dataframe(df)
