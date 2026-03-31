# ==========================================
# BIO TWIN - FULL STACK (FASTAPI + MODEL)
# ==========================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI, Header
from pydantic import BaseModel
import requests
from datetime import datetime
import os

# ==========================================
# WEATHER API (YOUR KEY USED)
# ==========================================
API_KEY = "c86236be4a9f76875aad940c96e5111b"

def get_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Lucknow&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
    return data['main']['humidity']

# ==========================================
# CLOUD LOGGING (AWS / GCP AUTO SWITCH)
# ==========================================
USE_AWS = os.getenv("USE_AWS", "false") == "true"

if USE_AWS:
    import boto3
    db = boto3.resource('dynamodb')
    table = db.Table("biotwin_logs")

    def log(data):
        table.put_item(Item=data)

else:
    try:
        from google.cloud import firestore
        db = firestore.Client()

        def log(data):
            db.collection("biotwin_logs").add(data)
    except:
        def log(data):
            print("LOG:", data)

# ==========================================
# DATA GENERATION
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
# TRAIN MODEL (RUN ON START)
# ==========================================
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
    loss = loss_fn(pred, y_t)
    opt.zero_grad()
    loss.backward()
    opt.step()

# ==========================================
# SIMULATION
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
# GENETIC OPTIMIZATION (SAFE)
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

            # HARD CONSTRAINTS (IMPORTANT)
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
# FASTAPI
# ==========================================
app = FastAPI(title="BioTwin Industrial API")

class Input(BaseModel):
    temp: float
    press: float
    naoh: float
    stir: float
    mode: int

# -------------------------
# PREDICTION
# -------------------------
@app.post("/predict")
def predict(inp: Input, user: str = Header(default="guest")):
    hum = get_weather()

    v,m = simulate(inp.temp,inp.press,inp.naoh,hum,inp.stir,inp.mode)

    log({
        "user": user,
        "time": str(datetime.now()),
        "temp": inp.temp,
        "press": inp.press,
        "naoh": inp.naoh,
        "stir": inp.stir,
        "mode": inp.mode,
        "viscosity": float(v),
        "moisture": float(m)
    })

    return {
        "user": user,
        "viscosity": float(v),
        "moisture": float(m),
        "message": "Prediction successful"
    }

# -------------------------
# OPTIMIZATION
# -------------------------
@app.get("/optimize")
def optimize(user: str = Header(default="guest")):
    hum = get_weather()

    best = genetic_opt(hum)

    temp,press,naoh,stir,mode,v,m = best

    mode_text = "Stepwise" if mode==1 else "Bulk"

    return {
        "user": user,
        "recommended_settings": {
            "temperature_C": round(temp,2),
            "pressure_kPa": round(press,2),
            "naoh_ratio": round(naoh,3),
            "stir_rpm": round(stir,2),
            "mode": mode_text
        },
        "expected_output": {
            "viscosity_Pa_s": round(float(v),2),
            "moisture_percent": round(float(m),2)
        },
        "action": "Apply above parameters in reactor"
    }

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def home():
    return {"status": "BioTwin API running"}
