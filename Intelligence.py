"""
AirSense Intelligence Engine
─────────────────────────────
Temporal forecasting  · Source attribution · Event detection · Alert generation
Models: XGBoost lag · RF autoregression · Prophet-style decomposition · LSTM-lite
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
def build_features(df: pd.DataFrame, target="pm10") -> pd.DataFrame:
    """
    Build the full temporal feature matrix from a historical DataFrame.
    Requires columns: timestamp, pm10, temp, humidity (or hum), wind_speed (or wind),
                       lat, lon  (at minimum).
    """
    df = df.copy().sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Normalise column names
    if "hum" in df.columns and "humidity" not in df.columns:
        df["humidity"] = df["hum"]
    if "wind" in df.columns and "wind_speed" not in df.columns:
        df["wind_speed"] = df["wind"]
    for col, default in [("temp",25),("humidity",55),("wind_speed",10),
                          ("pressure",1013),("congestion",20),
                          ("fire_hotspots",0),("pm25",np.nan)]:
        if col not in df.columns:
            df[col] = default

    # Calendar features
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month
    df["is_morning_rush"] = df["hour"].isin(range(7,10)).astype(int)
    df["is_evening_rush"] = df["hour"].isin(range(17,20)).astype(int)
    df["is_night"]        = df["hour"].isin(range(0,6)).astype(int)
    df["is_weekend"]      = (df["dayofweek"] >= 5).astype(int)

    # Lag features (t-1 to t-24)
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f"pm10_lag{lag}"] = df[target].shift(lag)

    # Rolling statistics
    for window in [3, 6, 12, 24]:
        df[f"pm10_roll_mean_{window}"] = df[target].shift(1).rolling(window).mean()
        df[f"pm10_roll_std_{window}"]  = df[target].shift(1).rolling(window).std()

    # Rate of change
    df["pm10_delta1"]  = df[target].diff(1)
    df["pm10_delta3"]  = df[target].diff(3)
    df["pm10_delta6"]  = df[target].diff(6)

    # Meteorological composite index
    # High pressure + low wind + low cloud = trapping conditions
    df["trap_index"] = (
        (df["pressure"].clip(1005, 1030) - 1005) / 25 * 0.4 +
        (1 / (df["wind_speed"].clip(0.5, 20))) * 5 * 0.4 +
        (df["humidity"].clip(0, 100) / 100) * 0.2
    )

    # Wind direction features (if available)
    if "wind_deg" in df.columns:
        df["wind_sin"] = np.sin(np.radians(df["wind_deg"]))
        df["wind_cos"] = np.cos(np.radians(df["wind_deg"]))
    else:
        df["wind_sin"] = 0.0
        df["wind_cos"] = 1.0

    return df

FEATURE_COLS = [
    "hour","dayofweek","month","is_morning_rush","is_evening_rush",
    "is_night","is_weekend",
    "pm10_lag1","pm10_lag2","pm10_lag3","pm10_lag6","pm10_lag12","pm10_lag24",
    "pm10_roll_mean_3","pm10_roll_mean_6","pm10_roll_mean_12","pm10_roll_mean_24",
    "pm10_roll_std_3","pm10_roll_std_6",
    "pm10_delta1","pm10_delta3","pm10_delta6",
    "temp","humidity","wind_speed","pressure","trap_index",
    "wind_sin","wind_cos",
    "congestion","fire_hotspots",
]

# ══════════════════════════════════════════════════════
# 2. XGBOOST-STYLE LAG MODEL (using GradientBoosting — no xgboost dep)
# ══════════════════════════════════════════════════════
class LagBoostModel:
    """Gradient-boosted lag autoregression — fast, explainable."""
    def __init__(self):
        self.model   = GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
        self.scaler  = StandardScaler()
        self.trained = False
        self.feature_importance_ = {}

    def fit(self, df: pd.DataFrame):
        df_f = build_features(df)
        cols = [c for c in FEATURE_COLS if c in df_f.columns]
        df_f = df_f.dropna(subset=cols + ["pm10"])
        if len(df_f) < 20:
            return self
        X = self.scaler.fit_transform(df_f[cols])
        y = np.log1p(df_f["pm10"].values)
        self.model.fit(X, y)
        self.cols    = cols
        self.trained = True
        self.feature_importance_ = dict(zip(cols, self.model.feature_importances_))
        return self

    def predict_one(self, state: dict) -> float:
        if not self.trained:
            return state.get("pm10_lag1", 80)
        row = pd.DataFrame([{c: state.get(c, 0) for c in self.cols}])
        X   = self.scaler.transform(row)
        return float(np.expm1(self.model.predict(X)[0]))


# ══════════════════════════════════════════════════════
# 3. RF AUTOREGRESSION WITH UNCERTAINTY BANDS
# ══════════════════════════════════════════════════════
class RFAutoRegModel:
    """RF autoregression — uses individual tree predictions for uncertainty."""
    def __init__(self):
        self.model   = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        self.scaler  = StandardScaler()
        self.trained = False

    def fit(self, df: pd.DataFrame):
        df_f = build_features(df)
        cols = [c for c in FEATURE_COLS if c in df_f.columns]
        df_f = df_f.dropna(subset=cols + ["pm10"])
        if len(df_f) < 20:
            return self
        X = self.scaler.fit_transform(df_f[cols])
        y = np.log1p(df_f["pm10"].values)
        self.model.fit(X, y)
        self.cols    = cols
        self.trained = True
        return self

    def predict_with_uncertainty(self, state: dict):
        """Returns (mean, lower_5, upper_95) using tree ensemble spread."""
        if not self.trained:
            v = state.get("pm10_lag1", 80)
            return v, v * 0.7, v * 1.3
        row   = pd.DataFrame([{c: state.get(c, 0) for c in self.cols}])
        X     = self.scaler.transform(row)
        preds = np.array([t.predict(X)[0] for t in self.model.estimators_])
        preds = np.expm1(preds)
        return (float(np.median(preds)),
                float(np.percentile(preds, 5)),
                float(np.percentile(preds, 95)))


# ══════════════════════════════════════════════════════
# 4. PROPHET-STYLE DECOMPOSITION (no Prophet dep)
# ══════════════════════════════════════════════════════
class ProphetLite:
    """
    Lightweight trend + seasonality decomposition without the Prophet package.
    Uses Fourier terms for diurnal/weekly cycles + linear trend.
    """
    def fit(self, df: pd.DataFrame):
        df = df.copy().sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["t"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() / 3600
        df = df.dropna(subset=["pm10"])
        if len(df) < 48:
            self.trained = False
            return self

        # Build design matrix: linear trend + hourly Fourier (24h period) + weekly (168h)
        t = df["t"].values
        X = [np.ones(len(t)), t]  # intercept + trend
        for k in [1, 2, 3]:      # 3 harmonics for diurnal cycle
            X.append(np.sin(2 * np.pi * k * t / 24))
            X.append(np.cos(2 * np.pi * k * t / 24))
        for k in [1, 2]:          # 2 harmonics for weekly cycle
            X.append(np.sin(2 * np.pi * k * t / 168))
            X.append(np.cos(2 * np.pi * k * t / 168))
        X = np.column_stack(X)
        y = df["pm10"].values

        # Least-squares fit
        self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.t0    = df["timestamp"].min()
        self.trained = True

        # Decompose components
        y_hat     = X @ self.coef_
        self.residuals = y - y_hat
        self.trend_component = self.coef_[0] + self.coef_[1] * t
        return self

    def predict(self, future_timestamps):
        if not getattr(self, "trained", False):
            return np.full(len(future_timestamps), 80.0)
        t = (pd.to_datetime(future_timestamps) - self.t0).total_seconds() / 3600
        X = [np.ones(len(t)), t]
        for k in [1, 2, 3]:
            X.append(np.sin(2 * np.pi * k * t / 24))
            X.append(np.cos(2 * np.pi * k * t / 24))
        for k in [1, 2]:
            X.append(np.sin(2 * np.pi * k * t / 168))
            X.append(np.cos(2 * np.pi * k * t / 168))
        X = np.column_stack(X)
        return np.clip(X @ self.coef_, 0, 1000)

    def get_components(self, df: pd.DataFrame):
        """Returns dict with trend, diurnal, weekly, residual arrays."""
        if not getattr(self, "trained", False):
            return {}
        df = df.copy().sort_values("timestamp")
        t  = (pd.to_datetime(df["timestamp"]) - self.t0).dt.total_seconds() / 3600
        t  = t.values
        trend   = self.coef_[0] + self.coef_[1] * t
        diurnal = np.zeros(len(t))
        weekly  = np.zeros(len(t))
        idx = 2
        for k in [1, 2, 3]:
            diurnal += (self.coef_[idx]   * np.sin(2*np.pi*k*t/24) +
                        self.coef_[idx+1] * np.cos(2*np.pi*k*t/24))
            idx += 2
        for k in [1, 2]:
            weekly  += (self.coef_[idx]   * np.sin(2*np.pi*k*t/168) +
                        self.coef_[idx+1] * np.cos(2*np.pi*k*t/168))
            idx += 2
        return {"trend": trend, "diurnal": diurnal,
                "weekly": weekly, "residual": df["pm10"].values - (trend+diurnal+weekly)}


# ══════════════════════════════════════════════════════
# 5. LSTM-LITE (pure numpy — no tensorflow dep)
# ══════════════════════════════════════════════════════
class LSTMLite:
    """
    Single-layer LSTM cell implemented in pure NumPy.
    Small (hidden_size=32) — trains fast, handles sequence memory.
    """
    def __init__(self, input_size=8, hidden_size=32, seq_len=24):
        self.H  = hidden_size
        self.I  = input_size
        self.SL = seq_len
        rng     = np.random.RandomState(42)
        scale   = 0.1
        # Combined weight matrix [input | hidden] → [4 * hidden]  (IFOG ordering)
        self.Wh = rng.randn(4*hidden_size, hidden_size) * scale
        self.Wx = rng.randn(4*hidden_size, input_size)  * scale
        self.b  = np.zeros(4*hidden_size)
        self.Wy = rng.randn(1, hidden_size) * scale
        self.by = np.zeros(1)
        self.trained = False

    def _sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    def _tanh(self, x):    return np.tanh(np.clip(x, -15, 15))

    def _forward(self, X_seq):
        """X_seq: (seq_len, input_size) → returns h_T (hidden_size,)"""
        h = np.zeros(self.H)
        c = np.zeros(self.H)
        for t in range(len(X_seq)):
            x   = X_seq[t]
            z   = self.Wx @ x + self.Wh @ h + self.b
            i   = self._sigmoid(z[:self.H])
            f   = self._sigmoid(z[self.H:2*self.H])
            o   = self._sigmoid(z[2*self.H:3*self.H])
            g   = self._tanh(z[3*self.H:])
            c   = f * c + i * g
            h   = o * self._tanh(c)
        return h

    def fit(self, df: pd.DataFrame, epochs=30, lr=0.001):
        df = df.copy().sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df_f = build_features(df)
        # Use a small subset of features for LSTM
        lstm_feats = ["pm10_lag1","pm10_lag2","pm10_lag3","pm10_lag6",
                      "temp","humidity","wind_speed","trap_index"]
        lstm_feats = [c for c in lstm_feats if c in df_f.columns]
        self.I = len(lstm_feats)
        df_f   = df_f.dropna(subset=lstm_feats + ["pm10"])
        if len(df_f) < self.SL + 10:
            return self

        scaler = StandardScaler()
        X_all  = scaler.fit_transform(df_f[lstm_feats].values)
        y_all  = np.log1p(df_f["pm10"].values)
        self.scaler     = scaler
        self.lstm_feats = lstm_feats

        # Rebuild Wx with correct size
        rng    = np.random.RandomState(42)
        self.Wx = rng.randn(4*self.H, self.I) * 0.1

        # Simple SGD with gradient clipping (BPTT not implemented — use output layer only)
        # We treat LSTM as a fixed feature extractor and train only output layer
        features = []
        targets  = []
        for i in range(self.SL, len(X_all)):
            h = self._forward(X_all[i-self.SL:i])
            features.append(h)
            targets.append(y_all[i])

        if len(features) < 5:
            return self

        F = np.array(features)
        T = np.array(targets)
        # Ridge regression on hidden states
        A = F.T @ F + 1e-4 * np.eye(self.H)
        b = F.T @ T
        w = np.linalg.solve(A, b)
        self.Wy = w.reshape(1, -1)
        self.by = np.array([np.mean(T) - (self.Wy @ np.mean(F, axis=0))])

        self.trained  = True
        self.last_seq = X_all[-self.SL:]
        self.last_lag = float(np.expm1(y_all[-1]))
        return self

    def predict_steps(self, steps=24, weather_override=None):
        """Autoregressive forecast for `steps` hours ahead."""
        if not self.trained:
            return np.full(steps, 80.0)
        seq   = self.last_seq.copy()
        preds = []
        lag   = self.last_lag
        for _ in range(steps):
            h    = self._forward(seq)
            pred = float(np.expm1((self.Wy @ h + self.by)[0]))
            pred = max(pred, 1.0)
            preds.append(pred)
            # Slide window: build next input row with updated lag
            next_row = seq[-1].copy()
            # Update lag feature positions
            feat_names = self.lstm_feats
            if "pm10_lag1" in feat_names:
                idx = feat_names.index("pm10_lag1")
                next_row[idx] = self.scaler.transform(
                    [[lag if i == idx else seq[-1][i]
                      for i in range(len(feat_names))]])[0][idx]
            seq  = np.vstack([seq[1:], next_row])
            lag  = pred
        return np.array(preds)


# ══════════════════════════════════════════════════════
# 6. ENSEMBLE FUSION
# ══════════════════════════════════════════════════════
def fuse_forecasts(xgb_preds, rf_preds, rf_lower, rf_upper,
                   prophet_preds, lstm_preds, weights=(0.35, 0.30, 0.20, 0.15)):
    """Weighted ensemble with uncertainty propagation."""
    wx, wr, wp, wl = weights
    n = max(len(xgb_preds), len(rf_preds), len(prophet_preds), len(lstm_preds))

    def pad(arr, n):
        arr = np.array(arr, dtype=float)
        if len(arr) < n:
            arr = np.pad(arr, (0, n-len(arr)), constant_values=arr[-1] if len(arr)>0 else 80)
        return arr[:n]

    xgb_p    = pad(xgb_preds, n)
    rf_p     = pad(rf_preds, n)
    rf_lo    = pad(rf_lower, n)
    rf_hi    = pad(rf_upper, n)
    prop_p   = pad(prophet_preds, n)
    lstm_p   = pad(lstm_preds, n)

    mean = wx*xgb_p + wr*rf_p + wp*prop_p + wl*lstm_p
    # Uncertainty: spread between models + RF tree variance
    model_spread = np.std(np.vstack([xgb_p, rf_p, prop_p, lstm_p]), axis=0)
    rf_band      = (rf_hi - rf_lo) / 2
    uncertainty  = np.maximum(model_spread, rf_band)

    lower = np.clip(mean - 1.645 * uncertainty, 0, None)
    upper = mean + 1.645 * uncertainty

    return mean, lower, upper


# ══════════════════════════════════════════════════════
# 7. FULL 72-HOUR FORECAST PIPELINE
# ══════════════════════════════════════════════════════
def run_forecast_pipeline(df_hist: pd.DataFrame, weather: dict,
                           traffic: dict, firms_df, horizon_h=72):
    """
    Full pipeline. Returns dict with:
      - forecast_df: DataFrame with timestamp, pm10_mean, lower, upper
      - models_df:   per-model predictions
      - feature_importance: dict
      - decomposition: trend/diurnal/weekly components
    """
    result = {
        "forecast_df": pd.DataFrame(),
        "models_df": pd.DataFrame(),
        "feature_importance": {},
        "decomposition": {},
        "model_quality": {},
        "trained": False,
    }

    if df_hist.empty or len(df_hist) < 24:
        return result

    # Enrich hist with weather/traffic if missing
    for col, val in [("temp", weather.get("temp",25)),
                     ("humidity", weather.get("humidity",55)),
                     ("wind_speed", weather.get("wind_speed",10)),
                     ("pressure", weather.get("pressure",1013)),
                     ("congestion", traffic.get("congestion_index",20)),
                     ("fire_hotspots", len(firms_df) if not firms_df.empty else 0)]:
        if col not in df_hist.columns:
            df_hist[col] = val

    # ── Train models ──
    xgb = LagBoostModel().fit(df_hist)
    rf  = RFAutoRegModel().fit(df_hist)
    proph = ProphetLite().fit(df_hist)
    lstm  = LSTMLite().fit(df_hist)

    # ── Build state for autoregressive loop ──
    df_f = build_features(df_hist)
    df_f = df_f.dropna(subset=["pm10"])
    if df_f.empty:
        return result

    last_row   = df_f.iloc[-1]
    now        = pd.Timestamp.now().ceil("h")
    future_ts  = pd.date_range(start=now, periods=horizon_h, freq="h")

    # ── XGBoost autoregressive loop ──
    state = {c: float(last_row.get(c, 0)) for c in FEATURE_COLS if c in last_row}
    xgb_preds = []
    current_lag = float(last_row["pm10"])
    for ft in future_ts:
        state.update({
            "hour": ft.hour, "dayofweek": ft.dayofweek, "month": ft.month,
            "is_morning_rush": int(ft.hour in range(7,10)),
            "is_evening_rush": int(ft.hour in range(17,20)),
            "is_night": int(ft.hour in range(0,6)),
            "pm10_lag1": current_lag,
            "temp": weather.get("temp",25),
            "humidity": weather.get("humidity",55),
            "wind_speed": weather.get("wind_speed",10),
            "trap_index": _trap_index(weather),
            "congestion": traffic.get("congestion_index",20),
            "fire_hotspots": len(firms_df) if not firms_df.empty else 0,
        })
        pred = xgb.predict_one(state)
        pred = max(pred, 1.0)
        xgb_preds.append(pred)
        current_lag = pred

    # ── RF predictions with uncertainty ──
    rf_means, rf_lowers, rf_uppers = [], [], []
    current_lag = float(last_row["pm10"])
    for ft in future_ts:
        state.update({"hour": ft.hour, "dayofweek": ft.dayofweek,
                      "month": ft.month, "pm10_lag1": current_lag})
        m, lo, hi = rf.predict_with_uncertainty(state)
        m = max(m, 1.0)
        rf_means.append(m); rf_lowers.append(lo); rf_uppers.append(hi)
        current_lag = m

    # ── Prophet predictions ──
    prophet_preds = proph.predict(future_ts)

    # ── LSTM predictions ──
    lstm_preds = lstm.predict_steps(steps=horizon_h)

    # ── Fuse ──
    fused_mean, fused_lower, fused_upper = fuse_forecasts(
        xgb_preds, rf_means, rf_lowers, rf_uppers,
        prophet_preds, lstm_preds)

    forecast_df = pd.DataFrame({
        "timestamp": future_ts,
        "pm10_mean": np.round(fused_mean, 1),
        "pm10_lower": np.round(fused_lower, 1),
        "pm10_upper": np.round(fused_upper, 1),
        "xgb": np.round(xgb_preds, 1),
        "rf":  np.round(rf_means, 1),
        "prophet": np.round(prophet_preds[:horizon_h], 1),
        "lstm": np.round(lstm_preds[:horizon_h], 1),
    })

    # Decomposition
    decomp = proph.get_components(df_hist)

    # Feature importance from XGBoost
    fi = xgb.feature_importance_

    result.update({
        "forecast_df": forecast_df,
        "feature_importance": fi,
        "decomposition": decomp,
        "trained": True,
        "model_quality": {"xgb_trained": xgb.trained,
                           "rf_trained": rf.trained,
                           "prophet_trained": proph.trained,
                           "lstm_trained": lstm.trained},
    })
    return result


def _trap_index(weather):
    p  = weather.get("pressure", 1013)
    ws = weather.get("wind_speed", 10)
    rh = weather.get("humidity", 55)
    return ((p - 1005) / 25 * 0.4 + 1/max(ws/3.6, 0.5) * 5 * 0.4 + rh/100 * 0.2)


# ══════════════════════════════════════════════════════
# 8. SOURCE ATTRIBUTION
# ══════════════════════════════════════════════════════
def attribute_sources(pm10_val: float, weather: dict, traffic: dict,
                       firms_df, terrain: str, pop_millions: float,
                       feature_importance: dict = None):
    """
    Decompose PM10 into contribution fractions from each source.
    Returns dict: {source: (absolute µg/m³, fraction 0-1, explanation)}
    """
    ws   = weather.get("wind_speed_ms", weather.get("wind_speed",10)/3.6)
    rh   = weather.get("humidity", 55)
    p    = weather.get("pressure", 1013)
    t    = weather.get("temp", 25)
    cong = traffic.get("congestion_index", 20)
    fires = len(firms_df) if not firms_df.empty else 0

    # Raw weights before normalization
    w = {}

    # Traffic: function of congestion + peak hour
    hour = pd.Timestamp.now().hour
    rush = 1.4 if hour in list(range(7,10)) + list(range(17,20)) else 1.0
    w["traffic"] = (cong / 100) * 30 * rush

    # Meteorological trapping
    trap = (1.0 / max(ws, 0.3)) * 2 + max(0, p - 1013) * 0.5 + max(0, (rh-60)/40) * 5
    w["meteorology"] = trap * 3

    # Fire / biomass burning
    w["fire_biomass"] = fires * 4 + (1 if fires > 0 else 0) * 10

    # Dust / terrain
    dust_map = {"arid": 20, "semi-arid": 12, "plains": 6,
                "plateau": 5, "hilly": 3, "coastal": 4, "valley": 8}
    w["dust_terrain"] = dust_map.get(terrain, 6)

    # Industrial / background
    w["industrial_bg"] = max(0, pop_millions * 3)

    # If we have feature importance, modulate weights
    if feature_importance:
        fi = feature_importance
        traffic_fi = sum(v for k,v in fi.items() if "congestion" in k or "rush" in k)
        fire_fi    = sum(v for k,v in fi.items() if "fire" in k)
        met_fi     = sum(v for k,v in fi.items() if any(x in k for x in
                         ["wind","humid","pressure","trap"]))
        total_fi   = max(traffic_fi + fire_fi + met_fi, 0.01)
        w["traffic"]     *= (1 + traffic_fi / total_fi)
        w["fire_biomass"]*= (1 + fire_fi    / total_fi)
        w["meteorology"] *= (1 + met_fi     / total_fi)

    total_w = sum(w.values()) or 1.0
    fracs   = {k: v/total_w for k,v in w.items()}

    labels = {
        "traffic":       "Vehicle emissions & brake/tyre wear",
        "meteorology":   "Atmospheric trapping (low wind, high pressure)",
        "fire_biomass":  "Fire & biomass burning (NASA FIRMS)",
        "dust_terrain":  "Dust resuspension & soil erosion",
        "industrial_bg": "Industrial activity & background urban",
    }
    colors = {
        "traffic":       "#ff6b35",
        "meteorology":   "#3d5a80",
        "fire_biomass":  "#ff4500",
        "dust_terrain":  "#c8a96e",
        "industrial_bg": "#8f3f97",
    }

    return {
        k: {
            "µg/m³":   round(fracs[k] * pm10_val, 1),
            "fraction": round(fracs[k], 3),
            "pct":      round(fracs[k] * 100, 1),
            "label":    labels[k],
            "color":    colors[k],
        }
        for k in w
    }


# ══════════════════════════════════════════════════════
# 9. EVENT DETECTION
# ══════════════════════════════════════════════════════
EVENT_TYPES = {
    "pm_spike":        {"color": "#ff0000", "icon": "⚡", "severity": "high"},
    "stagnant_atm":    {"color": "#ff7e00", "icon": "🌫️", "severity": "medium"},
    "fire_influence":  {"color": "#ff4500", "icon": "🔥", "severity": "high"},
    "traffic_buildup": {"color": "#ffcc00", "icon": "🚗", "severity": "medium"},
    "pressure_inversion":{"color":"#8f3f97","icon":"⬇️", "severity": "medium"},
    "unusual_wind":    {"color": "#3d5a80", "icon": "💨", "severity": "low"},
}

def detect_events(df_hist: pd.DataFrame, weather: dict, traffic: dict,
                   firms_df, forecast_df: pd.DataFrame = None):
    """
    Scan current + recent data for anomalous events.
    Returns list of event dicts.
    """
    events = []
    now    = pd.Timestamp.now()

    # ── Rolling baseline (7-day) ──
    if not df_hist.empty and "pm10" in df_hist.columns and len(df_hist) >= 12:
        recent = df_hist.sort_values("timestamp").tail(168)  # 7 days
        baseline_mean = recent["pm10"].mean()
        baseline_std  = recent["pm10"].std()
        current_pm10  = recent["pm10"].iloc[-1]

        # z-score
        z = (current_pm10 - baseline_mean) / max(baseline_std, 1)

        # PM spike
        if z > 2.5:
            events.append({
                "type": "pm_spike",
                "title": "PM10 spike detected",
                "detail": f"Current PM10 is {z:.1f}σ above 7-day average "
                           f"({current_pm10:.0f} vs baseline {baseline_mean:.0f} µg/m³).",
                "time": now,
                "severity": "high",
                "z_score": z,
            })
        elif z > 1.5:
            events.append({
                "type": "pm_spike",
                "title": "Elevated PM10",
                "detail": f"PM10 is {z:.1f}σ above baseline. Monitor closely.",
                "time": now,
                "severity": "medium",
                "z_score": z,
            })

        # Check if 3-hour trend is rising fast
        if len(recent) >= 3:
            last3 = recent["pm10"].tail(3).values
            if len(last3) >= 2 and last3[-1] > last3[0] * 1.30:
                events.append({
                    "type": "pm_spike",
                    "title": "Rapid PM10 increase",
                    "detail": f"PM10 rose {((last3[-1]/last3[0]-1)*100):.0f}% in 3 hours "
                               f"({last3[0]:.0f} → {last3[-1]:.0f} µg/m³).",
                    "time": now,
                    "severity": "high",
                })

    # ── Stagnant atmosphere ──
    ws  = weather.get("wind_speed_ms", weather.get("wind_speed",10)/3.6)
    p   = weather.get("pressure", 1013)
    rh  = weather.get("humidity", 55)
    t_  = weather.get("temp", 25)

    if ws < 1.0:
        events.append({
            "type": "stagnant_atm",
            "title": "Near-calm atmosphere",
            "detail": f"Wind speed {ws:.1f} m/s — pollutants cannot disperse. "
                       f"Expect PM10 to accumulate over next 3-6 hours.",
            "time": now,
            "severity": "medium",
        })
    elif ws < 2.0:
        events.append({
            "type": "stagnant_atm",
            "title": "Low wind conditions",
            "detail": f"Wind speed {ws:.1f} m/s — limited dispersion. "
                       f"PM10 may rise if emissions continue.",
            "time": now,
            "severity": "low",
        })

    # ── Pressure inversion ──
    if p > 1018:
        events.append({
            "type": "pressure_inversion",
            "title": "High-pressure system",
            "detail": f"Pressure {p:.0f} hPa — subsidence inversion likely. "
                       f"Pollutants trapped near surface. Expect +10-25% PM10.",
            "time": now,
            "severity": "medium",
        })

    # ── Fire influence ──
    if not firms_df.empty:
        n_fires = len(firms_df)
        events.append({
            "type": "fire_influence",
            "title": f"{n_fires} fire hotspots detected (NASA FIRMS)",
            "detail": f"Active fire/biomass burning within 100 km. "
                       f"PM2.5 and PM10 likely elevated due to smoke plume. "
                       f"{'Wind direction may carry smoke toward city.' if ws > 2 else 'Low wind — smoke may linger.'}",
            "time": now,
            "severity": "high" if n_fires > 5 else "medium",
            "fire_count": n_fires,
        })

    # ── Traffic buildup ──
    cong = traffic.get("congestion_index", 0)
    hour = now.hour
    if cong > 60 and hour in list(range(7,11)) + list(range(17,21)):
        events.append({
            "type": "traffic_buildup",
            "title": "Severe traffic congestion",
            "detail": f"Congestion index {cong:.0f}% during {'morning' if hour < 12 else 'evening'} peak. "
                       f"Vehicle emissions estimated at +{cong/2:.0f}% above background. "
                       f"PM10 likely elevated near roads.",
            "time": now,
            "severity": "medium",
        })
    elif cong > 40:
        events.append({
            "type": "traffic_buildup",
            "title": "Moderate traffic congestion",
            "detail": f"Congestion {cong:.0f}% — contributing ~{cong/3:.0f}% to current PM10.",
            "time": now,
            "severity": "low",
        })

    # ── Forecast-based pre-emptive alert ──
    if forecast_df is not None and not forecast_df.empty and "pm10_mean" in forecast_df.columns:
        # Check if forecast crosses Very Poor threshold in next 12h
        next12 = forecast_df.head(12)
        max_fc = next12["pm10_mean"].max()
        max_t  = next12.loc[next12["pm10_mean"].idxmax(), "timestamp"]
        if max_fc > 250:
            events.append({
                "type": "pm_spike",
                "title": "⚠️ Forecast alert: Very Poor AQ expected",
                "detail": f"Models predict PM10 will reach {max_fc:.0f} µg/m³ "
                           f"around {pd.to_datetime(max_t).strftime('%H:%M')}. "
                           f"{'Fire plume arrival likely.' if not firms_df.empty else ''}"
                           f"{'Stagnant conditions will trap pollution.' if ws < 1.5 else ''}",
                "time": pd.to_datetime(max_t),
                "severity": "high",
                "forecast_peak": max_fc,
            })
        elif max_fc > 150:
            events.append({
                "type": "pm_spike",
                "title": "Forecast: Poor AQ expected",
                "detail": f"PM10 forecast to reach {max_fc:.0f} µg/m³ around "
                           f"{pd.to_datetime(max_t).strftime('%H:%M')}.",
                "time": pd.to_datetime(max_t),
                "severity": "medium",
                "forecast_peak": max_fc,
            })

    # Sort by severity
    sev_order = {"high": 0, "medium": 1, "low": 2}
    events.sort(key=lambda e: sev_order.get(e.get("severity","low"), 2))
    return events


# ══════════════════════════════════════════════════════
# 10. CAUSE ATTRIBUTION NARRATIVE
# ══════════════════════════════════════════════════════
def generate_narrative(pm10_val: float, attribution: dict, events: list,
                        weather: dict, traffic: dict, firms_df,
                        terrain: str, forecast_df: pd.DataFrame = None) -> str:
    """
    Returns a plain-language paragraph explaining:
    1. Why pollution is at current level
    2. Dominant driver
    3. Meteorological contribution
    4. Fire influence
    5. Traffic contribution
    6. What's expected next
    """
    lines = []
    cat   = _pm_cat(pm10_val)

    # 1. Current status
    lines.append(f"**Current status:** PM10 is {pm10_val:.0f} µg/m³ — classified as **{cat}**.")

    # 2. Dominant driver
    dom = max(attribution, key=lambda k: attribution[k]["fraction"])
    dom_info = attribution[dom]
    lines.append(f"**Dominant driver:** {dom_info['label']} accounts for an estimated "
                  f"{dom_info['pct']:.0f}% ({dom_info['µg/m³']:.0f} µg/m³) of current pollution.")

    # 3. Meteorological contribution
    met = attribution.get("meteorology", {})
    ws  = weather.get("wind_speed_ms", weather.get("wind_speed",10)/3.6)
    p   = weather.get("pressure", 1013)
    rh  = weather.get("humidity", 55)
    met_desc = "dispersing pollutants effectively" if ws > 5 else \
               "providing moderate dispersion" if ws > 2 else \
               "allowing pollutants to accumulate (near-calm)"
    lines.append(f"**Meteorological contribution:** Wind at {ws:.1f} m/s is {met_desc}. "
                  f"Pressure {p:.0f} hPa and humidity {rh:.0f}% contribute "
                  f"{met.get('pct',0):.0f}% to PM10 through atmospheric trapping.")

    # 4. Fire influence
    fire = attribution.get("fire_biomass", {})
    if not firms_df.empty:
        n = len(firms_df)
        lines.append(f"**Fire influence:** NASA FIRMS detected {n} active fire hotspot(s) "
                      f"within the region, contributing an estimated {fire.get('pct',0):.0f}% "
                      f"({fire.get('µg/m³',0):.0f} µg/m³). "
                      f"{'Low wind means smoke is not dispersing quickly.' if ws < 2 else 'Wind may be transporting smoke plume toward populated areas.'}")
    else:
        lines.append(f"**Fire influence:** No active fire hotspots detected by NASA FIRMS. "
                      f"Biomass burning contribution is minimal (<{fire.get('pct',0):.0f}%).")

    # 5. Traffic contribution
    traf = attribution.get("traffic", {})
    cong = traffic.get("congestion_index", 0)
    lines.append(f"**Traffic contribution:** Road traffic congestion at {cong:.0f}% "
                  f"contributes {traf.get('pct',0):.0f}% ({traf.get('µg/m³',0):.0f} µg/m³). "
                  f"{'Peak hour emissions are amplifying this significantly.' if cong > 50 else 'Off-peak conditions are keeping vehicular emissions lower.'}")

    # 6. Forecast outlook
    if forecast_df is not None and not forecast_df.empty and "pm10_mean" in forecast_df.columns:
        next6  = forecast_df.head(6)["pm10_mean"].mean()
        next24 = forecast_df.head(24)["pm10_mean"].mean()
        trend  = "worsening" if next24 > pm10_val * 1.15 else \
                 "improving" if next24 < pm10_val * 0.85 else "stable"
        lines.append(f"**Outlook:** Air quality is expected to be **{trend}** over the next 24 hours. "
                      f"6-hour mean forecast: {next6:.0f} µg/m³, 24-hour mean: {next24:.0f} µg/m³.")

    # 7. Active alerts summary
    high_events = [e for e in events if e.get("severity") == "high"]
    if high_events:
        lines.append(f"**Active high-priority alerts:** " +
                      " | ".join(e["title"] for e in high_events[:3]))

    return "\n\n".join(lines)


def _pm_cat(v):
    if v <= 50:  return "Good"
    if v <= 100: return "Moderate"
    if v <= 250: return "Poor"
    if v <= 350: return "Very Poor"
    if v <= 430: return "Severe"
    return "Hazardous"
