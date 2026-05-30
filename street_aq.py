"""
street_aq.py  — Street-level AQI without OSM dependency
─────────────────────────────────────────────────────────
OSM Overpass is blocked on Streamlit Cloud (403).
Strategy: generate a dense grid of points across the city,
fetch TomTom flow for each, blend with AQ dispersion grid,
then render as a fine-grained heatmap + traffic-weighted circles.

This gives genuine street-resolution AQ without needing OSM.
"""
import numpy as np
import pandas as pd
import requests
import folium
import time
import math
from scipy.interpolate import griddata

TOMTOM_KEY = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"

# ── AQ helpers ────────────────────────────────────────
def aqi_hex(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "#888888"
    v = float(val)
    if v <= 50:  return "#00e400"
    if v <= 100: return "#ffff00"
    if v <= 250: return "#ff7e00"
    if v <= 350: return "#ff0000"
    if v <= 430: return "#8f3f97"
    return "#7e0023"

def aqi_label(val):
    if val is None: return "N/A"
    v = float(val)
    if v <= 50:  return "Good"
    if v <= 100: return "Moderate"
    if v <= 250: return "Poor"
    if v <= 350: return "Very Poor"
    if v <= 430: return "Severe"
    return "Hazardous"

# ── Fetch TomTom flow for one point ──────────────────
def _tomtom_flow(lat, lon):
    try:
        url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
               f"relative0/14/json?point={lat},{lon}&key={TOMTOM_KEY}")
        r = requests.get(url, timeout=6).json()
        if "flowSegmentData" in r:
            fsd = r["flowSegmentData"]
            cs, ff = fsd.get("currentSpeed", 50), fsd.get("freeFlowSpeed", 50)
            return {
                "flow_ratio":    round(cs / max(ff, 1), 3),
                "congestion":    round((1 - cs / max(ff, 1)) * 100, 1),
                "current_speed": cs,
                "free_flow":     ff,
            }
    except:
        pass
    return {"flow_ratio": 1.0, "congestion": 0,
            "current_speed": 50, "free_flow": 50}

# ── Build street-resolution point grid ───────────────
def build_street_grid(lat, lon, radius_m, grid_n=18):
    """
    Create a uniform lat/lon grid of points covering the city area.
    Denser than the main AQ grid — acts as street-resolution proxy.
    """
    r_deg_lat = radius_m / 111320
    r_deg_lon = radius_m / (111320 * math.cos(math.radians(lat)))
    pts = []
    for i, la in enumerate(np.linspace(lat - r_deg_lat, lat + r_deg_lat, grid_n)):
        for j, lo in enumerate(np.linspace(lon - r_deg_lon, lon + r_deg_lon, grid_n)):
            pts.append({"lat": round(la, 5), "lon": round(lo, 5)})
    return pts

# ── Interpolate AQ grid onto street points ────────────
def interpolate_aq_to_points(pts, grids, lats, lons):
    lat_flat = np.array([[la]*len(lons) for la in lats]).ravel()
    lon_flat = np.tile(lons, len(lats))
    pt_lats  = np.array([p["lat"] for p in pts])
    pt_lons  = np.array([p["lon"] for p in pts])

    for param, grid in grids.items():
        try:
            vals = griddata(
                (lat_flat, lon_flat), grid.ravel(),
                (pt_lats, pt_lons), method="linear",
            )
            for i, p in enumerate(pts):
                p[param] = float(vals[i]) if vals[i] is not None and not np.isnan(vals[i]) else None
        except:
            for p in pts:
                p[param] = None
    return pts

# ── Fetch TomTom for sampled points ───────────────────
def enrich_with_traffic(pts, max_pts=40, delay=0.05):
    """
    Fetch TomTom traffic for a subsample of grid points.
    Propagate to neighbours by proximity.
    """
    # Sample evenly — every Nth point
    step = max(1, len(pts) // max_pts)
    sampled_idx = list(range(0, len(pts), step))[:max_pts]

    traffic_cache = {}
    for idx in sampled_idx:
        p = pts[idx]
        flow = _tomtom_flow(p["lat"], p["lon"])
        traffic_cache[idx] = flow
        time.sleep(delay)

    # Assign to all points (nearest sampled point)
    sampled_lats = np.array([pts[i]["lat"] for i in sampled_idx])
    sampled_lons = np.array([pts[i]["lon"] for i in sampled_idx])

    for j, p in enumerate(pts):
        dists = np.abs(sampled_lats - p["lat"]) + np.abs(sampled_lons - p["lon"])
        nearest = sampled_idx[int(np.argmin(dists))]
        p["traffic"] = traffic_cache.get(nearest,
                        {"flow_ratio":1.0,"congestion":0,
                         "current_speed":50,"free_flow":50})
    return pts

# ── Apply traffic weighting to AQ ─────────────────────
def apply_traffic_weight(pts):
    for p in pts:
        cong = p.get("traffic", {}).get("congestion", 0)
        mult = 1.0 + (cong / 100) * 0.5
        p["aq_road"] = {}
        for param in ["pm10","pm25","no2","o3","so2","co","aqi"]:
            v = p.get(param)
            if v is not None:
                # NO2 is most traffic-sensitive
                factor = mult * 1.3 if param == "no2" else mult
                p["aq_road"][param] = round(v * factor, 1)
            else:
                p["aq_road"][param] = None
    return pts

# ── Render as fine heatmap + clickable circles ────────
def add_street_layer(fmap, pts, active_param="pm10"):
    # High-resolution heatmap
    heat_data = []
    valid_vals = [p["aq_road"].get(active_param)
                  for p in pts if p.get("aq_road",{}).get(active_param) is not None]
    max_val = max(valid_vals) if valid_vals else 200

    for p in pts:
        v = p.get("aq_road", {}).get(active_param)
        if v is not None:
            heat_data.append([p["lat"], p["lon"], v / max_val])

    from folium.plugins import HeatMap
    street_heat = folium.FeatureGroup(
        name=f"🛣️ Street-level {active_param.upper()} (fine grid)", show=True)

    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_opacity=0.75,
        radius=18,
        blur=12,
        max_zoom=16,
        gradient={
            "0.0":"#00e400","0.25":"#ffff00",
            "0.5":"#ff7e00","0.75":"#ff0000",
            "1.0":"#8f3f97",
        },
    ).add_to(street_heat)
    street_heat.add_to(fmap)

    # Clickable circles at grid points (every 3rd point to avoid clutter)
    dot_group = folium.FeatureGroup(
        name=f"📍 Street AQ points (click for values)", show=False)

    for i, p in enumerate(pts):
        if i % 3 != 0:
            continue
        v    = p.get("aq_road", {}).get(active_param)
        if v is None:
            continue
        clr  = aqi_hex(v)
        traf = p.get("traffic", {})
        popup_html = f"""
        <div style='font-family:sans-serif;font-size:13px;min-width:170px'>
          <b style='color:{clr}'>{aqi_label(v)}</b><br>
          <hr style='margin:4px 0'>
          🌫️ {active_param.upper()}: <b>{v:.1f}</b><br>
          PM10: {p.get('aq_road',{}).get('pm10','N/A')}<br>
          PM2.5: {p.get('aq_road',{}).get('pm25','N/A')}<br>
          NO₂: {p.get('aq_road',{}).get('no2','N/A')}<br>
          🚗 Speed: {traf.get('current_speed','?')}/{traf.get('free_flow','?')} km/h<br>
          🚦 Congestion: {traf.get('congestion',0):.0f}%<br>
          📍 {p['lat']:.4f}, {p['lon']:.4f}
        </div>"""
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=4,
            color=clr,
            fill=True, fill_color=clr, fill_opacity=0.8,
            weight=1,
            popup=folium.Popup(popup_html, max_width=210),
            tooltip=f"{active_param.upper()}: {v:.0f}",
        ).add_to(dot_group)

    dot_group.add_to(fmap)
    return fmap

# ── Master function ───────────────────────────────────
def build_street_aq_layer(fmap, lat, lon, grids, lats_arr, lons_arr,
                            active_param="pm10", radius_m=2000,
                            fetch_traffic=True):
    """
    Build fine-grid street-level AQ layer using TomTom + dispersion grid.
    No OSM dependency. Always returns (fmap, pts).
    """
    pts = build_street_grid(lat, lon, radius_m=radius_m, grid_n=18)
    pts = interpolate_aq_to_points(pts, grids, lats_arr, lons_arr)

    if fetch_traffic:
        pts = enrich_with_traffic(pts, max_pts=35, delay=0.05)
    else:
        for p in pts:
            p["traffic"] = {"flow_ratio":1.0,"congestion":0,
                             "current_speed":50,"free_flow":50}

    pts = apply_traffic_weight(pts)
    fmap = add_street_layer(fmap, pts, active_param=active_param)

    return fmap, pts
