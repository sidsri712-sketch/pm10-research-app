"""
street_aq.py
─────────────
Street-level AQI mapping for India AirSense Pro.

Pipeline:
1. Fetch road network from OpenStreetMap Overpass API
2. Fetch TomTom traffic flow per road segment
3. Interpolate AQ grid values to each road segment midpoint
4. Weight by traffic density
5. Render as colored road polylines on Folium map
"""
import numpy as np
import pandas as pd
import requests
import folium
import time
import math
from scipy.interpolate import griddata

TOMTOM_KEY = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"

# ── AQI color helper ──────────────────────────────────
def aqi_hex(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "#888888"
    val = float(val)
    if val <= 50:  return "#00e400"
    if val <= 100: return "#ffff00"
    if val <= 250: return "#ff7e00"
    if val <= 350: return "#ff0000"
    if val <= 430: return "#8f3f97"
    return "#7e0023"

def aqi_label(val):
    if val is None: return "N/A"
    val = float(val)
    if val <= 50:  return "Good"
    if val <= 100: return "Moderate"
    if val <= 250: return "Poor"
    if val <= 350: return "Very Poor"
    if val <= 430: return "Severe"
    return "Hazardous"

# ── 1. Fetch OSM roads ────────────────────────────────
def fetch_osm_roads(lat, lon, radius_m=3000):
    """
    Query Overpass API for major roads within radius.
    Returns list of road segments: {name, highway, coords[(lat,lon),...]}
    """
    r_deg = radius_m / 111320
    bbox  = f"{lat-r_deg},{lon-r_deg},{lat+r_deg},{lon+r_deg}"
    query = f"""
    [out:json][timeout:25];
    (
      way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]
         ({bbox});
    );
    out geom;
    """
    roads = []
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query}, timeout=30
        )
        data = resp.json()
        for el in data.get("elements", []):
            if el.get("type") == "way" and "geometry" in el:
                coords = [(pt["lat"], pt["lon"]) for pt in el["geometry"]]
                if len(coords) >= 2:
                    roads.append({
                        "name":    el.get("tags", {}).get("name", ""),
                        "highway": el.get("tags", {}).get("highway", "road"),
                        "coords":  coords,
                        "osm_id":  el.get("id"),
                    })
    except Exception as e:
        pass
    return roads

# ── 2. Fetch TomTom flow for a road segment ───────────
def fetch_tomtom_flow_segment(lat, lon, zoom=15):
    """Get traffic flow ratio for a point."""
    try:
        url = (f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
               f"relative0/{zoom}/json?point={lat},{lon}&key={TOMTOM_KEY}")
        r = requests.get(url, timeout=6).json()
        if "flowSegmentData" in r:
            fsd = r["flowSegmentData"]
            cs  = fsd.get("currentSpeed", 50)
            ff  = fsd.get("freeFlowSpeed", 50)
            ratio = cs / max(ff, 1)
            return {
                "flow_ratio":      round(ratio, 3),
                "congestion_pct":  round((1 - ratio) * 100, 1),
                "current_speed":   cs,
                "free_flow_speed": ff,
            }
    except:
        pass
    return {"flow_ratio": 1.0, "congestion_pct": 0, "current_speed": 50, "free_flow_speed": 50}

# ── 3. Interpolate AQ grid → road midpoints ───────────
def interpolate_aq_to_roads(roads, grids, lats, lons):
    """
    For each road segment, interpolate AQ values from the grid to the midpoint.
    Returns roads list enriched with aq values.
    """
    # Build flat arrays for griddata
    lat_g_flat = np.array([[la] * len(lons) for la in lats]).ravel()
    lon_g_flat = np.tile(lons, len(lats))

    param_arrays = {}
    for param, grid in grids.items():
        param_arrays[param] = grid.ravel()

    enriched = []
    for road in roads:
        coords = road["coords"]
        mid_idx = len(coords) // 2
        mlat, mlon = coords[mid_idx]

        aq = {}
        for param, vals in param_arrays.items():
            try:
                v = griddata(
                    (lat_g_flat, lon_g_flat), vals,
                    ([mlat], [mlon]), method="nearest"
                )
                aq[param] = float(v[0]) if v is not None else None
            except:
                aq[param] = None

        road_copy = road.copy()
        road_copy["aq"] = aq
        road_copy["mid_lat"] = mlat
        road_copy["mid_lon"] = mlon
        enriched.append(road_copy)

    return enriched

# ── 4. Fetch traffic for sample roads (rate-limited) ──
def enrich_roads_with_traffic(roads, max_roads=40, delay=0.05):
    """
    Fetch TomTom flow for up to max_roads road midpoints.
    Skips residential roads to save API quota.
    """
    priority_types = ["motorway","trunk","primary","secondary","tertiary"]
    priority = [r for r in roads if r.get("highway") in priority_types]
    others   = [r for r in roads if r.get("highway") not in priority_types]
    sample   = (priority + others)[:max_roads]

    for road in sample:
        mlat = road.get("mid_lat", road["coords"][len(road["coords"])//2][0])
        mlon = road.get("mid_lon", road["coords"][len(road["coords"])//2][1])
        flow = fetch_tomtom_flow_segment(mlat, mlon)
        road["traffic"] = flow
        time.sleep(delay)

    # Fill rest with defaults
    for road in roads:
        if "traffic" not in road:
            road["traffic"] = {"flow_ratio":1.0,"congestion_pct":0,
                                "current_speed":50,"free_flow_speed":50}
    return roads

# ── 5. Apply traffic weighting to AQ ──────────────────
def apply_traffic_weighting(roads):
    """
    Adjust AQ values per road segment based on traffic congestion.
    More congestion → higher PM10/PM2.5/NO2 near that road.
    """
    for road in roads:
        aq   = road.get("aq", {})
        cong = road.get("traffic", {}).get("congestion_pct", 0)
        hw   = road.get("highway", "road")

        # Road-type emission factors
        hw_factor = {
            "motorway":    1.4,
            "trunk":       1.3,
            "primary":     1.2,
            "secondary":   1.1,
            "tertiary":    1.05,
            "residential": 1.0,
        }.get(hw, 1.0)

        # Traffic multiplier: congestion increases near-road PM
        traffic_mult = 1.0 + (cong / 100) * 0.5 * hw_factor

        road["aq_road"] = {}
        for param, val in aq.items():
            if val is not None:
                road["aq_road"][param] = round(val * traffic_mult, 1)
            else:
                road["aq_road"][param] = val

    return roads

# ── 6. Render street AQ layer on Folium map ───────────
def add_street_aq_layer(fmap, roads, active_param="pm10"):
    """
    Add colored road polylines to an existing Folium map.
    Line color = AQ category. Line weight = road importance.
    """
    street_group = folium.FeatureGroup(
        name=f"🛣️ Street-level {active_param.upper()} AQI", show=True)

    hw_weight = {
        "motorway": 5, "trunk": 4, "primary": 4,
        "secondary": 3, "tertiary": 2, "residential": 1.5,
    }

    for road in roads:
        aq_val = road.get("aq_road", road.get("aq", {})).get(active_param)
        if aq_val is None:
            continue

        color  = aqi_hex(aq_val)
        weight = hw_weight.get(road.get("highway","road"), 2)
        coords = road["coords"]

        traffic = road.get("traffic", {})
        popup_html = f"""
        <div style='font-family:sans-serif;min-width:180px;font-size:13px'>
          <b>{road.get('name') or road.get('highway','Road').title()}</b>
          <hr style='margin:4px 0'>
          <span style='background:{color};padding:2px 8px;border-radius:8px;
            color:{"#000" if color in ["#00e400","#ffff00"] else "#fff"};
            font-weight:700'>{aqi_label(aq_val)}</span><br><br>
          🌫️ {active_param.upper()}: <b>{aq_val:.1f} µg/m³</b><br>
          🚗 Speed: {traffic.get('current_speed','N/A')} / {traffic.get('free_flow_speed','N/A')} km/h<br>
          🚦 Congestion: {traffic.get('congestion_pct',0):.0f}%<br>
          🛣️ Type: {road.get('highway','').replace('_',' ').title()}
        </div>"""

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{road.get('name') or road.get('highway','Road')}: "
                    f"{active_param.upper()} {aq_val:.0f}",
        ).add_to(street_group)

    street_group.add_to(fmap)
    return fmap

# ── Master function ───────────────────────────────────
def build_street_aq_layer(fmap, lat, lon, grids, lats_arr, lons_arr,
                            active_param="pm10", radius_m=3000,
                            fetch_traffic=True):
    """
    Full pipeline: OSM roads → AQ interpolation → traffic weighting → map layer.
    Returns updated fmap.
    """
    import streamlit as st

    with st.spinner("🛣️ Fetching road network from OpenStreetMap..."):
        roads = fetch_osm_roads(lat, lon, radius_m=radius_m)

    if not roads:
        st.warning("⚠️ No roads fetched from OSM. Street AQ layer skipped.")
        return fmap

    with st.spinner(f"📍 Interpolating AQ to {len(roads)} road segments..."):
        roads = interpolate_aq_to_roads(roads, grids, lats_arr, lons_arr)

    if fetch_traffic:
        with st.spinner("🚗 Fetching TomTom traffic per road..."):
            roads = enrich_roads_with_traffic(roads, max_roads=50)

    roads = apply_traffic_weighting(roads)
    fmap  = add_street_aq_layer(fmap, roads, active_param=active_param)

    return fmap, roads
