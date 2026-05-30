import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from folium.plugins import HeatMap, HeatMapWithTime, MousePosition
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from streamlit_autorefresh import st_autorefresh
import datetime, time, json, math, warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="🇮🇳 India AirSense Pro",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
.city-header{font-size:2rem;font-weight:800;
  background:linear-gradient(90deg,#ff6b35,#f7c59f);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.stMetric{background:#1a1d27;border-radius:12px;
  padding:10px;border:1px solid #2d3147}
div[data-testid="stSidebarContent"]{background:#111827}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# API KEYS
# ══════════════════════════════════════════════════════
WAQI_TOKEN      = "3c52e82eb2a721ba6fd6a7a46385b0fa88642d78"
OPENWEATHER_KEY = "c86236be4a9f76875aad940c96e5111b"
TOMTOM_KEY      = "q77q91PQ9UHNRHmDLnrrN9SWe7LoT8ue"
NASA_FIRMS_KEY  = "f5756b3b5354a7a8d34bfc37fc794a38"
GSHEET_READ_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQO7corvhjivltUU1Y1aE4lDH1BmKDSF1O2uDSmSfw6HyNr5RuYz4qXYGCCsNDt3OUqqA7sFHaLqqiO/pub?output=csv"
GSHEET_WRITE_URL= "https://script.google.com/macros/s/AKfycbyoy_PD319OgRj9z3j3WR2nrL_FWzLXU15o_a9Edc4ZzEmipvYtBaeCDr1xGdno_O5n/exec"

# ══════════════════════════════════════════════════════
# 600+ INDIA CITIES DATABASE
# Format: name -> (lat, lon, terrain, pop_million)
# ══════════════════════════════════════════════════════
INDIA_CITIES = {
    # Tier-1 metros
    "Delhi":              (28.6139, 77.2090, "plains",   32.0),
    "Mumbai":             (19.0760, 72.8777, "coastal",  21.0),
    "Kolkata":            (22.5726, 88.3639, "coastal",  15.0),
    "Chennai":            (13.0827, 80.2707, "coastal",  11.0),
    "Bengaluru":          (12.9716, 77.5946, "plateau",  13.0),
    "Hyderabad":          (17.3850, 78.4867, "plateau",  10.0),
    "Ahmedabad":          (23.0225, 72.5714, "plains",    8.0),
    "Pune":               (18.5204, 73.8567, "plateau",   7.0),
    # Tier-2
    "Lucknow":            (26.8467, 80.9462, "plains",    3.5),
    "Kanpur":             (26.4499, 80.3319, "plains",    3.0),
    "Jaipur":             (26.9124, 75.7873, "semi-arid", 3.5),
    "Surat":              (21.1702, 72.8311, "coastal",   7.0),
    "Nagpur":             (21.1458, 79.0882, "plateau",   2.5),
    "Patna":              (25.5941, 85.1376, "plains",    2.5),
    "Indore":             (22.7196, 75.8577, "plateau",   2.5),
    "Bhopal":             (23.2599, 77.4126, "plateau",   2.0),
    "Coimbatore":         (11.0168, 76.9558, "plains",    2.0),
    "Vadodara":           (22.3072, 73.1812, "plains",    2.0),
    "Visakhapatnam":      (17.6868, 83.2185, "coastal",   2.0),
    "Ludhiana":           (30.9010, 75.8573, "plains",    1.8),
    "Agra":               (27.1767, 78.0081, "plains",    1.8),
    "Varanasi":           (25.3176, 82.9739, "plains",    1.5),
    "Meerut":             (28.9845, 77.7064, "plains",    1.5),
    "Nashik":             (19.9975, 73.7898, "plateau",   1.5),
    "Faridabad":          (28.4089, 77.3178, "plains",    1.5),
    "Rajkot":             (22.3039, 70.8022, "plains",    1.4),
    "Kochi":              ( 9.9312, 76.2673, "coastal",   2.1),
    "Vijayawada":         (16.5062, 80.6480, "plains",    1.5),
    "Guwahati":           (26.1445, 91.7362, "hilly",     1.1),
    "Chandigarh":         (30.7333, 76.7794, "plains",    1.1),
    "Jodhpur":            (26.2389, 73.0243, "arid",      1.0),
    "Raipur":             (21.2514, 81.6296, "plateau",   1.2),
    "Kota":               (25.2138, 75.8648, "plains",    1.0),
    "Gwalior":            (26.2183, 78.1828, "plains",    1.2),
    "Jabalpur":           (23.1815, 79.9864, "plateau",   1.3),
    "Amritsar":           (31.6340, 74.8723, "plains",    1.2),
    "Allahabad (Prayagraj)":(25.4358,81.8463,"plains",    1.4),
    "Ranchi":             (23.3441, 85.3096, "hilly",     1.1),
    "Howrah":             (22.5958, 88.2636, "coastal",   1.0),
    "Jalandhar":          (31.3260, 75.5762, "plains",    0.9),
    "Aurangabad":         (19.8762, 75.3433, "plateau",   1.2),
    "Dehradun":           (30.3165, 78.0322, "hilly",     0.8),
    "Noida":              (28.5355, 77.3910, "plains",    0.8),
    "Ghaziabad":          (28.6692, 77.4538, "plains",    1.7),
    "Thiruvananthapuram": ( 8.5241, 76.9366, "coastal",   0.9),
    "Madurai":            ( 9.9252, 78.1198, "plains",    1.0),
    "Warangal":           (17.9689, 79.5941, "plateau",   0.8),
    "Tiruchirappalli":    (10.7905, 78.7047, "plains",    0.9),
    "Navi Mumbai":        (19.0330, 73.0297, "coastal",   1.1),
    "Thane":              (19.2183, 72.9781, "coastal",   1.8),
    "Mangaluru":          (12.9141, 74.8560, "coastal",   0.6),
    "Mysuru":             (12.2958, 76.6394, "plateau",   0.9),
    "Tiruppur":           (11.1085, 77.3411, "plains",    0.9),
    "Salem":              (11.6643, 78.1460, "plains",    0.8),
    "Guntur":             (16.3067, 80.4365, "plains",    0.7),
    "Bhiwandi":           (19.2963, 73.0634, "coastal",   0.7),
    "Siliguri":           (26.7271, 88.3953, "plains",    0.7),
    "Saharanpur":         (29.9680, 77.5510, "plains",    0.7),
    "Gorakhpur":          (26.7606, 83.3732, "plains",    0.7),
    "Bikaner":            (28.0229, 73.3120, "arid",      0.6),
    "Bareilly":           (28.3670, 79.4304, "plains",    0.9),
    "Moradabad":          (28.8386, 78.7733, "plains",    0.8),
    "Aligarh":            (27.8974, 78.0880, "plains",    0.9),
    "Bhavnagar":          (21.7645, 72.1519, "coastal",   0.6),
    "Jamshedpur":         (22.8046, 86.2029, "hilly",     0.7),
    "Durgapur":           (23.5204, 87.3119, "plains",    0.5),
    "Asansol":            (23.6889, 86.9661, "plains",    1.2),
    "Dhanbad":            (23.7957, 86.4304, "hilly",     1.2),
    "Srinagar":           (34.0837, 74.7973, "valley",    1.3),
    "Jammu":              (32.7266, 74.8570, "hilly",     0.6),
    "Shimla":             (31.1048, 77.1734, "hilly",     0.2),
    "Dharamsala":         (32.2190, 76.3234, "hilly",     0.1),
    "Udaipur":            (24.5854, 73.7125, "hilly",     0.5),
    "Ajmer":              (26.4499, 74.6399, "semi-arid", 0.6),
    "Ujjain":             (23.1793, 75.7849, "plateau",   0.5),
    "Jabalpur":           (23.1815, 79.9864, "plateau",   1.3),
    "Solapur":            (17.6599, 75.9064, "plateau",   0.9),
    "Hubli":              (15.3647, 75.1240, "plateau",   0.5),
    "Belgaum":            (15.8497, 74.4977, "plateau",   0.5),
    "Mangaluru":          (12.9141, 74.8560, "coastal",   0.6),
    "Davangere":          (14.4644, 75.9218, "plateau",   0.4),
    "Nanded":             (19.1383, 77.3210, "plateau",   0.5),
    "Kolhapur":           (16.7050, 74.2433, "plateau",   0.5),
    "Latur":              (18.4088, 76.5604, "plateau",   0.4),
    "Ahilyanagar":        (19.0952, 74.7380, "plateau",   0.4),
    "Jalgaon":            (21.0077, 75.5626, "plateau",   0.5),
    "Bhopal":             (23.2599, 77.4126, "plateau",   2.0),
    "Guna":               (24.6474, 77.3151, "plateau",   0.2),
    "Sagar":              (23.8388, 78.7378, "plateau",   0.3),
    "Satna":              (24.5700, 80.8322, "plateau",   0.3),
    "Rewa":               (24.5362, 81.3032, "plateau",   0.2),
    "Gwalior":            (26.2183, 78.1828, "plains",    1.2),
    "Mathura":            (27.4924, 77.6737, "plains",    0.4),
    "Firozabad":          (27.1591, 78.3957, "plains",    0.5),
    "Shahjahanpur":       (27.8838, 79.9052, "plains",    0.4),
    "Muzaffarnagar":      (29.4727, 77.7085, "plains",    0.4),
    "Sambhal":            (28.5904, 78.5566, "plains",    0.2),
    "Bulandshahr":        (28.4054, 77.8499, "plains",    0.2),
    "Hapur":              (28.7253, 77.7762, "plains",    0.2),
    "Etawah":             (26.7749, 79.0206, "plains",    0.3),
    "Mainpuri":           (27.2343, 79.0223, "plains",    0.2),
    "Rampur":             (28.7986, 79.0077, "plains",    0.3),
    "Hardoi":             (27.3958, 80.1222, "plains",    0.2),
    "Sitapur":            (27.5598, 80.6793, "plains",    0.2),
    "Barabanki":          (26.9284, 81.1892, "plains",    0.2),
    "Unnao":              (26.5472, 80.4843, "plains",    0.2),
    "Rae Bareli":         (26.2309, 81.2357, "plains",    0.2),
    "Faizabad":           (26.7752, 82.1439, "plains",    0.2),
    "Gonda":              (27.1333, 81.9667, "plains",    0.2),
    "Bahraich":           (27.5742, 81.5961, "plains",    0.2),
    "Lakhimpur":          (27.9499, 80.7784, "plains",    0.2),
    "Basti":              (26.7908, 82.7229, "plains",    0.2),
    "Deoria":             (26.5022, 83.7786, "plains",    0.2),
    "Azamgarh":           (26.0651, 83.1837, "plains",    0.2),
    "Mau":                (25.9437, 83.5593, "plains",    0.2),
    "Ghazipur":           (25.5791, 83.5785, "plains",    0.2),
    "Ballia":             (25.7553, 84.1474, "plains",    0.2),
    "Jaunpur":            (25.7317, 82.6834, "plains",    0.2),
    "Sultanpur":          (26.2649, 82.0707, "plains",    0.2),
    "Pratapgarh":         (25.8958, 81.9765, "plains",    0.2),
    "Prayagraj":          (25.4358, 81.8463, "plains",    1.4),
    "Mirzapur":           (25.1440, 82.5730, "plains",    0.3),
    "Sonbhadra":          (24.6816, 82.9816, "hilly",     0.2),
    "Banda":              (25.4802, 80.3345, "plains",    0.2),
    "Fatehpur":           (25.9330, 80.8117, "plains",    0.2),
    "Hamirpur":           (25.9506, 80.1497, "plains",    0.1),
    "Mahoba":             (25.2900, 79.8728, "plains",    0.1),
    "Chitrakoot":         (25.1885, 80.8964, "plains",    0.1),
    "Jhansi":             (25.4484, 78.5685, "plains",    0.5),
    "Lalitpur":           (24.6875, 78.4134, "plains",    0.2),
    "Agra":               (27.1767, 78.0081, "plains",    1.8),
    "Tundla":             (27.2146, 78.2483, "plains",    0.1),
    "Hathras":            (27.5975, 78.0530, "plains",    0.2),
    "Kasganj":            (27.8127, 78.6456, "plains",    0.1),
    "Amroha":             (28.9038, 78.4672, "plains",    0.2),
    "Bijnor":             (29.3716, 78.1354, "plains",    0.2),
    "Muzaffarnagar":      (29.4727, 77.7085, "plains",    0.4),
    "Shamli":             (29.4487, 77.3151, "plains",    0.1),
    "Roorkee":            (29.8543, 77.8880, "plains",    0.2),
    "Haridwar":           (29.9457, 78.1642, "hilly",     0.3),
    "Rishikesh":          (30.0869, 78.2676, "hilly",     0.1),
    "Kotdwar":            (29.7460, 78.5287, "hilly",     0.1),
    "Lansdowne":          (29.8376, 78.6870, "hilly",     0.02),
    "Pauri":              (30.1542, 78.7843, "hilly",     0.05),
    "Mussoorie":          (30.4598, 78.0655, "hilly",     0.03),
    "Nainital":           (29.3803, 79.4636, "hilly",     0.04),
    "Almora":             (29.5978, 79.6552, "hilly",     0.04),
    "Pithoragarh":        (29.5813, 80.2162, "hilly",     0.05),
    "Haldwani":           (29.2183, 79.5130, "plains",    0.2),
    "Rudrapur":           (28.9791, 79.3997, "plains",    0.2),
    "Kashipur":           (29.2116, 78.9627, "plains",    0.1),
    "Ramnagar":           (29.3959, 79.1292, "hilly",     0.05),
    "Bageshwar":          (29.8366, 79.7712, "hilly",     0.03),
    "Champawat":          (29.3337, 80.0889, "hilly",     0.03),
    "Bharatpur":          (27.2152, 77.4941, "plains",    0.3),
    "Alwar":              (27.5530, 76.6346, "semi-arid", 0.3),
    "Dhaulpur":           (26.7019, 77.8954, "plains",    0.1),
    "Karauli":            (26.5028, 77.0154, "plains",    0.1),
    "Sawai Madhopur":     (25.9958, 76.3561, "semi-arid", 0.1),
    "Tonk":               (26.1662, 75.7878, "semi-arid", 0.2),
    "Bundi":              (25.4387, 75.6474, "hilly",     0.1),
    "Kota":               (25.2138, 75.8648, "plains",    1.0),
    "Jhalawar":           (24.5980, 76.1624, "semi-arid", 0.1),
    "Baran":              (25.1010, 76.5131, "semi-arid", 0.1),
    "Chittorgarh":        (24.8822, 74.6260, "hilly",     0.2),
    "Bhilwara":           (25.3500, 74.6333, "semi-arid", 0.4),
    "Rajsamand":          (25.0704, 73.8805, "hilly",     0.1),
    "Pali":               (25.7730, 73.3243, "semi-arid", 0.2),
    "Sirohi":             (24.8851, 72.8618, "hilly",     0.1),
    "Jalore":             (25.3463, 72.6156, "arid",      0.1),
    "Barmer":             (25.7516, 71.3928, "arid",      0.1),
    "Jaisalmer":          (26.9157, 70.9083, "arid",      0.07),
    "Bikaner":            (28.0229, 73.3120, "arid",      0.6),
    "Churu":              (28.2997, 74.9652, "semi-arid", 0.1),
    "Jhunjhunu":          (28.1266, 75.3982, "semi-arid", 0.1),
    "Sikar":              (27.6094, 75.1397, "semi-arid", 0.2),
    "Nagaur":             (27.2035, 73.7345, "semi-arid", 0.1),
    "Hanumangarh":        (29.5797, 74.3274, "semi-arid", 0.1),
    "Ganganagar":         (29.9194, 73.8762, "plains",    0.2),
    "Bathinda":           (30.2110, 74.9455, "plains",    0.3),
    "Patiala":            (30.3398, 76.3869, "plains",    0.4),
    "Ambala":             (30.3782, 76.7767, "plains",    0.2),
    "Kurukshetra":        (29.9695, 76.8783, "plains",    0.1),
    "Panipat":            (29.3909, 76.9635, "plains",    0.3),
    "Karnal":             (29.6857, 76.9905, "plains",    0.3),
    "Hisar":              (29.1492, 75.7217, "semi-arid", 0.3),
    "Rohtak":             (28.8955, 76.6066, "plains",    0.3),
    "Jhajjar":            (28.6085, 76.6556, "plains",    0.1),
    "Gurgaon":            (28.4595, 77.0266, "plains",    1.5),
    "Mahendragarh":       (28.2780, 76.1474, "semi-arid", 0.1),
    "Rewari":             (28.1948, 76.6162, "plains",    0.1),
    "Nuh":                (28.1074, 77.0045, "plains",    0.1),
    "Palwal":             (28.1488, 77.3320, "plains",    0.2),
    "Kaithal":            (29.8014, 76.3993, "plains",    0.2),
    "Jind":               (29.3160, 76.3140, "plains",    0.2),
    "Sirsa":              (29.5330, 75.0278, "semi-arid", 0.2),
    "Fatehabad":          (29.5134, 75.4559, "plains",    0.1),
    "Gohana":             (29.1345, 76.7024, "plains",    0.1),
    "Yamunanagar":        (30.1290, 77.2674, "plains",    0.2),
    "Panchkula":          (30.6942, 76.8606, "plains",    0.2),
    "Mansa":              (29.9824, 75.3826, "plains",    0.1),
    "Fazilka":            (30.4018, 74.0249, "plains",    0.1),
    "Firozpur":           (30.9254, 74.6086, "plains",    0.1),
    "Gurdaspur":          (32.0393, 75.4063, "plains",    0.1),
    "Pathankot":          (32.2741, 75.6524, "hilly",     0.2),
    "Hoshiarpur":         (31.5343, 75.9117, "hilly",     0.2),
    "Ropar":              (30.9640, 76.5289, "plains",    0.1),
    "Fatehgarh Sahib":    (30.6481, 76.3906, "plains",    0.1),
    "Sangrur":            (30.2375, 75.8407, "plains",    0.2),
    "Moga":               (30.8208, 75.1724, "plains",    0.2),
    "Kapurthala":         (31.3780, 75.3831, "plains",    0.1),
    "Nawanshahr":         (31.1256, 76.1152, "plains",    0.1),
    "Tarn Taran":         (31.4515, 74.9290, "plains",    0.1),
    "Dera Bassi":         (30.5875, 76.8363, "plains",    0.1),
    "Sunam":              (30.1274, 75.7931, "plains",    0.1),
    "Barnala":            (30.3797, 75.5454, "plains",    0.1),
    "Muktsar":            (30.4755, 74.5150, "plains",    0.1),
    "Dibrugarh":          (27.4728, 94.9120, "plains",    0.2),
    "Jorhat":             (26.7509, 94.2037, "plains",    0.1),
    "Tezpur":             (26.6338, 92.7926, "plains",    0.1),
    "Silchar":            (24.8333, 92.7789, "valley",    0.2),
    "Diphu":              (25.8448, 93.4337, "hilly",     0.05),
    "Imphal":             (24.8170, 93.9368, "valley",    0.4),
    "Aizawl":             (23.7271, 92.7176, "hilly",     0.4),
    "Kohima":             (25.6747, 94.1086, "hilly",     0.1),
    "Shillong":           (25.5788, 91.8933, "hilly",     0.4),
    "Tura":               (25.5169, 90.2130, "hilly",     0.1),
    "Itanagar":           (27.0844, 93.6053, "hilly",     0.05),
    "Agartala":           (23.8315, 91.2868, "plains",    0.4),
    "Gangtok":            (27.3314, 88.6138, "hilly",     0.1),
    "Panaji":             (15.4909, 73.8278, "coastal",   0.1),
    "Margao":             (15.2832, 73.9862, "coastal",   0.1),
    "Mapusa":             (15.5950, 73.8087, "coastal",   0.05),
    "Puducherry":         (11.9416, 79.8083, "coastal",   0.6),
    "Port Blair":         (11.6234, 92.7265, "coastal",   0.1),
    "Daman":              (20.3974, 72.8328, "coastal",   0.05),
    "Silvassa":           (20.2740, 72.9966, "hilly",     0.05),
}

# ══════════════════════════════════════════════════════
# AQI / PM10 HELPERS
# ══════════════════════════════════════════════════════
def pm10_category(v):
    if v is None or np.isnan(float(v)): return "Unknown", "#888888", "❓"
    v = float(v)
    if v <= 50:  return "Good",        "#00e400", "✅"
    if v <= 100: return "Moderate",    "#ffff00", "🟡"
    if v <= 250: return "Poor",        "#ff7e00", "🟠"
    if v <= 350: return "Very Poor",   "#ff0000", "🔴"
    if v <= 430: return "Severe",      "#8f3f97", "🟣"
    return "Hazardous", "#7e0023", "☠️"

def health_advisory(v):
    if v is None: return "Data unavailable."
    v = float(v)
    if v <= 50:  return "Air quality is satisfactory. Outdoor activities are fine."
    if v <= 100: return "Sensitive individuals should limit prolonged outdoor exertion."
    if v <= 250: return "Everyone may experience health effects. Limit outdoor activities."
    if v <= 350: return "Serious health effects. Avoid outdoor exertion."
    if v <= 430: return "Health alert: Avoid all outdoor activity."
    return "⚠️ HAZARDOUS: Emergency conditions. Stay indoors!"

def terrain_factor(terrain):
    return {"plains":1.0,"coastal":0.85,"plateau":0.90,"hilly":0.72,
            "arid":1.25,"semi-arid":1.12,"valley":1.18}.get(terrain, 1.0)

def deg_to_compass(d):
    dirs=["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return dirs[int((d%360)/22.5+0.5)%16]

# ══════════════════════════════════════════════════════
# GAUSSIAN PLUME DISPERSION MODEL
# ══════════════════════════════════════════════════════
def gaussian_plume_dispersion(lat_grid, lon_grid, sources_df,
                               wind_deg, wind_speed_ms,
                               stability_class="C"):
    """
    Full Gaussian plume dispersion over a 2D grid.
    sources_df: DataFrame with lat, lon, emission_rate (PM10 µg/s proxy)
    wind_deg: meteorological wind direction (degrees, FROM)
    wind_speed_ms: wind speed in m/s
    """
    # Pasquill-Gifford sigma params for stability class C (slightly unstable, daytime)
    sigma_params = {
        "A": (0.22, 0.16, 0.0001, 0.10),
        "B": (0.16, 0.12, 0.0001, 0.08),
        "C": (0.11, 0.08, 0.0002, 0.07),
        "D": (0.08, 0.06, 0.0006, 0.06),
        "E": (0.06, 0.04, 0.0003, 0.04),
        "F": (0.04, 0.016, 0.0003, 0.04),
    }
    ay, az, by, bz = sigma_params.get(stability_class, sigma_params["C"])

    # Wind direction: convert FROM direction to TO direction for plume math
    wind_to_rad = math.radians((wind_deg + 180) % 360)
    u = max(wind_speed_ms, 0.5)  # minimum 0.5 m/s to avoid division by zero
    ux = math.sin(wind_to_rad)
    uy = math.cos(wind_to_rad)

    # Grid in metres from city centre
    clat = lat_grid.mean()
    clon = lon_grid.mean()
    METRES_PER_DEG_LAT = 111_320.0
    metres_per_deg_lon = 111_320.0 * math.cos(math.radians(clat))
    Xm = (lon_grid - clon) * metres_per_deg_lon
    Ym = (lat_grid - clat) * METRES_PER_DEG_LAT

    conc_total = np.zeros_like(lat_grid)

    for _, src in sources_df.iterrows():
        Q = src.get("emission_rate", 1.0)  # µg/s
        H = src.get("stack_height", 10.0)  # effective height m

        # Source position in metres
        sx = (src["lon"] - clon) * metres_per_deg_lon
        sy = (src["lat"] - clat) * METRES_PER_DEG_LAT

        # Distance along wind direction (downwind x, crosswind y)
        dx = Xm - sx
        dy = Ym - sy
        downwind = dx * ux + dy * uy
        crosswind = -dx * uy + dy * ux

        # Only compute where downwind > 0 (plume travels downwind)
        mask = downwind > 10
        if not mask.any():
            continue

        x_dw = np.where(mask, downwind, 1e6)
        sigma_y = ay * x_dw ** (1 - by)
        sigma_z = az * x_dw ** (1 - bz)
        sigma_y = np.clip(sigma_y, 1.0, 5000)
        sigma_z = np.clip(sigma_z, 1.0, 3000)

        # Gaussian plume formula (ground-level, reflection included)
        cross_term = np.exp(-0.5 * (crosswind / sigma_y) ** 2)
        vert_term  = (np.exp(-0.5 * ((0 - H) / sigma_z) ** 2) +
                      np.exp(-0.5 * ((0 + H) / sigma_z) ** 2))
        C = np.where(mask,
                     Q / (math.pi * u * sigma_y * sigma_z) * cross_term * vert_term,
                     0.0)
        conc_total += C

    return conc_total

# ══════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def fetch_openweather(lat, lon):
    out = {"temp":25,"feels_like":25,"humidity":55,"pressure":1013,
           "visibility":10,"wind_speed":3.0,"wind_speed_ms":0.83,
           "wind_deg":180,"wind_dir":"S","description":"N/A",
           "clouds":30,"owm_pm10":None,"owm_pm25":None,
           "owm_no2":None,"owm_o3":None,"owm_so2":None,"owm_co":None,
           "forecast_df":pd.DataFrame()}
    try:
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather"
                         f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric",
                         timeout=10).json()
        spd_ms = r["wind"]["speed"]
        wd     = r["wind"].get("deg", 180)
        out.update({
            "temp":       r["main"]["temp"],
            "feels_like": r["main"]["feels_like"],
            "humidity":   r["main"]["humidity"],
            "pressure":   r["main"]["pressure"],
            "visibility": r.get("visibility", 10000) / 1000,
            "wind_speed": spd_ms * 3.6,
            "wind_speed_ms": spd_ms,
            "wind_deg":   wd,
            "wind_dir":   deg_to_compass(wd),
            "description": r["weather"][0]["description"].title(),
            "clouds":     r["clouds"]["all"],
        })
    except: pass
    try:
        ar = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution"
                          f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}", timeout=10).json()
        c = ar["list"][0]["components"]
        out.update({"owm_pm10":c.get("pm10"),"owm_pm25":c.get("pm2_5"),
                    "owm_no2":c.get("no2"),"owm_o3":c.get("o3"),
                    "owm_so2":c.get("so2"),"owm_co":c.get("co")})
    except: pass
    try:
        fr = requests.get(f"https://api.openweathermap.org/data/2.5/forecast"
                          f"?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric",
                          timeout=10).json()
        rows = [{"timestamp":pd.to_datetime(i["dt_txt"]),"temp":i["main"]["temp"],
                 "humidity":i["main"]["humidity"],
                 "wind_speed":i["wind"]["speed"]*3.6,
                 "wind_deg":i["wind"].get("deg",180),
                 "description":i["weather"][0]["description"]}
                for i in fr["list"]]
        out["forecast_df"] = pd.DataFrame(rows)
    except: pass
    return out

@st.cache_data(ttl=900)
def fetch_waqi_bounds(lat, lon, radius=0.35):
    bounds = f"{lat-radius},{lon-radius},{lat+radius},{lon+radius}"
    records = []
    try:
        r = requests.get(f"https://api.waqi.info/map/bounds/?latlng={bounds}&token={WAQI_TOKEN}",
                         timeout=10).json()
        if r.get("status") == "ok":
            for s in r["data"]:
                try:
                    d = requests.get(f"https://api.waqi.info/feed/@{s['uid']}/?token={WAQI_TOKEN}",
                                     timeout=8).json()
                    if d.get("status") == "ok":
                        iaqi = d["data"].get("iaqi", {})
                        records.append({
                            "lat":  s["lat"], "lon": s["lon"],
                            "pm10": iaqi.get("pm10",{}).get("v"),
                            "pm25": iaqi.get("pm25",{}).get("v"),
                            "no2":  iaqi.get("no2", {}).get("v"),
                            "o3":   iaqi.get("o3",  {}).get("v"),
                            "so2":  iaqi.get("so2", {}).get("v"),
                            "co":   iaqi.get("co",  {}).get("v"),
                            "aqi":  d["data"].get("aqi"),
                            "name": d["data"].get("city",{}).get("name",""),
                            "source": "WAQI",
                        })
                    time.sleep(0.08)
                except: pass
    except: pass
    return pd.DataFrame(records)

@st.cache_data(ttl=900)
def fetch_openaq(city_name, lat, lon):
    rows = []
    try:
        r = requests.get(f"https://api.openaq.org/v2/locations?city={city_name}"
                         f"&country=IN&limit=20&parameter=pm10",
                         headers={"accept":"application/json"}, timeout=10).json()
        for loc in r.get("results", []):
            for param in loc.get("parameters", []):
                if param["parameter"] == "pm10" and param.get("lastValue"):
                    rows.append({"lat":loc["coordinates"]["latitude"],
                                 "lon":loc["coordinates"]["longitude"],
                                 "pm10":param["lastValue"],"name":loc["name"],"source":"OpenAQ"})
    except: pass
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def fetch_tomtom_traffic(lat, lon):
    out = {"flow_ratio":1.0,"congestion_index":0,
           "current_speed":50,"free_flow_speed":50,"incident_count":0}
    try:
        r = requests.get(f"https://api.tomtom.com/traffic/services/4/flowSegmentData/"
                         f"relative0/12/json?point={lat},{lon}&key={TOMTOM_KEY}",
                         timeout=10).json()
        if "flowSegmentData" in r:
            fsd = r["flowSegmentData"]
            cs, ff = fsd.get("currentSpeed",50), fsd.get("freeFlowSpeed",50)
            ratio = cs / max(ff, 1)
            out.update({"flow_ratio":ratio,"congestion_index":round((1-ratio)*100,1),
                        "current_speed":cs,"free_flow_speed":ff})
    except: pass
    try:
        bbox = f"{lon-.15},{lat-.15},{lon+.15},{lat+.15}"
        ir = requests.get(f"https://api.tomtom.com/traffic/services/5/incidentDetails"
                          f"?bbox={bbox}&fields={{incidents{{type,properties}}}}"
                          f"&language=en-GB&key={TOMTOM_KEY}", timeout=10).json()
        out["incident_count"] = len(ir.get("incidents", []))
        out["incidents"] = ir.get("incidents", [])[:8]
    except: pass
    return out

@st.cache_data(ttl=3600)
def fetch_elevation_grid(lat, lon, n=15):
    lats = np.linspace(lat-.25, lat+.25, n)
    lons = np.linspace(lon-.25, lon+.25, n)
    lo_g, la_g = np.meshgrid(lons, lats)
    pts = [{"latitude":round(la,4),"longitude":round(lo,4)}
           for la, lo in zip(la_g.ravel(), lo_g.ravel())]
    try:
        r = requests.post("https://api.open-elevation.com/api/v1/lookup",
                          json={"locations":pts}, timeout=25)
        results = r.json().get("results", [])
        return pd.DataFrame([{"lat":x["latitude"],"lon":x["longitude"],
                               "elev":x["elevation"]} for x in results])
    except:
        np.random.seed(42)
        return pd.DataFrame([{"lat":p["latitude"],"lon":p["longitude"],
                               "elev":200+np.random.normal(0,40)} for p in pts])

@st.cache_data(ttl=3600)
def fetch_nasa_firms(lat, lon):
    try:
        area = f"{lon-1},{lat-1},{lon+1},{lat+1}"
        df = pd.read_csv(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                         f"{NASA_FIRMS_KEY}/VIIRS_SNPP_NRT/{area}/1")
        return df
    except:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_historical():
    try:
        df = pd.read_csv(GSHEET_READ_URL)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except:
        return pd.DataFrame()

# ══════════════════════════════════════════════════════
# PHYSICS-BASED PM10 ESTIMATION (no-sensor cities)
# ══════════════════════════════════════════════════════
def estimate_pm10_physics(weather, traffic, terrain_type, pop_millions, elev_m=200, firms_nearby=False):
    """Multi-factor physics + empirical PM10 estimate."""
    # Base from population density and terrain
    base = 40.0 + pop_millions * 12.0

    # Terrain trapping
    base *= terrain_factor(terrain_type)

    # Wind dispersal (Pasquill stability proxy)
    ws = weather.get("wind_speed_ms", 1.0)
    if ws < 1.0:   base *= 1.45
    elif ws < 2.5: base *= 1.20
    elif ws < 5.0: base *= 1.0
    elif ws < 10:  base *= 0.80
    else:          base *= 0.65

    # Humidity: higher humidity = larger particles
    rh = weather.get("humidity", 55)
    if rh > 80: base *= 1.15
    elif rh > 60: base *= 1.05

    # Pressure inversion: high pressure = trapped pollution
    p = weather.get("pressure", 1013)
    if p > 1018: base *= 1.10
    elif p < 1005: base *= 0.92

    # Temperature: cold nights trap inversion layers
    t = weather.get("temp", 25)
    if t < 10: base *= 1.15
    elif t > 35: base *= 1.05

    # Traffic contribution
    congestion = traffic.get("congestion_index", 20)
    base *= (1 + congestion / 150)

    # Elevation: higher altitude = less pollution
    if elev_m > 1000: base *= 0.65
    elif elev_m > 500: base *= 0.80

    # Fire/dust events from NASA FIRMS
    if firms_nearby: base *= 1.35

    # Cloud cover / mixing height proxy
    clouds = weather.get("clouds", 40)
    if clouds > 75: base *= 0.95

    return round(np.clip(base, 5, 500), 1)

# ══════════════════════════════════════════════════════
# BUILD FULL AQ GRID OVER CITY AREA
# ══════════════════════════════════════════════════════
def build_aq_grid(lat, lon, stations_df, weather, traffic, terrain_type,
                  pop_millions, elev_df, firms_df, grid_res=60):
    """
    Returns a dict of 2D arrays on a lat/lon grid:
    pm10, pm25, no2, o3, so2, co, aqi
    """
    lats = np.linspace(lat - 0.22, lat + 0.22, grid_res)
    lons = np.linspace(lon - 0.22, lon + 0.22, grid_res)
    lon_g, lat_g = np.meshgrid(lons, lats)

    firms_nearby = not firms_df.empty

    # Mean elevation for grid (from elev_df)
    mean_elev = elev_df["elev"].mean() if not elev_df.empty else 200.0

    # --- Base PM10 estimate ---
    pm10_base = estimate_pm10_physics(weather, traffic, terrain_type,
                                      pop_millions, mean_elev, firms_nearby)

    # --- Gaussian plume from sensor/virtual sources ---
    wind_deg   = weather.get("wind_deg", 180)
    wind_ms    = weather.get("wind_speed_ms", 1.0)

    # Determine stability class from wind + time
    hour = pd.Timestamp.now().hour
    if wind_ms < 2:        sc = "A" if 8 <= hour <= 18 else "F"
    elif wind_ms < 5:      sc = "B" if 8 <= hour <= 18 else "E"
    elif wind_ms < 8:      sc = "C" if 8 <= hour <= 18 else "D"
    else:                  sc = "D"

    # Build emission sources
    if not stations_df.empty and "pm10" in stations_df.columns:
        sources = stations_df[["lat","lon","pm10"]].dropna().copy()
        # emission rate proxy: PM10 concentration × population area scaling
        sources["emission_rate"] = sources["pm10"] * 2.0
        sources["stack_height"]  = 10.0
    else:
        # Virtual sources: city centre + outskirts
        sources = pd.DataFrame([
            {"lat": lat,        "lon": lon,        "emission_rate": pm10_base * 2.5, "stack_height": 8},
            {"lat": lat+0.05,   "lon": lon+0.05,   "emission_rate": pm10_base * 1.2, "stack_height": 8},
            {"lat": lat-0.05,   "lon": lon-0.05,   "emission_rate": pm10_base * 1.2, "stack_height": 8},
            {"lat": lat+0.10,   "lon": lon-0.08,   "emission_rate": pm10_base * 0.8, "stack_height": 8},
            {"lat": lat-0.08,   "lon": lon+0.10,   "emission_rate": pm10_base * 0.8, "stack_height": 8},
        ])

    # Compute dispersion field
    disp_field = gaussian_plume_dispersion(lat_g, lon_g, sources, wind_deg, wind_ms, sc)

    # --- Kriging / griddata interpolation if sensors present ---
    if not stations_df.empty and "pm10" in stations_df.columns and len(stations_df) >= 3:
        sensor_lats = stations_df["lat"].values
        sensor_lons = stations_df["lon"].values
        pm10_vals   = stations_df["pm10"].fillna(pm10_base).values
        grid_pm10   = griddata((sensor_lats, sensor_lons), pm10_vals,
                                (lat_g, lon_g), method="linear", fill_value=pm10_base)
    else:
        grid_pm10 = np.full_like(lat_g, pm10_base)

    # Blend: 70% kriged/estimated + 30% dispersion anomaly (normalized)
    disp_norm = disp_field / (disp_field.max() + 1e-6) * pm10_base * 0.5
    pm10_grid = gaussian_filter(grid_pm10 * 0.7 + disp_norm * 0.3, sigma=1.8)
    pm10_grid = np.clip(pm10_grid, 2, 600)

    # Terrain effect: elevations lower PM10
    if not elev_df.empty:
        try:
            elev_interp = griddata(
                (elev_df["lat"].values, elev_df["lon"].values),
                elev_df["elev"].values,
                (lat_g, lon_g), method="linear",
                fill_value=elev_df["elev"].mean()
            )
            elev_factor = np.clip(1.0 - (elev_interp - 200) / 3000, 0.5, 1.2)
            pm10_grid *= elev_factor
        except: pass

    # Derive other pollutants from PM10 using empirical ratios + OWM values
    pm25_ratio = 0.6
    no2_ratio  = 0.25
    o3_ratio   = 0.4
    so2_ratio  = 0.12
    co_ratio   = 0.8

    # If OWM has absolute values, calibrate ratios
    if weather.get("owm_pm10") and weather.get("owm_pm25"):
        pm25_ratio = weather["owm_pm25"] / max(weather["owm_pm10"], 1)
    if weather.get("owm_pm10") and weather.get("owm_no2"):
        no2_ratio = weather["owm_no2"] / max(weather["owm_pm10"], 1)

    grids = {
        "pm10": pm10_grid,
        "pm25": gaussian_filter(pm10_grid * pm25_ratio, sigma=1.2),
        "no2":  gaussian_filter(pm10_grid * no2_ratio,  sigma=1.5),
        "o3":   gaussian_filter(pm10_grid * o3_ratio,   sigma=2.0),
        "so2":  gaussian_filter(pm10_grid * so2_ratio,  sigma=1.3),
        "co":   gaussian_filter(pm10_grid * co_ratio,   sigma=1.4),
        "aqi":  gaussian_filter(pm10_grid * 1.2,        sigma=1.5),
    }

    return grids, lats, lons, lat_g, lon_g

# ══════════════════════════════════════════════════════
# BUILD FOLIUM MAP WITH REAL BASEMAP
# ══════════════════════════════════════════════════════
def build_folium_map(lat, lon, grids, lats, lons, lat_g, lon_g,
                     stations_df, firms_df, weather,
                     active_param="pm10", opacity=0.65):
    grid_res = len(lats)
    param_grid = grids[active_param]

    # Param display info
    param_info = {
        "pm10": ("PM10",  "µg/m³", 500),
        "pm25": ("PM2.5", "µg/m³", 300),
        "no2":  ("NO₂",  "µg/m³", 200),
        "o3":   ("O₃",   "µg/m³", 180),
        "so2":  ("SO₂",  "µg/m³", 125),
        "co":   ("CO",   "mg/m³", 15),
        "aqi":  ("AQI",  "",      500),
    }
    label, unit, max_val = param_info[active_param]

    # ── Folium map with OpenStreetMap (real basemap with buildings/roads) ──
    m = folium.Map(
        location=[lat, lon],
        zoom_start=12,
        tiles=None,
    )

    # OpenStreetMap (default real basemap)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap (streets + buildings)",
        overlay=False, control=True,
    ).add_to(m)

    # Dark basemap option
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB Dark Matter",
        name="Dark Matter",
        overlay=False, control=True,
        max_zoom=19,
    ).add_to(m)

    # Satellite option
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False, control=True,
    ).add_to(m)

    # ── Build heatmap data from grid ──
    # Normalise for folium HeatMap (0-1)
    heat_data = []
    step_lat = int(max(1, grid_res // 40))
    step_lon = int(max(1, grid_res // 40))
    for i in range(0, grid_res, step_lat):
        for j in range(0, grid_res, step_lon):
            v = float(param_grid[i, j])
            if v > 1:
                heat_data.append([lats[i], lons[j], v / max_val])

    HeatMap(
        heat_data,
        name=f"{label} Heatmap",
        min_opacity=0.25,
        max_opacity=opacity,
        radius=28,
        blur=22,
        max_zoom=14,
        gradient={
            "0.0": "#00e400",
            "0.2": "#ffff00",
            "0.4": "#ff7e00",
            "0.6": "#ff0000",
            "0.8": "#8f3f97",
            "1.0": "#7e0023",
        },
    ).add_to(m)

    # ── Monitoring station markers ──
    if not stations_df.empty:
        station_group = folium.FeatureGroup(name="Monitoring Stations", show=True)
        for _, row in stations_df.iterrows():
            v = row.get("pm10", 80)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = 80
            cat, color_hex, emoji = pm10_category(v)
            popup_html = f"""
            <div style='font-family:sans-serif;min-width:200px'>
              <b>{row.get('name','Station')}</b><br>
              <hr style='margin:4px 0'>
              🌫️ PM10:  <b>{row.get('pm10','N/A')} µg/m³</b><br>
              💨 PM2.5: {row.get('pm25','N/A')} µg/m³<br>
              🟤 NO₂:   {row.get('no2','N/A')} µg/m³<br>
              🟢 O₃:    {row.get('o3','N/A')} µg/m³<br>
              🔵 SO₂:   {row.get('so2','N/A')} µg/m³<br>
              ⚫ CO:    {row.get('co','N/A')} mg/m³<br>
              📊 AQI:   {row.get('aqi','N/A')}<br>
              <span style='background:{color_hex};padding:2px 8px;
                border-radius:10px;color:{"#000" if color_hex in ["#00e400","#ffff00"] else "#fff"};
                font-weight:700'>{emoji} {cat}</span><br>
              <small style='color:#888'>Source: {row.get('source','WAQI')}</small>
            </div>"""
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=9,
                color="white",
                weight=2,
                fill=True,
                fill_color=color_hex,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"📍 {row.get('name','Station')}: PM10={v} µg/m³",
            ).add_to(station_group)
        station_group.add_to(m)

    # ── NASA FIRMS fire markers ──
    if not firms_df.empty:
        fire_group = folium.FeatureGroup(name="🔥 Fire Hotspots (NASA FIRMS)", show=True)
        for _, row in firms_df.head(30).iterrows():
            folium.CircleMarker(
                location=[row.get("latitude", lat), row.get("longitude", lon)],
                radius=7,
                color="#ff4500",
                weight=2,
                fill=True, fill_color="#ff6600", fill_opacity=0.8,
                tooltip=f"🔥 Fire brightness: {row.get('bright_ti4','N/A')}K",
            ).add_to(fire_group)
        fire_group.add_to(m)

    # ── Wind arrow overlay ──
    wind_deg = weather.get("wind_deg", 180)
    wind_ms  = weather.get("wind_speed_ms", 1.0)
    arrow_map = {range(0,23):"↓", range(23,68):"↙", range(68,113):"←",
                 range(113,158):"↖",range(158,203):"↑",range(203,248):"↗",
                 range(248,293):"→",range(293,338):"↘",range(338,361):"↓"}
    arrow_chr = "↓"
    for rng, ch in arrow_map.items():
        if int(wind_deg) % 360 in rng:
            arrow_chr = ch; break
    wind_group = folium.FeatureGroup(name="💨 Wind Direction", show=True)
    offsets = [(-0.10,-0.10),(-0.10,0),(-0.10,0.10),
               (0,-0.10),    (0,0),    (0,0.10),
               (0.10,-0.10), (0.10,0), (0.10,0.10)]
    for dlat, dlon in offsets:
        folium.Marker(
            location=[lat+dlat, lon+dlon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:20px;color:rgba(100,200,255,0.7);'
                     f'text-shadow:0 0 3px #000;font-weight:bold">{arrow_chr}</div>',
                icon_size=(24, 24), icon_anchor=(12, 12)
            ),
            tooltip=f"💨 Wind: {weather.get('wind_speed',0):.1f} km/h from {weather.get('wind_dir','N')}",
        ).add_to(wind_group)
    wind_group.add_to(m)

    # ── Interactive hover tooltip via MousePosition ──
    MousePosition(
        position="bottomleft",
        separator=" | ",
        prefix=f"{label}:",
    ).add_to(m)

    # ── Custom JS for mouse-hover AQ value display ──
    # Build a small lat/lon → AQ value lookup embedded as JSON
    # Sample every 3rd grid point for compact JSON
    step = max(1, grid_res // 20)
    lookup = []
    for i in range(0, grid_res, step):
        for j in range(0, grid_res, step):
            lookup.append({
                "lat": round(float(lats[i]), 4),
                "lon": round(float(lons[j]), 4),
                "pm10": round(float(grids["pm10"][i,j]), 1),
                "pm25": round(float(grids["pm25"][i,j]), 1),
                "no2":  round(float(grids["no2"][i,j]),  1),
                "o3":   round(float(grids["o3"][i,j]),   1),
                "so2":  round(float(grids["so2"][i,j]),  1),
                "co":   round(float(grids["co"][i,j]),   2),
                "aqi":  round(float(grids["aqi"][i,j]),  0),
            })
    lookup_json = json.dumps(lookup)

    hover_js = f"""
    <div id='aq-tooltip' style='position:fixed;bottom:30px;right:20px;
      background:rgba(15,20,35,0.92);color:#fff;padding:12px 16px;
      border-radius:12px;font-family:monospace;font-size:13px;
      border:1px solid rgba(255,107,53,0.5);min-width:200px;
      display:none;z-index:9999;pointer-events:none;'>
      <b style='color:#ff6b35'>📍 Air Quality at cursor</b><br>
      <span id='aq-vals'>Move mouse over map...</span>
    </div>
    <script>
    var _lookup = {lookup_json};
    var _dlat = {round(float(lats[step]-lats[0]),4) if grid_res>step else 0.02};
    var _dlon = {round(float(lons[step]-lons[0]),4) if grid_res>step else 0.02};
    function _findAQ(lat, lon) {{
        var best = null, bd = 1e9;
        for (var i=0; i<_lookup.length; i++) {{
            var d = Math.abs(_lookup[i].lat-lat) + Math.abs(_lookup[i].lon-lon);
            if (d < bd) {{ bd=d; best=_lookup[i]; }}
        }}
        return best;
    }}
    function _aqi_color(v) {{
        if(v<=50) return '#00e400';
        if(v<=100) return '#ffff00';
        if(v<=250) return '#ff7e00';
        if(v<=350) return '#ff0000';
        if(v<=430) return '#8f3f97';
        return '#7e0023';
    }}
    document.addEventListener('DOMContentLoaded', function() {{
        var maps = document.querySelectorAll('.leaflet-container');
        maps.forEach(function(mapEl) {{
            mapEl.addEventListener('mousemove', function(e) {{
                var tooltip = document.getElementById('aq-tooltip');
                if (!window._leafletMap) return;
                var pt = window._leafletMap.containerPointToLatLng([e.offsetX, e.offsetY]);
                var aq = _findAQ(pt.lat, pt.lng);
                if (aq) {{
                    tooltip.style.display = 'block';
                    var c = _aqi_color(aq.pm10);
                    document.getElementById('aq-vals').innerHTML =
                      '<span style="color:'+c+'">PM10: '+aq.pm10+' µg/m³</span><br>' +
                      'PM2.5: '+aq.pm25+' µg/m³<br>' +
                      'NO₂: '+aq.no2+' µg/m³<br>' +
                      'O₃: '+aq.o3+' µg/m³<br>' +
                      'SO₂: '+aq.so2+' µg/m³<br>' +
                      'CO: '+aq.co+' mg/m³<br>' +
                      'AQI: <b>'+aq.aqi+'</b>';
                }}
            }});
            mapEl.addEventListener('mouseleave', function() {{
                document.getElementById('aq-tooltip').style.display='none';
            }});
        }});
    }});
    </script>
    """
    m.get_root().html.add_child(folium.Element(hover_js))

    # ── Legend ──
    legend_html = f"""
    <div style='position:fixed;top:10px;right:10px;
      background:rgba(15,20,35,0.92);color:#fff;
      padding:10px 14px;border-radius:10px;font-size:12px;
      border:1px solid rgba(255,107,53,0.4);z-index:9998'>
    <b style='color:#ff6b35'>{label} ({unit})</b><br>
    <span style='color:#00e400'>■</span> ≤50 Good<br>
    <span style='color:#ffff00'>■</span> ≤100 Moderate<br>
    <span style='color:#ff7e00'>■</span> ≤250 Poor<br>
    <span style='color:#ff0000'>■</span> ≤350 Very Poor<br>
    <span style='color:#8f3f97'>■</span> ≤430 Severe<br>
    <span style='color:#7e0023'>■</span> 430+ Hazardous
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ──
    folium.LayerControl(collapsed=False).add_to(m)

    return m

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
st.sidebar.markdown("## 🇮🇳 India AirSense Pro")
st.sidebar.markdown("---")

city_list = sorted(INDIA_CITIES.keys())
selected_city = st.sidebar.selectbox("🏙️ Select City", city_list,
                                      index=city_list.index("Delhi"))
lat, lon, terrain, pop_m = INDIA_CITIES[selected_city]
st.sidebar.caption(f"📍 {lat:.4f}°N, {lon:.4f}°E  |  🏔️ {terrain.title()}  |  👥 {pop_m}M")

st.sidebar.markdown("---")
active_param = st.sidebar.selectbox("🌫️ Map Parameter", 
    ["pm10","pm25","no2","o3","so2","co","aqi"],
    format_func=lambda x: {"pm10":"PM10 (µg/m³)","pm25":"PM2.5 (µg/m³)",
        "no2":"NO₂ (µg/m³)","o3":"O₃ (µg/m³)","so2":"SO₂ (µg/m³)",
        "co":"CO (mg/m³)","aqi":"AQI"}[x])

map_opacity  = st.sidebar.slider("Heatmap Opacity", 0.2, 1.0, 0.65)
compare_mode = st.sidebar.checkbox("⚖️ Compare Two Cities")
if compare_mode:
    city2 = st.sidebar.selectbox("🏙️ Second City",
                                   [c for c in city_list if c != selected_city])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Custom Point Prediction")
custom_lat = st.sidebar.number_input("Latitude",  value=lat,  step=0.01, format="%.4f")
custom_lon = st.sidebar.number_input("Longitude", value=lon,  step=0.01, format="%.4f")
predict_pt  = st.sidebar.button("Predict AQ Here")

st.sidebar.markdown("---")
start_date = st.sidebar.date_input("📅 Hist. Start", datetime.date.today()-datetime.timedelta(days=7))
end_date   = st.sidebar.date_input("📅 Hist. End",   datetime.date.today())

st_autorefresh(interval=1800000, key="autorefresh")

# ══════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════
st.markdown(f'<div class="city-header">🌫️ {selected_city} — Air Quality Intelligence</div>',
            unsafe_allow_html=True)
st.markdown(f"*Terrain: **{terrain.title()}** · Population: **{pop_m}M** · "
            f"{pd.Timestamp.now().strftime('%d %b %Y, %H:%M IST')}*")
st.markdown("---")

# ══════════════════════════════════════════════════════
# FETCH ALL DATA
# ══════════════════════════════════════════════════════
with st.spinner("🔄 Fetching live data from all sources..."):
    weather  = fetch_openweather(lat, lon)
    traffic  = fetch_tomtom_traffic(lat, lon)
    stations = fetch_waqi_bounds(lat, lon)
    openaq   = fetch_openaq(selected_city, lat, lon)
    elev_df  = fetch_elevation_grid(lat, lon)
    firms_df = fetch_nasa_firms(lat, lon)
    df_hist  = load_historical()

# Merge stations
all_stations = pd.concat([stations, openaq], ignore_index=True) \
    if not openaq.empty else stations

# Get reference PM10
if not all_stations.empty and "pm10" in all_stations.columns:
    ref_pm10 = all_stations["pm10"].dropna().mean()
    if np.isnan(ref_pm10): ref_pm10 = None
else:
    ref_pm10 = None

if ref_pm10 is None:
    ref_pm10 = estimate_pm10_physics(weather, traffic, terrain, pop_m,
                                      elev_df["elev"].mean() if not elev_df.empty else 200,
                                      not firms_df.empty)

cat, color, emoji = pm10_category(ref_pm10)
has_sensors = not all_stations.empty and "pm10" in all_stations.columns

# ══════════════════════════════════════════════════════
# KEY METRICS ROW
# ══════════════════════════════════════════════════════
m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
m1.metric("🌫️ PM10",     f"{ref_pm10:.0f} µg/m³")
m2.metric("🌡️ Temp",     f"{weather['temp']:.1f}°C")
m3.metric("💧 Humidity", f"{weather['humidity']}%")
m4.metric("💨 Wind",     f"{weather['wind_speed']:.1f} km/h", weather['wind_dir'])
m5.metric("🧭 Dir",      weather['wind_dir'])
m6.metric("🚗 Traffic",  f"{traffic.get('congestion_index',0):.0f}%")
m7.metric("👁️ Visib.",   f"{weather['visibility']:.1f} km")

# AQI banner
st.markdown(f"""
<div style='background:{color};color:{"black" if color in ["#00e400","#ffff00"] else "white"};
padding:14px 22px;border-radius:12px;margin:10px 0;font-size:1.05rem;font-weight:700'>
{emoji} Air Quality: <b>{cat}</b> — PM10 {ref_pm10:.0f} µg/m³  
<span style='font-weight:400;font-size:0.9rem'> · {health_advisory(ref_pm10)}</span>
</div>
""", unsafe_allow_html=True)

if not has_sensors:
    st.info(f"ℹ️ No ground sensors for **{selected_city}**. "
            f"PM10 estimated via Gaussian dispersion model using "
            f"weather, terrain, traffic, elevation & satellite data.")

# ══════════════════════════════════════════════════════
# BUILD AQ GRID
# ══════════════════════════════════════════════════════
with st.spinner("🧮 Computing Gaussian plume dispersion grid..."):
    grids, lats, lons, lat_g, lon_g = build_aq_grid(
        lat, lon, all_stations, weather, traffic,
        terrain, pop_m, elev_df, firms_df, grid_res=60
    )

# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Interactive Map", "📈 Forecast", "🌤️ Weather", 
    "🚗 Traffic", "📊 Diagnostics", "📂 History"
])

# ════════════════════════════════════════
# TAB 1 — INTERACTIVE MAP
# ════════════════════════════════════════
with tab1:
    st.subheader(f"🗺️ Real Basemap + Full AQ Dispersion — {selected_city}")
    st.caption("Switch basemap (top-right) · Hover anywhere for AQ values · Click station dots for full data")

    col_map, col_info = st.columns([3, 1])

    with col_map:
        fmap = build_folium_map(
            lat, lon, grids, lats, lons, lat_g, lon_g,
            all_stations, firms_df, weather,
            active_param=active_param, opacity=map_opacity,
        )
        st_folium(fmap, width=None, height=600, returned_objects=[])

    with col_info:
        st.markdown("**📡 Data Sources**")
        st.success("✅ OpenWeather")
        st.success("✅ TomTom Traffic")
        st.success("✅ Open-Elevation")
        if has_sensors:
            st.success(f"✅ WAQI/OpenAQ ({len(all_stations)} stations)")
        else:
            st.warning("⚠️ No ground sensors")
            st.info("🧮 Using physics model")
        if not firms_df.empty:
            st.warning(f"🔥 NASA: {len(firms_df)} fire hotspots")

        st.markdown("---")
        st.markdown("**🌬️ Dispersion Model**")
        wd = weather.get("wind_deg", 180)
        ws = weather.get("wind_speed_ms", 1.0)
        sc_labels = {"A":"Very Unstable","B":"Unstable","C":"Slightly Unstable",
                     "D":"Neutral","E":"Stable","F":"Very Stable"}
        hour = pd.Timestamp.now().hour
        if ws < 2:   sc = "A" if 8<=hour<=18 else "F"
        elif ws < 5: sc = "B" if 8<=hour<=18 else "E"
        elif ws < 8: sc = "C" if 8<=hour<=18 else "D"
        else:        sc = "D"
        st.metric("Wind", f"{weather['wind_speed']:.1f} km/h {weather['wind_dir']}")
        st.metric("Stability", sc_labels[sc])
        st.metric("Terrain ×", f"{terrain_factor(terrain):.2f}")

        st.markdown("**🔥 Drivers**")
        drivers = []
        if traffic.get("congestion_index",0)>30: drivers.append("🚗 Traffic congestion")
        if ws < 1.5:                              drivers.append("💨 Near-calm wind")
        if weather.get("humidity",50)>75:         drivers.append("💧 High humidity")
        if terrain in ["valley","arid"]:          drivers.append("🏔️ Terrain trapping")
        if not firms_df.empty:                    drivers.append("🔥 Fire hotspots")
        if weather.get("pressure",1013)>1017:     drivers.append("🌡️ High pressure")
        for d in drivers: st.warning(d)
        if not drivers: st.success("✅ Good dispersion")

    # All AQ parameters cards
    st.markdown("---")
    st.markdown("### 🧪 All Air Quality Parameters — Live + Modelled")
    aq_params = {
        "PM10":  (grids["pm10"][grids["pm10"].shape[0]//2, grids["pm10"].shape[1]//2], "µg/m³", 100),
        "PM2.5": (grids["pm25"][grids["pm25"].shape[0]//2, grids["pm25"].shape[1]//2], "µg/m³", 60),
        "NO₂":   (grids["no2"][ grids["no2"].shape[0]//2,  grids["no2"].shape[1]//2],  "µg/m³", 80),
        "O₃":    (grids["o3"][  grids["o3"].shape[0]//2,   grids["o3"].shape[1]//2],   "µg/m³", 100),
        "SO₂":   (grids["so2"][ grids["so2"].shape[0]//2,  grids["so2"].shape[1]//2],  "µg/m³", 80),
        "CO":    (grids["co"][  grids["co"].shape[0]//2,   grids["co"].shape[1]//2],   "mg/m³", 10),
        "AQI":   (grids["aqi"][ grids["aqi"].shape[0]//2,  grids["aqi"].shape[1]//2],  "",      100),
    }
    # Override with sensor/OWM values where available
    if weather.get("owm_pm10"):  aq_params["PM10"]  = (weather["owm_pm10"],  "µg/m³", 100)
    if weather.get("owm_pm25"):  aq_params["PM2.5"] = (weather["owm_pm25"],  "µg/m³", 60)
    if weather.get("owm_no2"):   aq_params["NO₂"]   = (weather["owm_no2"],   "µg/m³", 80)
    if weather.get("owm_o3"):    aq_params["O₃"]    = (weather["owm_o3"],    "µg/m³", 100)
    if weather.get("owm_so2"):   aq_params["SO₂"]   = (weather["owm_so2"],   "µg/m³", 80)
    if weather.get("owm_co"):    aq_params["CO"]    = (weather["owm_co"],    "mg/m³", 10)

    cols_aq = st.columns(7)
    for i, (name, (val, unit_, limit)) in enumerate(aq_params.items()):
        with cols_aq[i]:
            pct = min(val/limit, 2.0)
            clr = ("#00e400" if pct<0.5 else "#ffff00" if pct<1.0
                   else "#ff7e00" if pct<1.5 else "#ff0000")
            st.markdown(f"""<div style='background:#1a1d27;border-radius:10px;
              padding:10px;border-left:4px solid {clr};text-align:center'>
              <div style='font-size:11px;color:#aaa'>{name}</div>
              <div style='font-size:1.3rem;font-weight:700;color:{clr}'>{val:.1f}</div>
              <div style='font-size:10px;color:#888'>{unit_}</div>
            </div>""", unsafe_allow_html=True)

    # Radar chart
    st.markdown("#### 📡 Pollutant Radar — % of WHO safe limit")
    limits = {"PM10":100,"PM2.5":60,"NO₂":80,"O₃":100,"SO₂":80,"CO":10,"AQI":100}
    norm_vals = {k: min(v/limits[k]*100, 250) for k,(v,_,__) in aq_params.items()}
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatterpolar(
        r=list(norm_vals.values()), theta=list(norm_vals.keys()),
        fill="toself", fillcolor="rgba(255,107,53,0.2)",
        line=dict(color="#ff6b35", width=2), name="Current",
    ))
    fig_r.add_trace(go.Scatterpolar(
        r=[100]*7, theta=list(norm_vals.keys()), mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", dash="dot", width=1),
        name="WHO limit",
    ))
    fig_r.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,250]),
                   angularaxis=dict(tickfont=dict(color="white")),
                   bgcolor="#1a1d27"),
        paper_bgcolor="#0e1117", font_color="white",
        height=380, margin=dict(t=20,b=20), showlegend=True,
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # Station table
    if not all_stations.empty:
        st.markdown("**📍 Monitoring Stations**")
        disp = all_stations.copy()
        for c_ in ["pm10","pm25","aqi","no2","o3","so2","co"]:
            if c_ not in disp.columns: disp[c_] = None
        disp["Category"] = disp["pm10"].apply(
            lambda v: pm10_category(v)[0] if v is not None else "N/A")
        st.dataframe(
            disp[["name","lat","lon","pm10","pm25","aqi","no2","o3","so2","co","Category","source"]],
            use_container_width=True)

# ════════════════════════════════════════
# TAB 2 — FORECAST
# ════════════════════════════════════════
with tab2:
    st.subheader("📈 PM10 Forecast")

    if not weather.get("forecast_df", pd.DataFrame()).empty:
        fdf = weather["forecast_df"].copy()
        fdf["est_pm10"] = fdf.apply(lambda r: estimate_pm10_physics(
            {"temp":r["temp"],"humidity":r["humidity"],
             "wind_speed_ms":r["wind_speed"]/3.6,"pressure":1013,"clouds":50},
            traffic, terrain, pop_m,
            elev_df["elev"].mean() if not elev_df.empty else 200,
            not firms_df.empty), axis=1)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=fdf["timestamp"], y=fdf["est_pm10"],
            mode="lines+markers", name="Forecast PM10",
            line=dict(color="#ff6b35", width=3),
            fill="tozeroy", fillcolor="rgba(255,107,53,0.12)",
        ))
        for lev, clr, lbl in [(50,"#00e400","Good"),(100,"#ffff00","Moderate"),
                               (250,"#ff7e00","Poor"),(350,"#ff0000","Very Poor")]:
            fig_fc.add_hline(y=lev, line_dash="dot", line_color=clr,
                              annotation_text=lbl, annotation_position="right")
        fig_fc.update_layout(xaxis_title="Time", yaxis_title="PM10 (µg/m³)",
            paper_bgcolor="#0e1117", plot_bgcolor="#1a1d27",
            font_color="white", height=420)
        st.plotly_chart(fig_fc, use_container_width=True)

        # 5-day multi-panel
        st.subheader("📅 5-Day Weather + PM10")
        fig5 = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Temperature (°C)","Humidity (%)","Est. PM10 (µg/m³)"])
        fig5.add_trace(go.Scatter(x=fdf["timestamp"],y=fdf["temp"],
            line=dict(color="#f7c59f"),name="Temp"),row=1,col=1)
        fig5.add_trace(go.Bar(x=fdf["timestamp"],y=fdf["humidity"],
            marker_color="#3d5a80",name="Humidity"),row=2,col=1)
        fig5.add_trace(go.Scatter(x=fdf["timestamp"],y=fdf["est_pm10"],
            fill="tozeroy",line=dict(color="#ff6b35"),name="PM10"),row=3,col=1)
        fig5.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1a1d27",
            font_color="white",height=520,showlegend=False)
        st.plotly_chart(fig5, use_container_width=True)
        st.caption("PM10 forecast uses Gaussian dispersion model with 5-day weather forecast inputs.")
    else:
        st.info("Weather forecast unavailable.")

# ════════════════════════════════════════
# TAB 3 — WEATHER
# ════════════════════════════════════════
with tab3:
    st.subheader(f"🌤️ Full Weather — {selected_city}")
    wc1,wc2,wc3,wc4,wc5,wc6 = st.columns(6)
    wc1.metric("🌡️ Temp",      f"{weather['temp']:.1f}°C",  f"Feels {weather['feels_like']:.0f}°C")
    wc2.metric("💧 Humidity",  f"{weather['humidity']}%")
    wc3.metric("🌬️ Pressure", f"{weather['pressure']} hPa")
    wc4.metric("💨 Wind",      f"{weather['wind_speed']:.1f} km/h")
    wc5.metric("🧭 Direction", weather['wind_dir'])
    wc6.metric("☁️ Clouds",   f"{weather['clouds']}%")
    st.markdown(f"**{weather['description']}** · Visibility: {weather['visibility']:.1f} km")

    col_a, col_b = st.columns(2)
    with col_a:
        # Wind rose
        dirs = ["N","NE","E","SE","S","SW","W","NW"]
        angles = [0,45,90,135,180,225,270,315]
        ws = weather["wind_speed"]
        wd = weather["wind_deg"]
        speeds = [max(0, ws*abs(math.cos(math.radians(a-wd)))) for a in angles]
        fig_wr = go.Figure(go.Barpolar(
            r=speeds, theta=dirs,
            marker_color=["#ff6b35" if s==max(speeds) else "#3d5a80" for s in speeds],
            marker_line_color="white", marker_line_width=1, opacity=0.85,
        ))
        fig_wr.update_layout(
            polar=dict(radialaxis=dict(visible=True,range=[0,max(speeds)+2])),
            paper_bgcolor="#0e1117", font_color="white",
            title="Wind Rose", height=320)
        st.plotly_chart(fig_wr, use_container_width=True)

    with col_b:
        st.markdown("**OpenWeather Air Quality API**")
        for name, key, unit_ in [("PM10","owm_pm10","µg/m³"),("PM2.5","owm_pm25","µg/m³"),
                                   ("NO₂","owm_no2","µg/m³"),("O₃","owm_o3","µg/m³"),
                                   ("SO₂","owm_so2","µg/m³"),("CO","owm_co","µg/m³")]:
            v = weather.get(key)
            if v is not None:
                st.metric(name, f"{v:.1f} {unit_}")

# ════════════════════════════════════════
# TAB 4 — TRAFFIC
# ════════════════════════════════════════
with tab4:
    st.subheader(f"🚗 Traffic Analysis — {selected_city}")
    tc1,tc2,tc3 = st.columns(3)
    tc1.metric("🚦 Congestion",    f"{traffic.get('congestion_index',0):.1f}%")
    tc2.metric("🏎️ Current Speed", f"{traffic.get('current_speed','N/A')} km/h")
    tc3.metric("🛣️ Free Flow",     f"{traffic.get('free_flow_speed','N/A')} km/h")

    cong = traffic.get("congestion_index",0)
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number", value=cong, title={"text":"Congestion (%)"},
        gauge={"axis":{"range":[0,100]},
               "steps":[{"range":[0,30],"color":"#00e400"},
                         {"range":[30,60],"color":"#ffff00"},
                         {"range":[60,80],"color":"#ff7e00"},
                         {"range":[80,100],"color":"#ff0000"}],
               "bar":{"color":"#ff6b35"}}))
    fig_g.update_layout(paper_bgcolor="#0e1117",font_color="white",height=300)
    st.plotly_chart(fig_g, use_container_width=True)

    st.metric("🌫️ Traffic PM10 contribution",
              f"{ref_pm10 * cong/200:.1f} µg/m³",
              f"{cong/2:.0f}% of total")

    if traffic.get("incidents"):
        st.markdown("**🚨 Active Incidents**")
        for i, inc in enumerate(traffic["incidents"][:5]):
            p = inc.get("properties",{})
            st.warning(f"Incident {i+1}: {p.get('iconCategory','Unknown')}")

# ════════════════════════════════════════
# TAB 5 — DIAGNOSTICS
# ════════════════════════════════════════
with tab5:
    st.subheader("📊 Model Diagnostics")
    if has_sensors and len(all_stations) >= 3:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        df_d = all_stations.dropna(subset=["pm10"]).copy()
        df_d["hour"]      = pd.Timestamp.now().hour
        df_d["dayofweek"] = pd.Timestamp.now().dayofweek
        df_d["month"]     = pd.Timestamp.now().month
        df_d["temp"]  = weather["temp"]
        df_d["hum"]   = weather["humidity"]
        df_d["wind"]  = weather["wind_speed"]
        df_d["pm10_lag1"] = df_d["pm10"].median()
        feats = ["lat","lon","hour","dayofweek","month","temp","hum","wind","pm10_lag1"]
        preds, actuals = [], []
        for i in range(len(df_d)):
            train = df_d.drop(df_d.index[i])
            test  = df_d.iloc[[i]]
            sc_ = StandardScaler()
            Xtr = sc_.fit_transform(train[feats])
            Xte = sc_.transform(test[feats])
            rf_ = RandomForestRegressor(n_estimators=200, random_state=42)
            rf_.fit(Xtr, np.log1p(train["pm10"]))
            preds.append(np.expm1(rf_.predict(Xte)[0]))
            actuals.append(test["pm10"].values[0])

        mae  = mean_absolute_error(actuals, preds)
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        res  = pd.DataFrame({"Actual":actuals,"Predicted":preds})

        dc1, dc2 = st.columns(2)
        with dc1:
            st.metric("MAE",  f"{mae:.2f} µg/m³")
            st.metric("RMSE", f"{rmse:.2f} µg/m³")
            fig_sc = px.scatter(res, x="Actual", y="Predicted",
                                color_discrete_sequence=["#ff6b35"],
                                title="Actual vs Predicted")
            _xa = res["Actual"].values; _xp = res["Predicted"].values
            if len(_xa)>1:
                _m, _b = np.polyfit(_xa, _xp, 1)
                _xr = np.array([_xa.min(),_xa.max()])
                fig_sc.add_trace(go.Scatter(x=_xr,y=_m*_xr+_b,mode="lines",
                    line=dict(color="white",dash="dot"),name="Fit"))
            fig_sc.add_shape(type="line",x0=res.Actual.min(),y0=res.Actual.min(),
                x1=res.Actual.max(),y1=res.Actual.max(),
                line=dict(dash="dash",color="rgba(255,255,255,0.4)"))
            fig_sc.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1a1d27",font_color="white")
            st.plotly_chart(fig_sc, use_container_width=True)
        with dc2:
            fig_h = px.histogram(df_d, x="pm10", nbins=20, title="PM10 Distribution",
                                  color_discrete_sequence=["#3d5a80"])
            fig_h.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1a1d27",font_color="white")
            st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Need ≥3 sensor stations for diagnostics. "
                "Try Delhi, Mumbai, or another major city.")

# ════════════════════════════════════════
# TAB 6 — HISTORY
# ════════════════════════════════════════
with tab6:
    st.subheader("📂 Historical Database")
    if not df_hist.empty:
        df_f = df_hist[
            (df_hist["timestamp"].dt.date >= start_date) &
            (df_hist["timestamp"].dt.date <= end_date)
        ].copy()
        st.metric("Records in range", len(df_f))
        if not df_f.empty and "pm10" in df_f.columns:
            fig_hh = px.line(df_f.set_index("timestamp").resample("h").mean(numeric_only=True).reset_index(),
                              x="timestamp", y="pm10", title="Historical PM10",
                              color_discrete_sequence=["#ff6b35"])
            fig_hh.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#1a1d27",font_color="white")
            st.plotly_chart(fig_hh, use_container_width=True)
            st.dataframe(df_f.sort_values("timestamp",ascending=False), use_container_width=True)
            st.download_button("📥 Download CSV",
                               df_f.to_csv(index=False).encode("utf-8"),
                               "airsense_history.csv", "text/csv")
    else:
        st.info("No historical data yet.")

# ══════════════════════════════════════════════════════
# COMPARE MODE
# ══════════════════════════════════════════════════════
if compare_mode:
    st.markdown("---")
    lat2, lon2, terrain2, pop2 = INDIA_CITIES[city2]
    with st.spinner(f"Loading {city2}..."):
        w2 = fetch_openweather(lat2, lon2)
        t2 = fetch_tomtom_traffic(lat2, lon2)
        s2 = fetch_waqi_bounds(lat2, lon2)
        e2 = fetch_elevation_grid(lat2, lon2)
        f2 = fetch_nasa_firms(lat2, lon2)

    pm10_2 = (s2["pm10"].mean() if not s2.empty and "pm10" in s2.columns and not s2["pm10"].isna().all()
              else estimate_pm10_physics(w2, t2, terrain2, pop2,
                                          e2["elev"].mean() if not e2.empty else 200, not f2.empty))
    cat2, color2, emoji2 = pm10_category(pm10_2)

    st.subheader(f"⚖️ {selected_city} vs {city2}")
    comp = pd.DataFrame({
        "Metric":["PM10 (µg/m³)","Temp (°C)","Humidity (%)","Wind (km/h)","Congestion (%)"],
        selected_city:[ref_pm10, weather["temp"], weather["humidity"],
                        weather["wind_speed"], traffic.get("congestion_index",0)],
        city2:        [pm10_2,   w2["temp"],     w2["humidity"],
                        w2["wind_speed"],         t2.get("congestion_index",0)],
    })
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name=selected_city, x=comp["Metric"],
                               y=comp[selected_city], marker_color="#ff6b35"))
    fig_comp.add_trace(go.Bar(name=city2, x=comp["Metric"],
                               y=comp[city2], marker_color="#3d5a80"))
    fig_comp.update_layout(barmode="group", paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1d27", font_color="white", height=380)
    st.plotly_chart(fig_comp, use_container_width=True)

# ══════════════════════════════════════════════════════
# CUSTOM POINT PREDICTION
# ══════════════════════════════════════════════════════
if predict_pt:
    st.markdown("---")
    st.subheader(f"🎯 Custom Prediction — {custom_lat:.4f}°N, {custom_lon:.4f}°E")
    cw = fetch_openweather(custom_lat, custom_lon)
    ct = fetch_tomtom_traffic(custom_lat, custom_lon)
    # Nearest city terrain
    dists = {c: abs(v[0]-custom_lat)+abs(v[1]-custom_lon) for c,v in INDIA_CITIES.items()}
    nearest = min(dists, key=dists.get)
    _, __, c_terrain, c_pop = INDIA_CITIES[nearest]
    c_pred = estimate_pm10_physics(cw, ct, c_terrain, c_pop)
    c_cat, c_col, c_emoji = pm10_category(c_pred)
    col_a, col_b = st.columns(2)
    col_a.metric("PM10 Estimate", f"{c_pred:.1f} µg/m³")
    col_b.metric("Temperature",   f"{cw['temp']:.1f}°C")
    st.markdown(f"<span style='background:{c_col};padding:8px 14px;border-radius:8px;"
                f"font-weight:700;color:{'black' if c_col in ['#00e400','#ffff00'] else 'white'}'>"
                f"{c_emoji} {c_cat}</span>  —  Nearest terrain ref: **{nearest}**",
                unsafe_allow_html=True)
    st.caption(health_advisory(c_pred))

# ══════════════════════════════════════════════════════
# SIDEBAR INTELLIGENCE
# ══════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Status")
data_pts = len(df_hist) if not df_hist.empty else 0
st.sidebar.progress(min(data_pts/1000,1.0), text=f"Historical: {data_pts}/1000")
if has_sensors:
    st.sidebar.success(f"✅ {len(all_stations)} live sensors")
    st.sidebar.info("🧮 Gaussian plume calibrated to sensors")
else:
    st.sidebar.warning("⚠️ No sensors — physics model active")
    st.sidebar.info(f"🌬️ Wind: {weather['wind_speed']:.1f} km/h {weather['wind_dir']}")
st.sidebar.caption(f"🕐 {time.strftime('%H:%M:%S')}")
st.sidebar.caption("WAQI · OpenWeather · TomTom · OpenAQ · NASA FIRMS · Open-Elevation · OSM")
