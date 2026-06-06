"""
Datos de clima por estadio via Open-Meteo (gratis, sin API key).
Retorna wind_mph, temp_f, precip_mm para la ventana de juego (13-21h local).
"""
import json, time
import urllib.request
from functools import lru_cache

# Coordenadas + tipo de techo para los 30 estadios MLB
# roof: "open" | "retractable" | "dome"
STADIUMS = {
    # Outdoor / open
    "Fenway Park":               (42.3467, -71.0972, "open"),
    "Wrigley Field":             (41.9484, -87.6553, "open"),
    "Oriole Park at Camden Yards":(39.2839,-76.6218,"open"),
    "Yankee Stadium":            (40.8296, -73.9262, "open"),
    "PNC Park":                  (40.4469, -80.0057, "open"),
    "Dodger Stadium":            (34.0739,-118.2400, "open"),
    "Oracle Park":               (37.7786,-122.3893, "open"),
    "Angel Stadium":             (33.8003,-117.8827, "open"),
    "Petco Park":                (32.7076,-117.1570, "open"),
    "Coors Field":               (39.7559,-104.9942, "open"),  # altitude factor
    "Great American Ball Park":  (39.0979, -84.5069, "open"),
    "Busch Stadium":             (38.6226, -90.1928, "open"),
    "Nationals Park":            (38.8730, -77.0074, "open"),
    "Citizens Bank Park":        (39.9061, -75.1665, "open"),
    "Citi Field":                (40.7571, -73.8458, "open"),
    "Truist Park":               (33.8908, -84.4678, "open"),
    "American Family Field":     (43.0280, -87.9712, "retractable"),
    "Target Field":              (44.9817, -93.2781, "open"),
    "Kauffman Stadium":          (39.0517, -94.4803, "open"),
    "Guaranteed Rate Field":     (41.8300, -87.6339, "open"),
    "Progressive Field":         (41.4962, -81.6852, "open"),
    "Comerica Park":             (42.3390, -83.0485, "open"),
    "Daikin Park":               (29.7573, -95.3555, "retractable"),
    "Minute Maid Park":          (29.7573, -95.3555, "retractable"),
    "loanDepot park":            (25.7781, -80.2197, "retractable"),
    "Chase Field":               (33.4453,-112.0667, "retractable"),
    "T-Mobile Park":             (47.5914,-122.3325, "retractable"),
    "Oakland Coliseum":          (37.7516,-122.2007, "open"),
    "Globe Life Field":          (32.7513, -97.0824, "retractable"),
    "Rogers Centre":             (43.6414, -79.3894, "dome"),
    "Tropicana Field":           (27.7683, -82.6534, "dome"),
    "LoanDepot Park":            (25.7781, -80.2197, "retractable"),
}

def _fetch(url: str) -> dict:
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MLB-Picks/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(1)
    return {}

@lru_cache(maxsize=512)
def get_weather(lat: float, lon: float, date_str: str) -> dict:
    """
    Returns {"wind_mph": float, "temp_f": float, "precip_mm": float}
    for game-time window (13h-21h UTC-6 ≈ 19-03h UTC → use local hours 13-21).
    Caches in memory per (lat, lon, date).
    """
    # Use archive for past dates, forecast for today/future
    from datetime import date as Date
    d = Date.fromisoformat(date_str)
    today = Date.today()

    if d <= today:
        base = "https://archive-api.open-meteo.com/v1/archive"
    else:
        base = "https://api.open-meteo.com/v1/forecast"

    url = (f"{base}?latitude={lat}&longitude={lon}"
           f"&start_date={date_str}&end_date={date_str}"
           f"&hourly=windspeed_10m,temperature_2m,precipitation"
           f"&windspeed_unit=mph&temperature_unit=fahrenheit"
           f"&timezone=auto")

    data = _fetch(url)
    hourly = data.get("hourly", {})
    hours  = hourly.get("time", [])
    winds  = hourly.get("windspeed_10m", [])
    temps  = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation", [])

    if not hours:
        return {"wind_mph": 7.0, "temp_f": 72.0, "precip_mm": 0.0}

    # Game window: 13:00–20:00 local
    game_idx = [i for i, h in enumerate(hours) if "T13:" <= h[10:] <= "T20:"]
    if not game_idx:
        game_idx = list(range(len(hours)))

    def safe_mean(lst, idxs):
        vals = [lst[i] for i in idxs if i < len(lst) and lst[i] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "wind_mph":  round(safe_mean(winds,  game_idx), 1),
        "temp_f":    round(safe_mean(temps,  game_idx), 1),
        "precip_mm": round(safe_mean(precip, game_idx), 2),
    }

def get_venue_weather(venue_name: str, date_str: str) -> dict:
    """
    Looks up stadium by venue name and fetches weather.
    Returns weather dict + roof_type.
    """
    # fuzzy match
    match = None
    for name, (lat, lon, roof) in STADIUMS.items():
        if name.lower() in venue_name.lower() or venue_name.lower() in name.lower():
            match = (lat, lon, roof)
            break

    if match is None:
        # default: moderate conditions, unknown
        return {"wind_mph": 7.0, "temp_f": 72.0, "precip_mm": 0.0, "roof": "unknown"}

    lat, lon, roof = match
    if roof == "dome":
        return {"wind_mph": 0.0, "temp_f": 72.0, "precip_mm": 0.0, "roof": roof}

    w = get_weather(lat, lon, date_str)
    w["roof"] = roof
    return w

def wind_factor(wind_mph: float, roof: str) -> float:
    """
    Returns run adjustment for wind.
    +/- based on speed; direction unknown so we use absolute value effect.
    Wrigley effect: strong winds dramatically affect scoring.
    """
    if roof in ("dome", "retractable"):
        return 0.0
    # Wind > 15mph adds variance — slight upward pressure on totals
    if wind_mph >= 20: return  0.6
    if wind_mph >= 15: return  0.3
    if wind_mph >= 10: return  0.1
    return 0.0

def temp_factor(temp_f: float, roof: str) -> float:
    """Cold suppresses offense; heat at altitude (Coors) increases it."""
    if roof == "dome":
        return 0.0
    if temp_f < 45: return -0.8
    if temp_f < 55: return -0.4
    if temp_f < 65: return -0.1
    if temp_f > 90: return  0.2   # hot = slightly more offense
    return 0.0
