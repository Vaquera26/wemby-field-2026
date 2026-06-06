"""
Clima batch — UNA sola llamada a Open-Meteo por estadio por temporada.
Guarda en cache/weather_YYYY.json para reutilizar.
"""
import json, time, urllib.request
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

STADIUMS = {
    "Fenway Park":                (42.3467, -71.0972, "open"),
    "Wrigley Field":              (41.9484, -87.6553, "open"),
    "Oriole Park at Camden Yards":(39.2839, -76.6218, "open"),
    "Yankee Stadium":             (40.8296, -73.9262, "open"),
    "PNC Park":                   (40.4469, -80.0057, "open"),
    "Dodger Stadium":             (34.0739,-118.2400, "open"),
    "Oracle Park":                (37.7786,-122.3893, "open"),
    "Angel Stadium":              (33.8003,-117.8827, "open"),
    "Petco Park":                 (32.7076,-117.1570, "open"),
    "Coors Field":                (39.7559,-104.9942, "open"),
    "Great American Ball Park":   (39.0979, -84.5069, "open"),
    "Busch Stadium":              (38.6226, -90.1928, "open"),
    "Nationals Park":             (38.8730, -77.0074, "open"),
    "Citizens Bank Park":         (39.9061, -75.1665, "open"),
    "Citi Field":                 (40.7571, -73.8458, "open"),
    "Truist Park":                (33.8908, -84.4678, "open"),
    "American Family Field":      (43.0280, -87.9712, "retractable"),
    "Target Field":               (44.9817, -93.2781, "open"),
    "Kauffman Stadium":           (39.0517, -94.4803, "open"),
    "Guaranteed Rate Field":      (41.8300, -87.6339, "open"),
    "Progressive Field":          (41.4962, -81.6852, "open"),
    "Comerica Park":              (42.3390, -83.0485, "open"),
    "Daikin Park":                (29.7573, -95.3555, "retractable"),
    "Minute Maid Park":           (29.7573, -95.3555, "retractable"),
    "loanDepot park":             (25.7781, -80.2197, "retractable"),
    "LoanDepot Park":             (25.7781, -80.2197, "retractable"),
    "Chase Field":                (33.4453,-112.0667, "retractable"),
    "T-Mobile Park":              (47.5914,-122.3325, "retractable"),
    "Globe Life Field":           (32.7513, -97.0824, "retractable"),
    "Rogers Centre":              (43.6414, -79.3894, "dome"),
    "Tropicana Field":            (27.7683, -82.6534, "dome"),
    "Oakland Coliseum":           (37.7516,-122.2007, "open"),
}

def _fetch(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"MLB-Picks/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep(2)
    return {}

def _match_venue(venue_name: str):
    if not venue_name:
        return None, None, "open"
    vl = venue_name.lower()
    for name, (lat, lon, roof) in STADIUMS.items():
        if name.lower() in vl or vl in name.lower():
            return lat, lon, roof
    return None, None, "open"

def build_season_weather_cache(season: int, start_date: str, end_date: str):
    """
    Para cada estadio: UNA llamada a Open-Meteo con todo el rango de la temporada.
    Devuelve y guarda: {venue_name: {date_str: {wind_mph, temp_f, precip_mm}}}
    """
    cache_path = CACHE_DIR / f"weather_{season}.json"
    if cache_path.exists():
        print(f"  Clima {season}: cargando desde cache...")
        with open(cache_path) as f:
            return json.load(f)

    print(f"  Descargando clima {season} ({start_date} → {end_date})...")
    result = {}
    venues = list(STADIUMS.items())

    for i, (venue, (lat, lon, roof)) in enumerate(venues):
        if roof == "dome":
            result[venue] = {}
            continue

        url = (f"https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}"
               f"&start_date={start_date}&end_date={end_date}"
               f"&daily=windspeed_10m_max,temperature_2m_mean,precipitation_sum"
               f"&windspeed_unit=mph&temperature_unit=fahrenheit&timezone=auto")

        data = _fetch(url)
        daily = data.get("daily", {})
        dates  = daily.get("time", [])
        winds  = daily.get("windspeed_10m_max", [])
        temps  = daily.get("temperature_2m_mean", [])
        precip = daily.get("precipitation_sum", [])

        venue_wx = {}
        for j, d in enumerate(dates):
            venue_wx[d] = {
                "wind_mph":  float(winds[j])  if j < len(winds)  and winds[j]  is not None else 7.0,
                "temp_f":    float(temps[j])  if j < len(temps)  and temps[j]  is not None else 72.0,
                "precip_mm": float(precip[j]) if j < len(precip) and precip[j] is not None else 0.0,
                "roof":      roof,
            }
        result[venue] = venue_wx
        print(f"    {i+1}/{len(venues)} {venue[:30]:<30} {len(venue_wx)} dias", end="\r")
        time.sleep(0.15)  # rate limit

    print(f"\n  {season} completo — {len(result)} estadios")
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result

def lookup(wx_cache: dict, venue_name: str, date_str: str) -> dict:
    """
    Busca clima para un venue+fecha en el cache precargado.
    """
    # match venue
    for name in wx_cache:
        if name.lower() in venue_name.lower() or venue_name.lower() in name.lower():
            day = wx_cache[name].get(date_str)
            if day:
                return day
            # dome
            if not wx_cache[name]:
                return {"wind_mph":0.0,"temp_f":72.0,"precip_mm":0.0,"roof":"dome"}

    # no match — default outdoor moderate
    return {"wind_mph":7.0,"temp_f":72.0,"precip_mm":0.0,"roof":"open"}

def wind_factor(wind_mph: float, roof: str) -> float:
    if roof in ("dome","retractable"): return 0.0
    if wind_mph >= 20: return  0.6
    if wind_mph >= 15: return  0.3
    if wind_mph >= 10: return  0.1
    return 0.0

def temp_factor(temp_f: float, roof: str) -> float:
    if roof == "dome": return 0.0
    if temp_f < 45:  return -0.8
    if temp_f < 55:  return -0.4
    if temp_f < 65:  return -0.1
    if temp_f > 90:  return  0.2
    return 0.0
