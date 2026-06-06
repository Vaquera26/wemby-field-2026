"""
ERA/WHIP individual del pitcher titular — via MLB Stats API.
Cachea en memoria para no repetir llamadas.
"""
import json, time, urllib.request
from functools import lru_cache

BASE = "https://statsapi.mlb.com/api/v1"

def _fetch(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except:
            time.sleep(1)
    return {}

@lru_cache(maxsize=1024)
def get_pitcher_season_stats(person_id: int, season: int) -> dict:
    """
    Returns {"era": float, "whip": float, "ip": float, "k9": float}
    for a specific pitcher in a season.
    Falls back to league averages if not found.
    """
    url = (f"{BASE}/people/{person_id}/stats"
           f"?stats=season&season={season}&group=pitching&sportId=1")
    data = _fetch(url)

    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            s = split.get("stat", {})
            try:
                return {
                    "era":  float(s.get("era", 4.20)),
                    "whip": float(s.get("whip", 1.30)),
                    "ip":   float(s.get("inningsPitched", 0) or 0),
                    "k9":   float(s.get("strikeoutsPer9Inn", 8.0) or 8.0),
                    "bb9":  float(s.get("walksPer9Inn", 3.0) or 3.0),
                    "hr9":  float(s.get("homeRunsPer9", 1.0) or 1.0),
                    "found": True,
                }
            except (TypeError, ValueError):
                pass

    return {"era": 4.20, "whip": 1.30, "ip": 0.0,
            "k9": 8.0, "bb9": 3.0, "hr9": 1.0, "found": False}


def pitcher_quality_score(stats: dict) -> float:
    """
    0.0 = terrible pitcher (ERA 7+)
    1.0 = average (ERA ~4.2)
    2.0 = elite (ERA ~2.5)
    Combines ERA, WHIP, K/9.
    """
    era  = stats.get("era", 4.20)
    whip = stats.get("whip", 1.30)
    k9   = stats.get("k9", 8.0)

    era_score  = max(0, min(2, (7.0 - era)  / 4.5))   # 7→0, 2.5→1
    whip_score = max(0, min(2, (2.0 - whip) / 0.70))  # 2.0→0, 1.0→1.4
    k9_score   = max(0, min(2, k9 / 9.0))             # 9K/9→1.0

    return round((era_score * 0.50 + whip_score * 0.30 + k9_score * 0.20), 3)
