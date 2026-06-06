"""
Todas las llamadas a statsapi.mlb.com.
Se usa httpx con async para no bloquear FastAPI.
"""
import httpx
from datetime import date, timedelta

BASE = "https://statsapi.mlb.com/api/v1"
TIMEOUT = 30


async def _get(client: httpx.AsyncClient, url: str) -> dict:
    r = await client.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


async def fetch_schedule(client, start: str, end: str, hydrate: str) -> list[dict]:
    url = (f"{BASE}/schedule?sportId=1&startDate={start}&endDate={end}"
           f"&hydrate={hydrate}&gameType=R")
    data = await _get(client, url)
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            g["_date"] = d["date"]
            games.append(g)
    return games


async def fetch_team_stats(client, season: int, group: str) -> dict:
    """Returns {teamId: stat_dict}."""
    url = (f"{BASE}/teams/stats?season={season}&sportId=1"
           f"&stats=season&group={group}")
    data = await _get(client, url)
    out = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        tid = split["team"]["id"]
        out[tid] = split["stat"]
    return out


async def fetch_standings(client, season: int) -> dict:
    """Returns {teamId: {wins, losses, streak, winPct}}."""
    url = f"{BASE}/standings?leagueId=103,104&season={season}&standingsTypes=regularSeason"
    data = await _get(client, url)
    out = {}
    for record in data.get("records", []):
        for tr in record.get("teamRecords", []):
            tid = tr["team"]["id"]
            out[tid] = {
                "wins":    tr.get("wins", 0),
                "losses":  tr.get("losses", 0),
                "winPct":  float(tr.get("winningPercentage", "0") or 0),
                "streak":  tr.get("streak", {}).get("streakCode", ""),
            }
    return out


def finished(g: dict) -> bool:
    return g.get("status", {}).get("statusCode", "") in ("F", "FR", "FT", "FO")


def extract_innings(g: dict) -> dict[int, tuple[int, int]]:
    """Returns {inning_num: (away_runs, home_runs)}."""
    inn = {}
    for iv in g.get("linescore", {}).get("innings", []):
        n = iv.get("num", 0)
        if 1 <= n <= 9:
            ar = iv.get("away", {}).get("runs") or 0
            hr = iv.get("home", {}).get("runs") or 0
            inn[n] = (int(ar), int(hr))
    return inn


def game_totals(g: dict) -> tuple[int, int]:
    t = g.get("linescore", {}).get("teams", {})
    return (t.get("away", {}).get("runs") or 0,
            t.get("home", {}).get("runs") or 0)
