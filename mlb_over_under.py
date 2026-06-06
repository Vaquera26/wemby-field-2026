#!/usr/bin/env python3
"""MLB Over/Under analysis for today's games using season stats."""

import urllib.request
import json
from datetime import datetime

TODAY = "2026-06-04"

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())

def get_team_season_stats():
    """Get batting/pitching stats for all teams this season."""
    url = "https://statsapi.mlb.com/api/v1/teams/stats?season=2026&sportId=1&stats=season&group=hitting"
    data = fetch(url)
    hitting = {}
    for t in data.get("stats", [{}])[0].get("splits", []):
        tid = t["team"]["id"]
        s = t["stat"]
        hitting[tid] = {
            "name": t["team"]["name"],
            "runs_per_game": float(s.get("runs", 0)) / max(float(s.get("gamesPlayed", 1)), 1),
            "avg": s.get("avg", ".000"),
            "ops": s.get("ops", ".000"),
            "hr": s.get("homeRuns", 0),
            "games": s.get("gamesPlayed", 0),
            "total_runs": s.get("runs", 0),
        }

    url2 = "https://statsapi.mlb.com/api/v1/teams/stats?season=2026&sportId=1&stats=season&group=pitching"
    data2 = fetch(url2)
    pitching = {}
    for t in data2.get("stats", [{}])[0].get("splits", []):
        tid = t["team"]["id"]
        s = t["stat"]
        pitching[tid] = {
            "era": s.get("era", "0.00"),
            "runs_allowed_per_game": float(s.get("runs", 0)) / max(float(s.get("gamesPlayed", 1)), 1),
            "whip": s.get("whip", "0.00"),
        }
    return hitting, pitching

def get_todays_games():
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={TODAY}&hydrate=probablePitcher,linescore,team,venue,weather"
    data = fetch(url)
    games = []
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            games.append(g)
    return games

def get_last_10_runs(team_id):
    """Get runs scored in last 10 games for a team."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}&startDate=2026-05-01&endDate={TODAY}&hydrate=linescore"
    try:
        data = fetch(url)
        runs_scored = []
        runs_allowed = []
        for date_obj in data.get("dates", []):
            for g in date_obj.get("games", []):
                if g.get("status", {}).get("statusCode", "") not in ("F", "FR", "FT", "FO"):
                    continue
                ls = g.get("linescore", {})
                teams_ls = ls.get("teams", {})
                if g["teams"]["home"]["team"]["id"] == team_id:
                    rs = teams_ls.get("home", {}).get("runs")
                    ra = teams_ls.get("away", {}).get("runs")
                else:
                    rs = teams_ls.get("away", {}).get("runs")
                    ra = teams_ls.get("home", {}).get("runs")
                if rs is not None:
                    runs_scored.append(rs)
                    runs_allowed.append(ra if ra is not None else 0)
        return runs_scored[-10:], runs_allowed[-10:]
    except:
        return [], []

def format_bar(val, max_val=10, width=20):
    filled = int((val / max_val) * width) if max_val else 0
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)

def main():
    print("=" * 72)
    print(f"  MLB TODAY — {TODAY}  |  OVER/UNDER TOTAL RUNS ANALYSIS")
    print("=" * 72)

    print("\n⏳ Cargando stats de la temporada 2026...")
    hitting, pitching = get_team_season_stats()

    print("⏳ Obteniendo partidos de hoy...")
    games = get_todays_games()

    if not games:
        print("\n❌ No hay partidos programados para hoy o no hay datos disponibles.")
        return

    print(f"\n✅ {len(games)} partido(s) encontrado(s) para hoy\n")

    for i, g in enumerate(games, 1):
        away_team = g["teams"]["away"]["team"]
        home_team = g["teams"]["home"]["team"]
        away_id = away_team["id"]
        home_id = home_team["id"]
        away_name = away_team["name"]
        home_name = home_team["name"]

        game_time = g.get("gameDate", "")
        venue = g.get("venue", {}).get("name", "Unknown")
        status = g.get("status", {}).get("detailedState", "")

        # Probable pitchers
        away_pitcher = g["teams"]["away"].get("probablePitcher", {}).get("fullName", "TBD")
        home_pitcher = g["teams"]["home"].get("probablePitcher", {}).get("fullName", "TBD")

        # Season stats
        ah = hitting.get(away_id, {})
        hh = hitting.get(home_id, {})
        ap = pitching.get(away_id, {})
        hp = pitching.get(home_id, {})

        away_rpg = ah.get("runs_per_game", 0)
        home_rpg = hh.get("runs_per_game", 0)
        away_era = float(ap.get("era", 0))
        home_era = float(hp.get("era", 0))

        # Last 10 games
        away_scored, away_allowed = get_last_10_runs(away_id)
        home_scored, home_allowed = get_last_10_runs(home_id)

        away_l10_avg = sum(away_scored) / len(away_scored) if away_scored else away_rpg
        home_l10_avg = sum(home_scored) / len(home_scored) if home_scored else home_rpg

        # Proyección de carreras
        # Formula: promedio de (ofensiva visitante vs pitcheo local) + (ofensiva local vs pitcheo visitante)
        # Ajuste por ERA: si ERA del pitcher contrario es alta -> más carreras esperadas
        era_factor_away = min(away_era / 4.0, 1.8) if away_era > 0 else 1.0
        era_factor_home = min(home_era / 4.0, 1.8) if home_era > 0 else 1.0

        projected_away_runs = (away_rpg * 0.5 + away_l10_avg * 0.5) * era_factor_home
        projected_home_runs = (home_rpg * 0.5 + home_l10_avg * 0.5) * era_factor_away
        projected_total = projected_away_runs + projected_home_runs

        # Línea típica de over/under
        common_lines = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
        estimated_line = min(common_lines, key=lambda x: abs(x - projected_total))

        confidence = "ALTA" if abs(projected_total - estimated_line) > 1.0 else \
                     "MEDIA" if abs(projected_total - estimated_line) > 0.4 else "BAJA"

        rec = "OVER ⬆️ " if projected_total > estimated_line + 0.3 else \
              "UNDER ⬇️" if projected_total < estimated_line - 0.3 else "PUSH ↔️ "

        print("─" * 72)
        print(f" PARTIDO #{i}  |  {status}")
        print(f" {away_name:30s}  @  {home_name}")
        print(f" 🏟️  {venue}")
        print(f" ⚾ Pitchers: {away_pitcher}  vs  {home_pitcher}")
        print()
        print(f" {'ESTADÍSTICAS':40s}  {'VISITANTE':>14}  {'LOCAL':>10}")
        print(f" {'Carreras/juego (temporada)':40s}  {away_rpg:>14.2f}  {home_rpg:>10.2f}")
        print(f" {'ERA titular (equipo pitcheo)':40s}  {ap.get('era','N/A'):>14}  {hp.get('era','N/A'):>10}")
        print(f" {'WHIP':40s}  {ap.get('whip','N/A'):>14}  {hp.get('whip','N/A'):>10}")
        print(f" {'OPS ofensivo':40s}  {ah.get('ops','N/A'):>14}  {hh.get('ops','N/A'):>10}")
        print(f" {'Últ 10 juegos - promedio carreras':40s}  {away_l10_avg:>14.2f}  {home_l10_avg:>10.2f}")

        if away_scored:
            print(f" Últ 10 ({away_name[:20]}): {away_scored}")
        if home_scored:
            print(f" Últ 10 ({home_name[:20]}): {home_scored}")

        print()
        print(f" 🔮 PROYECCIÓN DE CARRERAS:")
        print(f"    {away_name[:25]:25s}: {projected_away_runs:.2f} carreras esperadas")
        print(f"    {home_name[:25]:25s}: {projected_home_runs:.2f} carreras esperadas")
        print(f"    TOTAL PROYECTADO: {projected_total:.2f} carreras")
        print(f"    Línea estimada:   {estimated_line}")
        print()

        bar = format_bar(projected_total, 14)
        print(f"    {bar} {projected_total:.1f} runs")
        print(f"    {'0':1s}{'─'*9}{'7.5':^4}{'─'*9}{'14':>2}")

        print()
        print(f"  ┌─────────────────────────────────────────────────┐")
        print(f"  │  RECOMENDACIÓN:  {rec}  (Línea ~{estimated_line})            │")
        print(f"  │  Confianza: {confidence}  |  Proyección: {projected_total:.1f} runs         │")
        print(f"  └─────────────────────────────────────────────────┘")
        print()

    print("=" * 72)
    print("  RESUMEN LEAGUE-WIDE TEMPORADA 2026")
    print("=" * 72)
    all_teams = sorted(hitting.items(), key=lambda x: -x[1].get("runs_per_game", 0))
    print(f"\n {'#':3} {'EQUIPO':30} {'R/G':>6} {'OPS':>8} {'ERA':>7} {'WHIP':>6}")
    print(f" {'─'*3} {'─'*30} {'─'*6} {'─'*8} {'─'*7} {'─'*6}")
    for rank, (tid, stats) in enumerate(all_teams, 1):
        p = pitching.get(tid, {})
        print(f" {rank:3}. {stats['name']:30} {stats['runs_per_game']:6.2f} {stats['ops']:>8} {p.get('era','N/A'):>7} {p.get('whip','N/A'):>6}")

    print("\n⚠️  Análisis estadístico — no es consejo financiero.")
    print(f"   Datos: statsapi.mlb.com | Temporada 2026 | {TODAY}")

if __name__ == "__main__":
    main()
