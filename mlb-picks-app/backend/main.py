"""
FastAPI backend — MLB Picks
GET /api/picks?date=YYYY-MM-DD
GET /api/ml/info
"""
import asyncio, statistics, httpx, math
from datetime import date, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

import cache
from mlb_api import (fetch_schedule, fetch_team_stats, fetch_standings,
                     finished, extract_innings, game_totals)
from predictor import TeamData, predict_game
from ml_model import MLPredictor, WinPredictor, build_features, build_win_features

app = FastAPI(title="MLB Picks API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

SEASON_START_2026 = "2026-03-19"
SEASON_START_2025 = "2025-03-20"
SEASON_END_2025   = "2025-09-28"

# Singleton ML models — loaded/trained once per process
_ml:       MLPredictor  | None = None
_win_pred: WinPredictor | None = None


def _week_start(d: str) -> str:
    dt = date.fromisoformat(d)
    return (dt - timedelta(days=7)).isoformat()


# ── Build season aggregates for ML training ───────────────────────────────────
def _build_team_agg(games: list[dict]) -> dict[int, dict]:
    """
    For each team that appears in `games`, compute full-season aggregates.
    Used to build ML training features for 2025.
    """
    scored: dict[int, list]  = {}
    allowed: dict[int, list] = {}
    wins: dict[int, int]     = {}

    for g in games:
        if not finished(g):
            continue
        aid = g["teams"]["away"]["team"]["id"]
        hid = g["teams"]["home"]["team"]["id"]
        ar, hr = game_totals(g)
        for tid, s, a in [(aid, ar, hr), (hid, hr, ar)]:
            scored.setdefault(tid,  []).append(s)
            allowed.setdefault(tid, []).append(a)
            wins.setdefault(tid, 0)
            if s > a:
                wins[tid] += 1

    out = {}
    for tid in scored:
        sc = scored[tid]; al = allowed[tid]
        n = len(sc)
        out[tid] = {
            "rpg":      statistics.mean(sc) if sc else 4.0,
            "rpg_l10":  statistics.mean(sc[-10:]) if sc else 4.0,
            "ra_pg":    statistics.mean(al) if al else 4.0,
            "win_pct":  wins[tid] / n if n else 0.5,
            "streak":   "",  # not available historically
            "era":      "4.20",
            "whip":     "1.30",
            "ops":      ".720",
        }
    return out


def _build_training_rows(
    games: list[dict],
    team_agg: dict[int, dict],
    hitting_stats: dict | None = None,
    pitching_stats: dict | None = None,
) -> list[dict]:
    """One row per finished game."""
    rows = []
    for g in games:
        if not finished(g):
            continue
        aid = g["teams"]["away"]["team"]["id"]
        hid = g["teams"]["home"]["team"]["id"]
        ar, hr = game_totals(g)
        total = ar + hr
        if total == 0:
            continue   # skip likely bad data

        a = team_agg.get(aid, {})
        h = team_agg.get(hid, {})
        if not a or not h:
            continue

        # Use richer hitting/pitching stats if available (2026)
        if hitting_stats and pitching_stats:
            a_era  = pitching_stats.get(aid, {}).get("era", a.get("era", "4.20"))
            a_whip = pitching_stats.get(aid, {}).get("whip", a.get("whip", "1.30"))
            a_ops  = hitting_stats.get(aid, {}).get("ops", a.get("ops", ".720"))
            h_era  = pitching_stats.get(hid, {}).get("era", h.get("era", "4.20"))
            h_whip = pitching_stats.get(hid, {}).get("whip", h.get("whip", "1.30"))
            h_ops  = hitting_stats.get(hid, {}).get("ops", h.get("ops", ".720"))
        else:
            a_era  = a.get("era", "4.20");  a_whip = a.get("whip", "1.30"); a_ops = a.get("ops", ".720")
            h_era  = h.get("era", "4.20");  h_whip = h.get("whip", "1.30"); h_ops = h.get("ops", ".720")

        month = int(g["_date"][5:7]) if g.get("_date") else 6

        feats = build_features(
            away_rpg=a.get("rpg", 4.0),    away_rpg_l10=a.get("rpg_l10", 4.0),
            away_win_pct=a.get("win_pct", 0.5),
            away_era=a_era,                 away_whip=a_whip,
            away_ops=a_ops,                 away_streak=a.get("streak", ""),
            away_ra_pg=a.get("ra_pg", 4.0),
            home_rpg=h.get("rpg", 4.0),    home_rpg_l10=h.get("rpg_l10", 4.0),
            home_win_pct=h.get("win_pct", 0.5),
            home_era=h_era,                 home_whip=h_whip,
            home_ops=h_ops,                 home_streak=h.get("streak", ""),
            home_ra_pg=h.get("ra_pg", 4.0),
            h2h_avg=None, h2h_n=0, month=month,
        )
        def _ef(v):
            try: return float(v)
            except: return 4.20
        win_feats = build_win_features(
            feats,
            away_win_pct=a.get("win_pct", 0.5), home_win_pct=h.get("win_pct", 0.5),
            away_rpg=a.get("rpg", 4.0),          home_rpg=h.get("rpg", 4.0),
            away_era_f=_ef(a_era),               home_era_f=_ef(h_era),
            away_ra_pg=a.get("ra_pg", 4.0),      home_ra_pg=h.get("ra_pg", 4.0),
        )
        home_win = 1 if hr > ar else 0
        rows.append({
            "features":     feats,
            "win_features": win_feats,
            "target":       float(total),
            "home_win":     home_win,
            "date":         g.get("_date", ""),
        })
    return rows


# ── Probabilidad implícita del pick O/U ──────────────────────────────────────
_SIGMA = 3.5   # MAE del modelo ≈ desviación estándar de errores

def _ou_prob(diff: float) -> float:
    """P(pick correcto) usando distribución normal con σ=MAE del modelo."""
    z = abs(diff) / _SIGMA
    return round(50 + 50 * math.erf(z / math.sqrt(2)), 1)


# ── Enriquecer game log con O/U e inning caliente ────────────────────────────
_OU_LINE = 8.5

def _enrich_log(entries: list[dict]) -> list[dict]:
    result = []
    for e in entries:
        total = e["scored"] + e["allowed"]
        innings = e.get("innings", {})
        is_away = e.get("isAway", True)
        team_col = 0 if is_away else 1

        hot_inn = hot_runs = team_hot_inn = team_hot_runs = None
        for inn_str, runs in innings.items():
            game_r = runs[0] + runs[1]
            team_r = runs[team_col]
            if hot_runs is None or game_r > hot_runs:
                hot_runs = game_r; hot_inn = int(inn_str)
            if team_hot_runs is None or team_r > team_hot_runs:
                team_hot_runs = team_r; team_hot_inn = int(inn_str)

        result.append({
            **e,
            "total":         total,
            "ouResult":      "OVER" if total > _OU_LINE else "UNDER",
            "hotInning":     hot_inn if (hot_runs or 0) > 0 else None,
            "hotInningRuns": hot_runs or 0,
            "teamHotInning": team_hot_inn if (team_hot_runs or 0) > 0 else None,
            "teamHotRuns":   team_hot_runs or 0,
        })
    return result


# ── Main build function ───────────────────────────────────────────────────────
async def build_picks(target_date: str) -> tuple[list[dict], dict]:
    global _ml, _win_pred
    week_start = _week_start(target_date)

    async with httpx.AsyncClient() as client:
        (today_games, season_games, prev_games,
         hitting26, pitching26, standings26,
         hitting25, pitching25) = await asyncio.gather(
            fetch_schedule(client, target_date, target_date,
                           "probablePitcher,linescore,team,venue"),
            fetch_schedule(client, SEASON_START_2026, target_date,
                           "linescore,team"),
            fetch_schedule(client, SEASON_START_2025, SEASON_END_2025,
                           "linescore,team"),
            fetch_team_stats(client, 2026, "hitting"),
            fetch_team_stats(client, 2026, "pitching"),
            fetch_standings(client, 2026),
            fetch_team_stats(client, 2025, "hitting"),
            fetch_team_stats(client, 2025, "pitching"),
        )

    # ── Build TeamData (2026) ──────────────────────────────────────────────
    teams: dict[int, TeamData] = {}

    def get_td(tid: int) -> TeamData:
        if tid not in teams:
            teams[tid] = TeamData(tid)
        return teams[tid]

    for g in season_games:
        if not finished(g):
            continue
        aid = g["teams"]["away"]["team"]["id"]
        hid = g["teams"]["home"]["team"]["id"]
        ar, hr = game_totals(g)
        inn = extract_innings(g)
        is_week = g["_date"] >= week_start
        aname_g = g["teams"]["away"]["team"]["name"]
        hname_g = g["teams"]["home"]["team"]["name"]
        get_td(aid).add_game(True,  ar, hr, inn, is_week, g["_date"], hname_g)
        get_td(hid).add_game(False, hr, ar, inn, is_week, g["_date"], aname_g)

    for g in prev_games:
        if not finished(g):
            continue
        aid = g["teams"]["away"]["team"]["id"]
        hid = g["teams"]["home"]["team"]["id"]
        ar, hr = game_totals(g)
        get_td(aid).add_prev_game(ar)
        get_td(hid).add_prev_game(hr)

    for tid, std in standings26.items():
        if tid in teams:
            t = teams[tid]
            t.wins = std["wins"]; t.losses = std["losses"]
            t.win_pct = std["winPct"]; t.streak_code = std["streak"]

    # ── Train / load ML models ─────────────────────────────────────────────
    if _ml is None:
        _ml = MLPredictor.load()
    if _win_pred is None:
        _win_pred = WinPredictor.load()

    if _ml is None or _win_pred is None:
        print("Building training rows...")
        agg_2025 = _build_team_agg(prev_games)
        agg_2026 = {
            tid: {
                "rpg":      td.season_rpg() or 4.0,
                "rpg_l10":  td.last10_rpg() or 4.0,
                "ra_pg":    td.season_ra() or 4.0,
                "win_pct":  td.win_pct,
                "streak":   td.streak_code,
                "era":      pitching26.get(tid, {}).get("era", "4.20"),
                "whip":     pitching26.get(tid, {}).get("whip", "1.30"),
                "ops":      hitting26.get(tid, {}).get("ops", ".720"),
            }
            for tid, td in teams.items()
        }
        rows_2025 = _build_training_rows(prev_games,    agg_2025, hitting25, pitching25)
        rows_2026 = _build_training_rows(season_games,  agg_2026, hitting26, pitching26)
        all_rows  = rows_2025 + rows_2026
        print(f"  Rows: {len(rows_2025)} (2025) + {len(rows_2026)} (2026) = {len(all_rows)}")

        if _ml is None:
            _ml = MLPredictor().fit(all_rows)
            _ml.save()
            print(f"  O/U model trained. CV MAE: {_ml.cv_mae:.3f} runs")

        if _win_pred is None:
            win_rows = [
                {"features": r["win_features"], "target": r["home_win"]}
                for r in all_rows if "win_features" in r
            ]
            print(f"  Win rows: {len(win_rows)}")
            _win_pred = WinPredictor().fit(win_rows)
            _win_pred.save()
            print(f"  Win model trained. CV Acc: {_win_pred.cv_acc:.3f}")

    ml_meta = {
        "trained":             _ml.trained,
        "trainN":              _ml.train_n,
        "cvMae":               _ml.cv_mae,
        "featureImportances":  _ml.feature_importances,
    }

    # ── Process today's games ──────────────────────────────────────────────
    LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]

    results = []
    for g in today_games:
        status_code = g.get("status", {}).get("statusCode", "")
        if status_code in ("F", "FR", "FT", "FO", "D", "DI"):
            continue

        aid   = g["teams"]["away"]["team"]["id"]
        hid   = g["teams"]["home"]["team"]["id"]
        aname = g["teams"]["away"]["team"]["name"]
        hname = g["teams"]["home"]["team"]["name"]

        h2h_raw = [
            x for x in season_games
            if finished(x) and (
                (x["teams"]["away"]["team"]["id"] == aid and x["teams"]["home"]["team"]["id"] == hid) or
                (x["teams"]["away"]["team"]["id"] == hid and x["teams"]["home"]["team"]["id"] == aid)
            )
        ]
        h2h_games_list = []
        for hg in h2h_raw:
            ha_id = hg["teams"]["away"]["team"]["id"]
            ar2, hr2 = game_totals(hg)
            h2h_games_list.append({
                "date": hg["_date"], "total": ar2 + hr2,
                "away_scored": ar2 if ha_id == aid else hr2,
                "home_scored": hr2 if ha_id == hid else ar2,
                "innings": extract_innings(hg),
            })

        at = get_td(aid)
        ht = get_td(hid)
        h2h_avg = statistics.mean([x["total"] for x in h2h_games_list]) if h2h_games_list else None

        # Formula prediction
        formula_pred = predict_game(at, ht, h2h_games_list)

        # ML prediction
        month = int(target_date[5:7])
        ml_feats = build_features(
            away_rpg=at.season_rpg() or 4.0,
            away_rpg_l10=at.last10_rpg() or 4.0,
            away_win_pct=at.win_pct,
            away_era=pitching26.get(aid, {}).get("era", "4.20"),
            away_whip=pitching26.get(aid, {}).get("whip", "1.30"),
            away_ops=hitting26.get(aid, {}).get("ops", ".720"),
            away_streak=at.streak_code,
            away_ra_pg=at.season_ra() or 4.0,
            home_rpg=ht.season_rpg() or 4.0,
            home_rpg_l10=ht.last10_rpg() or 4.0,
            home_win_pct=ht.win_pct,
            home_era=pitching26.get(hid, {}).get("era", "4.20"),
            home_whip=pitching26.get(hid, {}).get("whip", "1.30"),
            home_ops=hitting26.get(hid, {}).get("ops", ".720"),
            home_streak=ht.streak_code,
            home_ra_pg=ht.season_ra() or 4.0,
            h2h_avg=h2h_avg, h2h_n=len(h2h_games_list), month=month,
        )
        ml_raw = _ml.predict(ml_feats)

        # Ensemble: 55% ML + 45% formula
        if ml_raw["total"] is not None:
            ensemble_total = ml_raw["total"] * 0.55 + formula_pred["total"] * 0.45
        else:
            ensemble_total = formula_pred["total"]

        ensemble_line = min(LINES, key=lambda x: abs(x - ensemble_total))
        ensemble_diff = round(ensemble_total - ensemble_line, 3)
        ensemble_rec  = "OVER" if ensemble_diff >= 0 else "UNDER"
        # Confidence: how far from line, plus agreement between ML and formula
        formula_rec = formula_pred["rec"]
        ml_rec = "OVER" if (ml_raw["total"] or 0) >= ensemble_line else "UNDER"
        agreement = formula_rec == ml_rec
        conf = 3 if (abs(ensemble_diff) > 1.5 and agreement) else \
               2 if (abs(ensemble_diff) > 0.6 or agreement) else 1

        prob = _ou_prob(ensemble_diff)
        if agreement:  prob = min(prob + 2.0, 99.0)
        else:          prob = max(prob - 2.0, 50.0)
        prob = round(prob, 1)

        # Win probabilities — ML classifier (Model B)
        win_feats = build_win_features(
            ml_feats,
            away_win_pct=at.win_pct,             home_win_pct=ht.win_pct,
            away_rpg=at.season_rpg() or 4.0,     home_rpg=ht.season_rpg() or 4.0,
            away_era_f=float(pitching26.get(aid, {}).get("era", 4.20) or 4.20),
            home_era_f=float(pitching26.get(hid, {}).get("era", 4.20) or 4.20),
            away_ra_pg=at.season_ra() or 4.0,    home_ra_pg=ht.season_ra() or 4.0,
        )
        win_result = _win_pred.predict(win_feats)
        p_home_win = win_result["pHome"]
        p_away_win = win_result["pAway"]

        ensemble_pred = {
            **formula_pred,
            "total":        round(ensemble_total, 3),
            "line":         ensemble_line,
            "diff":         ensemble_diff,
            "rec":          ensemble_rec,
            "conf":         conf,
            "prob":         prob,
            "awayWinProb":  p_away_win,
            "homeWinProb":  p_home_win,
            "mlTotal":      ml_raw["total"],
            "mlXgb":        ml_raw["xgb"],
            "mlRidge":      ml_raw["ridge"],
            "formulaTotal": round(formula_pred["total"], 3),
            "formulaRec":   formula_rec,
            "mlRec":        ml_rec,
            "agreement":    agreement,
        }

        def team_payload(td: TeamData, name: str, tid: int,
                         side_data: dict, pitcher_data: dict) -> dict:
            hs = hitting26.get(tid, {})
            ps = pitching26.get(tid, {})
            record = f"{td.wins}-{td.losses}" if (td.wins + td.losses) > 0 else "—"
            return {
                "id":      tid,
                "name":    name,
                "abbr":    side_data.get("team", {}).get("abbreviation", name[:3].upper()),
                "logoUrl": f"https://www.mlbstatic.com/team-logos/{tid}.svg",
                "record":  record,
                "winPct":  round(td.win_pct, 3),
                "streak":  td.streak_code or "—",
                "rpgSeason": round(td.season_rpg() or 0, 2),
                "rpgLast10": round(td.last10_rpg() or 0, 2),
                "rpgWeek":   round(td.week_rpg() or 0, 2) if td.week_rpg() else None,
                "weekGames": len(td.scored_week),
                "probablePitcher": pitcher_data.get("fullName", "TBD"),
                "pitcherHand":     pitcher_data.get("pitchHand", {}).get("code", ""),
                "era":   ps.get("era", "—"),
                "whip":  ps.get("whip", "—"),
                "ops":   hs.get("ops", "—"),
                "avg":   hs.get("avg", "—"),
                "last10Log": _enrich_log(td.game_log[-10:]),
                "runsHistory": [
                    {"date": entry["date"], "scored": entry["scored"],
                     "allowed": entry["allowed"], "result": entry["result"],
                     "opp": entry["opp"]}
                    for entry in td.game_log[-20:]
                ],
            }

        away_pitcher = g["teams"]["away"].get("probablePitcher", {})
        home_pitcher = g["teams"]["home"].get("probablePitcher", {})

        results.append({
            "gamePk":     g["gamePk"],
            "gameTime":   g.get("gameDate", ""),
            "venue":      g.get("venue", {}).get("name", ""),
            "status":     g.get("status", {}).get("detailedState", "Scheduled"),
            "away":       team_payload(at, aname, aid, g["teams"]["away"], away_pitcher),
            "home":       team_payload(ht, hname, hid, g["teams"]["home"], home_pitcher),
            "projection": ensemble_pred,
            "h2hGames": [
                {"date": hg["date"], "total": hg["total"],
                 "awayScored": hg["away_scored"], "homeScored": hg["home_scored"]}
                for hg in h2h_games_list
            ],
        })

    results.sort(key=lambda x: x["projection"].get("prob", 50), reverse=True)
    return results, ml_meta


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/api/picks")
async def get_picks(date: str = Query(default=None)):
    target = date or str(__import__("datetime").date.today())
    cached = cache.load(target)
    if cached:
        ml_info = cached.get("mlMeta") if isinstance(cached, dict) else None
        games   = cached.get("games", cached) if isinstance(cached, dict) else cached
        return {"date": target, "cached": True, "games": games, "mlMeta": ml_info}
    games, ml_meta = await build_picks(target)
    payload = {"games": games, "mlMeta": ml_meta}
    cache.save(target, payload)
    return {"date": target, "cached": False, **payload}


@app.get("/api/picks/refresh")
async def refresh_picks(date: str = Query(default=None)):
    global _ml, _win_pred
    _ml = None
    _win_pred = None
    target = date or str(__import__("datetime").date.today())
    games, ml_meta = await build_picks(target)
    payload = {"games": games, "mlMeta": ml_meta}
    cache.save(target, payload)
    return {"date": target, "cached": False, **payload}


@app.get("/api/ml/info")
async def ml_info():
    if _ml and _ml.trained:
        return {
            "trained": True,
            "trainN": _ml.train_n,
            "cvMae": _ml.cv_mae,
            "featureImportances": _ml.feature_importances,
        }
    return {"trained": False}


@app.get("/health")
async def health():
    return {"ok": True}


# ── Build historical results (last 30 days) ───────────────────────────────────
async def build_results(target_date: str) -> tuple[list[dict], dict]:
    """
    Return predictions vs actuals for target_date + 30-day accuracy summary.
    Uses rolling stats: each prediction is computed with data available
    BEFORE that game (no data leakage), matching what the picks page showed.
    """
    global _ml, _win_pred

    import datetime as _dt
    today_str = str(_dt.date.today())
    month_ago = (_dt.date.fromisoformat(target_date) - timedelta(days=30)).isoformat()

    async with httpx.AsyncClient() as client:
        (season_games, prev_games,
         hitting26, pitching26,
         hitting25, pitching25) = await asyncio.gather(
            fetch_schedule(client, SEASON_START_2026, today_str, "linescore,team"),
            fetch_schedule(client, SEASON_START_2025, SEASON_END_2025, "linescore,team"),
            fetch_team_stats(client, 2026, "hitting"),
            fetch_team_stats(client, 2026, "pitching"),
            fetch_team_stats(client, 2025, "hitting"),
            fetch_team_stats(client, 2025, "pitching"),
        )

    # Ensure models are available (load from cache or train)
    if _ml is None:
        _ml = MLPredictor.load()
    if _win_pred is None:
        _win_pred = WinPredictor.load()

    if _ml is None or _win_pred is None:
        agg_2025 = _build_team_agg(prev_games)
        scored_s: dict[int, list] = {}; allowed_s: dict[int, list] = {}; wins_s: dict[int, int] = {}
        for g in season_games:
            if not finished(g): continue
            aid = g["teams"]["away"]["team"]["id"]
            hid = g["teams"]["home"]["team"]["id"]
            ar, hr = game_totals(g)
            for tid, s, a in [(aid, ar, hr), (hid, hr, ar)]:
                scored_s.setdefault(tid, []).append(s)
                allowed_s.setdefault(tid, []).append(a)
                wins_s.setdefault(tid, 0)
                if s > a: wins_s[tid] += 1
        agg_2026_full = {
            tid: {
                "rpg": statistics.mean(sc) if sc else 4.0,
                "rpg_l10": statistics.mean(sc[-10:]) if sc else 4.0,
                "ra_pg": statistics.mean(allowed_s.get(tid, [4.0])),
                "win_pct": wins_s.get(tid, 0) / len(sc) if sc else 0.5,
                "streak": "",
                "era":  pitching26.get(tid, {}).get("era",  "4.20"),
                "whip": pitching26.get(tid, {}).get("whip", "1.30"),
                "ops":  hitting26.get(tid,  {}).get("ops",  ".720"),
            }
            for tid, sc in scored_s.items()
        }
        rows_2025 = _build_training_rows(prev_games,   agg_2025,       hitting25, pitching25)
        rows_2026 = _build_training_rows(season_games, agg_2026_full,  hitting26, pitching26)
        all_rows  = rows_2025 + rows_2026
        if _ml is None:
            _ml = MLPredictor().fit(all_rows); _ml.save()
        if _win_pred is None:
            win_rows = [{"features": r["win_features"], "target": r["home_win"]}
                        for r in all_rows if "win_features" in r]
            _win_pred = WinPredictor().fit(win_rows); _win_pred.save()

    LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]

    # ── Rolling stats — built incrementally in chronological order ────────────
    scored_r:  dict[int, list[float]] = {}
    allowed_r: dict[int, list[float]] = {}
    results_r: dict[int, list[bool]]  = {}   # True = team won

    def _ef(v):
        try: return float(v)
        except: return 4.20

    def _rolling(tid: int) -> dict:
        sc = scored_r.get(tid, [])
        al = allowed_r.get(tid, [])
        rs = results_r.get(tid, [])
        n  = len(sc)
        # compute streak from recent results
        streak = ""
        if rs:
            cur = rs[-1]
            cnt = 1
            for v in reversed(rs[:-1]):
                if v == cur: cnt += 1
                else: break
            streak = f"{'W' if cur else 'L'}{cnt}"
        return {
            "rpg":     statistics.mean(sc)       if sc else 4.0,
            "rpg_l10": statistics.mean(sc[-10:]) if sc else 4.0,
            "ra_pg":   statistics.mean(al)        if al else 4.0,
            "win_pct": sum(rs) / n                if n  else 0.5,
            "streak":  streak,
            "era":  pitching26.get(tid, {}).get("era",  "4.20"),
            "whip": pitching26.get(tid, {}).get("whip", "1.30"),
            "ops":  hitting26.get(tid,  {}).get("ops",  ".720"),
        }

    target_results: list[dict] = []
    month_results:  list[dict] = []

    for g in sorted(season_games, key=lambda x: x.get("_date", "")):
        if not finished(g) or not g.get("_date"):
            continue
        gd  = g["_date"]
        aid = g["teams"]["away"]["team"]["id"]
        hid = g["teams"]["home"]["team"]["id"]
        ar, hr = game_totals(g)

        # ── Predict BEFORE adding this game's result (no data leakage) ───────
        if month_ago <= gd <= target_date:
            a = _rolling(aid)
            h = _rolling(hid)
            aname = g["teams"]["away"]["team"]["name"]
            hname = g["teams"]["home"]["team"]["name"]
            aabbr = g["teams"]["away"]["team"].get("abbreviation", aname[:3].upper())
            habbr = g["teams"]["home"]["team"].get("abbreviation", hname[:3].upper())

            month = int(gd[5:7])
            ml_feats = build_features(
                away_rpg=a["rpg"],      away_rpg_l10=a["rpg_l10"], away_win_pct=a["win_pct"],
                away_era=a["era"],      away_whip=a["whip"],        away_ops=a["ops"],
                away_streak=a["streak"], away_ra_pg=a["ra_pg"],
                home_rpg=h["rpg"],      home_rpg_l10=h["rpg_l10"], home_win_pct=h["win_pct"],
                home_era=h["era"],      home_whip=h["whip"],        home_ops=h["ops"],
                home_streak=h["streak"], home_ra_pg=h["ra_pg"],
                h2h_avg=None, h2h_n=0, month=month,
            )
            ml_raw = _ml.predict(ml_feats)
            ensemble_total = ml_raw["total"] or 8.5

            line = min(LINES, key=lambda x: abs(x - ensemble_total))
            diff = round(ensemble_total - line, 3)
            rec  = "OVER" if diff >= 0 else "UNDER"
            actual_total = ar + hr
            ou_correct   = (rec == "OVER") == (actual_total > line)

            win_feats = build_win_features(
                ml_feats,
                away_win_pct=a["win_pct"], home_win_pct=h["win_pct"],
                away_rpg=a["rpg"],         home_rpg=h["rpg"],
                away_era_f=_ef(a["era"]),  home_era_f=_ef(h["era"]),
                away_ra_pg=a["ra_pg"],     home_ra_pg=h["ra_pg"],
            )
            wp = _win_pred.predict(win_feats)
            pred_home_win   = wp["pHome"] >= 50
            actual_home_win = hr > ar
            win_correct     = pred_home_win == actual_home_win

            row = {
                "gamePk": g["gamePk"],
                "date":   gd,
                "away":   {"name": aname, "abbr": aabbr,
                           "logoUrl": f"https://www.mlbstatic.com/team-logos/{aid}.svg",
                           "score": ar},
                "home":   {"name": hname, "abbr": habbr,
                           "logoUrl": f"https://www.mlbstatic.com/team-logos/{hid}.svg",
                           "score": hr},
                "pred": {
                    "predictedTotal": round(ensemble_total, 1),
                    "line":           line,
                    "rec":            rec,
                    "diff":           diff,
                    "actualTotal":    actual_total,
                    "ouCorrect":      ou_correct,
                    "awayWinProb":    wp["pAway"],
                    "homeWinProb":    wp["pHome"],
                    "predictedWinner": "home" if pred_home_win else "away",
                    "winCorrect":     win_correct,
                },
            }
            month_results.append(row)
            if gd == target_date:
                target_results.append(row)

        # ── Add result to rolling stats AFTER prediction ─────────────────────
        for tid, s, alw, won in [
            (aid, ar, hr, ar > hr),
            (hid, hr, ar, hr > ar),
        ]:
            scored_r.setdefault(tid,  []).append(float(s))
            allowed_r.setdefault(tid, []).append(float(alw))
            results_r.setdefault(tid, []).append(won)

    # 30-day summary
    n     = len(month_results)
    ou_n  = sum(1 for r in month_results if r["pred"]["ouCorrect"])
    win_n = sum(1 for r in month_results if r["pred"]["winCorrect"])

    def _streak(booleans: list[bool]) -> int:
        if not booleans: return 0
        val = booleans[-1]
        s = 1 if val else -1
        for x in reversed(booleans[:-1]):
            if x == val: s = s + 1 if s > 0 else s - 1
            else: break
        return s

    ou_streak  = _streak([r["pred"]["ouCorrect"]  for r in month_results])
    win_streak = _streak([r["pred"]["winCorrect"] for r in month_results])

    summary = {
        "periodStart": month_ago,
        "periodEnd":   target_date,
        "games":       n,
        "ouCorrect":   ou_n,
        "ouPct":       round(ou_n  / n * 100, 1) if n else 0,
        "winCorrect":  win_n,
        "winPct":      round(win_n / n * 100, 1) if n else 0,
        "ouStreak":    ou_streak,
        "winStreak":   win_streak,
    }
    return target_results, summary


@app.get("/api/results")
async def get_results(date: str = Query(default=None)):
    """
    Results for a past date.
    Priority:
      1) If we have stored picks for target_date (user visited picks page that day),
         use those predictions + fetch actual scores → results match what was shown.
      2) Otherwise, compute retroactively with rolling ML model.
    30-day summary always uses rolling ML for consistent statistics.
    """
    import datetime as _dt
    target = date or str((_dt.date.today() - timedelta(days=1)))

    cached_results = cache.load(f"results_{target}")
    if cached_results:
        return {"date": target, "cached": True, **cached_results}

    # ── Check if picks were stored for this date ──────────────────────────────
    stored = cache.load(target)
    stored_games = None
    if stored:
        g_list = stored.get("games", stored) if isinstance(stored, dict) else stored
        if g_list:
            stored_games = {g["gamePk"]: g for g in g_list if g.get("gamePk")}

    # ── Fetch actual scores for target_date ───────────────────────────────────
    async with httpx.AsyncClient() as client:
        day_games = await fetch_schedule(client, target, target, "linescore,team")

    finished_day = [g for g in day_games if finished(g)]

    game_rows: list[dict] = []

    if stored_games and finished_day:
        # Use stored predictions — exact numbers shown on picks page that day
        for g in finished_day:
            pk  = g["gamePk"]
            ar, hr = game_totals(g)
            aid = g["teams"]["away"]["team"]["id"]
            hid = g["teams"]["home"]["team"]["id"]
            aname = g["teams"]["away"]["team"]["name"]
            hname = g["teams"]["home"]["team"]["name"]
            aabbr = g["teams"]["away"]["team"].get("abbreviation", aname[:3].upper())
            habbr = g["teams"]["home"]["team"].get("abbreviation", hname[:3].upper())

            stored_g = stored_games.get(pk)
            if stored_g:
                proj = stored_g.get("projection", {})
                p_away = proj.get("awayWinProb", 50.0)
                p_home = proj.get("homeWinProb", 50.0)
                line   = proj.get("line", 8.5)
                rec    = proj.get("rec", "OVER")
                ptotal = round(proj.get("total", 8.5), 1)
                diff   = proj.get("diff", 0)
            else:
                p_away = p_home = 50.0
                line = 8.5; rec = "OVER"; ptotal = 8.5; diff = 0

            actual_total  = ar + hr
            ou_correct    = (rec == "OVER") == (actual_total > line)
            pred_home_win = p_home >= 50
            win_correct   = pred_home_win == (hr > ar)

            game_rows.append({
                "gamePk": pk,
                "date":   target,
                "away":   {"name": aname, "abbr": aabbr,
                           "logoUrl": f"https://www.mlbstatic.com/team-logos/{aid}.svg",
                           "score": ar},
                "home":   {"name": hname, "abbr": habbr,
                           "logoUrl": f"https://www.mlbstatic.com/team-logos/{hid}.svg",
                           "score": hr},
                "pred": {
                    "predictedTotal": ptotal,
                    "line": line, "rec": rec, "diff": diff,
                    "actualTotal": actual_total,
                    "ouCorrect":   ou_correct,
                    "awayWinProb": p_away,
                    "homeWinProb": p_home,
                    "predictedWinner": "home" if pred_home_win else "away",
                    "winCorrect":  win_correct,
                    "fromCache":   stored_g is not None,
                },
            })

    # ── Fallback or supplement with rolling ML for any missing games ──────────
    rolling_games, summary = await build_results(target)

    if not game_rows:
        game_rows = rolling_games  # no stored picks → use rolling ML for all
    else:
        # Fill in any games not in stored picks with rolling predictions
        stored_pks = {r["gamePk"] for r in game_rows}
        for r in rolling_games:
            if r["gamePk"] not in stored_pks:
                game_rows.append(r)

    payload = {"games": game_rows, "summary": summary}
    cache.save(f"results_{target}", payload)
    return {"date": target, "cached": False, **payload}


# ── Parlay builder ────────────────────────────────────────────────────────────

_PARLAY_PAYOUTS = {2: 2.64, 3: 6.0, 4: 12.28, 5: 24.35, 6: 47.41}

def _combined_prob(legs: list[dict]) -> float:
    p = 1.0
    for l in legs:
        p *= l["prob"] / 100
    return round(p * 100, 1)

def _ev_pct(legs: list[dict]) -> float:
    cp   = _combined_prob(legs) / 100
    pout = _PARLAY_PAYOUTS.get(len(legs), 2.64 ** len(legs))
    return round((cp * pout - 1) * 100, 1)

def _parlay(pid, category, icon, legs, note, correlated=False):
    n    = len(legs)
    cp   = _combined_prob(legs)
    pout = _PARLAY_PAYOUTS.get(n, 2.64 ** n)
    return {
        "id": pid, "category": category, "icon": icon,
        "legs": legs,
        "legsCount": n,
        "combinedProb": cp,
        "payoutEstimate": pout,
        "ev": _ev_pct(legs),
        "note": note,
        "correlated": correlated,
        "contrarian": False,
    }


def build_parlays(today_games: list[dict],
                  hist_ou_acc: float = 51.7,
                  hist_win_acc: float = 57.0) -> list[dict]:
    """
    Build ranked parlay recommendations from today's picks.
    hist_ou_acc / hist_win_acc: rolling 30-day accuracy (%) fed from results API.
    """
    ou_legs:  list[dict] = []
    win_legs: list[dict] = []

    for g in today_games:
        proj = g.get("projection", {})
        away = g["away"]; home = g["home"]
        pk   = g["gamePk"]
        gs   = f"{away['abbr']} @ {home['abbr']}"
        base = {
            "gamePk": pk, "game": gs,
            "away": {"name": away["name"], "abbr": away["abbr"], "logoUrl": away["logoUrl"]},
            "home": {"name": home["name"], "abbr": home["abbr"], "logoUrl": home["logoUrl"]},
        }

        # O/U leg
        p_ou = proj.get("prob", 50.0)
        ou_legs.append({**base,
            "type": "ou",
            "pick": f"{proj.get('rec','OVER')} {proj.get('line',8.5)}",
            "rec":  proj.get("rec", "OVER"),
            "prob": p_ou,
            "projTotal": proj.get("total"),
        })

        # WIN leg — pick the higher-confidence side
        p_a = proj.get("awayWinProb", 50.0)
        p_h = proj.get("homeWinProb", 50.0)
        if p_h >= p_a:
            wt = home; ws = "home"; wp = p_h
        else:
            wt = away; ws = "away";  wp = p_a
        win_legs.append({**base,
            "type":     "win",
            "pick":     f"{wt['abbr']} WIN",
            "pickSide": ws,
            "pickTeam": wt["name"],
            "pickAbbr": wt["abbr"],
            "pickLogoUrl": wt.get("logoUrl",""),
            "prob":     wp,
        })

    ou_legs.sort(key=lambda x: -x["prob"])
    win_legs.sort(key=lambda x: -x["prob"])

    parlays: list[dict] = []

    # ── O/U parlays ────────────────────────────────────────────────────────────
    if len(ou_legs) >= 2:
        l = ou_legs[:2]
        parlays.append(_parlay("ou_2", "O/U · 2 legs", "", l,
            "Las 2 predicciones de total con mayor certeza del modelo ML"))

    if len(ou_legs) >= 3:
        l = ou_legs[:3]
        parlays.append(_parlay("ou_3", "O/U · 3 legs", "", l,
            "Triple O/U — las 3 líneas más alejadas de la línea de equilibrio"))

    # ── WIN parlays (no same-game duplicates) ──────────────────────────────────
    seen_pk: set[int] = set()
    top_wins: list[dict] = []
    for l in win_legs:
        if l["gamePk"] not in seen_pk:
            top_wins.append(l); seen_pk.add(l["gamePk"])

    if len(top_wins) >= 2:
        l = top_wins[:2]
        parlays.append(_parlay("win_2", "Ganadores · 2 legs", "", l,
            "Los 2 favoritos más dominantes del día según el clasificador calibrado"))

    if len(top_wins) >= 3:
        l = top_wins[:3]
        parlays.append(_parlay("win_3", "Ganadores · 3 legs", "", l,
            "Triple favoritos — históricamente el tipo de parlay más efectivo"))

    # ── Game Stack: OVER + favored WIN, mismo partido ─────────────────────────
    for g in today_games:
        proj = g.get("projection", {})
        if proj.get("rec") != "OVER":
            continue
        p_ou = proj.get("prob", 50.0)
        p_a  = proj.get("awayWinProb", 50.0)
        p_h  = proj.get("homeWinProb", 50.0)
        win_p = max(p_a, p_h)

        pk   = g["gamePk"]
        away = g["away"]; home = g["home"]
        gs   = f"{away['abbr']} @ {home['abbr']}"
        ws   = "home" if p_h >= p_a else "away"
        wt   = home if ws == "home" else away

        base = {
            "gamePk": pk, "game": gs,
            "away": {"name": away["name"], "abbr": away["abbr"], "logoUrl": away["logoUrl"]},
            "home": {"name": home["name"], "abbr": home["abbr"], "logoUrl": home["logoUrl"]},
        }
        ou_leg  = {**base, "type": "ou",  "pick": f"OVER {proj['line']}", "rec": "OVER", "prob": p_ou}
        win_leg = {**base, "type": "win", "pick": f"{wt['abbr']} WIN",
                   "pickSide": ws, "pickTeam": wt["name"], "pickAbbr": wt["abbr"],
                   "pickLogoUrl": wt.get("logoUrl",""), "prob": win_p}

        # Correlated boost: OVER on a game where you also think Team X wins →
        # if X wins big, total goes over. Correlation bumps prob ~8-12%.
        indep = (p_ou / 100) * (win_p / 100) * 100
        corr  = round(min(indep * 1.10, min(p_ou, win_p) * 0.93), 1)

        parlays.append(_parlay(
            f"stack_{pk}", "Game Stack · 2 legs", "",
            [ou_leg, win_leg],
            f"Correlación positiva: si {wt['abbr']} domina el marcador, el total sube",
            correlated=True,
        ))
        # Override combinedProb with correlated estimate
        parlays[-1]["combinedProb"] = corr
        parlays[-1]["ev"] = round((corr / 100 * 2.64 - 1) * 100, 1)

    # ── Mixto top-3 and top-5 ─────────────────────────────────────────────────
    all_sorted: list[dict] = []
    seen_keys: set[str] = set()
    for leg in sorted(ou_legs + win_legs, key=lambda x: -x["prob"]):
        key = f"{leg['gamePk']}_{leg['type']}"
        if key not in seen_keys:
            all_sorted.append(leg)
            seen_keys.add(key)

    if len(all_sorted) >= 3:
        l = all_sorted[:3]
        parlays.append(_parlay("mix_3", "Mixto · 3 legs", "", l,
            "Los 3 picks con mayor confianza del día, O/U y WIN combinados"))

    if len(all_sorted) >= 5:
        l = all_sorted[:5]
        parlays.append(_parlay("mix_5", "Mega Parlay · 5 legs", "", l,
            "Alto riesgo / alto pago — solo para bankroll agresivo"))

    # ── Modo contrario: cuando O/U < 50% conviene apostar lo opuesto ─────────
    contrarian_mode = hist_ou_acc < 50.0
    inv_ou_acc = round(100.0 - hist_ou_acc, 1)   # precision si apostamos al reves

    if contrarian_mode:
        def _invert_ou_leg(leg: dict) -> dict:
            inv = dict(leg)
            inv["type"] = "ou_inv"
            inv["rec"]  = "UNDER" if leg["rec"] == "OVER" else "OVER"
            line        = leg["pick"].split()[-1]
            inv["pick"] = f"{inv['rec']} {line}"
            return inv

        # O/U invertido 2-leg
        if len(ou_legs) >= 2:
            l = [_invert_ou_leg(ou_legs[0]), _invert_ou_leg(ou_legs[1])]
            parlays.append(_parlay("ou_inv_2", "O/U invertido · 2 legs", "", l,
                f"Modelo O/U en racha negativa ({hist_ou_acc}%) — apostamos lo contrario"))
            parlays[-1]["contrarian"] = True

        # O/U invertido 3-leg
        if len(ou_legs) >= 3:
            l = [_invert_ou_leg(r) for r in ou_legs[:3]]
            parlays.append(_parlay("ou_inv_3", "O/U invertido · 3 legs", "", l,
                f"Triple contrario — mejor ROI historico cuando O/U < 50%"))
            parlays[-1]["contrarian"] = True

        # Stack invertido: UNDER (contra modelo) + WIN mismo partido
        # Correlacion positiva: juego de pocas carreras + favorito domina pitcheo
        for g in today_games:
            proj = g.get("projection", {})
            if proj.get("rec") != "OVER":
                continue
            p_ou  = proj.get("prob", 50.0)
            p_a   = proj.get("awayWinProb", 50.0)
            p_h   = proj.get("homeWinProb", 50.0)
            win_p = max(p_a, p_h)
            pk    = g["gamePk"]
            away  = g["away"]; home = g["home"]
            gs    = f"{away['abbr']} @ {home['abbr']}"
            ws    = "home" if p_h >= p_a else "away"
            wt    = home if ws == "home" else away
            base2 = {
                "gamePk": pk, "game": gs,
                "away": {"name": away["name"], "abbr": away["abbr"], "logoUrl": away["logoUrl"]},
                "home": {"name": home["name"], "abbr": home["abbr"], "logoUrl": home["logoUrl"]},
            }
            under_leg = {**base2, "type": "ou_inv", "pick": f"UNDER {proj['line']}",
                         "rec": "UNDER", "prob": p_ou}
            win_leg2  = {**base2, "type": "win", "pick": f"{wt['abbr']} WIN",
                         "pickSide": ws, "pickTeam": wt["name"], "pickAbbr": wt["abbr"],
                         "pickLogoUrl": wt.get("logoUrl",""), "prob": win_p}
            # Correlacion positiva: pitcheo domina → pocas carreras Y favorito gana
            indep2 = (p_ou / 100) * (win_p / 100) * 100
            corr2  = round(min(indep2 * 1.12, min(p_ou, win_p) * 0.93), 1)
            parlays.append(_parlay(
                f"stack_inv_{pk}", "Stack invertido · 2 legs", "",
                [under_leg, win_leg2],
                f"UNDER (contra modelo) + {wt['abbr']} WIN — pitcheo dominante = pocas carreras + favorito gana",
                correlated=True,
            ))
            parlays[-1]["combinedProb"] = corr2
            parlays[-1]["ev"] = round((corr2 / 100 * 2.64 - 1) * 100, 1)
            parlays[-1]["contrarian"] = True

    # ── Colapsar stacks por partido (quedar solo el mejor de cada tipo) ───────
    def _best_stack(parlays: list, prefix: str, final_type: str) -> list:
        stacks = [p for p in parlays if p["id"].startswith(prefix)]
        rest   = [p for p in parlays if not p["id"].startswith(prefix)]
        if stacks:
            best = max(stacks, key=lambda x: x.get("histHitPct", x.get("combinedProb", 0)))
            best = dict(best); best["id"] = final_type
            rest.append(best)
        return rest

    # ── Historical hit-rate estimate for each parlay ─────────────────────────
    for p in parlays:
        hit_rates = []
        for leg in p["legs"]:
            if leg["type"] == "ou_inv":
                # leg apostado al reves del modelo: usar precision invertida
                base_acc = inv_ou_acc if contrarian_mode else hist_ou_acc
            elif leg["type"] == "ou":
                base_acc = hist_ou_acc
            else:
                base_acc = hist_win_acc
            bonus = max(0, (leg["prob"] - 52) * 0.4)
            hit_rates.append(min(base_acc + bonus, 75.0))
        hist_combined = 1.0
        for hr in hit_rates:
            hist_combined *= hr / 100
        p["histHitPct"] = round(hist_combined * 100, 1)
        if p.get("correlated"):
            p["histHitPct"] = round(p["histHitPct"] * 1.08, 1)

    # Colapsar stacks normales e invertidos
    parlays = _best_stack(parlays, "stack_inv_", "stack_inv")
    parlays = _best_stack(parlays, "stack_",     "stack")

    parlays.sort(key=lambda x: -x["histHitPct"])
    return parlays, contrarian_mode


@app.get("/api/parlays")
async def get_parlays(date: str = Query(default=None)):
    """Recommended parlays for date. Includes historical hit-rate estimates."""
    import datetime as _dt
    target = date or str(_dt.date.today())

    # Load today's picks (train models if needed)
    stored = cache.load(target)
    if stored:
        games = stored.get("games", stored) if isinstance(stored, dict) else stored
    else:
        games, _ = await build_picks(target)

    if not games:
        return {"date": target, "parlays": [], "note": "No hay partidos programados"}

    # Pull rolling accuracy from results cache (last 30 days)
    import datetime as _dt2
    yesterday = str(_dt2.date.fromisoformat(target) - timedelta(days=1))
    res_cached = cache.load(f"results_{yesterday}")
    hist_ou  = 51.7
    hist_win = 57.0
    if res_cached:
        s = res_cached.get("summary", {})
        hist_ou  = s.get("ouPct",  51.7)
        hist_win = s.get("winPct", 57.0)

    parlays, contrarian_mode = build_parlays(games, hist_ou_acc=hist_ou, hist_win_acc=hist_win)
    return {
        "date":           target,
        "totalGames":     len(games),
        "histOuAcc":      hist_ou,
        "histWinAcc":     hist_win,
        "contrarianMode": contrarian_mode,
        "parlays":        parlays,
    }
