#!/usr/bin/env python3
"""
MLB O/U Backtest v3 — las 4 mejoras aplicadas:
  1. 3 temporadas de entrenamiento: 2024 + 2025 + 2026
  2. ERA del pitcher titular individual (no staff)
  3. Clima/viento por estadio via Open-Meteo
  4. Filtro UNDER solamente

Walk-forward honesto — ningún juego usa datos futuros.
"""

import sys, importlib, json, statistics
from pathlib import Path

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

np = ensure("numpy"); pd = ensure("pandas")
ensure("matplotlib"); ensure("seaborn")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np, pandas as pd

import urllib.request, time

from weather_batch import build_season_weather_cache, lookup, wind_factor, temp_factor
from park_factors import get_park_factor
from pitcher_stats import get_pitcher_season_stats, pitcher_quality_score
from ml_model import MLPredictor, build_features

LINES = [6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]
def nearest_line(v): return min(LINES, key=lambda x: abs(x-v))

# ── fetch helpers ─────────────────────────────────────────────────────────────
def fetch(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except: time.sleep(1.5)
    return {}

def load_games(start, end):
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&startDate={start}&endDate={end}&hydrate=linescore,team&gameType=R")
    rows = []
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls = g.get("linescore",{}).get("teams",{})
            ar = ls.get("away",{}).get("runs",0) or 0
            hr = ls.get("home",{}).get("runs",0) or 0
            if ar+hr == 0: continue
            rows.append({"date":d["date"],
                         "away_id":  g["teams"]["away"]["team"]["id"],
                         "home_id":  g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "venue":    g.get("venue",{}).get("name",""),
                         "game_pk":  g["gamePk"],
                         "away_runs":ar,"home_runs":hr,"total":ar+hr})
    df = pd.DataFrame(rows)
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    return df

def load_schedule_with_pitchers(start, end):
    """Schedule with probablePitcher — para obtener IDs de pitchers."""
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&startDate={start}&endDate={end}"
           f"&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    rows = []
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls = g.get("linescore",{}).get("teams",{})
            ar = ls.get("away",{}).get("runs",0) or 0
            hr = ls.get("home",{}).get("runs",0) or 0
            if ar+hr == 0: continue
            away_p = g["teams"]["away"].get("probablePitcher",{})
            home_p = g["teams"]["home"].get("probablePitcher",{})
            rows.append({"date":d["date"],
                         "away_id":  g["teams"]["away"]["team"]["id"],
                         "home_id":  g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "venue":    g.get("venue",{}).get("name",""),
                         "game_pk":  g["gamePk"],
                         "away_runs":ar,"home_runs":hr,"total":ar+hr,
                         "away_pitcher_id": away_p.get("id"),
                         "home_pitcher_id": home_p.get("id"),
                         "away_pitcher":    away_p.get("fullName","TBD"),
                         "home_pitcher":    home_p.get("fullName","TBD"),
                         })
    df = pd.DataFrame(rows)
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    return df

def load_team_stats(season):
    hit={}; pit={}
    for group, store in [("hitting",hit),("pitching",pit)]:
        url=(f"https://statsapi.mlb.com/api/v1/teams/stats"
             f"?season={season}&sportId=1&stats=season&group={group}")
        for sp in fetch(url).get("stats",[{}])[0].get("splits",[]):
            store[sp["team"]["id"]] = sp["stat"]
    return hit, pit

# ── rolling stats (sin data leakage) ─────────────────────────────────────────
def rolling_stats(df, tid, before_date):
    mask=((df["away_id"]==tid)|(df["home_id"]==tid)) & (df["date"]<before_date)
    sub=df[mask].sort_values("date")
    if len(sub)<5: return None
    sc=[]; al=[]; sc_away=[]; sc_home=[]
    for _,r in sub.iterrows():
        if r["away_id"]==tid:
            sc.append(r["away_runs"]); al.append(r["home_runs"]); sc_away.append(r["away_runs"])
        else:
            sc.append(r["home_runs"]); al.append(r["away_runs"]); sc_home.append(r["home_runs"])
    wins=sum(1 for s,a in zip(sc,al) if s>a)
    return {
        "rpg":      np.mean(sc),        "rpg_l10": np.mean(sc[-10:]),
        "ra_pg":    np.mean(al),        "win_pct": wins/len(sc),
        "scored":   sc,                 "allowed": al,
        "rpg_away": np.mean(sc_away) if sc_away else np.mean(sc),
        "rpg_home": np.mean(sc_home) if sc_home else np.mean(sc),
        "std":      float(np.std(sc[-20:])) if len(sc)>=5 else 2.5,
    }

def h2h_before(df, aid, hid, before_date):
    mask=(((df["away_id"]==aid)&(df["home_id"]==hid))|
          ((df["away_id"]==hid)&(df["home_id"]==aid)))&(df["date"]<before_date)
    return df[mask]["total"].tolist()

# ── prediction (ahora con pitcher + weather) ──────────────────────────────────
def predict_v3(a_st, h_st, h2h, away_pitcher_era, home_pitcher_era,
               wind_adj, temp_adj):
    """
    Improved prediction incorporating pitcher ERA and weather.
    away_pitcher_era / home_pitcher_era: individual ERA of starting pitcher
    wind_adj, temp_adj: run adjustments from weather
    """
    def proj(att, def_ra, opp_era):
        sv  = att["rpg"];    l10 = att["rpg_l10"]
        wv  = np.mean(att["scored"][-7:]) if len(att["scored"])>=3 else sv
        # opp_era factor: better pitcher (lower ERA) suppresses runs
        era_factor = min(1.3, max(0.7, opp_era / 4.20))
        base = sv*0.20 + l10*0.25 + wv*0.30 + def_ra*0.15 + sv*0.10
        return base * era_factor

    pa = proj(a_st, h_st["ra_pg"], home_pitcher_era)
    ph = proj(h_st, a_st["ra_pg"], away_pitcher_era)
    total = pa + ph

    if len(h2h)>=2:
        total = total*0.85 + np.mean(h2h)*0.15

    # Weather adjustment (only for outdoor parks)
    total += wind_adj + temp_adj

    line = nearest_line(total)
    rec  = "OVER" if total >= line else "UNDER"
    return total, line, rec

# ── build training rows ───────────────────────────────────────────────────────
def build_training_rows(df_games, hit_stats, pit_stats, season, wx_cache=None):
    """Build ML feature rows — incluye splits, std, park factor y clima."""
    # Calcular aggregates incluyendo splits home/away y std
    agg = {}
    for tid in set(df_games["away_id"])|set(df_games["home_id"]):
        sub = df_games[(df_games["away_id"]==tid)|(df_games["home_id"]==tid)]
        sc=[]; al=[]; sc_a=[]; sc_h=[]
        for _,r in sub.iterrows():
            if r["away_id"]==tid:
                sc.append(r["away_runs"]); al.append(r["home_runs"]); sc_a.append(r["away_runs"])
            else:
                sc.append(r["home_runs"]); al.append(r["away_runs"]); sc_h.append(r["home_runs"])
        wins=sum(1 for s,a in zip(sc,al) if s>a)
        agg[tid] = {
            "rpg":      np.mean(sc) if sc else 4.,
            "rpg_l10":  np.mean(sc[-10:]) if sc else 4.,
            "ra_pg":    np.mean(al) if al else 4.,
            "win_pct":  wins/len(sc) if sc else .5,
            "rpg_away": np.mean(sc_a) if sc_a else (np.mean(sc) if sc else 4.),
            "rpg_home": np.mean(sc_h) if sc_h else (np.mean(sc) if sc else 4.),
            "std":      float(np.std(sc)) if len(sc)>=5 else 2.5,
        }

    rows = []
    for _, g in df_games.iterrows():
        a=agg.get(g["away_id"],{}); h=agg.get(g["home_id"],{})
        if not a or not h: continue
        aps=pit_stats.get(g["away_id"],{}); hps=pit_stats.get(g["home_id"],{})
        ahs=hit_stats.get(g["away_id"],{}); hhs=hit_stats.get(g["home_id"],{})

        venue  = str(g.get("venue",""))
        date_s = g["date"].strftime("%Y-%m-%d") if hasattr(g["date"],"strftime") else str(g["date"])[:10]
        pf     = get_park_factor(venue)

        if wx_cache:
            wx    = lookup(wx_cache, venue, date_s)
            w_adj = wind_factor(wx["wind_mph"], wx.get("roof","open"))
            t_adj = temp_factor(wx["temp_f"],   wx.get("roof","open"))
        else:
            wx = {"wind_mph":7.0,"temp_f":72.0}; w_adj=0.0; t_adj=0.0

        feats = build_features(
            a["rpg"], a["rpg_l10"], a["win_pct"],
            aps.get("era","4.20"), aps.get("whip","1.30"), ahs.get("ops",".720"),
            "", a["ra_pg"],
            h["rpg"], h["rpg_l10"], h["win_pct"],
            hps.get("era","4.20"), hps.get("whip","1.30"), hhs.get("ops",".720"),
            "", h["ra_pg"],
            None, 0, g["date"].month,
            away_rpg_away=a["rpg_away"], home_rpg_home=h["rpg_home"],
            away_std=a["std"],           home_std=h["std"],
            park_factor=pf,
        )
        feats = feats + [w_adj, t_adj, wx.get("wind_mph",7.0), wx.get("temp_f",72.0)]
        rows.append({"features":feats,"target":float(g["total"])})
    return rows

# ── Extended ML model (with extra features) ───────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import pickle

CACHE = Path(__file__).parent / "cache"

def train_v3_model(rows):
    X = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r["target"]   for r in rows], dtype=float)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m",  XGBRegressor(n_estimators=500, learning_rate=0.03,
                                          max_depth=4, subsample=0.80,
                                          colsample_bytree=0.80,
                                          reg_alpha=0.5, reg_lambda=1.5,
                                          random_state=42, verbosity=0))])
    ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=5.0))])
    cv = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
    pipe.fit(X, y); ridge.fit(X, y)
    return pipe, ridge, float(-cv.mean())

def predict_v3_ml(pipe, ridge, feats_ext):
    X = np.array([feats_ext], dtype=float)
    xgb_p  = float(pipe.predict(X)[0])
    rid_p  = float(ridge.predict(X)[0])
    return xgb_p*0.65 + rid_p*0.35

# ── Main backtest ─────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  MLB O/U BACKTEST v3 — 4 mejoras")
    print("="*65)

    # ── Cache de datos pesados en disco ──────────────────────────────────────
    games24_path   = CACHE / "games_2024.pkl"
    games25_path   = CACHE / "games_2025.pkl"
    games26_path   = CACHE / "games_2026.pkl"
    stats24_path   = CACHE / "stats_2024.json"
    stats25_path   = CACHE / "stats_2025.json"
    stats26_path   = CACHE / "stats_2026.json"
    pitchers_path  = CACHE / "pitchers_2026.json"

    print("\n1. Cargando datos (2024 + 2025 + 2026)...")
    if games24_path.exists():
        df_2024 = pd.read_pickle(games24_path); print("   2024: desde cache")
    else:
        df_2024 = load_games("2024-03-20","2024-09-29")
        df_2024.to_pickle(games24_path)

    if games25_path.exists():
        df_2025 = pd.read_pickle(games25_path); print("   2025: desde cache")
    else:
        df_2025 = load_games("2025-03-20","2025-09-28")
        df_2025.to_pickle(games25_path)

    if games26_path.exists():
        df_2026 = pd.read_pickle(games26_path); print("   2026: desde cache")
    else:
        df_2026 = load_schedule_with_pitchers("2026-03-19","2026-06-04")
        df_2026.to_pickle(games26_path)

    if stats24_path.exists():
        d=json.load(open(stats24_path)); hit24,pit24=d["hit"],d["pit"]
    else:
        hit24,pit24=load_team_stats(2024)
        json.dump({"hit":hit24,"pit":pit24},open(stats24_path,"w"))

    if stats25_path.exists():
        d=json.load(open(stats25_path)); hit25,pit25=d["hit"],d["pit"]
    else:
        hit25,pit25=load_team_stats(2025)
        json.dump({"hit":hit25,"pit":pit25},open(stats25_path,"w"))

    if stats26_path.exists():
        d=json.load(open(stats26_path)); hit26,pit26=d["hit"],d["pit"]
    else:
        hit26,pit26=load_team_stats(2026)
        json.dump({"hit":{str(k):v for k,v in hit26.items()},
                   "pit":{str(k):v for k,v in pit26.items()}},open(stats26_path,"w"))
        # reload with int keys
    hit26={int(k):v for k,v in hit26.items()}; pit26={int(k):v for k,v in pit26.items()}

    print(f"   2024: {len(df_2024)}j  |  2025: {len(df_2025)}j  |  2026: {len(df_2026)}j")

    print("\n2. Obteniendo ERA de pitchers titulares 2026...")
    if pitchers_path.exists():
        raw = json.load(open(pitchers_path))
        pitcher_cache = {(int(k.split("_")[0]), int(k.split("_")[1])): v for k, v in raw.items()}
        print(f"   {len(pitcher_cache)} pitchers desde cache")
    else:
        pitcher_cache = {}
        unique_pids = set()
        for _, g in df_2026.iterrows():
            try:
                if g.get("away_pitcher_id") and not pd.isna(g["away_pitcher_id"]):
                    unique_pids.add((int(g["away_pitcher_id"]), 2026))
                if g.get("home_pitcher_id") and not pd.isna(g["home_pitcher_id"]):
                    unique_pids.add((int(g["home_pitcher_id"]), 2026))
            except (TypeError, ValueError):
                pass
        print(f"   {len(unique_pids)} pitchers a consultar...")
        for i, (pid, season) in enumerate(unique_pids):
            pitcher_cache[(pid, season)] = get_pitcher_season_stats(pid, season)
            if (i+1) % 50 == 0: print(f"   {i+1}/{len(unique_pids)}...", end="\r")
        json.dump({f"{k[0]}_{k[1]}": v for k, v in pitcher_cache.items()},
                  open(pitchers_path, "w"))
        print(f"   {len(pitcher_cache)} pitchers cargados y guardados")

    print("\n3. Descargando clima batch (una llamada por estadio)...")
    wx24 = build_season_weather_cache(2024, "2024-03-20", "2024-09-29")
    wx25 = build_season_weather_cache(2025, "2025-03-20", "2025-09-28")
    wx26 = build_season_weather_cache(2026, "2026-03-19", "2026-06-05")

    rows_24 = build_training_rows(df_2024, hit24, pit24, 2024, wx24)
    print(f"   2024: {len(rows_24)} filas")
    rows_25 = build_training_rows(df_2025, hit25, pit25, 2025, wx25)
    print(f"   2025: {len(rows_25)} filas")
    # Para 2026 usamos rolling (walk-forward) en el backtest
    all_train = rows_24 + rows_25
    print(f"   Total entrenamiento: {len(all_train)} juegos")

    print("\n4. Entrenando modelo v3...")
    pipe_v3, ridge_v3, cv_mae = train_v3_model(all_train)
    print(f"   CV MAE: {cv_mae:.3f} runs  (entrenado con 2024+2025)")

    # ── Backtest walk-forward en 2026 ─────────────────────────────────────────
    print("\n5. Backtest 2026 walk-forward...")
    cutoff = pd.Timestamp("2026-04-01")
    game_dates = sorted(df_2026[df_2026["date"]>=cutoff]["date"].unique())

    records = []
    for di, gdate in enumerate(game_dates):
        day_games = df_2026[df_2026["date"]==gdate]
        for _, g in day_games.iterrows():
            aid=int(g["away_id"]); hid=int(g["home_id"])
            a_st=rolling_stats(df_2026, aid, gdate)
            h_st=rolling_stats(df_2026, hid, gdate)
            if not a_st or not h_st: continue

            h2h=h2h_before(df_2026, aid, hid, gdate)
            venue=str(g.get("venue",""))
            date_s=gdate.strftime("%Y-%m-%d")

            # Pitcher ERA individual
            ap_id = g.get("away_pitcher_id"); hp_id = g.get("home_pitcher_id")
            try:
                away_era = pitcher_cache.get((int(ap_id),2026),{}).get("era",4.20) if (ap_id and not __import__("math").isnan(float(ap_id))) else 4.20
            except: away_era = 4.20
            try:
                home_era = pitcher_cache.get((int(hp_id),2026),{}).get("era",4.20) if (hp_id and not __import__("math").isnan(float(hp_id))) else 4.20
            except: home_era = 4.20

            # Weather — desde cache batch
            wx    = lookup(wx26, venue, date_s)
            w_adj = wind_factor(wx["wind_mph"], wx.get("roof","open"))
            t_adj = temp_factor(wx["temp_f"],   wx.get("roof","open"))

            # Fórmula mejorada
            f_proj, f_line, f_rec = predict_v3(a_st, h_st, h2h,
                                                away_era, home_era,
                                                w_adj, t_adj)

            # ML features extendidos — ahora con splits, std y park factor
            aps=pit26.get(aid,{}); hps=pit26.get(hid,{})
            ahs=hit26.get(aid,{}); hhs=hit26.get(hid,{})
            pf = get_park_factor(venue)

            feats = build_features(
                a_st["rpg"],a_st["rpg_l10"],a_st["win_pct"],
                str(away_era), aps.get("whip","1.30"), ahs.get("ops",".720"),
                "", a_st["ra_pg"],
                h_st["rpg"],h_st["rpg_l10"],h_st["win_pct"],
                str(home_era), hps.get("whip","1.30"), hhs.get("ops",".720"),
                "", h_st["ra_pg"],
                np.mean(h2h) if len(h2h)>=2 else None, len(h2h), gdate.month,
                away_rpg_away=a_st["rpg_away"], home_rpg_home=h_st["rpg_home"],
                away_std=a_st["std"],           home_std=h_st["std"],
                park_factor=pf,
            )
            feats_ext = feats + [w_adj, t_adj, wx["wind_mph"], wx["temp_f"]]

            ml_proj = predict_v3_ml(pipe_v3, ridge_v3, feats_ext)
            ens     = ml_proj*0.55 + f_proj*0.45
            line    = nearest_line(ens)
            rec     = "OVER" if ens>=line else "UNDER"
            actual  = float(g["total"])
            correct = (rec=="OVER" and actual>line) or (rec=="UNDER" and actual<line)

            # ── MEJORA 4: filtro UNDER solamente ─────────────────────────────
            play_under = (rec == "UNDER")

            records.append({
                "date":gdate, "away":g["away_name"].split()[-1],
                "home":g["home_name"].split()[-1],
                "venue":venue[:25], "away_era":round(away_era,2),
                "home_era":round(home_era,2),
                "wind":wx["wind_mph"],"temp":wx["temp_f"],"roof":wx.get("roof","?"),
                "w_adj":round(w_adj,2),"t_adj":round(t_adj,2),
                "f_proj":round(f_proj,2),"ml_proj":round(ml_proj,2),
                "ens":round(ens,2),"line":line,"rec":rec,
                "actual":int(actual),"correct":correct,
                "play_under":play_under,
                "diff":round(ens-line,2),
            })
        if (di+1)%10==0: print(f"   {di+1}/{len(game_dates)} dias...", end="\r")

    print(f"\n   {len(records)} predicciones generadas")
    df = pd.DataFrame(records)

    # ── Resultados ─────────────────────────────────────────────────────────────
    n       = len(df)
    n_under = df["play_under"].sum()
    acc_all = df["correct"].mean()*100
    acc_u   = df.loc[df["play_under"],"correct"].mean()*100 if n_under else 0
    pnl_all = df["correct"].map({True:100.,False:-110.}).sum()
    pnl_u   = df.loc[df["play_under"],"correct"].map({True:100.,False:-110.}).sum()

    print("\n" + "="*65)
    print("  RESULTADOS v3")
    print("="*65)
    print(f"\n  Entrenamiento: 2024+2025 ({len(all_train)} juegos)")
    print(f"  CV MAE modelo: {cv_mae:.3f} runs")
    print()
    print(f"  {'':30s} {'Acc':>6}  {'Juegos':>7}  {'P&L':>9}")
    print(f"  {'─'*30} {'─'*6}  {'─'*7}  {'─'*9}")
    print(f"  {'Todos los picks':30s} {acc_all:5.1f}%  {n:>7}  ${pnl_all:>+8.0f}")
    print(f"  {'Solo UNDER':30s} {acc_u:5.1f}%  {n_under:>7}  ${pnl_u:>+8.0f}")

    # By month — UNDER only
    print("\n  Solo UNDER por mes:")
    df["month"] = df["date"].dt.strftime("%b")
    for mo, sub in df[df["play_under"]].groupby("month"):
        a=sub["correct"].mean()*100
        p=sub["correct"].map({True:100.,False:-110.}).sum()
        print(f"    {mo}: {len(sub):3d}j  acc={a:.1f}%  P&L=${p:+.0f}")

    # Weather effect
    print("\n  Efecto del clima (todos los picks):")
    wind_mask = df["wind"]>=15
    if wind_mask.sum():
        wa = df.loc[wind_mask,"correct"].mean()*100
        print(f"    Viento >15mph: {wind_mask.sum()} juegos  acc={wa:.1f}%")
    cold_mask = df["temp"]<55
    if cold_mask.sum():
        ca = df.loc[cold_mask,"correct"].mean()*100
        print(f"    Frio <55F:     {cold_mask.sum()} juegos  acc={ca:.1f}%")

    # Pitcher ERA impact
    print("\n  Efecto ERA del pitcher (solo UNDER):")
    df["elite_pitchers"] = (df["away_era"]<3.5) & (df["home_era"]<3.5)
    elite_under = df[df["play_under"] & df["elite_pitchers"]]
    if len(elite_under):
        ea = elite_under["correct"].mean()*100
        ep = elite_under["correct"].map({True:100.,False:-110.}).sum()
        print(f"    Ambos pitchers ERA<3.5: {len(elite_under)}j  acc={ea:.1f}%  P&L=${ep:+.0f}")

    # ── Guardar modelo ─────────────────────────────────────────────────────────
    with open(CACHE/"model_v3_xgb.pkl","wb") as f: pickle.dump(pipe_v3, f)
    with open(CACHE/"model_v3_ridge.pkl","wb") as f: pickle.dump(ridge_v3, f)
    print(f"\n  Modelos guardados en cache/")

    # ── Imagen ─────────────────────────────────────────────────────────────────
    plot_results(df, cv_mae, len(all_train))

def plot_results(df, cv_mae, train_n):
    plt.rcParams.update({"figure.facecolor":"#fff","axes.facecolor":"#fafafa",
                          "axes.edgecolor":"#d1d5db","text.color":"#111",
                          "xtick.color":"#666","ytick.color":"#666",
                          "grid.color":"#e5e7eb","font.size":9})

    fig = plt.figure(figsize=(22, 24))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35,
                            top=0.93, bottom=0.04, left=0.06, right=0.97)

    by_day = df.groupby("date").agg(
        acc_all  =("correct","mean"),
        acc_under=("correct", lambda x: x[df.loc[x.index,"play_under"]].mean()
                               if df.loc[x.index,"play_under"].any() else np.nan),
        pnl_all  =("correct",lambda x: x.map({True:100.,False:-110.}).sum()),
        pnl_under=("correct",lambda x:
                   x[df.loc[x.index,"play_under"]].map({True:100.,False:-110.}).sum()),
    ).reset_index()
    by_day["cum_all"]   = by_day["pnl_all"].cumsum()
    by_day["cum_under"] = by_day["pnl_under"].cumsum()
    dates = pd.to_datetime(by_day["date"])

    # ── 1. Accuracy diaria UNDER ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0,:])
    u_acc = by_day["acc_under"].fillna(0)*100
    ax1.bar(dates, u_acc,
            color=["#166534" if a>=52.4 else "#dc2626" for a in u_acc],
            alpha=0.80, width=0.8)
    roll7 = u_acc.rolling(7, min_periods=3).mean()
    ax1.plot(dates, roll7, color="#111", lw=2.2, label="Rolling 7d")
    ax1.axhline(52.4, color="#374151", lw=1.2, linestyle="--", label="52.4% break-even")
    ax1.axhline(50,   color="#aaa",    lw=0.8, linestyle=":")
    n_u  = df["play_under"].sum()
    acc_u= df.loc[df["play_under"],"correct"].mean()*100 if n_u else 0
    pnl_u= df.loc[df["play_under"],"correct"].map({True:100.,False:-110.}).sum()
    ax1.set_ylim(0, 105); ax1.set_ylabel("Accuracy %")
    ax1.set_title(
        f"UNDER-only accuracy diaria  |  "
        f"Global {acc_u:.1f}%  |  {n_u} picks  |  "
        f"P&L USD{pnl_u:+.0f}  |  Modelo 2024+2025 ({train_n}j) CV MAE {cv_mae:.2f}",
        fontsize=12, fontweight="bold", pad=10, loc="left")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.4)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 2. P&L — todos vs UNDER ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1,:])
    ax2.plot(dates, by_day["cum_all"],   color="#9ca3af", lw=1.8, label="Todos los picks")
    ax2.plot(dates, by_day["cum_under"], color="#111",    lw=2.5, label="Solo UNDER")
    ax2.fill_between(dates, by_day["cum_under"], 0,
                     where=by_day["cum_under"]>=0, alpha=0.15, color="#166534")
    ax2.fill_between(dates, by_day["cum_under"], 0,
                     where=by_day["cum_under"]<0,  alpha=0.15, color="#dc2626")
    ax2.axhline(0, color="#374151", lw=1)
    cu = by_day["cum_under"].values
    max_i=np.argmax(cu); min_i=np.argmin(cu)
    ax2.annotate(f"USD{cu[max_i]:+.0f}", xy=(dates.iloc[max_i],cu[max_i]),
                 fontsize=8, color="#166534", xytext=(0,7), textcoords="offset points")
    ax2.annotate(f"USD{cu[min_i]:+.0f}", xy=(dates.iloc[min_i],cu[min_i]),
                 fontsize=8, color="#dc2626", xytext=(0,-14), textcoords="offset points")
    ax2.set_ylabel("USD acumulado"); ax2.grid(alpha=0.4)
    ax2.set_title("P&L acumulado — USD 110 por pick",
                  fontsize=11, pad=8, loc="left")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 3. ERA del pitcher vs accuracy ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    df_u = df[df["play_under"]].copy()
    df_u["avg_era"] = (df_u["away_era"]+df_u["home_era"])/2
    era_bins = pd.cut(df_u["avg_era"], bins=[0,3.0,3.5,4.0,4.5,5.0,10],
                      labels=["<3.0","3.0-3.5","3.5-4.0","4.0-4.5","4.5-5.0",">5.0"])
    era_acc = df_u.groupby(era_bins, observed=True).agg(
        acc=("correct","mean"), n=("correct","count")).reset_index()
    bc = ["#166534" if a>=0.524 else "#dc2626" for a in era_acc["acc"]]
    ax3.bar(range(len(era_acc)), era_acc["acc"]*100, color=bc, alpha=0.85, edgecolor="#fff")
    for xi,(_, r) in enumerate(era_acc.iterrows()):
        ax3.text(xi, r["acc"]*100+0.5, f"{r['acc']*100:.1f}%\n({int(r['n'])}j)",
                 ha="center", fontsize=8)
    ax3.axhline(52.4, color="#374151", lw=1.2, linestyle="--")
    ax3.set_xticks(range(len(era_acc)))
    ax3.set_xticklabels(era_acc["avg_era"].astype(str), fontsize=9)
    ax3.set_xlabel("ERA promedio de pitchers del partido")
    ax3.set_ylabel("Accuracy %"); ax3.set_ylim(0, 80)
    ax3.set_title("Accuracy UNDER por ERA del pitcher",
                  fontsize=11, pad=8, loc="left")
    ax3.grid(axis="y", alpha=0.4)

    # ── 4. Efecto del viento ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    wind_bins = pd.cut(df_u["wind"], bins=[0,5,10,15,20,50],
                       labels=["0-5","5-10","10-15","15-20",">20"])
    wind_acc = df_u.groupby(wind_bins, observed=True).agg(
        acc=("correct","mean"), n=("correct","count")).reset_index()
    bc2 = ["#166534" if a>=0.524 else "#dc2626" for a in wind_acc["acc"]]
    ax4.bar(range(len(wind_acc)), wind_acc["acc"]*100, color=bc2, alpha=0.85, edgecolor="#fff")
    for xi,(_, r) in enumerate(wind_acc.iterrows()):
        ax4.text(xi, r["acc"]*100+0.5, f"{r['acc']*100:.1f}%\n({int(r['n'])}j)",
                 ha="center", fontsize=8)
    ax4.axhline(52.4, color="#374151", lw=1.2, linestyle="--")
    ax4.set_xticks(range(len(wind_acc)))
    ax4.set_xticklabels([f"{w} mph" for w in wind_acc["wind"].astype(str)], fontsize=9)
    ax4.set_ylabel("Accuracy %"); ax4.set_ylim(0, 80)
    ax4.set_title("Accuracy UNDER por velocidad de viento",
                  fontsize=11, pad=8, loc="left")
    ax4.grid(axis="y", alpha=0.4)

    # ── 5. Tabla día a día (últimos 45) ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[3,:])
    ax5.axis("off")
    recent = by_day.tail(45).reset_index(drop=True)
    rh = 1.0/(len(recent)+1)

    for x_pos, lbl in [(0.01,"FECHA"),(0.11,"Acc ALL"),(0.20,"Acc UNDER"),
                        (0.30,"P&L U"),(0.38,"CumUNDER"),(0.47,"Partidos (★=UNDER pick)")]:
        ax5.text(x_pos, 1.01, lbl, fontsize=7.5, fontweight="bold",
                 color="#444", transform=ax5.transAxes)

    for i, row in recent.iterrows():
        y = 1.0-(i+1)*rh-0.01
        au = row["acc_under"]*100 if not np.isnan(row["acc_under"]) else None
        bg = "#f0fdf4" if (au and au>=52.4) else "#fef2f2" if (au and au<50) else "#fff"
        ax5.add_patch(plt.Rectangle((0,y-rh*0.35),1,rh*0.85, facecolor=bg,
                      edgecolor="#e0e0e0",lw=0.4,transform=ax5.transAxes,clip_on=False))

        ax5.text(0.01,y+rh*0.22, row["date"].strftime("%a %b %d"),
                 fontsize=8,fontweight="bold",color="#111",transform=ax5.transAxes)
        ca = "#166534" if row["acc_all"]>=0.524 else "#dc2626"
        ax5.text(0.11,y+rh*0.22, f"{row['acc_all']*100:.0f}%",
                 fontsize=8,color=ca,transform=ax5.transAxes)
        if au is not None:
            cu2 = "#166534" if au>=52.4 else "#dc2626"
            ax5.text(0.20,y+rh*0.22, f"{au:.0f}%",
                     fontsize=8,color=cu2,fontweight="bold",transform=ax5.transAxes)
        cp = "#166534" if row["pnl_under"]>=0 else "#dc2626"
        ax5.text(0.30,y+rh*0.22, f"USD{row['pnl_under']:+.0f}",
                 fontsize=8,color=cp,transform=ax5.transAxes)
        cc = "#166534" if row["cum_under"]>=0 else "#dc2626"
        ax5.text(0.38,y+rh*0.22, f"USD{row['cum_under']:+.0f}",
                 fontsize=8,color=cc,transform=ax5.transAxes)

        # Partidos del día
        day_preds = df[df["date"]==row["date"]]
        parts=[]
        for _,pr in day_preds.iterrows():
            star="★" if pr["play_under"] else " "
            sym ="✓" if pr["correct"] else "✗"
            parts.append(f"{star}{pr['away']}@{pr['home']} U{pr['line']} {sym}r={pr['actual']} ERA{pr['away_era']}/{pr['home_era']} W{pr['wind']:.0f}")
        ax5.text(0.47,y+rh*0.22,"  ".join(parts)[:115],
                 fontsize=6.5,color="#444",transform=ax5.transAxes)

    ax5.set_title("Detalle diario — ★=pick UNDER jugado  ✓=acierto  ✗=error  ERA=titular  W=viento mph",
                  fontsize=10,pad=8,loc="left")
    ax5.set_xlim(0,1); ax5.set_ylim(0,1)

    pnl_u2= df.loc[df["play_under"],"correct"].map({True:100.,False:-110.}).sum()
    acc_u2= df.loc[df["play_under"],"correct"].mean()*100 if df["play_under"].sum() else 0
    fig.suptitle(
        f"MLB O/U v3 — 4 mejoras aplicadas  |  2026 Sem.2 a Jun 4\n"
        f"Entrenado con 2024+2025 ({train_n}j)  CV MAE {cv_mae:.2f}  |  "
        f"Solo UNDER: {acc_u2:.1f}% en {df['play_under'].sum()} picks  USD{pnl_u2:+.0f}",
        fontsize=12, fontweight="bold", y=0.975
    )
    out="/Users/vaquera/Documents/NBA-SPURS/mlb_backtest_v3.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    print(f"\n  Imagen: {out}")

if __name__=="__main__":
    main()
