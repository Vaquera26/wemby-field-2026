#!/usr/bin/env python3
"""
MLB Backtest v5 — O/U + Prediccion de victorias (Moneyline).

Dos modelos entrenados sobre los mismos datos:
  Modelo A: XGBRegressor  → predice carreras totales  → O/U pick
  Modelo B: XGBClassifier → predice quien gana (1=home, 0=away) → ML pick

Combinacion de senales:
  SIGNAL 1 — Solo UNDER (modelo A)
  SIGNAL 2 — Solo Moneyline (modelo B, P >= umbral)
  SIGNAL 3 — UNDER + ganador con alta confianza (ambos modelos alinados)
              → el pick mas poderoso: juego de pocas carreras + pitcheo dominante
"""

import sys, importlib, json, math
from pathlib import Path

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

np=ensure("numpy"); pd=ensure("pandas")
ensure("matplotlib"); ensure("seaborn")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import numpy as np, pandas as pd

import urllib.request, time, pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor, XGBClassifier

from weather_batch import build_season_weather_cache, lookup, wind_factor, temp_factor
from park_factors   import get_park_factor
from ml_model       import build_features, FEATURE_NAMES

CACHE=Path(__file__).parent/"cache"
LINES=[6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]
def nearest_line(v): return min(LINES,key=lambda x:abs(x-v))

# ── fetch helpers ─────────────────────────────────────────────────────────────
def fetch(url):
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=25) as r:
                return json.loads(r.read().decode())
        except: time.sleep(1.5)
    return {}

def load_games(start,end):
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
         f"&startDate={start}&endDate={end}&hydrate=linescore,team&gameType=R")
    rows=[]
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls=g.get("linescore",{}).get("teams",{})
            ar=ls.get("away",{}).get("runs",0) or 0
            hr=ls.get("home",{}).get("runs",0) or 0
            if ar+hr==0: continue
            rows.append({"date":d["date"],
                         "away_id":g["teams"]["away"]["team"]["id"],
                         "home_id":g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "venue":g.get("venue",{}).get("name",""),
                         "away_runs":ar,"home_runs":hr,"total":ar+hr,
                         "home_win":1 if hr>ar else 0})
    df=pd.DataFrame(rows)
    if not df.empty: df["date"]=pd.to_datetime(df["date"])
    return df

def load_with_pitchers(start,end):
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
         f"&startDate={start}&endDate={end}"
         f"&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    rows=[]
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls=g.get("linescore",{}).get("teams",{})
            ar=ls.get("away",{}).get("runs",0) or 0
            hr=ls.get("home",{}).get("runs",0) or 0
            if ar+hr==0: continue
            ap=g["teams"]["away"].get("probablePitcher",{})
            hp=g["teams"]["home"].get("probablePitcher",{})
            rows.append({"date":d["date"],
                         "away_id":g["teams"]["away"]["team"]["id"],
                         "home_id":g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "venue":g.get("venue",{}).get("name",""),
                         "away_runs":ar,"home_runs":hr,"total":ar+hr,
                         "home_win":1 if hr>ar else 0,
                         "away_pitcher_id":ap.get("id"),
                         "home_pitcher_id":hp.get("id"),
                         "away_pitcher":ap.get("fullName","TBD"),
                         "home_pitcher":hp.get("fullName","TBD")})
    df=pd.DataFrame(rows)
    if not df.empty: df["date"]=pd.to_datetime(df["date"])
    return df

def load_team_stats(season):
    hit={}; pit={}
    for group,store in [("hitting",hit),("pitching",pit)]:
        url=(f"https://statsapi.mlb.com/api/v1/teams/stats"
             f"?season={season}&sportId=1&stats=season&group={group}")
        for sp in fetch(url).get("stats",[{}])[0].get("splits",[]):
            store[sp["team"]["id"]]=sp["stat"]
    return hit,pit

def cached_df(path,fn):
    p=Path(path)
    if p.exists(): return pd.read_pickle(p)
    df=fn(); df.to_pickle(p); return df

def cached_stats(path,fn):
    p=Path(path)
    if p.exists():
        d=json.load(open(p))
        return {int(k):v for k,v in d["hit"].items()},{int(k):v for k,v in d["pit"].items()}
    hit,pit=fn()
    json.dump({"hit":{str(k):v for k,v in hit.items()},"pit":{str(k):v for k,v in pit.items()}},open(p,"w"))
    return hit,pit

# ── rolling stats ─────────────────────────────────────────────────────────────
def rolling_stats(df,tid,before_date):
    mask=((df["away_id"]==tid)|(df["home_id"]==tid))&(df["date"]<before_date)
    sub=df[mask].sort_values("date")
    if len(sub)<5: return None
    sc=[]; al=[]; sc_a=[]; sc_h=[]; hw=0
    for _,r in sub.iterrows():
        if r["away_id"]==tid:
            sc.append(r["away_runs"]); al.append(r["home_runs"]); sc_a.append(r["away_runs"])
            # como visitante: gana si scored > allowed
        else:
            sc.append(r["home_runs"]); al.append(r["away_runs"]); sc_h.append(r["home_runs"])
    wins=sum(1 for s,a in zip(sc,al) if s>a)
    home_games=df[(df["home_id"]==tid)&(df["date"]<before_date)]
    home_wins=home_games["home_win"].sum() if "home_win" in home_games.columns else 0
    home_n=len(home_games)
    away_games=df[(df["away_id"]==tid)&(df["date"]<before_date)]
    away_wins=(away_games["away_runs"]>away_games["home_runs"]).sum()
    away_n=len(away_games)
    return {
        "rpg":np.mean(sc),"rpg_l10":np.mean(sc[-10:]),
        "ra_pg":np.mean(al),"win_pct":wins/len(sc),
        "scored":sc,"allowed":al,
        "rpg_away":np.mean(sc_a) if sc_a else np.mean(sc),
        "rpg_home":np.mean(sc_h) if sc_h else np.mean(sc),
        "std":float(np.std(sc[-20:])) if len(sc)>=5 else 2.5,
        "home_win_pct":home_wins/home_n if home_n>=5 else wins/len(sc),
        "away_win_pct":away_wins/away_n if away_n>=5 else wins/len(sc),
    }

def h2h_before(df,aid,hid,before_date):
    mask=(((df["away_id"]==aid)&(df["home_id"]==hid))|
          ((df["away_id"]==hid)&(df["home_id"]==aid)))&(df["date"]<before_date)
    sub=df[mask]
    totals=sub["total"].tolist()
    # tasa de victoria del home team en H2H
    h2h_home_wins=(sub[sub["home_id"]==hid]["home_win"]).sum() if len(sub)>0 else 0
    return totals, len(sub), h2h_home_wins/len(sub) if len(sub)>0 else 0.5

# ── features extendidas (O/U + Winner) ───────────────────────────────────────
def make_features(a_st,h_st,away_era,home_era,h2h_totals,
                  h2h_n,h2h_home_win_rate,wx,pf,month):
    """
    Devuelve features base (para O/U) + features extra (para winner).
    """
    w_adj=wind_factor(wx["wind_mph"],wx.get("roof","open"))
    t_adj=temp_factor(wx["temp_f"],wx.get("roof","open"))
    h2h_avg=np.mean(h2h_totals) if len(h2h_totals)>=2 else None

    base=build_features(
        a_st["rpg"],a_st["rpg_l10"],a_st["win_pct"],
        str(away_era),"1.30",".720","",a_st["ra_pg"],
        h_st["rpg"],h_st["rpg_l10"],h_st["win_pct"],
        str(home_era),"1.30",".720","",h_st["ra_pg"],
        h2h_avg,h2h_n,month,
        away_rpg_away=a_st["rpg_away"],home_rpg_home=h_st["rpg_home"],
        away_std=a_st["std"],home_std=h_st["std"],park_factor=pf,
    )+[w_adj,t_adj,wx["wind_mph"],wx["temp_f"]]

    win_diff   = h_st["home_win_pct"] - a_st["away_win_pct"]
    rpg_diff   = h_st["rpg_home"]     - a_st["rpg_away"]
    era_adv    = away_era - home_era
    ra_diff    = a_st["ra_pg"] - h_st["ra_pg"]
    h2h_hw     = h2h_home_win_rate - 0.5
    home_adv   = 0.04

    extra=[win_diff, rpg_diff, era_adv, ra_diff, h2h_hw, home_adv,
           h_st["home_win_pct"], a_st["away_win_pct"],
           h_st["std"], a_st["std"]]

    return base, base+extra

# ── Modelos ───────────────────────────────────────────────────────────────────
def train_ou_model(rows):
    X=np.array([r["features"] for r in rows],dtype=float)
    y=np.array([r["target"]   for r in rows],dtype=float)
    pipe=Pipeline([("sc",StandardScaler()),
                   ("m",XGBRegressor(n_estimators=500,learning_rate=0.03,
                                      max_depth=4,subsample=0.80,
                                      reg_alpha=0.5,reg_lambda=1.5,
                                      random_state=42,verbosity=0))])
    ridge=Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=5.0))])
    tscv=TimeSeriesSplit(n_splits=5)
    cv=cross_val_score(pipe,X,y,cv=tscv,scoring="neg_mean_absolute_error")
    pipe.fit(X,y); ridge.fit(X,y)
    return pipe,ridge,float(-cv.mean())

def train_winner_model(rows):
    """
    Clasificador calibrado: devuelve P(home_team_wins).
    Calibrado = la probabilidad tiene significado real (60% → gana 60% del tiempo).
    """
    X=np.array([r["features"] for r in rows],dtype=float)
    y=np.array([r["target"]   for r in rows],dtype=int)

    xgb=XGBClassifier(n_estimators=500,learning_rate=0.03,max_depth=4,
                       subsample=0.80,reg_alpha=0.5,reg_lambda=1.5,
                       use_label_encoder=False,eval_metric="logloss",
                       random_state=42,verbosity=0)
    # Calibracion isotonica — hace que P=0.60 realmente gane 60% del tiempo
    pipe=Pipeline([("sc",StandardScaler()),
                   ("m",CalibratedClassifierCV(xgb,method="isotonic",cv=3))])
    tscv=TimeSeriesSplit(n_splits=5)
    cv=cross_val_score(pipe,X,y,cv=tscv,scoring="accuracy")
    pipe.fit(X,y)
    return pipe,float(cv.mean())

# ── build training rows ───────────────────────────────────────────────────────
def build_rows(df_games, wx_cache):
    """
    Construye features de entrenamiento sin fuga de datos.
    Para cada juego usa solo datos ANTERIORES a la fecha del juego
    (rolling_stats + h2h_before), igual que el walk-forward de 2026.
    ERA se aproxima con ra_pg del rolling de cada equipo (no hay
    cache de pitchers individuales para 2024/2025).
    """
    ou_rows=[]; ml_rows=[]
    n=len(df_games)
    for i,(_,g) in enumerate(df_games.sort_values("date").iterrows()):
        if (i+1)%500==0: print(f"   build_rows {i+1}/{n}...",end="\r")
        gdate=g["date"]
        aid=int(g["away_id"]); hid=int(g["home_id"])

        a_st=rolling_stats(df_games,aid,gdate)
        h_st=rolling_stats(df_games,hid,gdate)
        if not a_st or not h_st: continue

        h2h_totals,h2h_n,h2h_hw=h2h_before(df_games,aid,hid,gdate)

        # ERA proxy: carreras permitidas por juego hasta esa fecha
        away_era=a_st["ra_pg"]; home_era=h_st["ra_pg"]

        venue=str(g.get("venue",""))
        date_s=gdate.strftime("%Y-%m-%d") if hasattr(gdate,"strftime") else str(gdate)[:10]
        pf=get_park_factor(venue)
        wx=lookup(wx_cache,venue,date_s) if wx_cache else {"wind_mph":7.,"temp_f":72.,"roof":"open"}

        base,extended=make_features(a_st,h_st,away_era,home_era,
                                     h2h_totals,h2h_n,h2h_hw,wx,pf,gdate.month)
        ou_rows.append({"features":base,"target":float(g["total"])})
        ml_rows.append({"features":extended,"target":int(g.get("home_win",0))})

    print()
    return ou_rows,ml_rows

# ── prediccion ────────────────────────────────────────────────────────────────
def predict_ou(pipe_ou,ridge_ou,feats_base):
    X=np.array([feats_base],dtype=float)
    return float(pipe_ou.predict(X)[0])*0.65+float(ridge_ou.predict(X)[0])*0.35

def predict_winner(pipe_ml,feats_ext):
    X=np.array([feats_ext],dtype=float)
    proba=pipe_ml.predict_proba(X)[0]
    # proba[1] = P(home wins)
    return float(proba[1])

# ── walk-forward reutilizable ─────────────────────────────────────────────────
def walkforward(df_games, pipe_ou, ridge_ou, pipe_ml, wx_cache,
                cutoff_str, pitcher_cache=None):
    """
    Walk-forward sobre df_games desde cutoff_str.
    pitcher_cache: dict {(pitcher_id, year): {era: ...}} o None.
    Si es None usa ra_pg como proxy de ERA (modo sin cache de pitchers).
    """
    cutoff=pd.Timestamp(cutoff_str)
    game_dates=sorted(df_games[df_games["date"]>=cutoff]["date"].unique())
    records=[]
    for di,gdate in enumerate(game_dates):
        for _,g in df_games[df_games["date"]==gdate].iterrows():
            aid=int(g["away_id"]); hid=int(g["home_id"])
            a_st=rolling_stats(df_games,aid,gdate)
            h_st=rolling_stats(df_games,hid,gdate)
            if not a_st or not h_st: continue

            h2h_totals,h2h_n,h2h_hw=h2h_before(df_games,aid,hid,gdate)
            venue=str(g.get("venue",""))
            date_s=gdate.strftime("%Y-%m-%d")
            pf=get_park_factor(venue)
            wx=lookup(wx_cache,venue,date_s)

            if pitcher_cache is not None:
                def get_era(col):
                    try:
                        pid=g.get(col)
                        if pid and not pd.isna(pid):
                            return float(pitcher_cache.get((int(pid),2026),{}).get("era",4.20))
                    except: pass
                    return 4.20
                away_era=get_era("away_pitcher_id"); home_era=get_era("home_pitcher_id")
            else:
                away_era=a_st["ra_pg"]; home_era=h_st["ra_pg"]

            feats_base,feats_ext=make_features(a_st,h_st,away_era,home_era,
                                                h2h_totals,h2h_n,h2h_hw,wx,pf,gdate.month)
            ou_proj=predict_ou(pipe_ou,ridge_ou,feats_base)
            ou_line=nearest_line(ou_proj)
            ou_rec="OVER" if ou_proj>=ou_line else "UNDER"
            ou_diff=abs(ou_proj-ou_line)

            p_home=predict_winner(pipe_ml,feats_ext)
            p_away=1-p_home
            if p_home>=0.55:   ml_pick="HOME"; ml_conf=p_home
            elif p_away>=0.55: ml_pick="AWAY"; ml_conf=p_away
            else:               ml_pick="SKIP"; ml_conf=max(p_home,p_away)

            actual_total=float(g["total"])
            actual_winner=int(g.get("home_win",0))

            ou_correct=(ou_rec=="OVER" and actual_total>ou_line) or \
                       (ou_rec=="UNDER" and actual_total<ou_line)
            if ml_pick=="HOME":  ml_correct=(actual_winner==1)
            elif ml_pick=="AWAY": ml_correct=(actual_winner==0)
            else:                ml_correct=None

            combined=None
            if ou_rec=="UNDER" and ml_pick!="SKIP":
                combined={"ou_correct":ou_correct,"ml_correct":ml_correct}

            records.append({
                "date":gdate,
                "away":g["away_name"].split()[-1],"home":g["home_name"].split()[-1],
                "away_era":away_era,"home_era":home_era,
                "ou_proj":round(ou_proj,2),"ou_line":ou_line,"ou_rec":ou_rec,
                "ou_diff":round(ou_diff,2),"ou_correct":ou_correct,
                "p_home":round(p_home,3),"ml_pick":ml_pick,"ml_conf":round(ml_conf,3),
                "ml_correct":ml_correct,
                "actual_total":int(actual_total),"actual_winner":actual_winner,
                "has_combined":combined is not None,
                "comb_ou_ok":combined["ou_correct"] if combined else None,
                "comb_ml_ok":combined["ml_correct"] if combined else None,
            })
        if (di+1)%10==0: print(f"   {di+1}/{len(game_dates)} dias...",end="\r")
    print()
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  MLB BACKTEST v5 — O/U + VICTORIAS (MONEYLINE)")
    print("="*65)

    print("\n1. Cargando datos...")
    df24=cached_df(CACHE/"games_2024.pkl",lambda:load_games("2024-03-20","2024-09-29"))
    df25=cached_df(CACHE/"games_2025.pkl",lambda:load_games("2025-03-20","2025-09-28"))
    df26=cached_df(CACHE/"games_2026.pkl",lambda:load_with_pitchers("2026-03-19","2026-06-04"))
    # Agregar home_win si no existe
    for df_ in [df24,df25,df26]:
        if "home_win" not in df_.columns:
            df_["home_win"]=(df_["home_runs"]>df_["away_runs"]).astype(int)
    h24,p24=cached_stats(CACHE/"stats_2024.json",lambda:load_team_stats(2024))
    h25,p25=cached_stats(CACHE/"stats_2025.json",lambda:load_team_stats(2025))
    h26,p26=cached_stats(CACHE/"stats_2026.json",lambda:load_team_stats(2026))
    print(f"   2024:{len(df24)}j  2025:{len(df25)}j  2026:{len(df26)}j")

    print("\n2. Pitchers 2026...")
    pc_path=CACHE/"pitchers_2026.json"
    pitcher_cache={}
    if pc_path.exists():
        raw=json.load(open(pc_path))
        pitcher_cache={(int(k.split("_")[0]),int(k.split("_")[1])):v for k,v in raw.items()}
    print(f"   {len(pitcher_cache)} pitchers cacheados")

    print("\n3. Clima...")
    wx24=build_season_weather_cache(2024,"2024-03-20","2024-09-29")
    wx25=build_season_weather_cache(2025,"2025-03-20","2025-09-28")
    wx26=build_season_weather_cache(2026,"2026-03-19","2026-06-05")

    print("\n4. Features de entrenamiento...")
    print("   2024 (sin fuga de datos)...")
    ou24,ml24=build_rows(df24,wx24)
    print("   2025 (sin fuga de datos)...")
    ou25,ml25=build_rows(df25,wx25)
    all_ou=ou24+ou25; all_ml=ml24+ml25
    print(f"   O/U rows: {len(all_ou)}  |  Winner rows: {len(all_ml)}")

    # ── VALIDACION: entrena solo en 2024, evalua 2025 completo ──────────────────
    print("\n5. [VALIDACION] Entrenando solo con 2024 → test 2025...")
    pipe_ou_val,ridge_ou_val,mae_val=train_ou_model(ou24)
    pipe_ml_val,acc_val=train_winner_model(ml24)
    print(f"   CV MAE: {mae_val:.3f} runs  |  CV Accuracy: {acc_val*100:.1f}%")
    print("   Walk-forward 2025 (entrenado solo en 2024)...")
    df_val=walkforward(df25,pipe_ou_val,ridge_ou_val,pipe_ml_val,
                       wx25,"2025-04-01",pitcher_cache=None)
    print(f"   {len(df_val)} predicciones 2025")

    # ── PRODUCCION: entrena en 2024+2025, evalua 2026 ───────────────────────────
    print("\n6. Entrenando Modelo A (O/U regresion) [2024+2025]...")
    pipe_ou,ridge_ou,mae_cv=train_ou_model(all_ou)
    print(f"   CV MAE: {mae_cv:.3f} runs")

    print("\n7. Entrenando Modelo B (Winner clasificacion) [2024+2025]...")
    pipe_ml,acc_cv=train_winner_model(all_ml)
    print(f"   CV Accuracy: {acc_cv*100:.1f}%")

    print("\n8. Backtest walk-forward 2026...")
    df26_res=walkforward(df26,pipe_ou,ridge_ou,pipe_ml,
                         wx26,"2026-04-01",pitcher_cache=pitcher_cache)
    print(f"   {len(df26_res)} predicciones 2026")

    print_results(df_val, mae_val, acc_val, label="VALIDACION — train:2024 test:2025 (temporada completa)")
    print_results(df26_res, mae_cv, acc_cv, label="PRODUCCION — train:2024+2025 test:2026")
    plot_v5(df26_res,mae_cv,acc_cv)

# ── resultados ────────────────────────────────────────────────────────────────
def print_results(df,mae_cv,acc_cv,label="RESULTADOS v5"):
    n=len(df)

    # O/U
    under_df=df[df["ou_rec"]=="UNDER"]
    ou_all_acc=df["ou_correct"].mean()*100
    under_acc=under_df["ou_correct"].mean()*100 if len(under_df) else 0
    under_pnl=under_df["ou_correct"].map({True:100.,False:-110.}).sum()

    # Moneyline
    ml_df=df[df["ml_pick"]!="SKIP"]
    ml_acc=ml_df["ml_correct"].mean()*100 if len(ml_df) else 0
    ml_pnl=ml_df["ml_correct"].map({True:100.,False:-110.}).sum()

    # Alta confianza ML
    ml_hi=df[df["ml_conf"]>=0.60]
    ml_hi_acc=ml_hi["ml_correct"].mean()*100 if len(ml_hi) else 0
    ml_hi_pnl=ml_hi["ml_correct"].map({True:100.,False:-110.}).sum()

    # Combinado
    comb_df=df[df["has_combined"]]
    if len(comb_df):
        comb_ou_acc=comb_df["comb_ou_ok"].mean()*100
        comb_ml_acc=comb_df["comb_ml_ok"].mean()*100
        comb_pnl_ou=comb_df["comb_ou_ok"].map({True:100.,False:-110.}).sum()
        comb_pnl_ml=comb_df["comb_ml_ok"].map({True:100.,False:-110.}).sum()
    else:
        comb_ou_acc=comb_ml_acc=comb_pnl_ou=comb_pnl_ml=0

    print("\n"+"="*68)
    print(f"  {label}")
    print("="*68)
    print(f"\n  {'':42s} {'Acc':>6}  {'Picks':>6}  {'P&L':>9}")
    print(f"  {'─'*42} {'─'*6}  {'─'*6}  {'─'*9}")
    print(f"  {'MODELO A — O/U':42s}")
    print(f"  {'  Todos los picks':42s} {ou_all_acc:5.1f}%  {n:>6}")
    print(f"  {'  Solo UNDER':42s} {under_acc:5.1f}%  {len(under_df):>6}  ${under_pnl:>+8.0f}")
    print()
    print(f"  {'MODELO B — VICTORIAS (Moneyline)':42s}")
    print(f"  {'  Todos los picks (P>=0.55)':42s} {ml_acc:5.1f}%  {len(ml_df):>6}  ${ml_pnl:>+8.0f}")
    print(f"  {'  Alta confianza (P>=0.60)':42s} {ml_hi_acc:5.1f}%  {len(ml_hi):>6}  ${ml_hi_pnl:>+8.0f}")
    print()
    print(f"  {'SENAL COMBINADA (UNDER + Ganador)':42s}")
    print(f"  {'  O/U accuracy cuando hay senal':42s} {comb_ou_acc:5.1f}%  {len(comb_df):>6}  ${comb_pnl_ou:>+8.0f}")
    print(f"  {'  ML accuracy cuando hay senal':42s} {comb_ml_acc:5.1f}%  {len(comb_df):>6}  ${comb_pnl_ml:>+8.0f}")

    # Por mes
    print(f"\n  Moneyline por mes (P>=0.55):")
    df_m=df[df["ml_pick"]!="SKIP"].copy()
    df_m["month"]=pd.to_datetime(df_m["date"]).dt.strftime("%b")
    for mo,sub in df_m.groupby("month"):
        a=sub["ml_correct"].mean()*100
        p=sub["ml_correct"].map({True:100.,False:-110.}).sum()
        print(f"    {mo}: {len(sub):3d}j  acc={a:.1f}%  P&L=${p:+.0f}")

    # ERA elite
    era_el=df[(df["ml_pick"]!="SKIP")&(df["away_era"]<3.5)&(df["home_era"]<3.5)]
    if len(era_el):
        a=era_el["ml_correct"].mean()*100
        p=era_el["ml_correct"].map({True:100.,False:-110.}).sum()
        print(f"\n  ML con ERA<3.5 ambos: {len(era_el)}j  acc={a:.1f}%  P&L=${p:+.0f}")

    # Home vs Away win rate global
    home_wins=df["actual_winner"].mean()*100
    print(f"\n  Tasa real de victoria del local: {home_wins:.1f}%")
    print(f"  Modelo B CV accuracy: {acc_cv*100:.1f}%")
    print(f"  Modelo A CV MAE: {mae_cv:.3f} runs")

# ── imagen ─────────────────────────────────────────────────────────────────────
def plot_v5(df,mae_cv,acc_cv):
    plt.rcParams.update({"figure.facecolor":"#fff","axes.facecolor":"#fafafa",
                          "axes.edgecolor":"#d1d5db","text.color":"#111",
                          "xtick.color":"#666","ytick.color":"#666",
                          "grid.color":"#e5e7eb","font.size":9})
    fig=plt.figure(figsize=(24,22))
    gs=gridspec.GridSpec(3,2,figure=fig,hspace=0.50,wspace=0.38,
                         top=0.93,bottom=0.04,left=0.06,right=0.97)

    # ── 1. P&L comparativo ───────────────────────────────────────────────────
    ax1=fig.add_subplot(gs[0,:])
    by_day_ou=df[df["ou_rec"]=="UNDER"].groupby("date").agg(
        pnl=("ou_correct",lambda x:x.map({True:100.,False:-110.}).sum())).reset_index()
    by_day_ml=df[df["ml_pick"]!="SKIP"].groupby("date").agg(
        pnl=("ml_correct",lambda x:x.map({True:100.,False:-110.}).sum())).reset_index()
    by_day_hi=df[df["ml_conf"]>=0.60].groupby("date").agg(
        pnl=("ml_correct",lambda x:x.map({True:100.,False:-110.}).sum())).reset_index()

    all_dates=pd.to_datetime(sorted(df["date"].unique()))
    def cum_series(byday):
        s=pd.Series(0.,index=all_dates)
        bd=byday.set_index(pd.to_datetime(byday["date"]))
        for d,row in bd.iterrows():
            if d in s.index: s[d]=row["pnl"]
        return s.cumsum()

    c_ou=cum_series(by_day_ou); c_ml=cum_series(by_day_ml); c_hi=cum_series(by_day_hi)
    ax1.plot(all_dates,c_ou,color="#374151",lw=2,   label=f"UNDER O/U ({df['ou_rec'].eq('UNDER').sum()}j)")
    ax1.plot(all_dates,c_ml,color="#dc2626",lw=1.8, label=f"Moneyline P>=0.55 ({(df['ml_pick']!='SKIP').sum()}j)",alpha=0.7)
    ax1.plot(all_dates,c_hi,color="#166534",lw=2.5, label=f"Moneyline P>=0.60 ({(df['ml_conf']>=0.60).sum()}j)")
    ax1.fill_between(all_dates,c_hi,0,where=c_hi>=0,alpha=0.12,color="#166534")
    ax1.fill_between(all_dates,c_hi,0,where=c_hi<0, alpha=0.12,color="#dc2626")
    ax1.axhline(0,color="#374151",lw=1)
    ax1.set_ylabel("P&L acumulado (USD)"); ax1.grid(alpha=0.4)
    under_acc=df[df["ou_rec"]=="UNDER"]["ou_correct"].mean()*100
    ml_hi_acc=df[df["ml_conf"]>=0.60]["ml_correct"].mean()*100 if (df["ml_conf"]>=0.60).sum() else 0
    ax1.set_title(
        f"P&L acumulado — UNDER {under_acc:.1f}% vs Moneyline {ml_hi_acc:.1f}% (P>=0.60)\n"
        f"Modelos: O/U MAE={mae_cv:.2f} runs  |  Winner CV={acc_cv*100:.1f}%",
        fontsize=12,fontweight="bold",pad=10,loc="left")
    ax1.legend(fontsize=8,loc="upper left")
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 2. Accuracy ML por nivel de confianza ────────────────────────────────
    ax2=fig.add_subplot(gs[1,0])
    conf_bins=pd.cut(df[df["ml_pick"]!="SKIP"]["ml_conf"],
                     bins=[0.55,0.57,0.60,0.63,0.67,1.0],
                     labels=["0.55","0.57","0.60","0.63","0.67+"])
    conf_acc=df[df["ml_pick"]!="SKIP"].groupby(conf_bins,observed=True).agg(
        acc=("ml_correct","mean"),n=("ml_correct","count")).reset_index()
    bc=["#166534" if a>=0.524 else "#dc2626" for a in conf_acc["acc"]]
    ax2.bar(range(len(conf_acc)),conf_acc["acc"]*100,color=bc,alpha=0.85,edgecolor="#fff",width=0.6)
    for xi,(_,r) in enumerate(conf_acc.iterrows()):
        ax2.text(xi,r["acc"]*100+0.5,f"{r['acc']*100:.1f}%\n({int(r['n'])}j)",ha="center",fontsize=8)
    ax2.axhline(52.4,color="#374151",lw=1.2,linestyle="--",label="Break-even")
    ax2.axhline(50,color="#aaa",lw=0.8,linestyle=":")
    ax2.set_xticks(range(len(conf_acc)))
    ax2.set_xticklabels([f"P={l}" for l in conf_acc["ml_conf"].astype(str)],fontsize=9)
    ax2.set_ylim(0,80); ax2.set_ylabel("Accuracy %")
    ax2.set_title("Moneyline accuracy por nivel de confianza del modelo",fontsize=11,pad=8,loc="left")
    ax2.legend(fontsize=8); ax2.grid(axis="y",alpha=0.4)

    # ── 3. UNDER vs ML — scatter (cuando coinciden) ──────────────────────────
    ax3=fig.add_subplot(gs[1,1])
    comb=df[df["has_combined"]].copy()
    if len(comb)>0:
        both_ok  = comb[(comb["comb_ou_ok"]==True)  & (comb["comb_ml_ok"]==True)]
        ou_only  = comb[(comb["comb_ou_ok"]==True)  & (comb["comb_ml_ok"]==False)]
        ml_only  = comb[(comb["comb_ou_ok"]==False) & (comb["comb_ml_ok"]==True)]
        both_bad = comb[(comb["comb_ou_ok"]==False) & (comb["comb_ml_ok"]==False)]
        for sub,col,lbl in [(both_ok,"#166534","Ambos correctos"),
                             (ou_only,"#58a6ff","Solo O/U correcto"),
                             (ml_only,"#ffa657","Solo ML correcto"),
                             (both_bad,"#dc2626","Ambos incorrectos")]:
            if len(sub):
                ax3.scatter(sub["ou_diff"],sub["ml_conf"],
                            color=col,alpha=0.65,s=28,label=f"{lbl} ({len(sub)})")
        ax3.axvline(0.5,color="#aaa",lw=1,linestyle="--")
        ax3.axhline(0.60,color="#aaa",lw=1,linestyle="--")
        ax3.set_xlabel("Diferencia O/U (cuanto bajo del line)")
        ax3.set_ylabel("Confianza modelo winner (P home/away)")
        ax3.set_title("Señal combinada: O/U diff vs confianza winner",fontsize=11,pad=8,loc="left")
        ax3.legend(fontsize=7); ax3.grid(alpha=0.3)
        # Zona de alta confianza
        ax3.fill_between([0.5,max(comb["ou_diff"].max(),1.5)],[0.60,0.60],[1.0,1.0],
                          alpha=0.08,color="#166534",label="Zona ideal")
        n_ideal=len(comb[(comb["ou_diff"]>=0.5)&(comb["ml_conf"]>=0.60)])
        if n_ideal:
            ideal=comb[(comb["ou_diff"]>=0.5)&(comb["ml_conf"]>=0.60)]
            ia=ideal["comb_ou_ok"].mean()*100
            ax3.text(0.55,0.995,f"Zona ideal: {n_ideal}j  O/U {ia:.1f}%",
                     transform=ax3.transAxes,fontsize=8,color="#166534",va="top")

    # ── 4. Home win rate real vs predicho por mes ─────────────────────────────
    ax4=fig.add_subplot(gs[2,0])
    df["month_n"]=pd.to_datetime(df["date"]).dt.month
    months=sorted(df["month_n"].unique())
    real_hw=[df[df["month_n"]==m]["actual_winner"].mean()*100 for m in months]
    pred_hw=[df[df["month_n"]==m]["p_home"].mean()*100 for m in months]
    ml_mo_acc=[]
    for m in months:
        sub=df[(df["month_n"]==m)&(df["ml_pick"]!="SKIP")]
        ml_mo_acc.append(sub["ml_correct"].mean()*100 if len(sub) else 50)
    month_names=["Mar","Abr","May","Jun","Jul","Ago","Sep"]
    x=np.arange(len(months)); w=0.28
    ax4.bar(x-w,real_hw,w,color="#374151",alpha=0.7,label="% victorias locales reales")
    ax4.bar(x,pred_hw,w,color="#58a6ff",alpha=0.7,label="% predicho (prob media)")
    ax4.bar(x+w,ml_mo_acc,w,color="#166534",alpha=0.85,label="Accuracy modelo ML")
    ax4.axhline(52.4,color="#dc2626",lw=1,linestyle="--")
    ax4.axhline(50,color="#aaa",lw=0.8,linestyle=":")
    ax4.set_xticks(x)
    ax4.set_xticklabels([month_names[m-3] for m in months],fontsize=9)
    ax4.set_ylim(0,80); ax4.set_ylabel("%")
    ax4.set_title("Victorias del local: real vs predicho vs accuracy por mes",fontsize=11,pad=8,loc="left")
    ax4.legend(fontsize=7); ax4.grid(axis="y",alpha=0.4)

    # ── 5. Señal combinada — tabla resumen ────────────────────────────────────
    ax5=fig.add_subplot(gs[2,1])
    ax5.axis("off")
    rows_tbl=[
        ["","Picks","Acc","P&L (flat)"],
        ["─"*18,"─"*6,"─"*6,"─"*10],
        ["UNDER (O/U)",
         str((df["ou_rec"]=="UNDER").sum()),
         f"{df[df['ou_rec']=='UNDER']['ou_correct'].mean()*100:.1f}%",
         f"${df[df['ou_rec']=='UNDER']['ou_correct'].map({True:100.,False:-110.}).sum():+.0f}"],
        ["ML P>=0.55",
         str((df["ml_pick"]!="SKIP").sum()),
         f"{df[df['ml_pick']!='SKIP']['ml_correct'].mean()*100:.1f}%",
         f"${df[df['ml_pick']!='SKIP']['ml_correct'].map({True:100.,False:-110.}).sum():+.0f}"],
        ["ML P>=0.60",
         str((df["ml_conf"]>=0.60).sum()),
         f"{df[df['ml_conf']>=0.60]['ml_correct'].mean()*100:.1f}%" if (df["ml_conf"]>=0.60).sum() else "—",
         f"${df[df['ml_conf']>=0.60]['ml_correct'].map({True:100.,False:-110.}).sum():+.0f}" if (df["ml_conf"]>=0.60).sum() else "$0"],
        ["─"*18,"─"*6,"─"*6,"─"*10],
        ["COMBINADO (UNDER+ML)","","",""],
        [" O/U acc en ese subset",
         str(len(df[df["has_combined"]])),
         f"{df[df['has_combined']]['comb_ou_ok'].mean()*100:.1f}%" if len(df[df["has_combined"]]) else "—",
         f"${df[df['has_combined']]['comb_ou_ok'].map({True:100.,False:-110.}).sum():+.0f}" if len(df[df["has_combined"]]) else "$0"],
        [" ML acc en ese subset",
         str(len(df[df["has_combined"]])),
         f"{df[df['has_combined']]['comb_ml_ok'].mean()*100:.1f}%" if len(df[df["has_combined"]]) else "—",
         f"${df[df['has_combined']]['comb_ml_ok'].map({True:100.,False:-110.}).sum():+.0f}" if len(df[df["has_combined"]]) else "$0"],
    ]
    tbl=ax5.table(cellText=rows_tbl,loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor("#fafafa"); cell.set_edgecolor("#d1d5db")
        cell.set_text_props(color="#111")
        if r==0: cell.set_text_props(color="#374151",fontweight="bold"); cell.set_facecolor("#f0f0f0")
        if r==1 or r==6: cell.set_facecolor("#e5e7eb")
        if c==2 and r>1:
            try:
                v=float(cell.get_text().get_text().replace("%",""))
                if v>=55: cell.set_facecolor("#f0fdf4"); cell.set_text_props(color="#166534",fontweight="bold")
                elif v<52: cell.set_facecolor("#fef2f2"); cell.set_text_props(color="#dc2626")
            except: pass
    ax5.set_title("Resumen de señales",fontsize=11,pad=8,loc="left")

    fig.suptitle(
        f"MLB v5 — O/U + Victorias (Moneyline)  |  2026 Sem.2 → Jun 4\n"
        f"Dos modelos: Regresion (O/U) + Clasificador calibrado (Winner)",
        fontsize=13,fontweight="bold",y=0.975)
    out="/Users/vaquera/Documents/NBA-SPURS/mlb_backtest_v5.png"
    plt.savefig(out,dpi=155,bbox_inches="tight",facecolor="#fff")
    print(f"\n  Imagen: {out}")

if __name__=="__main__":
    main()
