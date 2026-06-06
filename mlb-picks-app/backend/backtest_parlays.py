#!/usr/bin/env python3
"""
MLB Backtest Parlays — Simulacion de parlays recomendados, temporada 2026.

Replica la logica de build_parlays() de main.py sobre el walk-forward de v5:
  ou_2  : top-2 O/U por confianza
  ou_3  : top-3 O/U por confianza
  win_2 : top-2 WIN por confianza (sin duplicados por partido)
  win_3 : top-3 WIN por confianza
  stack : OVER + ganador del mismo partido (correlacion positiva)
  mix_3 : top-3 combinando O/U y WIN
  mix_5 : top-5 combinando O/U y WIN

Un parlay acierta solo si TODOS los legs aciertan.
P&L flat-bet: $10 por parlay por dia.
"""

import sys, importlib, json, math
from pathlib import Path
from collections import defaultdict

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

np=ensure("numpy"); pd=ensure("pandas")
ensure("matplotlib")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import numpy as np, pandas as pd

import urllib.request, time, pickle
from sklearn.pipeline    import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import Ridge

from weather_batch import build_season_weather_cache, lookup, wind_factor, temp_factor
from park_factors   import get_park_factor
from ml_model       import build_features, FEATURE_NAMES

CACHE = Path(__file__).parent / "cache"
LINES = [6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]
BET   = 10.0   # apuesta fija por parlay por dia
PAYOUTS = {2: 2.64, 3: 6.0, 4: 12.28, 5: 24.35}
SIGMA = 3.5

def nearest_line(v): return min(LINES, key=lambda x: abs(x-v))

def ou_prob(diff):
    z = abs(diff) / SIGMA
    return 50 + 50 * math.erf(z / math.sqrt(2))

# ── fetch helpers (identicos a v5) ────────────────────────────────────────────
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

def load_with_pitchers(start, end):
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
                         "home_pitcher_id":hp.get("id")})
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

def cached_df(path, fn):
    p=Path(path)
    if p.exists(): return pd.read_pickle(p)
    df=fn(); df.to_pickle(p); return df

def cached_stats(path, fn):
    p=Path(path)
    if p.exists():
        d=json.load(open(p))
        return {int(k):v for k,v in d["hit"].items()},{int(k):v for k,v in d["pit"].items()}
    hit,pit=fn()
    json.dump({"hit":{str(k):v for k,v in hit.items()},"pit":{str(k):v for k,v in pit.items()}},open(p,"w"))
    return hit,pit

# ── rolling stats ─────────────────────────────────────────────────────────────
def rolling_stats(df, tid, before_date):
    mask=((df["away_id"]==tid)|(df["home_id"]==tid))&(df["date"]<before_date)
    sub=df[mask].sort_values("date")
    if len(sub)<5: return None
    sc=[]; al=[]; sc_a=[]; sc_h=[]
    for _,r in sub.iterrows():
        if r["away_id"]==tid:
            sc.append(r["away_runs"]); al.append(r["home_runs"]); sc_a.append(r["away_runs"])
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

def h2h_before(df, aid, hid, before_date):
    mask=(((df["away_id"]==aid)&(df["home_id"]==hid))|
          ((df["away_id"]==hid)&(df["home_id"]==aid)))&(df["date"]<before_date)
    sub=df[mask]
    totals=sub["total"].tolist()
    h2h_home_wins=(sub[sub["home_id"]==hid]["home_win"]).sum() if len(sub)>0 else 0
    return totals, len(sub), h2h_home_wins/len(sub) if len(sub)>0 else 0.5

def make_features(a_st, h_st, away_era, home_era, h2h_totals, h2h_n, h2h_hw, wx, pf, month):
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
    h2h_hw     = h2h_hw - 0.5
    extra=[win_diff, rpg_diff, era_adv, ra_diff, h2h_hw, 0.04,
           h_st["home_win_pct"], a_st["away_win_pct"],
           h_st["std"], a_st["std"]]
    return base, base+extra

# ── modelos ───────────────────────────────────────────────────────────────────
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
    return pipe, ridge, float(-cv.mean())

def train_winner_model(rows):
    X=np.array([r["features"] for r in rows],dtype=float)
    y=np.array([r["target"]   for r in rows],dtype=int)
    xgb=XGBClassifier(n_estimators=500,learning_rate=0.03,max_depth=4,
                       subsample=0.80,reg_alpha=0.5,reg_lambda=1.5,
                       use_label_encoder=False,eval_metric="logloss",
                       random_state=42,verbosity=0)
    pipe=Pipeline([("sc",StandardScaler()),
                   ("m",CalibratedClassifierCV(xgb,method="isotonic",cv=3))])
    tscv=TimeSeriesSplit(n_splits=5)
    cv=cross_val_score(pipe,X,y,cv=tscv,scoring="accuracy")
    pipe.fit(X,y)
    return pipe, float(cv.mean())

def build_rows(df_games, wx_cache):
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
        away_era=a_st["ra_pg"]; home_era=h_st["ra_pg"]
        venue=str(g.get("venue",""))
        date_s=gdate.strftime("%Y-%m-%d") if hasattr(gdate,"strftime") else str(gdate)[:10]
        pf=get_park_factor(venue)
        wx=lookup(wx_cache,venue,date_s) if wx_cache else {"wind_mph":7.,"temp_f":72.,"roof":"open"}
        base,extended=make_features(a_st,h_st,away_era,home_era,h2h_totals,h2h_n,h2h_hw,wx,pf,gdate.month)
        ou_rows.append({"features":base,"target":float(g["total"])})
        ml_rows.append({"features":extended,"target":int(g.get("home_win",0))})
    print()
    return ou_rows, ml_rows

# ── walk-forward → registros por partido ─────────────────────────────────────
def walkforward(df_games, pipe_ou, ridge_ou, pipe_ml, wx_cache, cutoff_str, pitcher_cache=None):
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
            X_ou=np.array([feats_base],dtype=float)
            X_ml=np.array([feats_ext],dtype=float)

            ou_raw=float(pipe_ou.predict(X_ou)[0])*0.65+float(ridge_ou.predict(X_ou)[0])*0.35
            ou_line=nearest_line(ou_raw)
            ou_diff=ou_raw-ou_line
            ou_rec="OVER" if ou_diff>=0 else "UNDER"
            ou_prob_val=ou_prob(ou_diff)
            # agreement bonus
            ml_ou_rec="OVER" if ou_raw>=ou_line else "UNDER"
            if ml_ou_rec==ou_rec: ou_prob_val=min(ou_prob_val+2.0,99.0)
            else:                  ou_prob_val=max(ou_prob_val-2.0,50.0)

            proba=pipe_ml.predict_proba(X_ml)[0]
            p_home=float(proba[1]); p_away=1-p_home
            win_side="home" if p_home>=p_away else "away"
            win_conf=max(p_home,p_away)*100

            actual_total=float(g["total"])
            actual_home_win=int(g.get("home_win",0))
            ou_correct=(ou_rec=="OVER" and actual_total>ou_line) or \
                       (ou_rec=="UNDER" and actual_total<ou_line)
            win_correct=(win_side=="home" and actual_home_win==1) or \
                        (win_side=="away" and actual_home_win==0)

            records.append({
                "date":        gdate,
                "gamePk":      int(g.get("away_id",0))*1000+int(g.get("home_id",0)),
                "away":        g["away_name"].split()[-1],
                "home":        g["home_name"].split()[-1],
                "ou_rec":      ou_rec,
                "ou_line":     ou_line,
                "ou_diff":     round(ou_diff,3),
                "ou_prob":     round(ou_prob_val,1),
                "ou_correct":  ou_correct,
                "win_side":    win_side,
                "win_conf":    round(win_conf,1),
                "win_correct": win_correct,
                "actual_total":int(actual_total),
                "actual_home_win":actual_home_win,
            })
        if (di+1)%10==0: print(f"   {di+1}/{len(game_dates)} dias...",end="\r")
    print()
    return pd.DataFrame(records)

# ── logica de parlays (replica build_parlays) ─────────────────────────────────
def sim_parlays_day(day_df):
    """
    Dado el DataFrame de un dia, devuelve lista de parlays simulados.
    Cada parlay: {type, legs, hit, pnl}
    """
    rows=day_df.to_dict("records")
    if not rows: return []

    # Ordenar legs
    ou_legs=sorted(rows, key=lambda x: -x["ou_prob"])
    win_legs_all=sorted(rows, key=lambda x: -x["win_conf"])
    # Win legs: sin duplicados de partido
    seen=set(); win_legs=[]
    for r in win_legs_all:
        if r["gamePk"] not in seen:
            win_legs.append(r); seen.add(r["gamePk"])

    parlays=[]

    def leg_ou(r):
        return ("ou", r["ou_rec"], r["ou_correct"], r["ou_prob"])
    def leg_ou_inv(r):
        # apuesta lo contrario de lo que dice el modelo O/U
        inv_rec = "UNDER" if r["ou_rec"]=="OVER" else "OVER"
        inv_correct = not r["ou_correct"]
        return ("ou_inv", inv_rec, inv_correct, 100-r["ou_prob"])
    def leg_win(r):
        return ("win", r["win_side"], r["win_correct"], r["win_conf"])

    def parlay(ptype, legs):
        hit=all(l[2] for l in legs)
        n=len(legs)
        pout=PAYOUTS.get(n, PAYOUTS[5])
        pnl=BET*(pout-1) if hit else -BET
        probs=[l[3] for l in legs]
        combined=1.0
        for p in probs: combined*=p/100
        return {"type":ptype,"legs":legs,"hit":hit,"pnl":pnl,"payout":pout,"n":n,
                "combined_prob":round(combined*100,1)}

    # ou_2
    if len(ou_legs)>=2:
        parlays.append(parlay("ou_2",[leg_ou(ou_legs[0]),leg_ou(ou_legs[1])]))
    # ou_3
    if len(ou_legs)>=3:
        parlays.append(parlay("ou_3",[leg_ou(r) for r in ou_legs[:3]]))
    # win_2
    if len(win_legs)>=2:
        parlays.append(parlay("win_2",[leg_win(win_legs[0]),leg_win(win_legs[1])]))
    # win_3
    if len(win_legs)>=3:
        parlays.append(parlay("win_3",[leg_win(r) for r in win_legs[:3]]))
    # stacks normales: OVER + ganador del mismo partido
    for r in rows:
        if r["ou_rec"]=="OVER":
            parlays.append(parlay(f"stack_{r['gamePk']}",[leg_ou(r),leg_win(r)]))
    # ── estrategias invertidas ────────────────────────────────────────────────
    # ou_inv_2: top-2 O/U pero apostando lo contrario
    if len(ou_legs)>=2:
        parlays.append(parlay("ou_inv_2",[leg_ou_inv(ou_legs[0]),leg_ou_inv(ou_legs[1])]))
    if len(ou_legs)>=3:
        parlays.append(parlay("ou_inv_3",[leg_ou_inv(r) for r in ou_legs[:3]]))
    # stack invertido: UNDER (contra modelo) + WIN mismo partido
    # correlacion positiva: juego cerrado de pocas carreras → favorito suele ganar
    for r in rows:
        if r["ou_rec"]=="OVER":  # modelo dice OVER, nosotros apostamos UNDER
            parlays.append(parlay(f"stack_inv_{r['gamePk']}",[leg_ou_inv(r),leg_win(r)]))
    # mix_3: top-3 de todos (ou y win mezclados, sin mismo gamePk dos veces)
    all_mixed=[]
    seen_mix=set()
    for r in sorted(rows, key=lambda x: -max(x["ou_prob"],x["win_conf"])):
        if r["gamePk"] not in seen_mix:
            # elige el leg de mayor confianza para este partido
            if r["ou_prob"]>=r["win_conf"]:
                all_mixed.append(leg_ou(r))
            else:
                all_mixed.append(leg_win(r))
            seen_mix.add(r["gamePk"])
    if len(all_mixed)>=3:
        parlays.append(parlay("mix_3",all_mixed[:3]))
    if len(all_mixed)>=5:
        parlays.append(parlay("mix_5",all_mixed[:5]))

    return parlays

# ── agregar stacks por dia ────────────────────────────────────────────────────
def aggregate_stacks(day_parlays):
    """Colapsa stacks por partido en un representativo por tipo."""
    normal_stacks =[p for p in day_parlays if p["type"].startswith("stack_") and not p["type"].startswith("stack_inv_")]
    inv_stacks    =[p for p in day_parlays if p["type"].startswith("stack_inv_")]
    rest          =[p for p in day_parlays if not p["type"].startswith("stack_")]
    if normal_stacks:
        best=max(normal_stacks,key=lambda x:x.get("combined_prob",0))
        best=dict(best); best["type"]="stack"; rest.append(best)
    if inv_stacks:
        best=max(inv_stacks,key=lambda x:x.get("combined_prob",0))
        best=dict(best); best["type"]="stack_inv"; rest.append(best)
    return rest

# ── simulacion completa ───────────────────────────────────────────────────────
def simulate(df):
    """Simula parlays dia a dia. Devuelve DataFrame con resultados por dia/tipo."""
    results=[]
    for gdate,group in df.groupby("date"):
        day_parlays=sim_parlays_day(group)
        day_parlays=aggregate_stacks(day_parlays)
        for p in day_parlays:
            results.append({
                "date":     gdate,
                "type":     p["type"],
                "n_legs":   p["n"],
                "hit":      p["hit"],
                "pnl":      p["pnl"],
                "payout":   p["payout"],
                "combined_prob": p.get("combined_prob",0),
            })
    return pd.DataFrame(results)

# ── impresion de resultados ───────────────────────────────────────────────────
PARLAY_LABELS={
    "ou_2":     "O/U   2-leg (normal)",
    "ou_3":     "O/U   3-leg (normal)",
    "win_2":    "WIN   2-leg",
    "win_3":    "WIN   3-leg",
    "stack":    "Stack 2-leg  OVER+WIN",
    "mix_3":    "Mixto 3-leg",
    "mix_5":    "Mixto 5-leg",
    "ou_inv_2": "O/U   2-leg INVERTIDO",
    "ou_inv_3": "O/U   3-leg INVERTIDO",
    "stack_inv":"Stack 2-leg  UNDER+WIN (inv)",
}

def print_results(sim_df, ou_acc=0.0, win_acc=0.0, label="BACKTEST PARLAYS 2026"):
    print("\n"+"="*72)
    print(f"  {label}")
    print(f"  Apuesta fija: ${BET:.0f} por parlay por dia")
    print("="*72)
    print(f"  {'Tipo':<38}  {'Dias':>5}  {'Hit%':>6}  {'ROI':>7}  {'P&L':>10}")
    print(f"  {'─'*38}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*10}")

    for ptype,label_t in PARLAY_LABELS.items():
        sub=sim_df[sim_df["type"]==ptype]
        if sub.empty: continue
        n=len(sub)
        hits=sub["hit"].sum()
        hit_pct=hits/n*100
        total_pnl=sub["pnl"].sum()
        total_wagered=n*BET
        roi=total_pnl/total_wagered*100
        print(f"  {label_t:<38}  {n:>5}  {hit_pct:>5.1f}%  {roi:>+6.1f}%  ${total_pnl:>+9.2f}")

    print()
    # Mejor tipo
    best_roi=None; best_t=None
    for ptype in PARLAY_LABELS:
        sub=sim_df[sim_df["type"]==ptype]
        if sub.empty or len(sub)<10: continue
        r=sub["pnl"].sum()/(len(sub)*BET)*100
        if best_roi is None or r>best_roi:
            best_roi=r; best_t=ptype
    if best_t:
        print(f"  Mejor tipo: {PARLAY_LABELS[best_t]} (ROI {best_roi:+.1f}%)")

    # Por mes
    print(f"\n  O/U 2-leg por mes:")
    sub=sim_df[sim_df["type"]=="ou_2"].copy()
    sub["month"]=pd.to_datetime(sub["date"]).dt.strftime("%b")
    for mo,g in sub.groupby("month"):
        n=len(g); h=g["hit"].sum(); pnl=g["pnl"].sum()
        print(f"    {mo}: {n:2d} dias  hit={h/n*100:.0f}%  P&L=${pnl:+.1f}")

    print(f"\n  WIN 2-leg por mes:")
    sub=sim_df[sim_df["type"]=="win_2"].copy()
    sub["month"]=pd.to_datetime(sub["date"]).dt.strftime("%b")
    for mo,g in sub.groupby("month"):
        n=len(g); h=g["hit"].sum(); pnl=g["pnl"].sum()
        print(f"    {mo}: {n:2d} dias  hit={h/n*100:.0f}%  P&L=${pnl:+.1f}")

# ── grafica ───────────────────────────────────────────────────────────────────
def plot_parlays(sim_df, ou_acc, win_acc):
    plt.rcParams.update({"figure.facecolor":"#fff","axes.facecolor":"#fafafa",
                          "axes.edgecolor":"#d1d5db","text.color":"#111",
                          "xtick.color":"#666","ytick.color":"#666",
                          "grid.color":"#e5e7eb","font.size":9})

    fig=plt.figure(figsize=(22,20))
    gs_fig=gridspec.GridSpec(3,2,figure=fig,hspace=0.50,wspace=0.38,
                             top=0.92,bottom=0.04,left=0.06,right=0.97)

    all_dates=pd.to_datetime(sorted(sim_df["date"].unique()))

    def cum_pnl(ptype):
        sub=sim_df[sim_df["type"]==ptype].copy()
        s=pd.Series(0.,index=all_dates,name=ptype)
        for d,row in sub.groupby("date"):
            dt=pd.Timestamp(d)
            if dt in s.index: s[dt]=row["pnl"].sum()
        return s.cumsum()

    # ── 1. P&L acumulado todos los tipos ─────────────────────────────────────
    ax1=fig.add_subplot(gs_fig[0,:])
    COLORS={"ou_2":"#374151","ou_3":"#6b7280","win_2":"#166534","win_3":"#22c55e",
            "stack":"#b45309","mix_3":"#1d4ed8","mix_5":"#7c3aed"}
    WIDTHS={"win_2":2.5,"win_3":2.0,"ou_2":2.0,"mix_3":1.8,"stack":1.8,"ou_3":1.5,"mix_5":1.5}
    for ptype,label_t in PARLAY_LABELS.items():
        if ptype not in sim_df["type"].values: continue
        c=cum_pnl(ptype)
        sub=sim_df[sim_df["type"]==ptype]
        n=len(sub); pnl=sub["pnl"].sum()
        ax1.plot(all_dates,c,color=COLORS.get(ptype,"#888"),
                 lw=WIDTHS.get(ptype,1.5),
                 label=f"{label_t} ({n}d  P&L=${pnl:+.0f})")
    ax1.axhline(0,color="#374151",lw=1)
    ax1.set_ylabel("P&L acumulado ($)")
    ax1.grid(alpha=0.4)
    ax1.set_title(
        f"P&L acumulado por tipo de parlay — 2026 (desde Abr)\n"
        f"Precision individual: O/U {ou_acc:.1f}%  |  WIN {win_acc:.1f}%  |  Apuesta fija ${BET:.0f}/parlay/dia",
        fontsize=11,fontweight="bold",pad=8,loc="left")
    ax1.legend(fontsize=7.5,loc="upper left",ncol=2)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 2. Hit rate por tipo ──────────────────────────────────────────────────
    ax2=fig.add_subplot(gs_fig[1,0])
    ptypes=[t for t in PARLAY_LABELS if t in sim_df["type"].values]
    hit_rates=[sim_df[sim_df["type"]==t]["hit"].mean()*100 for t in ptypes]
    expected=[PAYOUTS.get(sim_df[sim_df["type"]==t]["n_legs"].mode()[0],2.64) for t in ptypes]
    break_evens=[1/e*100 for e in expected]
    bar_colors=["#166534" if h>=b else "#dc2626" for h,b in zip(hit_rates,break_evens)]
    xp=range(len(ptypes))
    ax2.bar(xp,hit_rates,color=bar_colors,alpha=0.8,edgecolor="#fff",width=0.6)
    for xi,(t,h,be) in enumerate(zip(ptypes,hit_rates,break_evens)):
        ax2.plot([xi-0.3,xi+0.3],[be,be],color="#374151",lw=1.5,linestyle="--")
        ax2.text(xi,h+0.5,f"{h:.0f}%",ha="center",fontsize=8)
    ax2.set_xticks(xp)
    ax2.set_xticklabels([PARLAY_LABELS[t].replace(" (OVER+WIN mismo partido)","") for t in ptypes],
                        rotation=15,ha="right",fontsize=8)
    ax2.set_ylabel("Hit rate %")
    ax2.set_title("Hit rate por tipo (linea punteada = break-even)",fontsize=11,pad=8,loc="left")
    ax2.grid(axis="y",alpha=0.4)

    # ── 3. ROI por tipo ───────────────────────────────────────────────────────
    ax3=fig.add_subplot(gs_fig[1,1])
    rois=[sim_df[sim_df["type"]==t]["pnl"].sum()/(len(sim_df[sim_df["type"]==t])*BET)*100
          for t in ptypes]
    roi_colors=["#166534" if r>0 else "#dc2626" for r in rois]
    ax3.bar(xp,rois,color=roi_colors,alpha=0.8,edgecolor="#fff",width=0.6)
    for xi,r in enumerate(rois):
        ax3.text(xi,r+(0.5 if r>=0 else -1.5),f"{r:+.1f}%",ha="center",fontsize=8)
    ax3.axhline(0,color="#374151",lw=1)
    ax3.set_xticks(xp)
    ax3.set_xticklabels([PARLAY_LABELS[t].replace(" (OVER+WIN mismo partido)","") for t in ptypes],
                        rotation=15,ha="right",fontsize=8)
    ax3.set_ylabel("ROI %")
    ax3.set_title("ROI total por tipo de parlay",fontsize=11,pad=8,loc="left")
    ax3.grid(axis="y",alpha=0.4)

    # ── 4. P&L acumulado WIN 2-leg vs OU 2-leg por mes ───────────────────────
    ax4=fig.add_subplot(gs_fig[2,0])
    for ptype,color,label_t in [("win_2","#166534","WIN 2-leg"),
                                  ("ou_2","#374151","O/U 2-leg"),
                                  ("stack","#b45309","Stack")]:
        sub=sim_df[sim_df["type"]==ptype].copy()
        if sub.empty: continue
        sub=sub.sort_values("date")
        sub["cum"]=sub["pnl"].cumsum()
        ax4.plot(pd.to_datetime(sub["date"]),sub["cum"],color=color,lw=2,label=label_t)
    ax4.axhline(0,color="#ccc",lw=1)
    ax4.set_ylabel("P&L acumulado ($)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.4)
    ax4.set_title("P&L acumulado: WIN 2-leg, O/U 2-leg y Stack",fontsize=11,pad=8,loc="left")
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 5. Tabla resumen ──────────────────────────────────────────────────────
    ax5=fig.add_subplot(gs_fig[2,1])
    ax5.axis("off")
    tbl_rows=[["Tipo","Dias","Hit%","ROI","P&L"]]
    for ptype,label_t in PARLAY_LABELS.items():
        sub=sim_df[sim_df["type"]==ptype]
        if sub.empty: continue
        n=len(sub); h=sub["hit"].mean()*100; pnl=sub["pnl"].sum()
        roi=pnl/(n*BET)*100
        tbl_rows.append([label_t.replace(" (OVER+WIN mismo partido)",""),
                         str(n),f"{h:.0f}%",f"{roi:+.1f}%",f"${pnl:+.1f}"])
    tbl=ax5.table(cellText=tbl_rows,loc="center",bbox=[0,0.1,1,0.9])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_facecolor("#fafafa"); cell.set_edgecolor("#d1d5db")
        cell.set_text_props(color="#111")
        if r==0:
            cell.set_facecolor("#f0f0f0"); cell.set_text_props(color="#374151",fontweight="bold")
        if c==3 and r>0:
            try:
                v=float(cell.get_text().get_text().replace("%","").replace("+",""))
                if v>0: cell.set_facecolor("#f0fdf4"); cell.set_text_props(color="#166534",fontweight="bold")
                else:   cell.set_facecolor("#fef2f2"); cell.set_text_props(color="#dc2626")
            except: pass
    ax5.set_title("Resumen completo",fontsize=11,pad=8,loc="left")

    fig.suptitle(
        f"MLB Backtest Parlays — 2026 (Abr–Jun)\n"
        f"Simulacion walk-forward identica al modelo de produccion",
        fontsize=13,fontweight="bold",y=0.97)
    out="/Users/vaquera/Documents/NBA-SPURS/mlb_backtest_parlays.png"
    plt.savefig(out,dpi=155,bbox_inches="tight",facecolor="#fff")
    print(f"\n  Imagen: {out}")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  MLB BACKTEST PARLAYS — 2026")
    print("="*65)

    print("\n1. Cargando datos...")
    df24=cached_df(CACHE/"games_2024.pkl",lambda:load_games("2024-03-20","2024-09-29"))
    df25=cached_df(CACHE/"games_2025.pkl",lambda:load_games("2025-03-20","2025-09-28"))
    df26=cached_df(CACHE/"games_2026_parlays.pkl",
                   lambda:load_with_pitchers("2026-03-19","2026-06-05"))
    for df_ in [df24,df25,df26]:
        if "home_win" not in df_.columns:
            df_["home_win"]=(df_["home_runs"]>df_["away_runs"]).astype(int)
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

    print("\n4. Features de entrenamiento (2024+2025)...")
    ou24,ml24=build_rows(df24,wx24)
    ou25,ml25=build_rows(df25,wx25)
    all_ou=ou24+ou25; all_ml=ml24+ml25
    print(f"   O/U: {len(all_ou)} rows  |  WIN: {len(all_ml)} rows")

    print("\n5. Entrenando modelo O/U...")
    pipe_ou,ridge_ou,mae_cv=train_ou_model(all_ou)
    print(f"   CV MAE: {mae_cv:.3f} runs")

    print("\n6. Entrenando modelo WIN...")
    pipe_ml,acc_cv=train_winner_model(all_ml)
    print(f"   CV Accuracy: {acc_cv*100:.1f}%")

    print("\n7. Walk-forward 2026...")
    df_preds=walkforward(df26,pipe_ou,ridge_ou,pipe_ml,wx26,"2026-04-01",
                         pitcher_cache=pitcher_cache)
    print(f"   {len(df_preds)} predicciones individuales")

    # Precision individual de referencia
    ou_acc=df_preds["ou_correct"].mean()*100
    win_acc=df_preds["win_correct"].mean()*100
    print(f"\n   Precision individual: O/U={ou_acc:.1f}%  WIN={win_acc:.1f}%")

    print("\n8. Simulando parlays...")
    sim_df=simulate(df_preds)
    print(f"   {len(sim_df)} registros de parlay simulados")

    print_results(sim_df, ou_acc, win_acc)
    plot_parlays(sim_df, ou_acc, win_acc)

if __name__=="__main__":
    main()
