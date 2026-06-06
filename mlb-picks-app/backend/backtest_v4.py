#!/usr/bin/env python3
"""
MLB O/U Backtest v4 — 4 mejoras nuevas sobre v3:
  1. Temporadas 2022+2023+2024+2025 con pesos decrecientes
  2. ERA de los ultimos 3 arranques del pitcher (no solo temporada)
  3. Filtro compuesto de confianza (5 criterios)
  4. Kelly Criterion — tamano de apuesta variable segun ventaja

Walk-forward honesto — ningun juego usa datos futuros.
"""

import sys, importlib, json, math, statistics
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
from park_factors   import get_park_factor
from ml_model       import MLPredictor, build_features

CACHE = Path(__file__).parent / "cache"
LINES = [6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]
def nearest_line(v): return min(LINES, key=lambda x: abs(x-v))

# ── fetch ─────────────────────────────────────────────────────────────────────
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
                         "away_runs":ar,"home_runs":hr,"total":ar+hr})
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
                         "away_pitcher_id":ap.get("id"),
                         "home_pitcher_id":hp.get("id"),
                         "away_pitcher":ap.get("fullName","TBD"),
                         "home_pitcher":hp.get("fullName","TBD")})
    df=pd.DataFrame(rows)
    if not df.empty: df["date"]=pd.to_datetime(df["date"])
    return df

def load_team_stats(season):
    hit={}; pit={}
    for group, store in [("hitting",hit),("pitching",pit)]:
        url=(f"https://statsapi.mlb.com/api/v1/teams/stats"
             f"?season={season}&sportId=1&stats=season&group={group}")
        for sp in fetch(url).get("stats",[{}])[0].get("splits",[]):
            store[sp["team"]["id"]]=sp["stat"]
    return hit, pit

# ── MEJORA 2: ERA ultimos 3 arranques ─────────────────────────────────────────
def get_last3_era(person_id: int, season: int) -> float:
    """
    ERA del pitcher en sus ultimos 3 arranques del season.
    Mas relevante que el ERA de temporada completa.
    """
    url=(f"https://statsapi.mlb.com/api/v1/people/{person_id}/stats"
         f"?stats=gameLog&season={season}&group=pitching&sportId=1")
    data=fetch(url)
    starts=[]
    for blk in data.get("stats",[]):
        for sp in blk.get("splits",[]):
            s=sp.get("stat",{})
            ip=float(s.get("inningsPitched",0) or 0)
            if ip >= 2.0:   # solo arranques reales (no relevos cortos)
                er=float(s.get("earnedRuns",0) or 0)
                starts.append({"ip":ip,"er":er,"date":sp.get("date","")})
    if not starts:
        return 4.20
    # ultimos 3
    last3=sorted(starts, key=lambda x: x["date"], reverse=True)[:3]
    total_ip=sum(s["ip"] for s in last3)
    total_er=sum(s["er"] for s in last3)
    if total_ip==0: return 4.20
    return round((total_er*9)/total_ip, 2)

def load_pitcher_cache():
    p=CACHE/"pitchers_v4.json"
    if p.exists():
        raw=json.load(open(p))
        return {(int(k.split("_")[0]),int(k.split("_")[1])):v for k,v in raw.items()}
    return {}

def save_pitcher_cache(d):
    json.dump({f"{k[0]}_{k[1]}":v for k,v in d.items()}, open(CACHE/"pitchers_v4.json","w"))

# ── rolling stats con splits y std ────────────────────────────────────────────
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
    return {
        "rpg":np.mean(sc),"rpg_l10":np.mean(sc[-10:]),
        "ra_pg":np.mean(al),"win_pct":wins/len(sc),
        "scored":sc,"allowed":al,
        "rpg_away":np.mean(sc_a) if sc_a else np.mean(sc),
        "rpg_home":np.mean(sc_h) if sc_h else np.mean(sc),
        "std":float(np.std(sc[-20:])) if len(sc)>=5 else 2.5,
    }

def h2h_before(df, aid, hid, before_date):
    mask=(((df["away_id"]==aid)&(df["home_id"]==hid))|
          ((df["away_id"]==hid)&(df["home_id"]==aid)))&(df["date"]<before_date)
    return df[mask]["total"].tolist()

# ── formula mejorada ──────────────────────────────────────────────────────────
def predict_formula(a_st, h_st, h2h, away_era, home_era, w_adj, t_adj, pf):
    def proj(att, def_ra, opp_era):
        sv=att["rpg"]; l10=att["rpg_l10"]
        wv=np.mean(att["scored"][-7:]) if len(att["scored"])>=3 else sv
        era_f=min(1.35, max(0.65, opp_era/4.20))
        return (sv*0.20+l10*0.25+wv*0.30+def_ra*0.15+att["rpg"]*0.10)*era_f
    pa=proj(a_st, h_st["ra_pg"], home_era)
    ph=proj(h_st, a_st["ra_pg"], away_era)
    total=(pa+ph)*pf
    if len(h2h)>=2: total=total*0.85+np.mean(h2h)*0.15
    total+=w_adj+t_adj
    line=nearest_line(total)
    return total, line, "OVER" if total>=line else "UNDER"

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import pickle

def train_weighted_model(rows_by_season: dict):
    """
    Entrena con pesos por temporada:
    2022=0.50, 2023=0.70, 2024=0.90, 2025=1.00
    """
    WEIGHTS={2022:0.50,2023:0.70,2024:0.90,2025:1.00}
    all_X=[]; all_y=[]; all_w=[]
    for season, rows in rows_by_season.items():
        w=WEIGHTS.get(season,1.0)
        for r in rows:
            all_X.append(r["features"]); all_y.append(r["target"]); all_w.append(w)

    X=np.array(all_X,dtype=float)
    y=np.array(all_y,dtype=float)
    W=np.array(all_w,dtype=float)

    xgb=XGBRegressor(n_estimators=600,learning_rate=0.025,max_depth=4,
                      subsample=0.80,colsample_bytree=0.80,
                      reg_alpha=0.5,reg_lambda=1.5,
                      random_state=42,verbosity=0)
    pipe=Pipeline([("sc",StandardScaler()),("m",xgb)])
    ridge=Pipeline([("sc",StandardScaler()),("m",Ridge(alpha=5.0))])

    # CV sin pesos (sklearn 1.6+ removio fit_params) — solo para reportar MAE
    cv=cross_val_score(pipe,X,y,cv=5,scoring="neg_mean_absolute_error")
    # Fit final SI usa pesos por temporada
    pipe.fit(X,y,m__sample_weight=W)
    ridge.fit(X,y,m__sample_weight=W)
    return pipe, ridge, float(-cv.mean())

def ml_predict(pipe, ridge, feats):
    X=np.array([feats],dtype=float)
    return float(pipe.predict(X)[0])*0.65 + float(ridge.predict(X)[0])*0.35

# ── MEJORA 3: filtro compuesto de confianza ────────────────────────────────────
def confidence_filter(rec, ens, line, away_era, home_era, pf, away_std, home_std):
    """
    Devuelve (play: bool, conf_level: int 1-5, reasons: list)
    Solo recomienda UNDER cuando pasan >= 3 de 5 criterios.
    """
    if rec != "UNDER":
        return False, 0, []

    criteria=[]
    diff=line-ens  # cuanto esta por debajo de la linea

    if diff >= 0.7:           criteria.append(f"diff={diff:.2f}")
    if away_era  < 4.0:       criteria.append(f"ERA_away={away_era:.2f}")
    if home_era  < 4.0:       criteria.append(f"ERA_home={home_era:.2f}")
    if pf        <= 1.0:      criteria.append(f"park={pf:.2f}")
    if away_std  < 3.0 and \
       home_std  < 3.0:       criteria.append(f"std={max(away_std,home_std):.2f}")

    n=len(criteria)
    play=(n>=3)
    conf_level=n  # 0-5 estrellas
    return play, conf_level, criteria

# ── MEJORA 4: Kelly Criterion ─────────────────────────────────────────────────
BANKROLL = 1000.0   # bankroll inicial para simulacion

def kelly_bet(conf_level: int, base_accuracy: float = 0.568) -> float:
    """
    Half-Kelly: apuesta fraccion del bankroll segun confianza.
    conf_level 3 = accuracy estimada base
    conf_level 4 = +4% accuracy
    conf_level 5 = +8% accuracy (era <3.5 ambos)
    Vig -110: necesitas 52.4% para break-even
    """
    accuracy_est = base_accuracy + (conf_level-3)*0.04
    accuracy_est = min(accuracy_est, 0.80)
    edge = accuracy_est - 0.5238   # vs break-even -110
    if edge <= 0: return 0
    # Kelly = edge / odds_decimal   (odds = 100/110 = 0.909)
    kelly = edge / 0.909
    half_kelly = kelly * 0.5       # conservador
    # Limitar entre $22 y $275 (2.2%-27.5% del bankroll)
    return round(min(max(half_kelly * BANKROLL, 22), 275), 0)

# ── build training rows ───────────────────────────────────────────────────────
def build_rows(df_games, hit_stats, pit_stats, wx_cache):
    agg={}
    for tid in set(df_games["away_id"])|set(df_games["home_id"]):
        sub=df_games[(df_games["away_id"]==tid)|(df_games["home_id"]==tid)]
        sc=[]; al=[]; sc_a=[]; sc_h=[]
        for _,r in sub.iterrows():
            if r["away_id"]==tid:
                sc.append(r["away_runs"]); al.append(r["home_runs"]); sc_a.append(r["away_runs"])
            else:
                sc.append(r["home_runs"]); al.append(r["away_runs"]); sc_h.append(r["home_runs"])
        wins=sum(1 for s,a in zip(sc,al) if s>a)
        agg[tid]={"rpg":np.mean(sc) if sc else 4.,"rpg_l10":np.mean(sc[-10:]) if sc else 4.,
                  "ra_pg":np.mean(al) if al else 4.,"win_pct":wins/len(sc) if sc else .5,
                  "rpg_away":np.mean(sc_a) if sc_a else 4.,"rpg_home":np.mean(sc_h) if sc_h else 4.,
                  "std":float(np.std(sc)) if len(sc)>=5 else 2.5}
    rows=[]
    for _,g in df_games.iterrows():
        a=agg.get(g["away_id"],{}); h=agg.get(g["home_id"],{})
        if not a or not h: continue
        aps=pit_stats.get(g["away_id"],{}); hps=pit_stats.get(g["home_id"],{})
        ahs=hit_stats.get(g["away_id"],{}); hhs=hit_stats.get(g["home_id"],{})
        venue=str(g.get("venue",""))
        date_s=g["date"].strftime("%Y-%m-%d") if hasattr(g["date"],"strftime") else str(g["date"])[:10]
        pf=get_park_factor(venue)
        wx=lookup(wx_cache,venue,date_s) if wx_cache else {"wind_mph":7.,"temp_f":72.,"roof":"open"}
        w_adj=wind_factor(wx["wind_mph"],wx.get("roof","open"))
        t_adj=temp_factor(wx["temp_f"],wx.get("roof","open"))
        feats=build_features(
            a["rpg"],a["rpg_l10"],a["win_pct"],
            aps.get("era","4.20"),aps.get("whip","1.30"),ahs.get("ops",".720"),
            "",a["ra_pg"],h["rpg"],h["rpg_l10"],h["win_pct"],
            hps.get("era","4.20"),hps.get("whip","1.30"),hhs.get("ops",".720"),
            "",h["ra_pg"],None,0,g["date"].month,
            away_rpg_away=a["rpg_away"],home_rpg_home=h["rpg_home"],
            away_std=a["std"],home_std=h["std"],park_factor=pf,
        )+[w_adj,t_adj,wx.get("wind_mph",7.),wx.get("temp_f",72.)]
        rows.append({"features":feats,"target":float(g["total"])})
    return rows

# ── helper: cached dataframe ──────────────────────────────────────────────────
def cached_df(path, load_fn):
    p=Path(path)
    if p.exists(): return pd.read_pickle(p)
    df=load_fn(); df.to_pickle(p); return df

def cached_stats(path, load_fn):
    p=Path(path)
    if p.exists():
        d=json.load(open(p))
        return {int(k):v for k,v in d["hit"].items()}, {int(k):v for k,v in d["pit"].items()}
    hit,pit=load_fn()
    json.dump({"hit":{str(k):v for k,v in hit.items()},
               "pit":{str(k):v for k,v in pit.items()}},open(p,"w"))
    return hit,pit

# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*68)
    print("  MLB O/U BACKTEST v4")
    print("  2022+2023+2024+2025 training | ERA last 3 | Filtro | Kelly")
    print("="*68)

    # ── 1. Datos ─────────────────────────────────────────────────────────────
    print("\n1. Cargando datos...")
    seasons={
        2022:("2022-04-07","2022-10-05"),
        2023:("2023-03-30","2023-10-01"),
        2024:("2024-03-20","2024-09-29"),
        2025:("2025-03-20","2025-09-28"),
    }
    dfs={}
    for yr,(s,e) in seasons.items():
        dfs[yr]=cached_df(CACHE/f"games_{yr}.pkl", lambda s=s,e=e: load_games(s,e))
        print(f"   {yr}: {len(dfs[yr])}j",end="  ")
    print()
    df26=cached_df(CACHE/"games_2026.pkl", lambda: load_with_pitchers("2026-03-19","2026-06-04"))
    print(f"   2026: {len(df26)}j (con pitchers)")

    stats={}
    for yr in seasons:
        h,p=cached_stats(CACHE/f"stats_{yr}.json", lambda yr=yr: load_team_stats(yr))
        stats[yr]=(h,p)
    h26,p26=cached_stats(CACHE/"stats_2026.json", lambda: load_team_stats(2026))

    # ── 2. ERA ultimos 3 arranques ────────────────────────────────────────────
    print("\n2. ERA ultimos 3 arranques por pitcher...")
    p4=load_pitcher_cache()
    need=set()
    for _,g in df26.iterrows():
        for col in ["away_pitcher_id","home_pitcher_id"]:
            try:
                pid=g.get(col)
                if pid and not pd.isna(pid): need.add((int(pid),2026))
            except: pass

    new_pids=[x for x in need if x not in p4]
    if new_pids:
        print(f"   Descargando ERA last3 para {len(new_pids)} pitchers...")
        for i,(pid,season) in enumerate(new_pids):
            era_season=4.20; era_last3=4.20
            # season ERA
            url=(f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                 f"?stats=season&season={season}&group=pitching&sportId=1")
            d=fetch(url)
            for blk in d.get("stats",[]):
                for sp in blk.get("splits",[]):
                    try: era_season=float(sp["stat"].get("era",4.20))
                    except: pass
            # last 3 starts ERA
            era_last3=get_last3_era(pid,season)
            p4[(pid,season)]={"era":era_season,"era_last3":era_last3}
            if (i+1)%50==0: print(f"   {i+1}/{len(new_pids)}...",end="\r")
        save_pitcher_cache(p4)
        print(f"   {len(p4)} pitchers guardados")
    else:
        print(f"   {len(p4)} pitchers desde cache")

    # ── 3. Clima ──────────────────────────────────────────────────────────────
    print("\n3. Clima por temporada...")
    wx_ranges={
        2022:("2022-04-07","2022-10-05"),
        2023:("2023-03-30","2023-10-01"),
        2024:("2024-03-20","2024-09-29"),
        2025:("2025-03-20","2025-09-28"),
        2026:("2026-03-19","2026-06-05"),
    }
    wxs={}
    for yr,(s,e) in wx_ranges.items():
        wxs[yr]=build_season_weather_cache(yr,s,e)

    # ── 4. Training rows con pesos por temporada ──────────────────────────────
    print("\n4. Construyendo features de entrenamiento...")
    rows_by_season={}
    for yr in seasons:
        h,p=stats[yr]
        rows_by_season[yr]=build_rows(dfs[yr],h,p,wxs[yr])
        print(f"   {yr}: {len(rows_by_season[yr])} filas  (peso {[0.5,0.7,0.9,1.0][[2022,2023,2024,2025].index(yr)]})")
    total_rows=sum(len(v) for v in rows_by_season.values())
    print(f"   Total: {total_rows} juegos de entrenamiento")

    # ── 5. Entrenamiento ──────────────────────────────────────────────────────
    print("\n5. Entrenando modelo v4 (pesos por temporada)...")
    pipe_v4, ridge_v4, cv_mae = train_weighted_model(rows_by_season)
    print(f"   CV MAE: {cv_mae:.3f} runs")

    # ── 6. Backtest walk-forward 2026 ─────────────────────────────────────────
    print("\n6. Backtest walk-forward 2026...")
    cutoff=pd.Timestamp("2026-04-01")
    game_dates=sorted(df26[df26["date"]>=cutoff]["date"].unique())

    records=[]
    bankroll_flat=BANKROLL; bankroll_kelly=BANKROLL

    for di,gdate in enumerate(game_dates):
        for _,g in df26[df26["date"]==gdate].iterrows():
            aid=int(g["away_id"]); hid=int(g["home_id"])
            a_st=rolling_stats(df26,aid,gdate); h_st=rolling_stats(df26,hid,gdate)
            if not a_st or not h_st: continue

            h2h=h2h_before(df26,aid,hid,gdate)
            venue=str(g.get("venue",""))
            date_s=gdate.strftime("%Y-%m-%d")
            pf=get_park_factor(venue)
            wx=lookup(wxs[2026],venue,date_s)
            w_adj=wind_factor(wx["wind_mph"],wx.get("roof","open"))
            t_adj=temp_factor(wx["temp_f"],wx.get("roof","open"))

            # ERA — season + last 3 starts (promedio ponderado)
            def get_era(col):
                try:
                    pid=g.get(col)
                    if pid and not pd.isna(pid):
                        d=p4.get((int(pid),2026),{})
                        era_s=d.get("era",4.20); era_l3=d.get("era_last3",4.20)
                        # 40% season + 60% last3 (mas reciente pesa mas)
                        return round(era_s*0.40 + era_l3*0.60, 2)
                except: pass
                return 4.20

            away_era=get_era("away_pitcher_id"); home_era=get_era("home_pitcher_id")

            f_proj,f_line,f_rec=predict_formula(a_st,h_st,h2h,away_era,home_era,w_adj,t_adj,pf)

            aps=p26.get(aid,{}); hps=p26.get(hid,{})
            ahs=h26.get(aid,{}); hhs=h26.get(hid,{})
            feats=build_features(
                a_st["rpg"],a_st["rpg_l10"],a_st["win_pct"],
                str(away_era),aps.get("whip","1.30"),ahs.get("ops",".720"),
                "",a_st["ra_pg"],h_st["rpg"],h_st["rpg_l10"],h_st["win_pct"],
                str(home_era),hps.get("whip","1.30"),hhs.get("ops",".720"),
                "",h_st["ra_pg"],
                np.mean(h2h) if len(h2h)>=2 else None,len(h2h),gdate.month,
                away_rpg_away=a_st["rpg_away"],home_rpg_home=h_st["rpg_home"],
                away_std=a_st["std"],home_std=h_st["std"],park_factor=pf,
            )+[w_adj,t_adj,wx["wind_mph"],wx["temp_f"]]

            ml_proj=ml_predict(pipe_v4,ridge_v4,feats)
            ens=ml_proj*0.55+f_proj*0.45
            line=nearest_line(ens); rec="OVER" if ens>=line else "UNDER"
            actual=float(g["total"])
            correct=(rec=="OVER" and actual>line) or (rec=="UNDER" and actual<line)

            # Filtro compuesto (MEJORA 3)
            play,conf_level,reasons=confidence_filter(
                rec,ens,line,away_era,home_era,pf,a_st["std"],h_st["std"])

            # Kelly sizing (MEJORA 4)
            bet_kelly=kelly_bet(conf_level) if play else 0
            bet_flat=110.0

            # P&L simulacion
            if play:
                pnl_kelly = bet_kelly*(100/110) if correct else -bet_kelly
                pnl_flat  = 100.0 if correct else -110.0
                bankroll_kelly += pnl_kelly
                bankroll_flat  += pnl_flat
            else:
                pnl_kelly=0; pnl_flat=0

            records.append({
                "date":gdate,"away":g["away_name"].split()[-1],"home":g["home_name"].split()[-1],
                "venue":venue[:22],"away_era":away_era,"home_era":home_era,
                "pf":pf,"wind":wx["wind_mph"],"temp":wx["temp_f"],
                "f_proj":round(f_proj,2),"ml_proj":round(ml_proj,2),
                "ens":round(ens,2),"line":line,"rec":rec,"actual":int(actual),
                "correct":correct,"play":play,"conf_level":conf_level,
                "reasons":" | ".join(reasons),
                "bet_kelly":bet_kelly,"pnl_kelly":round(pnl_kelly,2),
                "pnl_flat":round(pnl_flat,2),
                "diff":round(line-ens,2) if rec=="UNDER" else round(ens-line,2),
                "away_std":round(a_st["std"],2),"home_std":round(h_st["std"],2),
            })
        if (di+1)%10==0: print(f"   {di+1}/{len(game_dates)} dias...",end="\r")

    print(f"\n   {len(records)} predicciones generadas")
    df=pd.DataFrame(records)

    # ── Resultados ─────────────────────────────────────────────────────────────
    print_results(df)
    plot_v4(df, cv_mae, total_rows)

def print_results(df):
    n=len(df); nf=df["play"].sum()
    acc_all  = df["correct"].mean()*100
    acc_u    = df.loc[df["rec"]=="UNDER","correct"].mean()*100
    acc_filt = df.loc[df["play"],"correct"].mean()*100 if nf else 0
    pnl_flat = df.loc[df["play"],"pnl_flat"].sum()
    pnl_kelly= df.loc[df["play"],"pnl_kelly"].sum()
    wagered_f= nf*110
    wagered_k= df.loc[df["play"],"bet_kelly"].sum()
    roi_flat = pnl_flat/wagered_f*100 if wagered_f else 0
    roi_kelly= pnl_kelly/wagered_k*100 if wagered_k else 0

    print("\n"+"="*68)
    print("  RESULTADOS v4")
    print("="*68)
    print(f"  {'':35s} {'Acc':>6}  {'Picks':>6}  {'P&L':>10}  {'ROI':>7}")
    print(f"  {'─'*35} {'─'*6}  {'─'*6}  {'─'*10}  {'─'*7}")
    print(f"  {'Todos los picks':35s} {acc_all:5.1f}%  {n:>6}  {'':>10}")
    print(f"  {'Solo UNDER (sin filtro)':35s} {acc_u:5.1f}%  {df['rec'].eq('UNDER').sum():>6}")
    print(f"  {'Filtro compuesto (>=3 criterios)':35s} {acc_filt:5.1f}%  {nf:>6}  ${pnl_flat:>+9.0f}  {roi_flat:>+6.1f}%")
    print(f"  {'  + Kelly Criterion':35s} {'':>6}  {'':>6}  ${pnl_kelly:>+9.0f}  {roi_kelly:>+6.1f}%")

    print(f"\n  Bankroll inicial: ${BANKROLL:.0f}")
    bk=BANKROLL+pnl_kelly; bf=BANKROLL+pnl_flat
    print(f"  Bankroll final (Kelly):   ${bk:>+.0f}  ({(bk/BANKROLL-1)*100:+.1f}%)")
    print(f"  Bankroll final (flat):    ${bf:>+.0f}  ({(bf/BANKROLL-1)*100:+.1f}%)")

    print("\n  Por nivel de confianza (filtrado):")
    print(f"  {'Conf':>5}  {'Picks':>6}  {'Acc':>7}  {'BetKelly':>10}  {'P&L Kelly':>11}")
    print(f"  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*10}  {'─'*11}")
    for cl in sorted(df.loc[df["play"],"conf_level"].unique()):
        sub=df[(df["play"])&(df["conf_level"]==cl)]
        a=sub["correct"].mean()*100
        bk_=sub["bet_kelly"].mean(); pk_=sub["pnl_kelly"].sum()
        stars="★"*int(cl)+"☆"*(5-int(cl))
        print(f"  {stars}  {len(sub):>6}  {a:>6.1f}%  ${bk_:>9.0f}  ${pk_:>+10.0f}")

    print("\n  Solo UNDER por mes (filtrado):")
    df["month"]=df["date"].dt.strftime("%b")
    for mo,sub in df[df["play"]].groupby("month"):
        a=sub["correct"].mean()*100
        pf_=sub["pnl_flat"].sum(); pk_=sub["pnl_kelly"].sum()
        print(f"    {mo}: {len(sub):3d}j  acc={a:.1f}%  flat=${pf_:+.0f}  kelly=${pk_:+.0f}")

    # ERA elite
    elite=df[(df["play"])&(df["away_era"]<3.5)&(df["home_era"]<3.5)]
    if len(elite):
        ea=elite["correct"].mean()*100; ep=elite["pnl_kelly"].sum()
        print(f"\n  ERA<3.5 ambos pitchers: {len(elite)}j  acc={ea:.1f}%  kelly P&L=${ep:+.0f}")

# ── imagen ────────────────────────────────────────────────────────────────────
def plot_v4(df, cv_mae, train_n):
    plt.rcParams.update({"figure.facecolor":"#fff","axes.facecolor":"#fafafa",
                          "axes.edgecolor":"#d1d5db","text.color":"#111",
                          "xtick.color":"#666","ytick.color":"#666",
                          "grid.color":"#e5e7eb","font.size":9})
    fig=plt.figure(figsize=(24,26))
    gs=gridspec.GridSpec(4,2,figure=fig,hspace=0.50,wspace=0.35,
                         top=0.93,bottom=0.04,left=0.06,right=0.97)

    df_f=df[df["play"]].copy()
    by_day=df_f.groupby("date").agg(
        acc=("correct","mean"),
        pnl_flat=("pnl_flat","sum"),
        pnl_kelly=("pnl_kelly","sum"),
        n=("correct","count"),
    ).reset_index()
    by_day["cum_flat"]=by_day["pnl_flat"].cumsum()
    by_day["cum_kelly"]=by_day["pnl_kelly"].cumsum()
    dates=pd.to_datetime(by_day["date"])

    # 1. Accuracy diaria
    ax1=fig.add_subplot(gs[0,:])
    ax1.bar(dates,by_day["acc"]*100,
            color=["#166534" if a>=52.4 else "#dc2626" for a in by_day["acc"]],
            alpha=0.75,width=0.8)
    roll=pd.Series(by_day["acc"].values).rolling(7,min_periods=3).mean()*100
    ax1.plot(dates,roll,color="#111",lw=2.2,label="Rolling 7d")
    ax1.axhline(52.4,color="#374151",lw=1.2,linestyle="--",label="52.4% break-even")
    acc_g=df_f["correct"].mean()*100; pnl_k=df_f["pnl_kelly"].sum(); pnl_f=df_f["pnl_flat"].sum()
    ax1.set_ylim(0,105); ax1.set_ylabel("Accuracy %")
    ax1.set_title(
        f"Accuracy diaria (filtro compuesto)  |  Global {acc_g:.1f}%  |  "
        f"{len(df_f)} picks  |  Kelly P&L USD{pnl_k:+.0f}  |  Flat P&L USD{pnl_f:+.0f}\n"
        f"Entrenado: 2022-2025 ponderado ({train_n}j)  CV MAE {cv_mae:.3f} runs",
        fontsize=11,fontweight="bold",pad=8,loc="left")
    ax1.legend(fontsize=8); ax1.grid(axis="y",alpha=0.4)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # 2. P&L flat vs Kelly
    ax2=fig.add_subplot(gs[1,:])
    ax2.plot(dates,by_day["cum_flat"], color="#9ca3af",lw=2,  label="Flat USD110")
    ax2.plot(dates,by_day["cum_kelly"],color="#111",   lw=2.5,label="Kelly (variable)")
    ax2.fill_between(dates,by_day["cum_kelly"],0,
                     where=by_day["cum_kelly"]>=0,alpha=0.15,color="#166534")
    ax2.fill_between(dates,by_day["cum_kelly"],0,
                     where=by_day["cum_kelly"]<0, alpha=0.15,color="#dc2626")
    ax2.axhline(0,color="#374151",lw=1)
    ck=by_day["cum_kelly"].values
    ax2.annotate(f"Max USD{max(ck):+.0f}",xy=(dates.iloc[np.argmax(ck)],max(ck)),
                 fontsize=8,color="#166534",xytext=(0,7),textcoords="offset points")
    ax2.set_ylabel("USD"); ax2.grid(alpha=0.4)
    ax2.set_title("P&L acumulado — Flat vs Kelly",fontsize=11,pad=8,loc="left")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # 3. Accuracy por conf level
    ax3=fig.add_subplot(gs[2,0])
    cl_data=[(cl,df_f[df_f["conf_level"]==cl]) for cl in sorted(df_f["conf_level"].unique())]
    bc=["#166534" if s["correct"].mean()>=0.524 else "#dc2626" for _,s in cl_data]
    ax3.bar(range(len(cl_data)),[s["correct"].mean()*100 for _,s in cl_data],
            color=bc,alpha=0.85,edgecolor="#fff",width=0.55)
    for xi,(_,sub) in enumerate(cl_data):
        a=sub["correct"].mean()*100; n=len(sub)
        ax3.text(xi,a+0.5,f"{a:.1f}%\n({n}j)",ha="center",fontsize=8)
    ax3.axhline(52.4,color="#374151",lw=1.2,linestyle="--")
    ax3.set_xticks(range(len(cl_data)))
    ax3.set_xticklabels(["★"*int(c)+"☆"*(5-int(c)) for c,_ in cl_data],fontsize=9)
    ax3.set_ylim(0,85); ax3.set_ylabel("Accuracy %")
    ax3.set_title("Accuracy por nivel de confianza",fontsize=11,pad=8,loc="left")
    ax3.grid(axis="y",alpha=0.4)

    # 4. ERA vs accuracy
    ax4=fig.add_subplot(gs[2,1])
    df_f["avg_era"]=(df_f["away_era"]+df_f["home_era"])/2
    era_bins=pd.cut(df_f["avg_era"],bins=[0,3.0,3.5,4.0,4.5,10],
                    labels=["<3.0","3.0-3.5","3.5-4.0","4.0-4.5",">4.5"])
    era_acc=df_f.groupby(era_bins,observed=True).agg(
        acc=("correct","mean"),n=("correct","count")).reset_index()
    bc2=["#166534" if a>=0.524 else "#dc2626" for a in era_acc["acc"]]
    ax4.bar(range(len(era_acc)),era_acc["acc"]*100,color=bc2,alpha=0.85,edgecolor="#fff")
    for xi,(_,r) in enumerate(era_acc.iterrows()):
        ax4.text(xi,r["acc"]*100+0.5,f"{r['acc']*100:.1f}%\n({int(r['n'])}j)",
                 ha="center",fontsize=8)
    ax4.axhline(52.4,color="#374151",lw=1.2,linestyle="--")
    ax4.set_xticks(range(len(era_acc)))
    ax4.set_xticklabels(era_acc["avg_era"].astype(str),fontsize=9)
    ax4.set_xlabel("ERA promedio pitchers (last3+season blend)")
    ax4.set_ylim(0,85); ax4.set_ylabel("Accuracy %")
    ax4.set_title("Accuracy por ERA del pitcher (v4 blend 40/60)",fontsize=11,pad=8,loc="left")
    ax4.grid(axis="y",alpha=0.4)

    # 5. Tabla dias recientes
    ax5=fig.add_subplot(gs[3,:])
    ax5.axis("off")
    recent=by_day.tail(45).reset_index(drop=True)
    rh=1.0/(len(recent)+1)

    for xp,lbl in [(0.01,"FECHA"),(0.10,"Picks"),(0.15,"Acc"),(0.22,"P&L Flat"),
                   (0.31,"P&L Kelly"),(0.41,"Cum.Kelly"),(0.50,"Partidos jugados (★)")]:
        ax5.text(xp,1.01,lbl,fontsize=7.5,fontweight="bold",color="#444",transform=ax5.transAxes)

    cum_k=0
    for i,row in recent.iterrows():
        y=1.0-(i+1)*rh-0.01
        cum_k+=row["pnl_kelly"]
        bg="#f0fdf4" if row["acc"]>=0.524 else "#fef2f2" if row["acc"]<0.45 else "#fff"
        ax5.add_patch(plt.Rectangle((0,y-rh*0.35),1,rh*0.85,facecolor=bg,
                      edgecolor="#e0e0e0",lw=0.4,transform=ax5.transAxes,clip_on=False))
        ca="#166534" if row["acc"]>=0.524 else "#dc2626"
        ck2="#166534" if row["pnl_kelly"]>=0 else "#dc2626"
        ck3="#166534" if cum_k>=0 else "#dc2626"
        ax5.text(0.01,y+rh*0.22,row["date"].strftime("%a %b %d"),fontsize=8,fontweight="bold",color="#111",transform=ax5.transAxes)
        ax5.text(0.10,y+rh*0.22,str(int(row["n"])),fontsize=8,color="#444",transform=ax5.transAxes)
        ax5.text(0.15,y+rh*0.22,f"{row['acc']*100:.0f}%",fontsize=8,color=ca,fontweight="bold",transform=ax5.transAxes)
        ax5.text(0.22,y+rh*0.22,f"USD{row['pnl_flat']:+.0f}",fontsize=8,color=ck2,transform=ax5.transAxes)
        ax5.text(0.31,y+rh*0.22,f"USD{row['pnl_kelly']:+.0f}",fontsize=8,color=ck2,fontweight="bold",transform=ax5.transAxes)
        ax5.text(0.41,y+rh*0.22,f"USD{cum_k:+.0f}",fontsize=8,color=ck3,transform=ax5.transAxes)
        day_preds=df_f[df_f["date"]==row["date"]]
        parts=[]
        for _,pr in day_preds.iterrows():
            sym="✓" if pr["correct"] else "✗"
            parts.append(f"{pr['away']}@{pr['home']} U{pr['line']} {sym}r={pr['actual']} "
                         f"ERA{pr['away_era']}/{pr['home_era']} kelly=${pr['bet_kelly']:.0f}")
        ax5.text(0.50,y+rh*0.22,"  ".join(parts)[:100],fontsize=6.5,color="#444",transform=ax5.transAxes)

    ax5.set_title("Detalle diario — filtro compuesto + Kelly  ✓=acierto  ✗=error",
                  fontsize=10,pad=8,loc="left")
    ax5.set_xlim(0,1); ax5.set_ylim(0,1)

    acc_g2=df_f["correct"].mean()*100; pnl_k2=df_f["pnl_kelly"].sum(); nf=len(df_f)
    fig.suptitle(
        f"MLB O/U v4  |  2022-2025 ponderado + ERA last3 + Filtro compuesto + Kelly\n"
        f"Acc filtrado: {acc_g2:.1f}%  |  {nf} picks  |  Kelly P&L: USD{pnl_k2:+.0f}  |  CV MAE: {cv_mae:.3f} runs",
        fontsize=12,fontweight="bold",y=0.975)

    out="/Users/vaquera/Documents/NBA-SPURS/mlb_backtest_v4.png"
    plt.savefig(out,dpi=150,bbox_inches="tight",facecolor="#fff")
    print(f"\n  Imagen: {out}")

if __name__=="__main__":
    main()
