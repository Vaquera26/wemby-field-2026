#!/usr/bin/env python3
"""
Backtest día a día — v2 con Capa 2 (calibrador de confianza).

Mismo ejercicio que backtest_diario.py pero ahora:
  - Capa 1: XGBoost/Ridge entrenados con 2025
  - Capa 2: Calibrador entrenado con errores del backtest (hasta May 14)
  - Solo se "juega" cuando P(correcto) >= 0.54
  - Se muestra cada día: todos los picks vs picks filtrados
"""

import sys, importlib, json, pickle
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

import urllib.request

CACHE = Path(__file__).parent / "cache"
THRESHOLD = 0.54

def fetch(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except: __import__("time").sleep(1.5)
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
            rows.append({"date":d["date"],"away_id":g["teams"]["away"]["team"]["id"],
                         "home_id":g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "away_runs":ar,"home_runs":hr,"total":ar+hr})
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

LINES=[6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]
def nearest_line(v): return min(LINES, key=lambda x: abs(x-v))

def rolling_stats(df, tid, before_date):
    mask=((df["away_id"]==tid)|(df["home_id"]==tid)) & (df["date"]<before_date)
    sub=df[mask].sort_values("date")
    if len(sub)<5: return None
    sc=[r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
    al=[r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]
    wins=sum(1 for s,a in zip(sc,al) if s>a)
    return {"rpg":np.mean(sc),"rpg_l10":np.mean(sc[-10:]),"ra_pg":np.mean(al),
            "win_pct":wins/len(sc),"scored":sc,"allowed":al}

def h2h_before(df, aid, hid, before_date):
    mask=(((df["away_id"]==aid)&(df["home_id"]==hid))|
          ((df["away_id"]==hid)&(df["home_id"]==aid)))&(df["date"]<before_date)
    return df[mask]["total"].tolist()

def formula_predict(a, h, h2h):
    def proj(att, def_):
        sv=att["rpg"]; l10=att["rpg_l10"]
        wv=np.mean(att["scored"][-7:]) if len(att["scored"])>=3 else sv
        return sv*0.25+l10*0.25+wv*0.35+def_["ra_pg"]*0.15
    p=proj(a,h)+proj(h,a)
    if len(h2h)>=2: p=p*0.85+np.mean(h2h)*0.15
    line=nearest_line(p); return p,line,"OVER" if p>=line else "UNDER"

from ml_model import MLPredictor, build_features
from backtest_retrain import FEAT_COLS_L2, train_2025

def make_X_row(f_proj, ml_proj, ens_proj, h2h, a_st, h_st, month, dow, is_over):
    return [[
        f_proj, ml_proj, ens_proj,
        abs(ml_proj-f_proj),           # disagreement
        ens_proj-nearest_line(ens_proj), # delta_line
        abs(ens_proj-nearest_line(ens_proj)),
        len(h2h),
        np.mean(h2h) if h2h else ens_proj,
        a_st["rpg"], h_st["rpg"],
        a_st["ra_pg"], h_st["ra_pg"],
        a_st["win_pct"], h_st["win_pct"],
        float(month), float(dow),
        float(is_over),
    ]]

def main():
    print("Cargando datos...")
    df_2025 = load_games("2025-03-20","2025-09-28")
    df_2026 = load_games("2026-03-19","2026-06-04")
    hit25,pit25 = load_team_stats(2025)
    hit26,pit26 = load_team_stats(2026)
    print(f"  {len(df_2025)} j-2025  |  {len(df_2026)} j-2026")

    # Capa 1
    print("Cargando/entrenando capa 1...")
    ml_l1 = MLPredictor.load()
    if ml_l1 is None:
        ml_l1 = train_2025(df_2025, hit25, pit25)
        ml_l1.save()
    print(f"  Capa 1 lista  (CV MAE {ml_l1.cv_mae:.3f})")

    # Capa 2
    corr_path = CACHE/"corrector.pkl"; calib_path = CACHE/"calibrator.pkl"
    if not corr_path.exists() or not calib_path.exists():
        print("  Modelos de capa 2 no encontrados — ejecuta backtest_retrain.py primero")
        return
    with open(corr_path,"rb") as f:  corrector  = pickle.load(f)
    with open(calib_path,"rb") as f: calibrator = pickle.load(f)
    print("  Capa 2 cargada (corrector + calibrador)")

    # El calibrador fue entrenado con datos hasta May 14
    # Solo aplicamos el filtro desde May 15 en adelante
    CALIB_FROM = pd.Timestamp("2026-05-15")

    # Procesamos día a día
    cutoff = pd.Timestamp("2026-04-01")
    game_dates = sorted(df_2026[df_2026["date"]>=cutoff]["date"].unique())
    print(f"\nProcesando {len(game_dates)} días...\n")

    all_preds = []
    for di, gdate in enumerate(game_dates):
        for _, g in df_2026[df_2026["date"]==gdate].iterrows():
            aid=g["away_id"]; hid=g["home_id"]
            a_st=rolling_stats(df_2026,aid,gdate)
            h_st=rolling_stats(df_2026,hid,gdate)
            if not a_st or not h_st: continue
            h2h=h2h_before(df_2026,aid,hid,gdate)

            f_proj,f_line,f_rec=formula_predict(a_st,h_st,h2h)
            aps=pit26.get(aid,{}); hps=pit26.get(hid,{})
            ahs=hit26.get(aid,{}); hhs=hit26.get(hid,{})
            feats=build_features(a_st["rpg"],a_st["rpg_l10"],a_st["win_pct"],
                                  aps.get("era","4.20"),aps.get("whip","1.30"),ahs.get("ops",".720"),
                                  "",a_st["ra_pg"],h_st["rpg"],h_st["rpg_l10"],h_st["win_pct"],
                                  hps.get("era","4.20"),hps.get("whip","1.30"),hhs.get("ops",".720"),
                                  "",h_st["ra_pg"],
                                  np.mean(h2h) if len(h2h)>=2 else None,len(h2h),gdate.month)
            ml_r=ml_l1.predict(feats); ml_tot=ml_r["total"] or f_proj
            ens=ml_tot*0.55+f_proj*0.45
            line=nearest_line(ens); rec="OVER" if ens>=line else "UNDER"
            actual=float(g["total"])
            correct=(rec=="OVER" and actual>line) or (rec=="UNDER" and actual<line)

            # Capa 2 — solo si estamos en el periodo post-entrenamiento
            if gdate >= CALIB_FROM:
                X2 = np.array(make_X_row(f_proj,ml_tot,ens,h2h,a_st,h_st,
                                          gdate.month,gdate.dayofweek,rec=="OVER"))
                proba = float(calibrator.predict_proba(X2)[0,1])
                play  = proba >= THRESHOLD
            else:
                proba = 0.50   # sin calibrador en periodo de entrenamiento
                play  = True   # en periodo de entrenamiento jugamos todo

            all_preds.append({
                "date":gdate,"month":gdate.month,
                "away":g["away_name"].split()[-1],"home":g["home_name"].split()[-1],
                "proj":round(ens,2),"line":line,"rec":rec,
                "actual":int(actual),"correct":correct,
                "proba":round(proba,3),"play":play,
                "in_calib_period": gdate >= CALIB_FROM,
            })
        if (di+1)%10==0: print(f"  {di+1}/{len(game_dates)} dias...", end="\r")

    print(f"\n  {len(all_preds)} predicciones listas\n")
    df = pd.DataFrame(all_preds)

    # ── Tabla día a día ───────────────────────────────────────────────────────
    by_day = df.groupby("date").agg(
        games=("correct","count"),
        correct=("correct","sum"),
        plays=("play","sum"),
        plays_correct=("correct", lambda x: x[df.loc[x.index,"play"]].sum()),
        pnl_all=("correct",lambda x: x.map({True:100.,False:-110.}).sum()),
        pnl_filt=("correct",lambda x:
                  x[df.loc[x.index,"play"]].map({True:100.,False:-110.}).sum()),
    ).reset_index()
    by_day["acc_all"]   = by_day["correct"]       / by_day["games"]*100
    by_day["acc_filt"]  = by_day.apply(
        lambda r: r["plays_correct"]/r["plays"]*100 if r["plays"]>0 else np.nan, axis=1)
    by_day["cum_pnl_all"]  = by_day["pnl_all"].cumsum()
    by_day["cum_pnl_filt"] = by_day["pnl_filt"].cumsum()

    # ── Print ─────────────────────────────────────────────────────────────────
    print("="*100)
    print(f"  MLB O/U — DÍA A DÍA  |  Capa 1 (todos) vs Capa 2 (filtrado P≥{THRESHOLD})")
    print("="*100)
    print(f"  {'Fecha':<13} {'J':>3}  {'Todos':>12}  {'P&L':>7}  "
          f"{'Jugados':>8}  {'Filtrado':>12}  {'P&L':>7}  {'CumAll':>8}  {'CumFilt':>8}")
    print(f"  {'─'*13} {'─'*3}  {'─'*12}  {'─'*7}  "
          f"{'─'*8}  {'─'*12}  {'─'*7}  {'─'*8}  {'─'*8}")

    for _, r in by_day.iterrows():
        acc_all_str  = f"{int(r['correct'])}/{int(r['games'])} ({r['acc_all']:.0f}%)"
        if r['plays'] > 0 and not np.isnan(r['acc_filt']):
            acc_filt_str = f"{int(r['plays_correct'])}/{int(r['plays'])} ({r['acc_filt']:.0f}%)"
        else:
            acc_filt_str = "—"

        ok_all  = "✓" if r["acc_all"]  >= 52.4 else "·"
        ok_filt = "✓" if (not np.isnan(r["acc_filt"]) and r["acc_filt"] >= 52.4) else "·"
        calib_mark = " [L2]" if r["date"] >= CALIB_FROM else "     "

        c_cum_all  = "+" if r["cum_pnl_all"]  >= 0 else ""
        c_cum_filt = "+" if r["cum_pnl_filt"] >= 0 else ""

        print(f"  {r['date'].strftime('%a %b %d'):<13}"
              f" {int(r['games']):>3} "
              f" {ok_all} {acc_all_str:>12}"
              f"  ${r['pnl_all']:>+6.0f}"
              f"  {calib_mark}"
              f" {ok_filt} {acc_filt_str:>12}"
              f"  ${r['pnl_filt']:>+6.0f}"
              f"  ${r['cum_pnl_all']:>+7.0f}"
              f"  ${r['cum_pnl_filt']:>+7.0f}")

    # Totales
    n_all  = len(df); n_filt = df["play"].sum()
    acc_all  = df["correct"].mean()*100
    acc_filt = df.loc[df["play"],"correct"].mean()*100 if n_filt>0 else 0
    pnl_all  = df["correct"].map({True:100.,False:-110.}).sum()
    pnl_filt = df.loc[df["play"],"correct"].map({True:100.,False:-110.}).sum()

    print(f"\n  {'TOTAL':<13} {n_all:>3}  {'':>12}  ${pnl_all:>+6.0f}  "
          f"       {'':>12}  ${pnl_filt:>+6.0f}  ${pnl_all:>+7.0f}  ${pnl_filt:>+7.0f}")
    print(f"\n  Todos los picks:     {acc_all:.1f}% acc  ({n_all} juegos)  P&L ${pnl_all:+.0f}")
    print(f"  Picks filtrados L2:  {acc_filt:.1f}% acc  ({n_filt} jugados / {n_all} totales)  P&L ${pnl_filt:+.0f}")
    print(f"\n  [L2] = calibrador de capa 2 activo (entrenado con datos hasta May 14)")

    # ── IMAGEN ────────────────────────────────────────────────────────────────
    plot_diario_v2(by_day, df, CALIB_FROM)

def plot_diario_v2(by_day, df_all, calib_from):
    plt.rcParams.update({
        "figure.facecolor":"#ffffff","axes.facecolor":"#fafafa",
        "axes.edgecolor":"#d1d5db","text.color":"#111",
        "xtick.color":"#666","ytick.color":"#666","grid.color":"#e5e7eb","font.size":9,
    })

    fig = plt.figure(figsize=(24, 28))
    gs  = gridspec.GridSpec(5, 1, figure=fig, hspace=0.52,
                            top=0.94, bottom=0.03, left=0.06, right=0.97,
                            height_ratios=[2.2, 2.2, 1.8, 1.8, 5.0])

    dates    = pd.to_datetime(by_day["date"])
    acc_all  = by_day["acc_all"].values
    acc_filt = by_day["acc_filt"].values

    # ── Panel 1: Accuracy diaria — todos vs filtrado ──────────────────────────
    ax1 = fig.add_subplot(gs[0])
    w = 0.35
    x = np.arange(len(dates))
    ax1.bar(x-w/2, acc_all,  w, color=["#166534" if a>=52.4 else "#dc2626" for a in acc_all],
            alpha=0.55, label="Todos los picks")
    filt_vals = [a if not np.isnan(a) else 0 for a in acc_filt]
    ax1.bar(x+w/2, filt_vals, w,
            color=["#166534" if a>=52.4 else "#dc2626" for a in filt_vals],
            alpha=0.90, label=f"Filtrado (P≥{THRESHOLD})")
    roll7 = pd.Series(filt_vals).rolling(7, min_periods=3).mean()
    ax1.plot(x, roll7, color="#111", lw=2, label="Rolling 7d (filtrado)")
    ax1.axhline(52.4, color="#374151", lw=1.2, linestyle="--", label="52.4% break-even")
    ax1.axhline(50,   color="#aaa",    lw=0.8, linestyle=":")
    # Separador entre periodo entrenamiento y prueba
    calib_x = np.where(dates >= calib_from)[0]
    if len(calib_x):
        ax1.axvline(calib_x[0]-0.5, color="#6366f1", lw=2, linestyle="-", alpha=0.7)
        ax1.text(calib_x[0]-0.3, 98, "← entrenamiento  prueba →",
                 fontsize=8, color="#6366f1", va="top")
    ax1.set_xticks(x[::3]); ax1.set_xticklabels([d.strftime("%b %d") for d in dates[::3]],
                                                   fontsize=7.5, rotation=30, ha="right")
    ax1.set_ylim(0, 105); ax1.set_ylabel("Accuracy %")
    ax1.set_title("Accuracy diaria — Todos los picks vs Filtrado por calibrador (Capa 2)",
                  fontsize=12, pad=10, loc="left", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left"); ax1.grid(axis="y", alpha=0.4)

    # ── Panel 2: P&L acumulado — todos vs filtrado ────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, by_day["cum_pnl_all"],  color="#9ca3af", lw=1.8, label="P&L todos (sin filtro)")
    ax2.plot(dates, by_day["cum_pnl_filt"], color="#111",    lw=2.5, label="P&L filtrado (P≥0.54)")
    ax2.fill_between(dates, by_day["cum_pnl_filt"], 0,
                     where=by_day["cum_pnl_filt"]>=0, alpha=0.15, color="#166534")
    ax2.fill_between(dates, by_day["cum_pnl_filt"], 0,
                     where=by_day["cum_pnl_filt"]<0,  alpha=0.15, color="#dc2626")
    ax2.axhline(0, color="#374151", lw=1)
    if len(calib_x):
        ax2.axvline(pd.to_datetime(by_day.iloc[calib_x[0]]["date"]),
                    color="#6366f1", lw=2, linestyle="-", alpha=0.5, label="Inicio capa 2")
    pnl_vals = by_day["cum_pnl_filt"].values
    max_i=np.argmax(pnl_vals); min_i=np.argmin(pnl_vals)
    ax2.annotate(f"${pnl_vals[max_i]:+.0f}", xy=(dates.iloc[max_i], pnl_vals[max_i]),
                 fontsize=8, color="#166534", xytext=(0,7), textcoords="offset points")
    ax2.annotate(f"${pnl_vals[min_i]:+.0f}", xy=(dates.iloc[min_i], pnl_vals[min_i]),
                 fontsize=8, color="#dc2626", xytext=(0,-14), textcoords="offset points")
    ax2.set_ylabel("USD"); ax2.grid(alpha=0.4)
    ax2.set_title("P&L acumulado — $110 por pick",
                  fontsize=11, pad=8, loc="left")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Panel 3: picks jugados vs total por día ───────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(dates, by_day["games"], color="#d1d5db", width=0.8, label="Total partidos")
    ax3.bar(dates, by_day["plays"], color="#374151", width=0.8, alpha=0.9, label="Picks jugados (filtrado)")
    ax3.set_ylabel("# partidos"); ax3.grid(axis="y", alpha=0.4)
    ax3.set_title("Picks jugados por día (filtrado) vs total de partidos",
                  fontsize=11, pad=8, loc="left")
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Panel 4: OVER vs UNDER por semana (filtrado) ──────────────────────────
    ax4 = fig.add_subplot(gs[3])
    df_filt = df_all[df_all["play"]].copy()
    if len(df_filt) > 0:
        df_filt["week"] = pd.to_datetime(df_filt["date"]).dt.to_period("W").apply(
            lambda x: str(x.start_time.date()))
        wk = df_filt.groupby(["week","rec"]).agg(acc=("correct","mean"),n=("correct","count")).reset_index()
        for rec, col in [("OVER","#dc2626"),("UNDER","#166534")]:
            sub = wk[wk["rec"]==rec]
            if len(sub):
                ax4.plot(pd.to_datetime(sub["week"]), sub["acc"]*100,
                         "o-", color=col, lw=2, ms=6, label=f"{rec} (filtrado)")
    ax4.axhline(52.4, color="#374151", lw=1.2, linestyle="--")
    ax4.axhline(50,   color="#aaa", lw=0.8, linestyle=":")
    ax4.set_ylim(25, 85); ax4.set_ylabel("Accuracy %")
    ax4.set_title("OVER vs UNDER accuracy semanal — solo picks filtrados",
                  fontsize=11, pad=8, loc="left")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.4)
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Panel 5: Tabla calendario detallada ───────────────────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.axis("off")
    recent = by_day.tail(50).reset_index(drop=True)
    row_h  = 1.0 / (len(recent)+1)

    # header
    for x_pos, lbl in [(0.01,"FECHA"),(0.12,"J"),(0.15,"Todos"),(0.26,"P&L"),
                        (0.33,"Filtrado"),(0.46,"P&L"),(0.53,"Cum.All"),(0.61,"Cum.Filt"),
                        (0.69,"Partidos del día (★=jugado)")]:
        ax5.text(x_pos, 1.01, lbl, fontsize=7.5, fontweight="bold",
                 color="#444", transform=ax5.transAxes)

    for i, row in recent.iterrows():
        y = 1.0 - (i+1)*row_h - 0.01
        bg = "#f0fdf4" if (not np.isnan(row["acc_filt"]) and row["acc_filt"]>=52.4) \
             else "#fef2f2" if row["acc_all"]<50 else "#ffffff"
        ax5.add_patch(plt.Rectangle((0, y-row_h*0.35), 1, row_h*0.85,
                      facecolor=bg, edgecolor="#e0e0e0", lw=0.4,
                      transform=ax5.transAxes, clip_on=False))

        calib_mark = "★" if row["date"] >= calib_from else " "
        acc_all_s  = f"{int(row['correct'])}/{int(row['games'])} ({row['acc_all']:.0f}%)"
        ca_color   = "#166534" if row["acc_all"]>=52.4 else "#dc2626"

        if row["plays"]>0 and not np.isnan(row["acc_filt"]):
            acc_filt_s = f"{int(row['plays_correct'])}/{int(row['plays'])} ({row['acc_filt']:.0f}%)"
            cf_color   = "#166534" if row["acc_filt"]>=52.4 else "#dc2626"
        else:
            acc_filt_s = "—"; cf_color="#aaa"

        cc_all  = "#166534" if row["cum_pnl_all"] >=0 else "#dc2626"
        cc_filt = "#166534" if row["cum_pnl_filt"]>=0 else "#dc2626"

        ax5.text(0.01, y+row_h*0.22, row["date"].strftime("%a %b %d"),
                 fontsize=8, fontweight="bold", color="#111", transform=ax5.transAxes)
        ax5.text(0.12, y+row_h*0.22, str(int(row["games"])),
                 fontsize=8, color="#444", transform=ax5.transAxes)
        ax5.text(0.15, y+row_h*0.22, acc_all_s,
                 fontsize=8, color=ca_color, fontweight="bold", transform=ax5.transAxes)
        ax5.text(0.26, y+row_h*0.22, f"${row['pnl_all']:+.0f}",
                 fontsize=8, color=ca_color, transform=ax5.transAxes)
        ax5.text(0.33, y+row_h*0.22, f"{calib_mark} {acc_filt_s}",
                 fontsize=8, color=cf_color, fontweight="bold", transform=ax5.transAxes)
        ax5.text(0.46, y+row_h*0.22, f"${row['pnl_filt']:+.0f}",
                 fontsize=8, color=cf_color, transform=ax5.transAxes)
        ax5.text(0.53, y+row_h*0.22, f"${row['cum_pnl_all']:+.0f}",
                 fontsize=8, color=cc_all, transform=ax5.transAxes)
        ax5.text(0.61, y+row_h*0.22, f"${row['cum_pnl_filt']:+.0f}",
                 fontsize=8, color=cc_filt, transform=ax5.transAxes)

        # Partidos del día con marcador de jugado
        day_preds = df_all[df_all["date"]==row["date"]]
        parts = []
        for _, pr in day_preds.iterrows():
            sym  = "✓" if pr["correct"] else "✗"
            star = "★" if pr["play"] else " "
            parts.append(f"{star}{pr['away']}@{pr['home']} {pr['rec']}{pr['line']} {sym}r={pr['actual']}")
        ax5.text(0.69, y+row_h*0.22, "  ".join(parts)[:100],
                 fontsize=6.8, color="#444", transform=ax5.transAxes)

    ax5.set_title(f"Detalle por día — ★ = pick jugado (P≥{THRESHOLD})  ✓=acierto  ✗=error",
                  fontsize=10, pad=8, loc="left", color="#111")
    ax5.set_xlim(0,1); ax5.set_ylim(0,1)

    n_all  = len(df_all); n_filt = df_all["play"].sum()
    acc_a  = df_all["correct"].mean()*100
    acc_f  = df_all.loc[df_all["play"],"correct"].mean()*100 if n_filt>0 else 0
    pnl_a  = df_all["correct"].map({True:100.,False:-110.}).sum()
    pnl_f  = df_all.loc[df_all["play"],"correct"].map({True:100.,False:-110.}).sum()

    fig.suptitle(
        f"MLB O/U — Backtest Dia a Dia v2  |  Capa 1 + Calibrador Capa 2  |  2026 Sem.2 → Jun 4\n"
        f"Todos: {acc_a:.1f}% ({n_all}j) USD{pnl_a:+.0f}   |   "
        f"Filtrado: {acc_f:.1f}% ({n_filt}j jugados/{n_all}) USD{pnl_f:+.0f}   |   "
        f"Calibrador activo desde May 15",
        fontsize=12, fontweight="bold", y=0.975
    )

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_backtest_diario_v2.png"
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor="#ffffff")
    print(f"\nImagen guardada: {out}")

if __name__ == "__main__":
    main()
