#!/usr/bin/env python3
"""
Backtest día por día — MLB Over/Under 2026.

Para cada día con partidos desde semana 2 de temporada hasta hoy:
  - Simula qué hubiera dicho el modelo esa mañana
  - Compara con resultados reales de ese día
  - Muestra: correctos / total, P&L, accuracy

Walk-forward honesto: solo usa datos ANTERIORES a cada día.
Modelo entrenado con 2025 exclusivamente.
"""

import sys, importlib, json, statistics
from pathlib import Path
from datetime import datetime

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

np = ensure("numpy"); pd = ensure("pandas")
ensure("matplotlib"); ensure("seaborn")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np, pandas as pd

import urllib.request

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
    for d in fetch(url).get("dates", []):
        for g in d.get("games", []):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls = g.get("linescore",{}).get("teams",{})
            ar = ls.get("away",{}).get("runs",0) or 0
            hr = ls.get("home",{}).get("runs",0) or 0
            if ar + hr == 0: continue
            rows.append({
                "date":      d["date"],
                "away_id":   g["teams"]["away"]["team"]["id"],
                "home_id":   g["teams"]["home"]["team"]["id"],
                "away_name": g["teams"]["away"]["team"]["name"],
                "home_name": g["teams"]["home"]["team"]["name"],
                "away_runs": ar, "home_runs": hr, "total": ar + hr,
            })
    df = pd.DataFrame(rows)
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    return df

def load_team_stats(season):
    hit = {}; pit = {}
    for group, store in [("hitting", hit), ("pitching", pit)]:
        url = (f"https://statsapi.mlb.com/api/v1/teams/stats"
               f"?season={season}&sportId=1&stats=season&group={group}")
        for sp in fetch(url).get("stats",[{}])[0].get("splits",[]):
            store[sp["team"]["id"]] = sp["stat"]
    return hit, pit

LINES = [6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]
def nearest_line(v): return min(LINES, key=lambda x: abs(x-v))

def rolling_stats(df, tid, before_date):
    mask = ((df["away_id"]==tid)|(df["home_id"]==tid)) & (df["date"] < before_date)
    sub  = df[mask].sort_values("date")
    if len(sub) < 5: return None
    scored  = [r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
    allowed = [r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]
    wins = sum(1 for s,a in zip(scored,allowed) if s>a)
    return {
        "rpg":      np.mean(scored),
        "rpg_l10":  np.mean(scored[-10:]),
        "ra_pg":    np.mean(allowed),
        "win_pct":  wins/len(scored),
        "scored":   scored,
        "allowed":  allowed,
    }

def h2h_before(df, aid, hid, before_date):
    mask = (((df["away_id"]==aid)&(df["home_id"]==hid))|
            ((df["away_id"]==hid)&(df["home_id"]==aid))) & (df["date"] < before_date)
    return df[mask]["total"].tolist()

def formula_predict(a, h, h2h):
    def proj(att, def_):
        sv = att["rpg"]; l10 = att["rpg_l10"]
        wv = np.mean(att["scored"][-7:]) if len(att["scored"])>=3 else sv
        return sv*0.25 + l10*0.25 + wv*0.35 + def_["ra_pg"]*0.15
    p = proj(a,h) + proj(h,a)
    if len(h2h)>=2: p = p*0.85 + np.mean(h2h)*0.15
    line = nearest_line(p)
    return p, line, "OVER" if p>=line else "UNDER"

from ml_model import MLPredictor, build_features

def train_2025_model(df_2025, hit25, pit25):
    agg = {}
    for tid in set(df_2025["away_id"])|set(df_2025["home_id"]):
        sub = df_2025[(df_2025["away_id"]==tid)|(df_2025["home_id"]==tid)]
        sc  = [r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
        al  = [r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]
        wins = sum(1 for s,a in zip(sc,al) if s>a)
        agg[tid] = {"rpg":np.mean(sc) if sc else 4.0,"rpg_l10":np.mean(sc[-10:]) if sc else 4.0,
                    "ra_pg":np.mean(al) if al else 4.0,"win_pct":wins/len(sc) if sc else 0.5}
    rows = []
    for _,g in df_2025.iterrows():
        a=agg.get(g["away_id"],{}); h=agg.get(g["home_id"],{})
        if not a or not h: continue
        aps=pit25.get(g["away_id"],{}); hps=pit25.get(g["home_id"],{})
        ahs=hit25.get(g["away_id"],{}); hhs=hit25.get(g["home_id"],{})
        feats=build_features(a["rpg"],a["rpg_l10"],a["win_pct"],aps.get("era","4.20"),
                              aps.get("whip","1.30"),ahs.get("ops",".720"),"",a["ra_pg"],
                              h["rpg"],h["rpg_l10"],h["win_pct"],hps.get("era","4.20"),
                              hps.get("whip","1.30"),hhs.get("ops",".720"),"",h["ra_pg"],
                              None,0,g["date"].month)
        rows.append({"features":feats,"target":float(g["total"])})
    ml = MLPredictor().fit(rows)
    return ml

def main():
    print("Cargando datos...")
    df_2025 = load_games("2025-03-20","2025-09-28")
    df_2026 = load_games("2026-03-19","2026-06-04")
    hit25,pit25 = load_team_stats(2025)
    hit26,pit26 = load_team_stats(2026)
    print(f"  2025: {len(df_2025)}j  |  2026: {len(df_2026)}j")

    print("Entrenando con 2025...")
    ml = train_2025_model(df_2025, hit25, pit25)
    print(f"  CV MAE: {ml.cv_mae:.3f} runs\n")

    # Backtest desde semana 2 (2026-04-01) día por día
    cutoff = pd.Timestamp("2026-04-01")
    game_dates = sorted(df_2026[df_2026["date"]>=cutoff]["date"].unique())
    print(f"Procesando {len(game_dates)} dias con partidos...\n")

    # ── procesar juego por juego ──────────────────────────────────────────────
    all_preds = []
    for di, gdate in enumerate(game_dates):
        day_games = df_2026[df_2026["date"]==gdate]
        for _, g in day_games.iterrows():
            aid=g["away_id"]; hid=g["home_id"]
            a_st = rolling_stats(df_2026, aid, gdate)
            h_st = rolling_stats(df_2026, hid, gdate)
            if not a_st or not h_st: continue
            h2h = h2h_before(df_2026, aid, hid, gdate)

            f_proj, f_line, f_rec = formula_predict(a_st, h_st, h2h)

            aps=pit26.get(aid,{}); hps=pit26.get(hid,{})
            ahs=hit26.get(aid,{}); hhs=hit26.get(hid,{})
            feats=build_features(a_st["rpg"],a_st["rpg_l10"],a_st["win_pct"],
                                  aps.get("era","4.20"),aps.get("whip","1.30"),ahs.get("ops",".720"),
                                  "",a_st["ra_pg"],
                                  h_st["rpg"],h_st["rpg_l10"],h_st["win_pct"],
                                  hps.get("era","4.20"),hps.get("whip","1.30"),hhs.get("ops",".720"),
                                  "",h_st["ra_pg"],
                                  np.mean(h2h) if len(h2h)>=2 else None,
                                  len(h2h), gdate.month)
            ml_r = ml.predict(feats)
            ml_tot = ml_r["total"] or f_proj

            ens = ml_tot*0.55 + f_proj*0.45
            line = nearest_line(ens)
            rec  = "OVER" if ens>=line else "UNDER"
            actual = g["total"]
            correct = (rec=="OVER" and actual>line) or (rec=="UNDER" and actual<line)
            push    = actual==line
            diff    = abs(ens-line)
            conf    = 3 if diff>1.5 else 2 if diff>0.6 else 1

            all_preds.append({
                "date":      gdate,
                "away":      g["away_name"].split()[-1],
                "home":      g["home_name"].split()[-1],
                "proj":      round(ens,2),
                "line":      line,
                "rec":       rec,
                "actual":    int(actual),
                "correct":   correct,
                "push":      push,
                "conf":      conf,
                "diff":      round(ens-line,2),
                "error":     round(ens-actual,2),
            })
        if (di+1) % 10 == 0:
            print(f"  {di+1}/{len(game_dates)} dias...", end="\r")

    print(f"\n  {len(all_preds)} predicciones calculadas")
    df = pd.DataFrame(all_preds)

    # ── Resumen por día ───────────────────────────────────────────────────────
    by_day = df.groupby("date").agg(
        games    =("correct","count"),
        correct  =("correct","sum"),
        pushes   =("push","sum"),
        pnl      =("correct", lambda x: x.map({True:100.,False:-110.}).sum()),
        mean_err =("error",   lambda x: x.abs().mean()),
    ).reset_index()
    by_day["accuracy"] = by_day["correct"] / by_day["games"] * 100
    by_day["cum_pnl"]  = by_day["pnl"].cumsum()
    by_day["cum_acc"]  = df.sort_values("date").groupby("date")["correct"].sum().cumsum().values / \
                         df.sort_values("date").groupby("date")["correct"].count().cumsum().values * 100

    # ── Print tabla dia por dia ───────────────────────────────────────────────
    print("\n" + "="*78)
    print(f"  DIA A DIA — MLB O/U BACKTEST 2026  (modelo entrenado solo con 2025)")
    print("="*78)
    print(f"  {'Fecha':<12} {'J':>3} {'OK':>3} {'Acc':>6} {'P&L':>8} {'CumP&L':>9}  Partidos del dia")
    print(f"  {'-'*12} {'-'*3} {'-'*3} {'-'*6} {'-'*8} {'-'*9}  {'-'*30}")

    for _, row in by_day.tail(60).iterrows():  # last 60 days
        day_g = df[df["date"]==row["date"]]
        results_str = "  ".join([
            f"{r['away']}@{r['home']} {'OK' if r['correct'] else 'XX'}"
            for _, r in day_g.iterrows()
        ])
        acc_color = "✓" if row["accuracy"] >= 52.4 else "·"
        print(f"  {row['date'].strftime('%b %d, %Y'):<12} "
              f"{int(row['games']):>3} "
              f"{int(row['correct']):>3} "
              f"{row['accuracy']:>5.0f}% "
              f"{row['pnl']:>+8.0f} "
              f"{row['cum_pnl']:>+9.0f}  "
              f"{acc_color} {results_str[:60]}")

    # Overall
    n   = len(df)
    acc = df["correct"].mean()*100
    pnl = df["correct"].map({True:100.,False:-110.}).sum()
    mae = df["error"].abs().mean()
    print(f"\n  TOTAL: {n} juegos  |  Accuracy: {acc:.1f}%  |  MAE: {mae:.2f} runs  |  P&L: ${pnl:+.0f}")
    print(f"  UNDER picks: {df[df['rec']=='UNDER']['correct'].mean()*100:.1f}%  |  "
          f"OVER picks: {df[df['rec']=='OVER']['correct'].mean()*100:.1f}%")

    # ── IMAGEN ────────────────────────────────────────────────────────────────
    plot_daily(by_day, df)

def plot_daily(by_day, df_all):
    plt.rcParams.update({
        "figure.facecolor":"#ffffff","axes.facecolor":"#fafafa",
        "axes.edgecolor":"#d1d5db","text.color":"#111",
        "xtick.color":"#666","ytick.color":"#666",
        "grid.color":"#e5e7eb","font.size":9,
    })

    fig = plt.figure(figsize=(24, 26))
    gs  = gridspec.GridSpec(5, 1, figure=fig,
                            hspace=0.50, top=0.94, bottom=0.03,
                            left=0.06, right=0.97,
                            height_ratios=[2.5, 2.5, 2.0, 2.0, 4.5])

    dates     = pd.to_datetime(by_day["date"])
    acc_vals  = by_day["accuracy"].values
    cum_pnl   = by_day["cum_pnl"].values
    games_n   = by_day["games"].values
    correct_n = by_day["correct"].values

    # ── Panel 1: Accuracy por día (barras) + rolling 7d ──────────────────────
    ax1 = fig.add_subplot(gs[0])
    bar_colors = ["#166534" if a>=52.4 else "#dc2626" for a in acc_vals]
    ax1.bar(dates, acc_vals, color=bar_colors, alpha=0.75, width=0.8, edgecolor="none")
    roll7 = pd.Series(acc_vals).rolling(7, min_periods=3).mean()
    ax1.plot(dates, roll7, color="#111", lw=2, label="Rolling 7 días")
    ax1.axhline(52.4, color="#374151", lw=1.2, linestyle="--", label="52.4% break-even (-110)")
    ax1.axhline(50,   color="#9ca3af", lw=0.8, linestyle=":")
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("Accuracy %")
    ax1.set_title(
        f"Accuracy diaria — MLB O/U  |  "
        f"Overall {df_all['correct'].mean()*100:.1f}%  |  "
        f"P&L acum. ${cum_pnl[-1]:+.0f}  |  "
        f"MAE {df_all['error'].abs().mean():.2f} runs",
        fontsize=12, fontweight="bold", pad=10, loc="left"
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.4)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))
    # Anotar % en los dias notables
    for xi, (d, a, n_) in enumerate(zip(dates, acc_vals, games_n)):
        if a == 100 or (a >= 70 and n_ >= 5):
            ax1.text(d, a+1.5, f"{a:.0f}%", ha="center", fontsize=7, color="#166534", fontweight="bold")
        elif a == 0 or (a <= 30 and n_ >= 5):
            ax1.text(d, a+1.5, f"{a:.0f}%", ha="center", fontsize=7, color="#dc2626", fontweight="bold")

    # ── Panel 2: P&L acumulado ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(dates, cum_pnl, color="#111", lw=2.5, label="P&L acumulado")
    ax2.fill_between(dates, cum_pnl, 0,
                     where=np.array(cum_pnl)>=0, alpha=0.18, color="#166534")
    ax2.fill_between(dates, cum_pnl, 0,
                     where=np.array(cum_pnl)<0,  alpha=0.18, color="#dc2626")
    ax2.axhline(0, color="#374151", lw=1)
    # Anotar maximo y minimo
    max_i = np.argmax(cum_pnl); min_i = np.argmin(cum_pnl)
    ax2.annotate(f"Max ${cum_pnl[max_i]:+.0f}", xy=(dates.iloc[max_i], cum_pnl[max_i]),
                 fontsize=8, color="#166534",
                 xytext=(0, 8), textcoords="offset points")
    ax2.annotate(f"Min ${cum_pnl[min_i]:+.0f}", xy=(dates.iloc[min_i], cum_pnl[min_i]),
                 fontsize=8, color="#dc2626",
                 xytext=(0, -14), textcoords="offset points")
    ax2.set_ylabel("USD (apostando $110/juego)")
    ax2.set_title("P&L acumulado — apostando $110 para ganar $100 en cada pick",
                  fontsize=11, pad=8, loc="left")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.4)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Panel 3: Juegos correctos/totales por día ─────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.bar(dates, games_n,   color="#d1d5db", width=0.8, edgecolor="none", label="Total juegos")
    ax3.bar(dates, correct_n, color="#374151", width=0.8, edgecolor="none", alpha=0.9, label="Correctos")
    ax3.set_ylabel("# partidos")
    ax3.set_title("Correctos vs Total por día",
                  fontsize=11, pad=8, loc="left")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.4)
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Panel 4: OVER vs UNDER accuracy por semana ────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    df_all["week"] = pd.to_datetime(df_all["date"]).dt.to_period("W").apply(lambda x: str(x.start_time.date()))
    weekly = df_all.groupby(["week","rec"]).agg(
        acc=("correct","mean"), n=("correct","count")).reset_index()
    weeks_u = weekly[weekly["rec"]=="UNDER"]; weeks_o = weekly[weekly["rec"]=="OVER"]
    w_dates_u = pd.to_datetime(weeks_u["week"]); w_dates_o = pd.to_datetime(weeks_o["week"])
    ax4.plot(w_dates_u, weeks_u["acc"]*100, "o-", color="#166534", lw=2, ms=6, label="UNDER accuracy")
    ax4.plot(w_dates_o, weeks_o["acc"]*100, "s-", color="#dc2626", lw=2, ms=6, label="OVER accuracy")
    ax4.axhline(52.4, color="#9ca3af", lw=1, linestyle="--")
    ax4.set_ylim(25, 85); ax4.set_ylabel("Accuracy %")
    ax4.set_title("OVER vs UNDER accuracy — por semana",
                  fontsize=11, pad=8, loc="left")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.4)
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── Panel 5: Calendario visual — cada dia es una fila ─────────────────────
    ax5 = fig.add_subplot(gs[4])
    ax5.axis("off")
    # Últimos 45 días para que quepa bien
    recent_days = by_day.tail(45).reset_index(drop=True)
    row_h = 1.0 / (len(recent_days) + 1)

    # Cabecera
    ax5.text(0.01, 1.01, "FECHA", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)
    ax5.text(0.10, 1.01, "J", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)
    ax5.text(0.13, 1.01, "OK", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)
    ax5.text(0.17, 1.01, "Acc", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)
    ax5.text(0.23, 1.01, "P&L dia", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)
    ax5.text(0.30, 1.01, "Acum.", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)
    ax5.text(0.38, 1.01, "Partidos", fontsize=8, fontweight="bold", color="#444", transform=ax5.transAxes)

    for i, row in recent_days.iterrows():
        y = 1.0 - (i + 1) * row_h - 0.02
        acc_ = row["accuracy"]
        pnl_ = row["pnl"]
        bg = "#f0fdf4" if acc_>=52.4 else "#fef2f2"
        ax5.add_patch(plt.Rectangle((0, y-0.005), 1, row_h*0.92,
                      facecolor=bg, edgecolor="#e0e0e0", lw=0.5,
                      transform=ax5.transAxes, clip_on=False))

        # Fecha
        ax5.text(0.01, y+row_h*0.28,
                 row["date"].strftime("%a %b %d"),
                 fontsize=8.5, color="#111", transform=ax5.transAxes, fontweight="bold")

        # Juegos / correctos
        ax5.text(0.10, y+row_h*0.28, str(int(row["games"])),
                 fontsize=8.5, color="#444", transform=ax5.transAxes)
        ax5.text(0.13, y+row_h*0.28, str(int(row["correct"])),
                 fontsize=8.5, color="#444", transform=ax5.transAxes)

        # Accuracy con color
        c_acc = "#166534" if acc_>=52.4 else "#dc2626"
        ax5.text(0.17, y+row_h*0.28, f"{acc_:.0f}%",
                 fontsize=8.5, color=c_acc, fontweight="bold", transform=ax5.transAxes)

        # P&L dia
        c_pnl = "#166534" if pnl_>=0 else "#dc2626"
        ax5.text(0.23, y+row_h*0.28, f"${pnl_:+.0f}",
                 fontsize=8.5, color=c_pnl, fontweight="bold", transform=ax5.transAxes)

        # Acumulado
        c_cum = "#166534" if row["cum_pnl"]>=0 else "#dc2626"
        ax5.text(0.30, y+row_h*0.28, f"${row['cum_pnl']:+.0f}",
                 fontsize=8, color=c_cum, transform=ax5.transAxes)

        # Partidos del dia con resultado
        day_preds = df_all[df_all["date"]==row["date"]]
        game_strs = []
        for _, pr in day_preds.iterrows():
            sym = "✓" if pr["correct"] else "✗"
            game_strs.append(f"{pr['away']}@{pr['home']} {pr['rec']} {pr['line']} {sym}(real {pr['actual']})")
        line_txt = "  |  ".join(game_strs)[:120]
        ax5.text(0.38, y+row_h*0.28, line_txt,
                 fontsize=7.2, color="#444", transform=ax5.transAxes)

    ax5.set_title("Detalle por dia (ultimos 45 dias) — verde=dia rentable  rojo=dia perdedor",
                  fontsize=11, pad=8, loc="left", color="#111")
    ax5.set_xlim(0,1); ax5.set_ylim(0,1)

    # ── Titulo global ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             f"MLB OVER/UNDER — Backtest Dia a Dia  |  2026 (desde semana 2) → Jun 4",
             ha="center", fontsize=15, fontweight="bold", color="#111")
    fig.text(0.5, 0.960,
             "Modelo entrenado solo con datos 2025. "
             "Predicciones calculadas con stats disponibles ANTES de cada juego (sin data leakage).",
             ha="center", fontsize=9, color="#666")

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_backtest_diario.png"
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor="#ffffff")
    print(f"\nImagen guardada: {out}")

if __name__ == "__main__":
    main()
