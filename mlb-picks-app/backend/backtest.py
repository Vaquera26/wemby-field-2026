#!/usr/bin/env python3
"""
Backtest honesto del modelo MLB Over/Under.

Metodología walk-forward (sin data leakage):
  - Para cada juego en fecha D:
      * Features del equipo = promedio de sus juegos ANTERIORES a D
      * Modelo ML entrenado SOLO con temporada 2025 (nunca ve datos 2026)
      * Formula usa rolling stats hasta D-1
  - Se compara la predicción con el resultado real
  - Se calcula accuracy, MAE, P&L (apostando $110 para ganar $100)
"""

import sys, importlib, statistics, json
from pathlib import Path
from datetime import date, timedelta, datetime
from collections import defaultdict

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

np  = ensure("numpy"); pd = ensure("pandas")
ensure("matplotlib"); ensure("seaborn")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np, pandas as pd

# ── Inline fetch (sin depender del servidor) ──────────────────────────────────
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

# ── Rolling team stats up to (not including) a date ──────────────────────────
def rolling_stats(df_hist, tid, before_date):
    """Returns dict of team stats using only games before before_date."""
    mask = ((df_hist["away_id"]==tid) | (df_hist["home_id"]==tid)) & (df_hist["date"] < before_date)
    sub  = df_hist[mask].sort_values("date")
    if sub.empty: return None

    scored  = [r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
    allowed = [r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]

    wins   = sum(1 for s,a in zip(scored,allowed) if s > a)
    losses = len(scored) - wins

    return {
        "rpg":      np.mean(scored)       if scored  else 4.0,
        "rpg_l10":  np.mean(scored[-10:]) if scored  else 4.0,
        "ra_pg":    np.mean(allowed)      if allowed else 4.0,
        "win_pct":  wins / len(scored)    if scored  else 0.5,
        "games":    len(scored),
        "streak":   "",
        "scored":   scored,
        "allowed":  allowed,
    }

def h2h_before(df_hist, aid, hid, before_date):
    mask = (((df_hist["away_id"]==aid)&(df_hist["home_id"]==hid))|
            ((df_hist["away_id"]==hid)&(df_hist["home_id"]==aid))) & (df_hist["date"] < before_date)
    sub = df_hist[mask]
    return sub["total"].tolist()

# ── Formula prediction (same weights as main model) ──────────────────────────
LINES = [6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]

def nearest_line(v): return min(LINES, key=lambda x: abs(x-v))

def formula_predict(a_stats, h_stats, h2h_totals):
    W = {"season":0.20,"l10":0.25,"week":0.30,"prev":0.10,"def":0.05}
    def proj_team(att, def_):
        sv  = att["rpg"];     l10 = att["rpg_l10"]
        wv  = np.mean(att["scored"][-7:]) if len(att["scored"])>=2 else sv
        dv  = def_["ra_pg"]
        return sv*0.25 + l10*0.25 + wv*0.35 + dv*0.10 + att["rpg"]*0.05
    pa = proj_team(a_stats, h_stats)
    ph = proj_team(h_stats, a_stats)
    proj = pa + ph
    if len(h2h_totals) >= 2:
        proj = proj * 0.85 + np.mean(h2h_totals) * 0.15
    line = nearest_line(proj)
    return proj, line, "OVER" if proj >= line else "UNDER"

# ── ML prediction (2025-trained model) ───────────────────────────────────────
from ml_model import MLPredictor, build_features

def train_on_2025(df_2025, hit25, pit25):
    """Train ML model ONLY on 2025 data."""
    from ml_model import build_features, MLPredictor

    # Build 2025 team aggregates
    agg = {}
    for tid in set(df_2025["away_id"]) | set(df_2025["home_id"]):
        mask = (df_2025["away_id"]==tid) | (df_2025["home_id"]==tid)
        sub = df_2025[mask]
        sc  = [r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
        al  = [r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]
        wins = sum(1 for s,a in zip(sc,al) if s>a)
        agg[tid] = {
            "rpg": np.mean(sc) if sc else 4.0,
            "rpg_l10": np.mean(sc[-10:]) if sc else 4.0,
            "ra_pg": np.mean(al) if al else 4.0,
            "win_pct": wins/len(sc) if sc else 0.5,
        }

    rows = []
    for _, g in df_2025.iterrows():
        aid=g["away_id"]; hid=g["home_id"]
        a=agg.get(aid,{}); h=agg.get(hid,{})
        if not a or not h: continue
        aps = pit25.get(aid, {}); hps = pit25.get(hid, {})
        ahs = hit25.get(aid, {}); hhs = hit25.get(hid, {})
        month = g["date"].month
        feats = build_features(
            away_rpg=a["rpg"],          away_rpg_l10=a["rpg_l10"],
            away_win_pct=a["win_pct"],  away_era=aps.get("era","4.20"),
            away_whip=aps.get("whip","1.30"), away_ops=ahs.get("ops",".720"),
            away_streak="",             away_ra_pg=a["ra_pg"],
            home_rpg=h["rpg"],          home_rpg_l10=h["rpg_l10"],
            home_win_pct=h["win_pct"],  home_era=hps.get("era","4.20"),
            home_whip=hps.get("whip","1.30"), home_ops=hhs.get("ops",".720"),
            home_streak="",             home_ra_pg=h["ra_pg"],
            h2h_avg=None, h2h_n=0, month=month,
        )
        rows.append({"features": feats, "target": float(g["total"])})

    print(f"  2025 training rows: {len(rows)}")
    ml = MLPredictor().fit(rows)
    print(f"  2025-only model MAE (CV): {ml.cv_mae:.3f} runs")
    return ml

# ── Main backtest loop ────────────────────────────────────────────────────────
def run_backtest():
    print("Cargando datos...")
    df_2025 = load_games("2025-03-20", "2025-09-28")
    df_2026 = load_games("2026-03-19", "2026-06-04")
    hit25, pit25 = load_team_stats(2025)
    hit26, pit26 = load_team_stats(2026)
    print(f"  2025: {len(df_2025)} juegos  |  2026: {len(df_2026)} juegos")

    print("\nEntrenando modelo con solo datos 2025...")
    ml_2025 = train_on_2025(df_2025, hit25, pit25)

    # Backtest sobre 2026 — para cada juego usamos rolling stats hasta D-1
    # Solo probamos desde Apr 15 en adelante (suficientes juegos previos)
    cutoff = pd.Timestamp("2026-04-15")
    test_games = df_2026[df_2026["date"] >= cutoff].copy()
    print(f"\nBacktest sobre {len(test_games)} juegos (2026-04-15 → 2026-06-04)")
    print("Calculando predicciones rolling (puede tardar ~30s)...\n")

    results = []
    for i, (_, g) in enumerate(test_games.iterrows()):
        aid=g["away_id"]; hid=g["home_id"]
        gdate = g["date"]
        actual_total = g["total"]

        # Rolling stats for this game
        a_stats = rolling_stats(df_2026, aid, gdate)
        h_stats = rolling_stats(df_2026, hid, gdate)
        if a_stats is None or a_stats["games"] < 5: continue
        if h_stats is None or h_stats["games"] < 5: continue

        h2h = h2h_before(df_2026, aid, hid, gdate)

        # Formula prediction
        f_proj, f_line, f_rec = formula_predict(a_stats, h_stats, h2h)

        # ML prediction (2025-trained)
        aps = pit26.get(aid,{}); hps = pit26.get(hid,{})
        ahs = hit26.get(aid,{}); hhs = hit26.get(hid,{})
        feats = build_features(
            away_rpg=a_stats["rpg"],       away_rpg_l10=a_stats["rpg_l10"],
            away_win_pct=a_stats["win_pct"], away_era=aps.get("era","4.20"),
            away_whip=aps.get("whip","1.30"), away_ops=ahs.get("ops",".720"),
            away_streak="",                away_ra_pg=a_stats["ra_pg"],
            home_rpg=h_stats["rpg"],       home_rpg_l10=h_stats["rpg_l10"],
            home_win_pct=h_stats["win_pct"], home_era=hps.get("era","4.20"),
            home_whip=hps.get("whip","1.30"), home_ops=hhs.get("ops",".720"),
            home_streak="",                home_ra_pg=h_stats["ra_pg"],
            h2h_avg=np.mean(h2h) if len(h2h)>=2 else None,
            h2h_n=len(h2h), month=gdate.month,
        )
        ml_raw = ml_2025.predict(feats)
        ml_proj = ml_raw["total"] or f_proj

        # Ensemble
        ens_proj  = ml_proj * 0.55 + f_proj * 0.45
        ens_line  = nearest_line(ens_proj)
        ens_rec   = "OVER" if ens_proj >= ens_line else "UNDER"
        f_correct = (f_rec == "OVER" and actual_total > f_line) or \
                    (f_rec == "UNDER" and actual_total < f_line) or \
                    (actual_total == f_line)
        e_correct = (ens_rec == "OVER" and actual_total > ens_line) or \
                    (ens_rec == "UNDER" and actual_total < ens_line) or \
                    (actual_total == ens_line)

        # Confidence level
        ens_diff = abs(ens_proj - ens_line)
        conf = 3 if ens_diff > 1.5 else 2 if ens_diff > 0.6 else 1

        results.append({
            "date":         gdate,
            "away":         g["away_name"].split()[-1],
            "home":         g["home_name"].split()[-1],
            "actual":       actual_total,
            "f_proj":       round(f_proj, 2),
            "f_line":       f_line,
            "f_rec":        f_rec,
            "f_correct":    f_correct,
            "ens_proj":     round(ens_proj, 2),
            "ens_line":     ens_line,
            "ens_rec":      ens_rec,
            "ens_correct":  e_correct,
            "conf":         conf,
            "error":        round(ens_proj - actual_total, 2),
            "h2h_n":        len(h2h),
        })
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(test_games)} juegos procesados...")

    df_r = pd.DataFrame(results)
    print(f"\n  Backtest completado: {len(df_r)} juegos evaluados\n")
    return df_r, ml_2025

# ── Stats summary ─────────────────────────────────────────────────────────────
def print_summary(df_r):
    n      = len(df_r)
    f_acc  = df_r["f_correct"].mean()
    e_acc  = df_r["ens_correct"].mean()
    mae    = df_r["error"].abs().mean()
    push   = (df_r["actual"] == df_r["ens_line"]).sum()

    # P&L at -110 (bet $110 to win $100)
    pnl = df_r["ens_correct"].map({True: 100.0, False: -110.0}).sum()
    roi = pnl / (n * 110) * 100

    print("="*60)
    print(f"  BACKTEST RESULTS — {n} games (2026-04-15 → 2026-06-04)")
    print("="*60)
    print(f"  Formula accuracy:   {f_acc*100:.1f}%  ({int(df_r['f_correct'].sum())}/{n})")
    print(f"  Ensemble accuracy:  {e_acc*100:.1f}%  ({int(df_r['ens_correct'].sum())}/{n})")
    print(f"  Pushes (tie):       {push}")
    print(f"  MAE (total runs):   {mae:.2f}")
    print(f"  P&L ($110/game):    ${pnl:+.0f}  ROI {roi:+.1f}%")
    print()

    # By confidence
    print("  By confidence level:")
    for c in [1, 2, 3]:
        sub = df_r[df_r["conf"]==c]
        if len(sub) == 0: continue
        acc = sub["ens_correct"].mean()
        pnl_c = sub["ens_correct"].map({True:100., False:-110.}).sum()
        print(f"    Conf {c}: {len(sub):3d} games  "
              f"acc={acc*100:.1f}%  P&L=${pnl_c:+.0f}")

    print()
    # By pick type
    print("  By pick (ensemble):")
    for rec in ["OVER", "UNDER"]:
        sub = df_r[df_r["ens_rec"]==rec]
        if len(sub) == 0: continue
        acc = sub["ens_correct"].mean()
        print(f"    {rec}:  {len(sub):3d} games  acc={acc*100:.1f}%")

    print()
    # By month
    df_r["month"] = df_r["date"].dt.strftime("%b")
    print("  By month:")
    for mo, sub in df_r.groupby("month"):
        acc = sub["ens_correct"].mean()
        print(f"    {mo}: {len(sub):3d} games  acc={acc*100:.1f}%")

    return {"n":n,"f_acc":f_acc,"e_acc":e_acc,"mae":mae,"pnl":pnl,"roi":roi}

# ── Visualization ─────────────────────────────────────────────────────────────
def plot_backtest(df_r, stats):
    plt.rcParams.update({
        "figure.facecolor":"#ffffff","axes.facecolor":"#fafafa",
        "axes.edgecolor":"#d1d5db","text.color":"#111",
        "axes.labelcolor":"#444","xtick.color":"#666",
        "ytick.color":"#666","grid.color":"#e5e7eb","font.size":9,
    })

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35,
                            top=0.93, bottom=0.04, left=0.07, right=0.97)

    df_sorted = df_r.sort_values("date").copy()
    df_sorted["cum_correct"] = df_sorted["ens_correct"].cumsum()
    df_sorted["cum_total"]   = range(1, len(df_sorted)+1)
    df_sorted["rolling_acc"] = df_sorted["ens_correct"].rolling(20, min_periods=5).mean() * 100
    df_sorted["cum_pnl"]     = df_sorted["ens_correct"].map({True:100.,False:-110.}).cumsum()

    # ── 1. Rolling accuracy ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_sorted["date"], df_sorted["rolling_acc"],
             color="#111", lw=2, label="Rolling accuracy (20 games)")
    ax1.axhline(50, color="#dc2626", lw=1.2, linestyle="--", alpha=0.7, label="50% (break-even)")
    ax1.axhline(52.4, color="#166534", lw=1, linestyle=":", alpha=0.6, label="52.4% (profitable at -110)")
    ax1.fill_between(df_sorted["date"], df_sorted["rolling_acc"], 50,
                     where=df_sorted["rolling_acc"] >= 50,
                     alpha=0.12, color="#166534", label="Above break-even")
    ax1.fill_between(df_sorted["date"], df_sorted["rolling_acc"], 50,
                     where=df_sorted["rolling_acc"] < 50,
                     alpha=0.12, color="#dc2626")
    ax1.set_ylabel("Accuracy %")
    ax1.set_ylim(30, 75)
    ax1.set_title(
        f"Rolling Accuracy — {stats['n']} games  |  "
        f"Overall {stats['e_acc']*100:.1f}%  |  "
        f"MAE {stats['mae']:.2f} runs  |  "
        f"P&L ${stats['pnl']:+.0f}  ROI {stats['roi']:+.1f}%",
        fontsize=12, pad=10, loc="left", fontweight="bold"
    )
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.5)
    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 2. Cumulative P&L ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    colors_pnl = ["#166534" if v >= 0 else "#dc2626" for v in df_sorted["cum_pnl"]]
    ax2.plot(df_sorted["date"], df_sorted["cum_pnl"], color="#111", lw=2)
    ax2.fill_between(df_sorted["date"], df_sorted["cum_pnl"], 0,
                     where=df_sorted["cum_pnl"] >= 0, alpha=0.15, color="#166534")
    ax2.fill_between(df_sorted["date"], df_sorted["cum_pnl"], 0,
                     where=df_sorted["cum_pnl"] < 0, alpha=0.15, color="#dc2626")
    ax2.axhline(0, color="#444", lw=1)
    ax2.set_title("P&L acumulado ($110/juego)", fontsize=11, pad=8, loc="left")
    ax2.set_ylabel("USD")
    ax2.grid(alpha=0.4)
    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))

    # ── 3. Accuracy by confidence ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    conf_data = []
    for c in [1, 2, 3]:
        sub = df_r[df_r["conf"]==c]
        if len(sub): conf_data.append((f"Conf {c}\n({len(sub)} games)", sub["ens_correct"].mean()*100))
    if conf_data:
        labels_c, accs_c = zip(*conf_data)
        bc = ["#166534" if a >= 52.4 else "#dc2626" for a in accs_c]
        bars = ax3.bar(range(len(labels_c)), accs_c, color=bc, alpha=0.85, edgecolor="#fff", width=0.5)
        ax3.axhline(52.4, color="#666", lw=1.2, linestyle="--")
        ax3.axhline(50,   color="#aaa", lw=1,   linestyle=":")
        for xi, a in enumerate(accs_c):
            ax3.text(xi, a+0.5, f"{a:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax3.set_xticks(range(len(labels_c))); ax3.set_xticklabels(labels_c)
        ax3.set_ylim(0, 80); ax3.set_ylabel("Accuracy %")
        ax3.set_title("Accuracy por nivel de confianza", fontsize=11, pad=8, loc="left")
        ax3.grid(axis="y", alpha=0.4)

    # ── 4. Error distribution ────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    errors = df_r["error"].values
    ax4.hist(errors, bins=30, color="#374151", edgecolor="#fff", alpha=0.85)
    ax4.axvline(0,                   color="#dc2626", lw=2,   linestyle="-",  label="Perfect")
    ax4.axvline(errors.mean(),       color="#f59e0b", lw=1.5, linestyle="--", label=f"Mean {errors.mean():+.2f}")
    ax4.axvline(np.median(errors),   color="#6366f1", lw=1.5, linestyle=":",  label=f"Median {np.median(errors):+.2f}")
    ax4.set_xlabel("Error (proyectado − real)"); ax4.set_ylabel("# juegos")
    ax4.set_title(f"Distribución del error  |  MAE={abs(errors).mean():.2f}  SD={errors.std():.2f}",
                  fontsize=11, pad=8, loc="left")
    ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.4)

    # ── 5. Actual vs Predicted scatter ────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    correct_mask = df_r["ens_correct"].values
    ax5.scatter(df_r.loc[correct_mask,  "actual"], df_r.loc[correct_mask,  "ens_proj"],
                color="#166534", alpha=0.55, s=18, label="Correct")
    ax5.scatter(df_r.loc[~correct_mask, "actual"], df_r.loc[~correct_mask, "ens_proj"],
                color="#dc2626", alpha=0.55, s=18, label="Wrong")
    lims = [0, max(df_r["actual"].max(), df_r["ens_proj"].max()) + 1]
    ax5.plot(lims, lims, color="#aaa", lw=1, linestyle="--")
    ax5.set_xlabel("Carreras reales"); ax5.set_ylabel("Carreras proyectadas")
    ax5.set_title("Real vs Proyectado (verde = pick correcto)", fontsize=11, pad=8, loc="left")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # ── 6. Accuracy por pick (OVER vs UNDER) ─────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 0])
    pick_acc = {}
    for rec in ["OVER","UNDER"]:
        sub = df_r[df_r["ens_rec"]==rec]
        if len(sub): pick_acc[rec] = (sub["ens_correct"].mean()*100, len(sub))
    if pick_acc:
        labs = [f"{k}\n({v[1]} games)" for k,v in pick_acc.items()]
        vals = [v[0] for v in pick_acc.values()]
        bcs  = ["#166534" if a>=52.4 else "#dc2626" for a in vals]
        bars = ax6.bar(range(len(labs)), vals, color=bcs, alpha=0.85, edgecolor="#fff", width=0.4)
        ax6.axhline(52.4, color="#666", lw=1.2, linestyle="--")
        ax6.axhline(50,   color="#aaa", lw=1,   linestyle=":")
        for xi, a in enumerate(vals):
            ax6.text(xi, a+0.5, f"{a:.1f}%", ha="center", fontsize=11, fontweight="bold")
        ax6.set_xticks(range(len(labs))); ax6.set_xticklabels(labs, fontsize=10)
        ax6.set_ylim(0, 80); ax6.set_ylabel("Accuracy %")
        ax6.set_title("Accuracy OVER vs UNDER", fontsize=11, pad=8, loc="left")
        ax6.grid(axis="y", alpha=0.4)

    # ── 7. Resultados recientes (últimos 20 juegos) ───────────────────────────
    ax7 = fig.add_subplot(gs[3, 1])
    recent = df_sorted.tail(20).copy().reset_index(drop=True)
    for i, row in recent.iterrows():
        c = "#166534" if row["ens_correct"] else "#dc2626"
        ax7.barh(i, 1, color=c, alpha=0.8, edgecolor="#fff")
        label = f"{row['away']}@{row['home']}  {row['ens_rec']} {row['ens_line']}  real={int(row['actual'])}"
        ax7.text(1.05, i, label, va="center", fontsize=8, color="#111")
    ax7.set_xlim(0, 6)
    ax7.set_yticks(range(len(recent)))
    ax7.set_yticklabels([r["date"].strftime("%b %d") for _,r in recent.iterrows()], fontsize=8)
    ax7.set_title("Últimos 20 juegos — verde=correcto, rojo=incorrecto",
                  fontsize=11, pad=8, loc="left")
    ax7.axis("off")
    for i, (_, row) in enumerate(recent.iterrows()):
        c = "#166534" if row["ens_correct"] else "#dc2626"
        ax7.barh(i, 0.4, color=c, alpha=0.85)
        ax7.text(0.5, i,
                 f"{row['date'].strftime('%b %d')}  "
                 f"{row['away']}@{row['home']}  "
                 f"{row['ens_rec']} {row['ens_line']}  "
                 f"real={int(row['actual'])}",
                 va="center", fontsize=8, color="#111")
    ax7.set_xlim(0, 8)
    ax7.set_yticks([]); ax7.set_xticks([])
    ax7.set_title("Ultimos 20 juegos — verde=correcto  rojo=incorrecto",
                  fontsize=11, pad=8, loc="left")

    fig.suptitle(
        f"MLB OVER/UNDER — Backtest Walk-Forward  |  2026-04-15 → 2026-06-04\n"
        f"Modelo entrenado SOLO con 2025. Features calculados rolling (sin data leakage).",
        fontsize=13, fontweight="bold", y=0.975
    )

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_backtest.png"
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor="#ffffff")
    print(f"\nImagen guardada: {out}")
    return out

if __name__ == "__main__":
    df_results, ml = run_backtest()
    stats = print_summary(df_results)
    plot_backtest(df_results, stats)
