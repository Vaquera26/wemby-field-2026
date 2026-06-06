#!/usr/bin/env python3
"""
MLB 2026 - Análisis completo de TODOS los partidos de la temporada.
Calcula over/under histórico, tendencias, y genera gráficas visuales.
"""

import urllib.request
import json
import time
from datetime import datetime, date
import sys

TODAY = "2026-06-04"
SEASON_START = "2026-03-19"

# ── pip install silencioso si falta ──────────────────────────────────────────
def ensure(pkg, import_as=None):
    import importlib
    try:
        return importlib.import_module(import_as or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        return importlib.import_module(import_as or pkg)

pd  = ensure("pandas")
np  = ensure("numpy")
plt_mod = ensure("matplotlib")
plt = ensure("matplotlib.pyplot", "matplotlib.pyplot")
sns = ensure("seaborn")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────

def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries - 1:
                return {}
            time.sleep(1)

def get_all_games():
    """Descarga TODOS los juegos finalizados de la temporada 2026."""
    print("📦 Descargando calendario completo de la temporada 2026...")
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&startDate={SEASON_START}&endDate={TODAY}"
           f"&hydrate=linescore,team&gameType=R")
    data = fetch(url)

    rows = []
    total_dates = len(data.get("dates", []))
    for di, date_obj in enumerate(data.get("dates", [])):
        for g in date_obj.get("games", []):
            status = g.get("status", {}).get("statusCode", "")
            if status not in ("F", "FR", "FT", "FO"):
                continue
            ls = g.get("linescore", {})
            teams_ls = ls.get("teams", {})
            away_id   = g["teams"]["away"]["team"]["id"]
            home_id   = g["teams"]["home"]["team"]["id"]
            away_name = g["teams"]["away"]["team"]["name"]
            home_name = g["teams"]["home"]["team"]["name"]
            away_runs = teams_ls.get("away", {}).get("runs", 0) or 0
            home_runs = teams_ls.get("home", {}).get("runs", 0) or 0
            total     = away_runs + home_runs
            rows.append({
                "date":       date_obj["date"],
                "game_pk":    g["gamePk"],
                "away_id":    away_id,
                "home_id":    home_id,
                "away_name":  away_name,
                "home_name":  home_name,
                "away_runs":  away_runs,
                "home_runs":  home_runs,
                "total_runs": total,
            })
        if (di + 1) % 10 == 0:
            print(f"  ↳ Fechas procesadas: {di+1}/{total_dates} | Juegos: {len(rows)}", end="\r")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n✅ {len(df)} juegos finalizados cargados.")
    return df

def get_todays_games():
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={TODAY}&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    data = fetch(url)
    games = []
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            games.append(g)
    return games

# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS
# ─────────────────────────────────────────────────────────────────────────────

def build_team_stats(df):
    """Para cada equipo: carreras anotadas/permitidas por juego, over/under rate a distintas líneas."""
    teams = {}
    all_ids = set(df["away_id"]) | set(df["home_id"])

    for tid in all_ids:
        away = df[df["away_id"] == tid]
        home = df[df["home_id"] == tid]

        scored  = list(away["away_runs"]) + list(home["home_runs"])
        allowed = list(away["home_runs"]) + list(away["away_runs"])  # fixed below
        # carreras permitidas = las que anotó el rival
        allowed = list(away["home_runs"]) + list(home["away_runs"])
        totals  = list(away["total_runs"]) + list(home["total_runs"])

        name = ""
        if not away.empty:
            name = away.iloc[0]["away_name"]
        elif not home.empty:
            name = home.iloc[0]["home_name"]

        if not totals:
            continue

        teams[tid] = {
            "name":      name,
            "games":     len(totals),
            "scored":    scored,
            "allowed":   allowed,
            "totals":    totals,
            "avg_scored":  np.mean(scored),
            "avg_allowed": np.mean(allowed),
            "avg_total":   np.mean(totals),
            "std_total":   np.std(totals),
            # over rates a distintas líneas
            "over_6.5":  np.mean([1 if t > 6.5 else 0 for t in totals]),
            "over_7.5":  np.mean([1 if t > 7.5 else 0 for t in totals]),
            "over_8.5":  np.mean([1 if t > 8.5 else 0 for t in totals]),
            "over_9.5":  np.mean([1 if t > 9.5 else 0 for t in totals]),
        }
    return teams

def rolling_over_rate(df, team_id, line=8.5, window=10):
    """Rolling over rate (últimos N juegos) para un equipo."""
    sub = df[(df["away_id"] == team_id) | (df["home_id"] == team_id)].sort_values("date")
    sub = sub.copy()
    sub["over"] = (sub["total_runs"] > line).astype(int)
    sub["roll"] = sub["over"].rolling(window, min_periods=1).mean()
    return sub

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(df, team_stats, todays_games):
    sns.set_theme(style="dark", palette="deep")
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "text.color":       "#e6edf3",
        "axes.labelcolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "axes.titlecolor":  "#e6edf3",
    })

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Total runs distribution (histograma de todos los juegos) ──────────
    ax1 = fig.add_subplot(gs[0, 0])
    runs = df["total_runs"].values
    ax1.hist(runs, bins=range(0, int(max(runs))+2), color="#58a6ff", edgecolor="#0d1117", alpha=0.85)
    for line, col in [(6.5, "#ff7b72"), (7.5, "#ffa657"), (8.5, "#d2a8ff"), (9.5, "#3fb950")]:
        ax1.axvline(line, color=col, lw=1.8, linestyle="--", label=f"O/U {line}")
    ax1.axvline(np.mean(runs), color="white", lw=2, linestyle="-", label=f"Media {np.mean(runs):.2f}")
    ax1.set_title("Distribución de Carreras Totales — Temporada 2026", fontsize=12, pad=10)
    ax1.set_xlabel("Carreras totales en el juego")
    ax1.set_ylabel("# Juegos")
    ax1.legend(fontsize=7, loc="upper right")
    over_pct = {l: f"{np.mean(runs > l)*100:.1f}%" for l in [6.5, 7.5, 8.5, 9.5]}
    info = "\n".join([f"OVER {k}: {v}" for k, v in over_pct.items()])
    ax1.text(0.02, 0.97, info, transform=ax1.transAxes, va="top", fontsize=8,
             color="#8b949e", family="monospace")

    # ── 2. Over rate por equipo (línea 8.5) ──────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_teams = sorted(team_stats.items(), key=lambda x: -x[1]["over_8.5"])
    names85  = [v["name"].replace(" ", "\n") for _, v in sorted_teams]
    rates85  = [v["over_8.5"] * 100 for _, v in sorted_teams]
    colors85 = ["#3fb950" if r >= 55 else "#ffa657" if r >= 45 else "#ff7b72" for r in rates85]
    bars = ax2.bar(range(len(names85)), rates85, color=colors85, edgecolor="#0d1117", width=0.75)
    ax2.axhline(50, color="white", lw=1.2, linestyle="--", alpha=0.6, label="50% neutro")
    ax2.set_xticks(range(len(names85)))
    ax2.set_xticklabels(names85, fontsize=5, rotation=45, ha="right")
    ax2.set_title("% OVER 8.5 por Equipo — Toda la Temporada", fontsize=12, pad=10)
    ax2.set_ylabel("% de juegos OVER 8.5")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8)

    # ── 3. Heatmap: avg total runs HOME vs AWAY ──────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    # Top 15 equipos por promedio de carreras totales
    top15_ids = sorted(team_stats.keys(), key=lambda x: -team_stats[x]["avg_total"])[:15]
    top15_names = [team_stats[t]["name"] for t in top15_ids]

    matrix = np.zeros((15, 15))
    for i, tid_away in enumerate(top15_ids):
        for j, tid_home in enumerate(top15_ids):
            games = df[(df["away_id"] == tid_away) & (df["home_id"] == tid_home)]
            if not games.empty:
                matrix[i, j] = games["total_runs"].mean()

    # Máscara para celdas sin datos
    mask = matrix == 0
    short_names = [n.split()[-1] for n in top15_names]
    hm = sns.heatmap(matrix, mask=mask, annot=True, fmt=".1f", cmap="YlOrRd",
                     xticklabels=short_names, yticklabels=short_names,
                     ax=ax3, linewidths=0.5, linecolor="#0d1117",
                     cbar_kws={"shrink": 0.8, "label": "Carreras promedio"},
                     annot_kws={"size": 7})
    ax3.set_title("Promedio Carreras Totales — Top 15 Equipos Ofensivos (VISITANTE × LOCAL)", fontsize=12, pad=10)
    ax3.set_xlabel("Equipo LOCAL →")
    ax3.set_ylabel("← Equipo VISITANTE")
    ax3.tick_params(axis="x", labelsize=7, rotation=30)
    ax3.tick_params(axis="y", labelsize=7, rotation=0)

    # ── 4. Rolling over rate — equipos de hoy ────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    colors_cycle = ["#58a6ff", "#3fb950", "#ff7b72", "#ffa657", "#d2a8ff",
                    "#79c0ff", "#56d364", "#ffa198", "#ffb77c", "#c5a6f7",
                    "#a5d6ff", "#85e89d", "#ffaaa5", "#ffc56e", "#e2c5f7",
                    "#b6d9f7", "#99d1ab", "#ffc1c0", "#ffd0a0", "#ece0fa"]

    today_team_ids = set()
    for g in todays_games:
        today_team_ids.add(g["teams"]["away"]["team"]["id"])
        today_team_ids.add(g["teams"]["home"]["team"]["id"])

    for ci, tid in enumerate(today_team_ids):
        if tid not in team_stats:
            continue
        sub = rolling_over_rate(df, tid, line=8.5, window=10)
        if len(sub) < 5:
            continue
        name = team_stats[tid]["name"].split()[-1]
        ax4.plot(sub["date"], sub["roll"] * 100,
                 label=name, color=colors_cycle[ci % len(colors_cycle)],
                 lw=1.8, alpha=0.9)

    ax4.axhline(50, color="white", lw=1.2, linestyle="--", alpha=0.5)
    ax4.fill_between(pd.date_range(SEASON_START, TODAY), 50, 100, alpha=0.04, color="#3fb950")
    ax4.fill_between(pd.date_range(SEASON_START, TODAY), 0, 50, alpha=0.04, color="#ff7b72")
    ax4.set_title("Rolling OVER 8.5 Rate (últimos 10 juegos) — Equipos de Hoy", fontsize=12, pad=10)
    ax4.set_ylabel("% Over 8.5 (roll 10j)")
    ax4.set_xlabel("Fecha")
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=7, ncol=5, loc="upper left")
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 5. Proyecciones para los juegos de hoy ───────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])

    game_labels = []
    proj_totals = []
    estimated_lines = []
    colors_proj = []

    common_lines = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    for g in todays_games:
        away_id = g["teams"]["away"]["team"]["id"]
        home_id = g["teams"]["home"]["team"]["id"]
        away_s  = team_stats.get(away_id)
        home_s  = team_stats.get(home_id)
        if not away_s or not home_s:
            continue

        # Head-to-head histórico
        h2h = df[((df["away_id"] == away_id) & (df["home_id"] == home_id)) |
                 ((df["away_id"] == home_id) & (df["home_id"] == away_id))]

        # Proyección: 60% season avg, 30% últimos 10, 10% h2h
        away_last10 = np.mean(away_s["scored"][-10:]) if away_s["scored"] else away_s["avg_scored"]
        home_last10 = np.mean(home_s["scored"][-10:]) if home_s["scored"] else home_s["avg_scored"]
        away_allowed_last10 = np.mean(away_s["allowed"][-10:]) if away_s["allowed"] else away_s["avg_allowed"]
        home_allowed_last10 = np.mean(home_s["allowed"][-10:]) if home_s["allowed"] else home_s["avg_allowed"]

        proj_away = (away_s["avg_scored"] * 0.4 + away_last10 * 0.4 +
                     home_allowed_last10 * 0.2)
        proj_home = (home_s["avg_scored"] * 0.4 + home_last10 * 0.4 +
                     away_allowed_last10 * 0.2)
        proj = proj_away + proj_home

        if not h2h.empty and len(h2h) >= 2:
            proj = proj * 0.85 + h2h["total_runs"].mean() * 0.15

        est_line = min(common_lines, key=lambda x: abs(x - proj))

        away_short = g["teams"]["away"]["team"]["name"].split()[-1]
        home_short = g["teams"]["home"]["team"]["name"].split()[-1]
        label = f"{away_short}@{home_short}"
        game_labels.append(label)
        proj_totals.append(proj)
        estimated_lines.append(est_line)

        if proj > est_line + 0.5:
            colors_proj.append("#3fb950")  # verde = over
        elif proj < est_line - 0.5:
            colors_proj.append("#ff7b72")  # rojo = under
        else:
            colors_proj.append("#ffa657")  # naranja = push

    x = np.arange(len(game_labels))
    w = 0.35
    bars1 = ax5.bar(x - w/2, proj_totals, w, color=colors_proj, alpha=0.9,
                    edgecolor="#0d1117", label="Proyección carreras")
    bars2 = ax5.bar(x + w/2, estimated_lines, w, color="#8b949e", alpha=0.6,
                    edgecolor="#0d1117", label="Línea O/U estimada")

    for xi, (proj, line) in enumerate(zip(proj_totals, estimated_lines)):
        ax5.text(xi - w/2, proj + 0.15, f"{proj:.1f}", ha="center", va="bottom",
                 fontsize=8, color="white", fontweight="bold")
        ax5.text(xi + w/2, line + 0.15, f"{line}", ha="center", va="bottom",
                 fontsize=8, color="#8b949e")

    over_patch  = mpatches.Patch(color="#3fb950", label="OVER")
    push_patch  = mpatches.Patch(color="#ffa657", label="PUSH")
    under_patch = mpatches.Patch(color="#ff7b72", label="UNDER")
    line_patch  = mpatches.Patch(color="#8b949e", label="Línea estimada")
    ax5.legend(handles=[over_patch, push_patch, under_patch, line_patch],
               fontsize=9, loc="upper right")

    ax5.set_xticks(x)
    ax5.set_xticklabels(game_labels, fontsize=9, rotation=15)
    ax5.set_title(f"Proyección de Carreras vs Línea O/U — Juegos de Hoy ({TODAY})", fontsize=12, pad=10)
    ax5.set_ylabel("Carreras esperadas")
    ax5.set_ylim(0, max(max(proj_totals, default=10), 12) + 2)
    ax5.grid(axis="y", alpha=0.3)

    # ── Título global ──────────────────────────────────────────────────────
    fig.suptitle(
        f"MLB 2026 — Análisis Completo de la Temporada  ·  {TODAY}\n"
        f"Basado en {len(df)} juegos  ·  {len(team_stats)} equipos",
        fontsize=15, color="#e6edf3", y=0.995, fontweight="bold"
    )

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_analysis_2026.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n🖼️  Gráfica guardada en: {out}")
    return out

# ─────────────────────────────────────────────────────────────────────────────
# TABLA PANDAS RESUMEN
# ─────────────────────────────────────────────────────────────────────────────

def print_team_table(team_stats, todays_games):
    rows = []
    for tid, s in team_stats.items():
        rows.append({
            "Equipo":        s["name"],
            "JJ":            s["games"],
            "R/G ✦":         round(s["avg_scored"], 2),
            "RA/G":          round(s["avg_allowed"], 2),
            "Total/G":       round(s["avg_total"], 2),
            "σ Total":       round(s["std_total"], 2),
            "OVER 6.5%":     f"{s['over_6.5']*100:.0f}%",
            "OVER 7.5%":     f"{s['over_7.5']*100:.0f}%",
            "OVER 8.5%":     f"{s['over_8.5']*100:.0f}%",
            "OVER 9.5%":     f"{s['over_9.5']*100:.0f}%",
        })
    df_table = pd.DataFrame(rows).sort_values("Total/G", ascending=False).reset_index(drop=True)
    df_table.index += 1

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 35)
    print("\n" + "="*90)
    print(f"  TABLA COMPLETA — TODOS LOS EQUIPOS MLB 2026")
    print("="*90)
    print(df_table.to_string())
    return df_table

def print_today_detailed(df, team_stats, todays_games):
    common_lines = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

    print("\n" + "="*90)
    print(f"  PROYECCIONES HOY {TODAY} — ANÁLISIS CON TODOS LOS PARTIDOS DE LA TEMPORADA")
    print("="*90)
    print(f"  {'PARTIDO':<35} {'H2H':>5} {'Proj':>6} {'Línea':>6} {'Dif':>5}  {'REC':^8}  {'Conf':^7}")
    print(f"  {'─'*35} {'─'*5} {'─'*6} {'─'*6} {'─'*5}  {'─'*8}  {'─'*7}")

    picks = []
    for g in todays_games:
        away_id   = g["teams"]["away"]["team"]["id"]
        home_id   = g["teams"]["home"]["team"]["id"]
        away_name = g["teams"]["away"]["team"]["name"]
        home_name = g["teams"]["home"]["team"]["name"]
        away_s = team_stats.get(away_id)
        home_s = team_stats.get(home_id)
        if not away_s or not home_s:
            continue

        h2h = df[((df["away_id"] == away_id) & (df["home_id"] == home_id)) |
                 ((df["away_id"] == home_id) & (df["home_id"] == away_id))]
        h2h_count = len(h2h)
        h2h_avg   = h2h["total_runs"].mean() if h2h_count > 0 else None

        away_last10 = np.mean(away_s["scored"][-10:]) if away_s["scored"] else away_s["avg_scored"]
        home_last10 = np.mean(home_s["scored"][-10:]) if home_s["scored"] else home_s["avg_scored"]
        away_allowed_last10 = np.mean(away_s["allowed"][-10:]) if away_s["allowed"] else away_s["avg_allowed"]
        home_allowed_last10 = np.mean(home_s["allowed"][-10:]) if home_s["allowed"] else home_s["avg_allowed"]

        proj_away = (away_s["avg_scored"] * 0.4 + away_last10 * 0.4 + home_allowed_last10 * 0.2)
        proj_home = (home_s["avg_scored"] * 0.4 + home_last10 * 0.4 + away_allowed_last10 * 0.2)
        proj = proj_away + proj_home
        if h2h_count >= 2:
            proj = proj * 0.85 + h2h_avg * 0.15

        est_line = min(common_lines, key=lambda x: abs(x - proj))
        diff = proj - est_line

        if abs(diff) > 1.5:
            conf = "ALTA  🔥"
        elif abs(diff) > 0.7:
            conf = "MEDIA ⚡"
        else:
            conf = "BAJA  💤"

        rec = "OVER ⬆️ " if diff > 0.4 else "UNDER ⬇️" if diff < -0.4 else "PUSH ↔️ "

        matchup = f"{away_name.split()[-1]} @ {home_name.split()[-1]}"
        h2h_str = str(h2h_count) if h2h_count else " —"
        print(f"  {matchup:<35} {h2h_str:>5} {proj:>6.2f} {est_line:>6} {diff:>+5.2f}  {rec:^8}  {conf}")
        picks.append((matchup, proj, est_line, diff, conf))

    # Top picks
    top_over  = sorted([p for p in picks if p[3] > 0.4], key=lambda x: -x[3])
    top_under = sorted([p for p in picks if p[3] < -0.4], key=lambda x: x[3])

    print(f"\n  🔥 TOP PICKS OVER:")
    if top_over:
        for p in top_over:
            print(f"     OVER {p[2]} — {p[0]}  (proj {p[1]:.2f}, dif {p[3]:+.2f})  {p[4]}")
    else:
        print("     Ninguno con confianza alta hoy.")

    print(f"\n  ❄️  TOP PICKS UNDER:")
    if top_under:
        for p in top_under:
            print(f"     UNDER {p[2]} — {p[0]}  (proj {p[1]:.2f}, dif {p[3]:+.2f})  {p[4]}")
    else:
        print("     Ninguno con confianza alta hoy.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_all = get_all_games()

    print("📊 Calculando estadísticas por equipo...")
    team_stats = build_team_stats(df_all)

    print("📅 Obteniendo partidos de hoy...")
    todays = get_todays_games()
    print(f"  → {len(todays)} partidos hoy")

    df_table = print_team_table(team_stats, todays)
    print_today_detailed(df_all, team_stats, todays)

    print("\n🎨 Generando gráficas...")
    out_path = plot_all(df_all, team_stats, todays)

    # Guardar CSV también
    csv_path = "/Users/vaquera/Documents/NBA-SPURS/mlb_team_stats_2026.csv"
    df_table.to_csv(csv_path)
    print(f"📋 CSV guardado en: {csv_path}")
    print(f"\n✅ Listo. Abre la imagen: {out_path}")
