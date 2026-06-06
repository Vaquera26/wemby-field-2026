#!/usr/bin/env python3
"""
MLB Picks Hoy — OVER o UNDER definitivo para cada partido.
Usa datos reales de toda la temporada 2026 (926 juegos).
"""

import urllib.request, json, time, sys
import importlib

def ensure(pkg, imp=None):
    try:
        return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

pd  = ensure("pandas")
np  = ensure("numpy")
ensure("matplotlib")
ensure("seaborn")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd, numpy as np

TODAY        = "2026-06-04"
SEASON_START = "2026-03-19"
LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]

# ── helpers ───────────────────────────────────────────────────────────────────
def fetch(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except:
            time.sleep(1)
    return {}

# ── datos ─────────────────────────────────────────────────────────────────────
def get_all_games():
    print("📦 Cargando todos los juegos de la temporada 2026...")
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&startDate={SEASON_START}&endDate={TODAY}"
           f"&hydrate=linescore,team&gameType=R")
    data = fetch(url)
    rows = []
    for date_obj in data.get("dates", []):
        for g in date_obj.get("games", []):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"):
                continue
            ls    = g.get("linescore",{}).get("teams",{})
            ar, hr = ls.get("away",{}).get("runs",0) or 0, ls.get("home",{}).get("runs",0) or 0
            rows.append({
                "date":      date_obj["date"],
                "away_id":   g["teams"]["away"]["team"]["id"],
                "home_id":   g["teams"]["home"]["team"]["id"],
                "away_name": g["teams"]["away"]["team"]["name"],
                "home_name": g["teams"]["home"]["team"]["name"],
                "away_runs": ar, "home_runs": hr,
                "total":     ar + hr,
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    print(f"✅ {len(df)} juegos cargados.")
    return df

def get_todays_games():
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={TODAY}&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    data = fetch(url)
    return [g for d in data.get("dates",[]) for g in d.get("games",[])]

def team_stats(df, tid):
    away = df[df["away_id"]==tid]; home = df[df["home_id"]==tid]
    scored  = list(away["away_runs"]) + list(home["home_runs"])
    allowed = list(away["home_runs"]) + list(home["away_runs"])
    totals  = list(away["total"]) + list(home["total"])
    return {
        "scored":  scored,  "allowed": allowed, "totals": totals,
        "avg_s":   np.mean(scored)  if scored  else 0,
        "avg_a":   np.mean(allowed) if allowed else 0,
        "avg_t":   np.mean(totals)  if totals  else 0,
        "last10_s":  np.mean(scored[-10:])  if scored  else 0,
        "last10_a":  np.mean(allowed[-10:]) if allowed else 0,
        "last10_t":  np.mean(totals[-10:])  if totals  else 0,
        "over_8.5":  np.mean([t>8.5  for t in totals]) if totals else 0.5,
        "over_9.5":  np.mean([t>9.5  for t in totals]) if totals else 0.5,
        "games":     len(totals),
    }

def project(df, away_id, home_id):
    a = team_stats(df, away_id)
    h = team_stats(df, home_id)
    # Proj carreras visitante = 50% avg_scored temporada + 30% last10_scored + 20% avg_allowed_rival
    pa = a["avg_s"]*0.5 + a["last10_s"]*0.3 + h["avg_a"]*0.2
    ph = h["avg_s"]*0.5 + h["last10_s"]*0.3 + a["avg_a"]*0.2
    proj = pa + ph
    # Ajuste head-to-head
    h2h = df[((df["away_id"]==away_id)&(df["home_id"]==home_id))|
              ((df["away_id"]==home_id)&(df["home_id"]==away_id))]
    if len(h2h) >= 2:
        proj = proj*0.80 + h2h["total"].mean()*0.20
    # Línea de mercado más cercana
    line = min(LINES, key=lambda x: abs(x-proj))
    return proj, line, pa, ph, a, h, h2h

# ── gráfica ───────────────────────────────────────────────────────────────────
def make_graphic(picks, df):
    plt.rcParams.update({
        "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
        "axes.edgecolor":"#30363d","text.color":"#e6edf3",
        "axes.labelcolor":"#e6edf3","xtick.color":"#8b949e",
        "ytick.color":"#8b949e","grid.color":"#21262d","grid.alpha":"0.4",
        "font.family":"monospace",
    })

    fig = plt.figure(figsize=(20, 26))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                           top=0.94, bottom=0.04, left=0.06, right=0.97)

    # ── PANEL 1: Proyección vs Línea ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    labels   = [p["label"]      for p in picks]
    projs    = [p["proj"]       for p in picks]
    lines    = [p["line"]       for p in picks]
    recs     = [p["rec"]        for p in picks]
    confs    = [p["conf_score"] for p in picks]

    col_map = {"OVER":"#3fb950","UNDER":"#ff7b72"}
    bar_colors = [col_map[r] for r in recs]

    x = np.arange(len(labels))
    bars = ax1.bar(x, projs, 0.55, color=bar_colors, alpha=0.92, edgecolor="#0d1117", zorder=3)
    ax1.scatter(x, lines, marker="D", s=90, color="white", zorder=5, label="Línea O/U")

    for xi, (proj, line, rec, conf) in enumerate(zip(projs, lines, recs, confs)):
        # valor encima de barra
        ax1.text(xi, proj+0.18, f"{proj:.1f}", ha="center", va="bottom",
                 fontsize=9, color="white", fontweight="bold")
        # línea encima/debajo
        ax1.text(xi, line+0.18 if line>proj else line-0.45, f"◆{line}",
                 ha="center", va="bottom", fontsize=7.5, color="#8b949e")
        # recomendación
        emoji = "⬆️" if rec=="OVER" else "⬇️"
        ax1.text(xi, -0.8, f"{rec} {emoji}", ha="center", va="top",
                 fontsize=9, color=col_map[rec], fontweight="bold")
        # confianza
        stars = "★"*conf + "☆"*(3-conf)
        ax1.text(xi, -1.5, stars, ha="center", fontsize=9,
                 color="#ffa657" if conf==3 else "#8b949e")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9, rotation=18, ha="right")
    ax1.set_ylim(-2.5, max(projs)+2.5)
    ax1.set_title("Proyección de Carreras vs Línea O/U — Juegos de Hoy", fontsize=13, pad=12)
    ax1.set_ylabel("Carreras proyectadas")
    ax1.axhline(0, color="#30363d", lw=0.8)
    ax1.grid(axis="y", zorder=0)
    over_p  = mpatches.Patch(color="#3fb950", label="OVER")
    under_p = mpatches.Patch(color="#ff7b72", label="UNDER")
    line_p  = plt.Line2D([0],[0], marker="D", color="w", label="Línea O/U",
                          markerfacecolor="white", markersize=7, linestyle="None")
    ax1.legend(handles=[over_p, under_p, line_p], fontsize=9, loc="upper right")

    # ── PANEL 2: Desglose ofensivo por equipo ────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    all_teams_today = []
    for p in picks:
        all_teams_today.append((p["away_name"], p["away_stats"]["avg_s"],
                                 p["away_stats"]["last10_s"], p["away_stats"]["avg_a"]))
        all_teams_today.append((p["home_name"], p["home_stats"]["avg_s"],
                                 p["home_stats"]["last10_s"], p["home_stats"]["avg_a"]))

    tnames  = [t[0].split()[-1] for t in all_teams_today]
    avg_s   = [t[1] for t in all_teams_today]
    l10_s   = [t[2] for t in all_teams_today]
    avg_a   = [t[3] for t in all_teams_today]

    x2 = np.arange(len(tnames))
    w  = 0.26
    ax2.bar(x2-w,   avg_s, w, color="#58a6ff", alpha=0.9, label="R/juego (temporada)",  edgecolor="#0d1117")
    ax2.bar(x2,     l10_s, w, color="#3fb950", alpha=0.9, label="R/juego (últimos 10)", edgecolor="#0d1117")
    ax2.bar(x2+w,   avg_a, w, color="#ff7b72", alpha=0.7, label="RA/juego (temporada)", edgecolor="#0d1117")

    ax2.axhline(np.mean(avg_s), color="#58a6ff", lw=1, linestyle=":", alpha=0.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(tnames, fontsize=8, rotation=30, ha="right")
    ax2.set_title("Ofensiva / Carreras Permitidas — Equipos de Hoy", fontsize=13, pad=12)
    ax2.set_ylabel("Carreras por juego")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y")
    # separadores entre partidos
    for xi in range(1, len(picks)):
        ax2.axvline(xi*2-0.5, color="#30363d", lw=1, linestyle="--")

    # ── PANEL 3: Head-to-Head histórico + over rate ──────────────────────────
    ax3 = fig.add_subplot(gs[2])
    h2h_means = [p["h2h_avg"] for p in picks]
    h2h_count = [p["h2h_n"]   for p in picks]
    over85    = [(p["away_stats"]["over_8.5"]+p["home_stats"]["over_8.5"])/2*100 for p in picks]
    over95    = [(p["away_stats"]["over_9.5"]+p["home_stats"]["over_9.5"])/2*100 for p in picks]

    x3 = np.arange(len(labels))
    ax3b = ax3.twinx()

    bars_h2h = ax3.bar(x3-0.2, h2h_means, 0.35, color="#d2a8ff", alpha=0.85,
                        label="Promedio H2H (carreras)", edgecolor="#0d1117")
    ax3.bar(x3+0.15, [p["proj"] for p in picks], 0.35, color="#ffa657", alpha=0.75,
            label="Proyección hoy", edgecolor="#0d1117")

    ax3b.plot(x3, over85, "o--", color="#3fb950", lw=1.8, ms=7, label="% Over 8.5 (prom equipos)")
    ax3b.plot(x3, over95, "s--", color="#58a6ff", lw=1.8, ms=7, label="% Over 9.5 (prom equipos)")
    ax3b.axhline(50, color="white", lw=1, linestyle=":", alpha=0.4)
    ax3b.set_ylabel("% Over", color="#8b949e")
    ax3b.set_ylim(0, 100)
    ax3b.tick_params(colors="#8b949e")

    for xi, (hm, hn) in enumerate(zip(h2h_means, h2h_count)):
        if hn > 0:
            ax3.text(xi-0.2, hm+0.15, f"{hm:.1f}\n({hn}j)", ha="center",
                     fontsize=7, color="#d2a8ff")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels, fontsize=9, rotation=18, ha="right")
    ax3.set_title("Head-to-Head 2026 + % Over por Equipos", fontsize=13, pad=12)
    ax3.set_ylabel("Carreras promedio")
    ax3.grid(axis="y")

    lines_h  = [mpatches.Patch(color="#d2a8ff", label="Promedio H2H"),
                mpatches.Patch(color="#ffa657", label="Proyección hoy")]
    lines_b  = [plt.Line2D([0],[0],color="#3fb950",marker="o",lw=1.8,label="% Over 8.5"),
                plt.Line2D([0],[0],color="#58a6ff",marker="s",lw=1.8,label="% Over 9.5")]
    ax3.legend(handles=lines_h+lines_b, fontsize=8, loc="upper right")

    # ── Título global ─────────────────────────────────────────────────────────
    fig.suptitle(
        f"MLB PICKS — {TODAY}  |  OVER / UNDER por Partido\n"
        f"Basado en {len(df)} juegos de la Temporada 2026",
        fontsize=15, color="#e6edf3", fontweight="bold", y=0.975
    )

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_picks_hoy.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n🖼️  Imagen: {out}")
    return out

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    df      = get_all_games()
    todays  = get_todays_games()

    picks = []
    print("\n" + "━"*72)
    print(f"  MLB PICKS HOY — {TODAY}")
    print("━"*72)
    print(f"  {'PARTIDO':<34} {'PROJ':>6} {'LÍNEA':>6} {'DIF':>6}  {'PICK':^7}  {'★'}")
    print(f"  {'─'*34} {'─'*6} {'─'*6} {'─'*6}  {'─'*7}  {'─'*5}")

    for g in todays:
        away_id   = g["teams"]["away"]["team"]["id"]
        home_id   = g["teams"]["home"]["team"]["id"]
        away_name = g["teams"]["away"]["team"]["name"]
        home_name = g["teams"]["home"]["team"]["name"]
        away_p    = g["teams"]["away"].get("probablePitcher",{}).get("fullName","TBD")
        home_p    = g["teams"]["home"].get("probablePitcher",{}).get("fullName","TBD")
        venue     = g.get("venue",{}).get("name","")

        proj, line, pa, ph, a_stats, h_stats, h2h = project(df, away_id, home_id)
        diff = proj - line
        rec  = "OVER" if diff >= 0 else "UNDER"

        # Confianza: 1-3 estrellas
        conf = 3 if abs(diff) > 1.5 else 2 if abs(diff) > 0.6 else 1

        label = f"{away_name.split()[-1]}@{home_name.split()[-1]}"
        stars = "★"*conf + "☆"*(3-conf)

        arrow = "⬆️ " if rec=="OVER" else "⬇️ "
        color_start = "\033[92m" if rec=="OVER" else "\033[91m"
        reset = "\033[0m"
        print(f"  {label:<34} {proj:>6.2f} {line:>6.1f} {diff:>+6.2f}  "
              f"{color_start}{rec:^7}{reset}  {stars}")

        picks.append({
            "label":      label,
            "away_name":  away_name,
            "home_name":  home_name,
            "away_p":     away_p,
            "home_p":     home_p,
            "venue":      venue,
            "proj":       proj,
            "proj_away":  pa,
            "proj_home":  ph,
            "line":       line,
            "diff":       diff,
            "rec":        rec,
            "conf_score": conf,
            "stars":      stars,
            "h2h_avg":    h2h["total"].mean() if len(h2h)>=1 else 0.0,
            "h2h_n":      len(h2h),
            "away_stats": a_stats,
            "home_stats": h_stats,
        })

    print("━"*72)

    # Resumen narrativo
    print("\n📋 DETALLE POR PARTIDO:\n")
    for p in picks:
        col = "\033[92m" if p["rec"]=="OVER" else "\033[91m"
        reset = "\033[0m"
        print(f"  {p['away_name']} @ {p['home_name']}")
        print(f"  🏟️  {p['venue']}")
        print(f"  ⚾ {p['away_p']} vs {p['home_p']}")
        print(f"  Carreras proyectadas: {p['proj_away']:.2f} (visitante) + {p['proj_home']:.2f} (local) = {p['proj']:.2f}")
        print(f"  H2H 2026: {p['h2h_n']} juegos | prom {p['h2h_avg']:.1f} carreras")
        h = p["home_stats"]; a = p["away_stats"]
        print(f"  {p['away_name'].split()[-1]}: {a['avg_s']:.2f} R/G temp | {a['last10_s']:.2f} últ10  |  RA/G {a['avg_a']:.2f}")
        print(f"  {p['home_name'].split()[-1]}: {h['avg_s']:.2f} R/G temp | {h['last10_s']:.2f} últ10  |  RA/G {h['avg_a']:.2f}")
        print(f"  {col}👉 {p['rec']} {p['line']}  {p['stars']}{reset}  (dif {p['diff']:+.2f})\n")

    print("━"*72)
    print("🔥 PICKS ORDENADOS POR CONFIANZA:\n")
    sorted_picks = sorted(picks, key=lambda x: -abs(x["diff"]))
    for i, p in enumerate(sorted_picks, 1):
        col = "\033[92m" if p["rec"]=="OVER" else "\033[91m"
        reset = "\033[0m"
        print(f"  #{i}  {col}{p['rec']} {p['line']}{reset}  {p['label']:<28}  "
              f"proj {p['proj']:.2f}  {p['stars']}")

    print("\n🎨 Generando gráfica...")
    make_graphic(picks, df)
    print(f"\n⚠️  Análisis estadístico — no es consejo financiero.")

if __name__ == "__main__":
    main()
