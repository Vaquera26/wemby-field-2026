#!/usr/bin/env python3
"""
MLB — Compara rendimiento ESTA SEMANA vs temporada completa.
Detecta equipos calientes/fríos y recalcula picks O/U.
"""

import urllib.request, json, time, sys, importlib

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

pd = ensure("pandas"); np = ensure("numpy"); ensure("matplotlib"); ensure("seaborn")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns, pandas as pd, numpy as np

TODAY        = "2026-06-04"
WEEK_START   = "2026-05-28"   # últimos 7 días
SEASON_START = "2026-03-19"
LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]

def fetch(url):
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except: time.sleep(1)
    return {}

def get_games(start, end):
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&startDate={start}&endDate={end}"
           f"&hydrate=linescore,team&gameType=R")
    data = fetch(url)
    rows = []
    for d in data.get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls = g.get("linescore",{}).get("teams",{})
            ar = ls.get("away",{}).get("runs",0) or 0
            hr = ls.get("home",{}).get("runs",0) or 0
            rows.append({
                "date":      d["date"],
                "away_id":   g["teams"]["away"]["team"]["id"],
                "home_id":   g["teams"]["home"]["team"]["id"],
                "away_name": g["teams"]["away"]["team"]["name"],
                "home_name": g["teams"]["home"]["team"]["name"],
                "away_runs": ar, "home_runs": hr, "total": ar+hr,
            })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["date","away_id","home_id","away_name","home_name","away_runs","home_runs","total"])
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    return df

def get_todays_games():
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={TODAY}&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    return [g for d in fetch(url).get("dates",[]) for g in d.get("games",[])]

def team_slice(df, tid):
    """Stats de un equipo dentro de un DataFrame (temporada o semana)."""
    away = df[df["away_id"]==tid]; home = df[df["home_id"]==tid]
    scored  = list(away["away_runs"]) + list(home["home_runs"])
    allowed = list(away["home_runs"]) + list(home["away_runs"])
    totals  = list(away["total"])     + list(home["total"])
    return {
        "scored": scored, "allowed": allowed, "totals": totals,
        "avg_s":  np.mean(scored)  if scored  else None,
        "avg_a":  np.mean(allowed) if allowed else None,
        "avg_t":  np.mean(totals)  if totals  else None,
        "games":  len(totals),
    }

def project_weighted(df_season, df_week, away_id, home_id):
    """
    Proyección con 3 niveles de peso:
      - Esta semana (si hay >= 2 juegos): 50%
      - Últimos 10 juegos del season: 30%
      - Promedio temporada completa: 20%
    Si no hay datos de semana, cae a: últimos10=60%, temporada=40%
    """
    a_s = team_slice(df_season, away_id)
    h_s = team_slice(df_season, home_id)
    a_w = team_slice(df_week,   away_id)
    h_w = team_slice(df_week,   home_id)

    # últimos 10 de temporada
    a_all = df_season[(df_season["away_id"]==away_id)|(df_season["home_id"]==away_id)].sort_values("date")
    h_all = df_season[(df_season["away_id"]==home_id)|(df_season["home_id"]==home_id)].sort_values("date")

    def last10_scored(df_t, tid):
        rows = []
        for _, row in df_t.iterrows():
            if row["away_id"]==tid: rows.append(row["away_runs"])
            elif row["home_id"]==tid: rows.append(row["home_runs"])
        return np.mean(rows[-10:]) if rows else None

    def last10_allowed(df_t, tid):
        rows = []
        for _, row in df_t.iterrows():
            if row["away_id"]==tid: rows.append(row["home_runs"])
            elif row["home_id"]==tid: rows.append(row["away_runs"])
        return np.mean(rows[-10:]) if rows else None

    a_l10s = last10_scored(a_all, away_id)
    h_l10s = last10_scored(h_all, home_id)
    a_l10a = last10_allowed(a_all, away_id)
    h_l10a = last10_allowed(h_all, home_id)

    has_away_week = a_w["games"] >= 2
    has_home_week = h_w["games"] >= 2

    # Proyección carreras visitante
    if has_away_week and a_w["avg_s"] is not None:
        pa = (a_w["avg_s"]  * 0.50 +
              (a_l10s or a_s["avg_s"] or 0) * 0.30 +
              (a_s["avg_s"] or 0) * 0.20)
        pa_label = "season+semana"
    else:
        pa = ((a_l10s or a_s["avg_s"] or 0) * 0.60 +
              (a_s["avg_s"] or 0) * 0.40)
        pa_label = "season+últ10"

    # vs defensa local
    if has_home_week and h_w["avg_a"] is not None:
        pa_adj = (h_w["avg_a"]  * 0.50 +
                  (h_l10a or h_s["avg_a"] or 0) * 0.30 +
                  (h_s["avg_a"] or 0) * 0.20)
    else:
        pa_adj = ((h_l10a or h_s["avg_a"] or 0) * 0.60 + (h_s["avg_a"] or 0) * 0.40)
    pa = (pa + pa_adj) / 2

    # Proyección carreras local
    if has_home_week and h_w["avg_s"] is not None:
        ph = (h_w["avg_s"]  * 0.50 +
              (h_l10s or h_s["avg_s"] or 0) * 0.30 +
              (h_s["avg_s"] or 0) * 0.20)
    else:
        ph = ((h_l10s or h_s["avg_s"] or 0) * 0.60 + (h_s["avg_s"] or 0) * 0.40)

    if has_away_week and a_w["avg_a"] is not None:
        ph_adj = (a_w["avg_a"]  * 0.50 +
                  (a_l10a or a_s["avg_a"] or 0) * 0.30 +
                  (a_s["avg_a"] or 0) * 0.20)
    else:
        ph_adj = ((a_l10a or a_s["avg_a"] or 0) * 0.60 + (a_s["avg_a"] or 0) * 0.40)
    ph = (ph + ph_adj) / 2

    proj = pa + ph

    # H2H
    h2h = df_season[((df_season["away_id"]==away_id)&(df_season["home_id"]==home_id))|
                    ((df_season["away_id"]==home_id)&(df_season["home_id"]==away_id))]
    if len(h2h) >= 2:
        proj = proj * 0.80 + h2h["total"].mean() * 0.20

    line = min(LINES, key=lambda x: abs(x-proj))

    return {
        "proj": proj, "pa": pa, "ph": ph, "line": line,
        "diff": proj - line,
        "a_week": a_w, "h_week": h_w,
        "a_season": a_s, "h_season": h_s,
        "a_l10s": a_l10s, "h_l10s": h_l10s,
        "has_away_week": has_away_week, "has_home_week": has_home_week,
        "h2h_avg": h2h["total"].mean() if len(h2h)>=1 else None,
        "h2h_n": len(h2h),
    }

# ── gráfica ───────────────────────────────────────────────────────────────────
def make_graphic(picks, df_season, df_week):
    plt.rcParams.update({
        "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
        "axes.edgecolor":"#30363d","text.color":"#e6edf3",
        "axes.labelcolor":"#e6edf3","xtick.color":"#8b949e",
        "ytick.color":"#8b949e","grid.color":"#21262d",
    })

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55, top=0.94, bottom=0.04, left=0.06, right=0.97)

    labels       = [p["label"]      for p in picks]
    proj_new     = [p["proj_new"]   for p in picks]
    proj_old     = [p["proj_old"]   for p in picks]
    lines_new    = [p["line_new"]   for p in picks]
    recs_new     = [p["rec_new"]    for p in picks]
    recs_old     = [p["rec_old"]    for p in picks]
    changed      = [p["changed"]    for p in picks]
    col = {"OVER":"#3fb950","UNDER":"#ff7b72"}

    # ── PANEL 1: Proyección antigua vs nueva + cambios ────────────────────────
    ax1 = fig.add_subplot(gs[0])
    x = np.arange(len(labels))
    w = 0.30
    b1 = ax1.bar(x - w, proj_old, w, color="#58a6ff", alpha=0.60, edgecolor="#0d1117", label="Solo temporada (anterior)")
    b2 = ax1.bar(x,     proj_new, w, color=[col[r] for r in recs_new], alpha=0.92, edgecolor="#0d1117", label="Con semana actual (nuevo)")
    ax1.scatter(x, lines_new, marker="D", s=100, color="white", zorder=5, label="Línea O/U")

    for xi, (po, pn, ln, ro, rn, ch) in enumerate(zip(proj_old, proj_new, lines_new, recs_old, recs_new, changed)):
        ax1.text(xi-w, po+0.15, f"{po:.1f}", ha="center", fontsize=7.5, color="#8b949e")
        ax1.text(xi,   pn+0.15, f"{pn:.1f}", ha="center", fontsize=9,   color="white", fontweight="bold")
        lbl = f"{rn}"
        if ch:
            lbl += " ⚡CAMBIO"
        ax1.text(xi, -1.2, lbl, ha="center", fontsize=8,
                 color=col[rn], fontweight="bold" if ch else "normal")
        if ch:
            ax1.text(xi, -2.1, f"(era {ro})", ha="center", fontsize=7, color="#8b949e")

    ax1.set_xticks(x - w/2)
    ax1.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax1.set_ylim(-3, max(max(proj_new), max(proj_old)) + 3)
    ax1.set_title("Proyección: Solo Temporada vs Ponderada con Esta Semana", fontsize=13, pad=12)
    ax1.set_ylabel("Carreras proyectadas")
    ax1.grid(axis="y", alpha=0.4)
    over_p  = mpatches.Patch(color="#3fb950", label="OVER (nuevo)")
    under_p = mpatches.Patch(color="#ff7b72", label="UNDER (nuevo)")
    old_p   = mpatches.Patch(color="#58a6ff", alpha=0.6, label="Proyección solo temporada")
    line_p  = plt.Line2D([0],[0],marker="D",color="w",markerfacecolor="white",markersize=8,linestyle="None",label="Línea O/U")
    ax1.legend(handles=[over_p, under_p, old_p, line_p], fontsize=8, loc="upper right")

    # ── PANEL 2: Calor/Frío — carreras esta semana vs temporada ──────────────
    ax2 = fig.add_subplot(gs[1])
    team_names_plot = []
    season_avgs = []
    week_avgs   = []
    team_colors = []
    hot_cold    = []   # "HOT", "COLD", "NORMAL", "NO DATA"

    seen = set()
    for p in picks:
        for side in [("away_name","a_s_avg","a_w_avg","a_games_w","away"),
                     ("home_name","h_s_avg","h_w_avg","h_games_w","home")]:
            nm, sa, wa, gw, _ = side
            name = p[nm]
            if name in seen: continue
            seen.add(name)
            s_avg = p[sa]; w_avg = p[wa]; games_w = p[gw]
            team_names_plot.append(name.split()[-1])
            season_avgs.append(s_avg if s_avg is not None else 0)
            if w_avg is not None and games_w >= 2:
                week_avgs.append(w_avg)
                diff = w_avg - s_avg if s_avg else 0
                if diff >= 1.0:   hot_cold.append("HOT");    team_colors.append("#ff8c00")
                elif diff <= -1.0: hot_cold.append("COLD");   team_colors.append("#58a6ff")
                else:              hot_cold.append("NORMAL"); team_colors.append("#8b949e")
            else:
                week_avgs.append(0)
                hot_cold.append("NO DATA")
                team_colors.append("#3d444d")

    x2 = np.arange(len(team_names_plot))
    w2 = 0.35
    ax2.bar(x2-w2/2, season_avgs, w2, color="#8b949e", alpha=0.6, edgecolor="#0d1117", label="R/juego temporada")
    ax2.bar(x2+w2/2, week_avgs,   w2, color=team_colors, alpha=0.95, edgecolor="#0d1117", label="R/juego esta semana")

    for xi, (sa, wa, hc) in enumerate(zip(season_avgs, week_avgs, hot_cold)):
        if hc == "HOT":
            ax2.text(xi, max(sa,wa)+0.15, "🔥HOT", ha="center", fontsize=7.5, color="#ff8c00")
        elif hc == "COLD":
            ax2.text(xi, max(sa,wa)+0.15, "❄COLD", ha="center", fontsize=7.5, color="#58a6ff")
        elif hc == "NO DATA":
            ax2.text(xi+w2/2, 0.2, "?", ha="center", fontsize=9, color="#3d444d")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(team_names_plot, fontsize=8, rotation=30, ha="right")
    ax2.set_title("Carreras/Juego: Esta Semana vs Temporada — Equipos de Hoy", fontsize=13, pad=12)
    ax2.set_ylabel("Carreras por juego")
    ax2.grid(axis="y", alpha=0.4)
    hot_p  = mpatches.Patch(color="#ff8c00", label="HOT (+1 sobre avg)")
    cold_p = mpatches.Patch(color="#58a6ff", label="COLD (-1 bajo avg)")
    norm_p = mpatches.Patch(color="#8b949e", label="Normal")
    ax2.legend(handles=[hot_p, cold_p, norm_p], fontsize=8)

    # ── PANEL 3: Tabla resumen visual ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    col_headers = ["PARTIDO", "Proy\nTemporada", "Proy\nSemana", "Línea", "PICK FINAL", "Cambio?", "Motivo"]
    table_data = []
    for p in picks:
        motivo = p["motivo"]
        table_data.append([
            p["label"],
            f"{p['proj_old']:.2f}",
            f"{p['proj_new']:.2f}",
            f"{p['line_new']}",
            p["rec_new"],
            "⚡ SI" if p["changed"] else "—",
            motivo,
        ])

    table = ax3.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col_), cell in table.get_celld().items():
        cell.set_facecolor("#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#e6edf3")
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#58a6ff", fontweight="bold")
        elif col_ == 4 and row > 0:
            txt = cell.get_text().get_text()
            cell.set_facecolor("#1a3a1a" if txt=="OVER" else "#3a1a1a")
            cell.set_text_props(color="#3fb950" if txt=="OVER" else "#ff7b72", fontweight="bold")
        elif col_ == 5 and row > 0:
            if cell.get_text().get_text().startswith("⚡"):
                cell.set_facecolor("#2a2a00")
                cell.set_text_props(color="#ffa657")

    ax3.set_title("Tabla de Picks — Impacto del Rendimiento Semanal", fontsize=13, pad=12, color="#e6edf3")

    fig.suptitle(
        f"MLB PICKS {TODAY}  |  Temporada completa + Esta Semana ({WEEK_START} → {TODAY})\n"
        f"Basado en {len(df_season)} juegos totales  |  {len(df_week)} juegos esta semana",
        fontsize=14, color="#e6edf3", fontweight="bold", y=0.975
    )

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_semana_picks.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n   Imagen: {out}")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("📦 Cargando temporada completa...")
    df_s = get_games(SEASON_START, TODAY)
    print(f"   {len(df_s)} juegos de temporada")

    print("📅 Cargando juegos de esta semana...")
    df_w = get_games(WEEK_START, TODAY)
    print(f"   {len(df_w)} juegos esta semana ({WEEK_START} → {TODAY})")

    print("📅 Obteniendo partidos de hoy...")
    todays = get_todays_games()

    # Proyección SOLO temporada (modelo anterior)
    def project_old(away_id, home_id):
        a = team_slice(df_s, away_id); h = team_slice(df_s, home_id)
        def l10s(tid):
            rows=[]
            for _,row in df_s[(df_s["away_id"]==tid)|(df_s["home_id"]==tid)].sort_values("date").iterrows():
                rows.append(row["away_runs"] if row["away_id"]==tid else row["home_runs"])
            return np.mean(rows[-10:]) if rows else (a["avg_s"] or 0)
        def l10a(tid):
            rows=[]
            for _,row in df_s[(df_s["away_id"]==tid)|(df_s["home_id"]==tid)].sort_values("date").iterrows():
                rows.append(row["home_runs"] if row["away_id"]==tid else row["away_runs"])
            return np.mean(rows[-10:]) if rows else (a["avg_a"] or 0)
        pa = (a["avg_s"] or 0)*0.5 + l10s(away_id)*0.3 + (h["avg_a"] or 0)*0.2
        ph = (h["avg_s"] or 0)*0.5 + l10s(home_id)*0.3 + (a["avg_a"] or 0)*0.2
        proj = pa + ph
        h2h = df_s[((df_s["away_id"]==away_id)&(df_s["home_id"]==home_id))|
                   ((df_s["away_id"]==home_id)&(df_s["home_id"]==away_id))]
        if len(h2h)>=2: proj = proj*0.80 + h2h["total"].mean()*0.20
        line = min(LINES, key=lambda x: abs(x-proj))
        return proj, line

    picks = []
    print("\n" + "━"*80)
    print(f"  MLB PICKS HOY — TEMPORADA vs SEMANA — {TODAY}")
    print("━"*80)

    for g in todays:
        aid  = g["teams"]["away"]["team"]["id"]
        hid  = g["teams"]["home"]["team"]["id"]
        aname = g["teams"]["away"]["team"]["name"]
        hname = g["teams"]["home"]["team"]["name"]
        label = f"{aname.split()[-1]}@{hname.split()[-1]}"

        # modelo anterior (solo temporada)
        po, lo = project_old(aid, hid)
        ro = "OVER" if po >= lo else "UNDER"

        # modelo nuevo (temporada + semana)
        nr = project_weighted(df_s, df_w, aid, hid)
        pn = nr["proj"]; ln = nr["line"]
        rn = "OVER" if pn >= ln else "UNDER"
        changed = (ro != rn)

        # stats semana
        a_w = nr["a_week"]; h_w = nr["h_week"]
        a_s_avg = nr["a_season"]["avg_s"]; h_s_avg = nr["h_season"]["avg_s"]
        a_w_avg = a_w["avg_s"] if a_w["games"]>=2 else None
        h_w_avg = h_w["avg_s"] if h_w["games"]>=2 else None

        # motivo
        parts = []
        if a_w_avg is not None:
            diff_a = a_w_avg - (a_s_avg or 0)
            if abs(diff_a) >= 0.8:
                parts.append(f"{aname.split()[-1]} {'HOT' if diff_a>0 else 'COLD'} ({a_w_avg:.1f}vs{a_s_avg:.1f})")
        if h_w_avg is not None:
            diff_h = h_w_avg - (h_s_avg or 0)
            if abs(diff_h) >= 0.8:
                parts.append(f"{hname.split()[-1]} {'HOT' if diff_h>0 else 'COLD'} ({h_w_avg:.1f}vs{h_s_avg:.1f})")
        motivo = " / ".join(parts) if parts else "Sin cambio relevante"

        # print
        arr = "⬆️ " if rn=="OVER" else "⬇️ "
        cambio_str = " ← CAMBIO!" if changed else ""
        col_s = "\033[92m" if rn=="OVER" else "\033[91m"
        reset = "\033[0m"
        print(f"\n  {label}")
        print(f"  Temp: proj={po:.2f} ({ro} {lo}) → Semana: proj={pn:.2f} {col_s}{rn} {ln}{reset}{cambio_str}")
        a_str = f"{a_w_avg:.2f} R/j" if a_w_avg else f"sin datos ({a_w['games']}j)"
        h_str = f"{h_w_avg:.2f} R/j" if h_w_avg else f"sin datos ({h_w['games']}j)"
        print(f"  Esta semana: {aname.split()[-1]}={a_str} (temp {a_s_avg:.2f})  |  {hname.split()[-1]}={h_str} (temp {h_s_avg:.2f})")
        if parts: print(f"  ⚡ {motivo}")

        picks.append({
            "label":    label,
            "away_name": aname, "home_name": hname,
            "proj_old": po, "proj_new": pn,
            "line_old": lo, "line_new": ln,
            "rec_old":  ro, "rec_new":  rn,
            "changed":  changed, "motivo": motivo,
            "a_s_avg": a_s_avg, "h_s_avg": h_s_avg,
            "a_w_avg": a_w_avg, "h_w_avg": h_w_avg,
            "a_games_w": a_w["games"], "h_games_w": h_w["games"],
        })

    # Resumen de cambios
    cambios = [p for p in picks if p["changed"]]
    print("\n" + "━"*80)
    print(f"  RESUMEN: {len(cambios)}/{len(picks)} picks CAMBIARON al incluir esta semana")
    print("━"*80)
    if cambios:
        for p in cambios:
            print(f"  ⚡ {p['label']:<28}  {p['rec_old']} {p['line_old']} → {p['rec_new']} {p['line_new']}")
            print(f"     {p['motivo']}")
    else:
        print("  Ningún pick cambió de dirección.")

    print("\n  PICKS FINALES:")
    sorted_p = sorted(picks, key=lambda x: -abs(x["proj_new"]-x["line_new"]))
    for i, p in enumerate(sorted_p, 1):
        col_s = "\033[92m" if p["rec_new"]=="OVER" else "\033[91m"
        reset = "\033[0m"
        dif = p["proj_new"] - p["line_new"]
        print(f"  #{i}  {col_s}{p['rec_new']} {p['line_new']}{reset}  {p['label']:<28}  proj {p['proj_new']:.2f}  dif {dif:+.2f}")

    print("\n  Generando grafica...")
    make_graphic(picks, df_s, df_w)
    print("  Listo.")

if __name__ == "__main__":
    main()
