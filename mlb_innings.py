#!/usr/bin/env python3
"""
MLB — Análisis por INNING de cada partido de hoy.
Temporada completa + Esta semana + H2H → Dashboard de carreras por entrada.
"""
import urllib.request, json, time, sys, importlib

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

pd=ensure("pandas"); np=ensure("numpy"); ensure("matplotlib"); ensure("seaborn")
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns, pandas as pd, numpy as np

TODAY="2026-06-04"; SEASON_START="2026-03-19"; WEEK_START="2026-05-28"
BG="#0d1117"; PANEL="#161b22"; BORDER="#30363d"
GREEN="#3fb950"; RED="#ff7b72"; BLUE="#58a6ff"
ORANGE="#ffa657"; PURPLE="#d2a8ff"; WHITE="#e6edf3"; GREY="#8b949e"

def fetch(url):
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=25) as r: return json.loads(r.read().decode())
        except: time.sleep(1.5)
    return {}

# ── Carga juegos con linescore por inning ────────────────────────────────────
def load_games_innings(start, end, verbose=True):
    """
    Descarga todos los juegos entre start y end.
    Para cada juego extrae carreras por inning (1-9) de visitante y local.
    """
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
         f"&startDate={start}&endDate={end}"
         f"&hydrate=linescore(matchup),team&gameType=R")
    data=fetch(url)
    rows=[]
    total_dates=len(data.get("dates",[]))
    for di,d in enumerate(data.get("dates",[])):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls=g.get("linescore",{})
            innings_raw=ls.get("innings",[])
            # Construir dict inning→(away_runs, home_runs)
            inn={}
            for iv in innings_raw:
                n=iv.get("num",0)
                if 1<=n<=9:
                    ar=iv.get("away",{}).get("runs",0) or 0
                    hr=iv.get("home",{}).get("runs",0) or 0
                    inn[n]=(ar,hr)
            if not inn: continue   # sin detalle de innings
            aid=g["teams"]["away"]["team"]["id"]
            hid=g["teams"]["home"]["team"]["id"]
            an=g["teams"]["away"]["team"]["name"]
            hn=g["teams"]["home"]["team"]["name"]
            teams_ls=ls.get("teams",{})
            total_ar=teams_ls.get("away",{}).get("runs",0) or 0
            total_hr=teams_ls.get("home",{}).get("runs",0) or 0
            row={"date":d["date"],"game_pk":g["gamePk"],
                 "away_id":aid,"home_id":hid,"away_name":an,"home_name":hn,
                 "away_total":total_ar,"home_total":total_hr,"total":total_ar+total_hr}
            for n in range(1,10):
                ar,hr=inn.get(n,(0,0))
                row[f"away_i{n}"]=ar; row[f"home_i{n}"]=hr
                row[f"total_i{n}"]=ar+hr
            rows.append(row)
        if verbose and (di+1)%15==0:
            print(f"  {di+1}/{total_dates} fechas | {len(rows)} juegos", end="\r")
    df=pd.DataFrame(rows)
    if not df.empty: df["date"]=pd.to_datetime(df["date"])
    if verbose: print(f"\n  ✅ {len(df)} juegos con detalle de innings")
    return df

def load_today():
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={TODAY}"
         f"&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    return [g for d in fetch(url).get("dates",[]) for g in d.get("games",[])]

# ── Estadísticas por inning de un equipo ─────────────────────────────────────
def team_innings(df, tid):
    """
    Para cada inning 1-9 devuelve:
      scored[i]  = lista de carreras anotadas por este equipo en inning i
      allowed[i] = lista de carreras permitidas en inning i
    """
    scored={i:[] for i in range(1,10)}
    allowed={i:[] for i in range(1,10)}
    for _,row in df.iterrows():
        is_away = row["away_id"]==tid
        is_home = row["home_id"]==tid
        if not (is_away or is_home): continue
        for i in range(1,10):
            if is_away:
                scored[i].append(row[f"away_i{i}"])
                allowed[i].append(row[f"home_i{i}"])
            else:
                scored[i].append(row[f"home_i{i}"])
                allowed[i].append(row[f"away_i{i}"])
    avg_s={i: np.mean(v) if v else 0 for i,v in scored.items()}
    avg_a={i: np.mean(v) if v else 0 for i,v in allowed.items()}
    run_pct={i: np.mean([1 if x>0 else 0 for x in v]) if v else 0 for i,v in scored.items()}
    return {"scored":scored,"allowed":allowed,"avg_s":avg_s,"avg_a":avg_a,"run_pct":run_pct,
            "games":len(df[(df["away_id"]==tid)|(df["home_id"]==tid)])}

def h2h_innings(df, aid, hid):
    """Carreras totales por inning en todos los H2H entre aid y hid."""
    h2h=df[((df["away_id"]==aid)&(df["home_id"]==hid))|
           ((df["away_id"]==hid)&(df["home_id"]==aid))].copy()
    if h2h.empty: return h2h, {i:[] for i in range(1,10)}, {}
    total_inn={i: list(h2h[f"total_i{i}"]) for i in range(1,10)}
    avg_inn={i: np.mean(v) if v else 0 for i,v in total_inn.items()}
    return h2h, total_inn, avg_inn

def proj_innings(df_s, df_w, aid, hid):
    """Proyecta carreras totales por inning ponderando temporada + semana."""
    a_s=team_innings(df_s, aid); h_s=team_innings(df_s, hid)
    a_w=team_innings(df_w, aid); h_w=team_innings(df_w, hid)
    proj={}
    for i in range(1,10):
        # Visitante anota en inning i: vs defensa local
        def blend_off(s_avg, w_scored, s_allowed_opp, w_allowed_opp):
            sv=s_avg; wv=np.mean(w_scored) if w_scored else None
            sa=s_allowed_opp; wa=np.mean(w_allowed_opp) if w_allowed_opp else None
            off = sv*0.4 + (wv if wv is not None else sv)*0.6
            dfc = sa*0.4 + (wa if wa is not None else sa)*0.6
            return (off + dfc) / 2

        pa = blend_off(a_s["avg_s"][i], a_w["scored"][i],
                       h_s["avg_a"][i], h_w["allowed"][i])
        ph = blend_off(h_s["avg_s"][i], h_w["scored"][i],
                       a_s["avg_a"][i], a_w["allowed"][i])
        proj[i] = pa + ph
    return proj, a_s, h_s, a_w, h_w

# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def draw_dashboard(games_data, df_s, df_w):
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":PANEL,"axes.edgecolor":BORDER,
        "text.color":WHITE,"axes.labelcolor":WHITE,"xtick.color":GREY,
        "ytick.color":GREY,"grid.color":BORDER,"font.size":9,
    })

    N=len(games_data)
    COLS=3; ROWS_G=(N+COLS-1)//COLS

    # Figura grande: sección superior = heatmap global, luego grid de partidos
    fig=plt.figure(figsize=(26, 8 + ROWS_G*9))
    fig.patch.set_facecolor(BG)

    outer=gridspec.GridSpec(2+ROWS_G, 1, figure=fig,
                            hspace=0.55, top=0.96, bottom=0.02,
                            left=0.04, right=0.97,
                            height_ratios=[2.0, 2.0]+[3.5]*ROWS_G)

    innings_label=[f"Inn {i}" for i in range(1,10)]

    # ══════════════════════════════════════════════════════════════════
    # PANEL A — Heatmap: promedio carreras totales por inning × equipo (todos los de hoy)
    # ══════════════════════════════════════════════════════════════════
    ax_heat=fig.add_subplot(outer[0])
    team_ids=[]
    for gd in games_data:
        if gd["aid"] not in team_ids: team_ids.append(gd["aid"])
        if gd["hid"] not in team_ids: team_ids.append(gd["hid"])

    heat_data=[]; heat_labels=[]
    for tid in team_ids:
        ts=team_innings(df_s, tid)
        heat_data.append([ts["avg_s"][i] for i in range(1,10)])
        # buscar nombre
        nm=[gd["an"] if gd["aid"]==tid else gd["hn"]
            for gd in games_data if gd["aid"]==tid or gd["hid"]==tid]
        heat_labels.append(nm[0].split()[-1] if nm else str(tid))

    heat_arr=np.array(heat_data)
    cmap_custom=LinearSegmentedColormap.from_list("hot",["#161b22","#1a3a1a",GREEN],N=256)
    sns.heatmap(heat_arr, ax=ax_heat, xticklabels=innings_label,
                yticklabels=heat_labels, cmap=cmap_custom, annot=True, fmt=".2f",
                linewidths=0.8, linecolor=BG, cbar_kws={"shrink":0.6,"label":"R/juego"},
                annot_kws={"size":7.5,"color":WHITE})
    ax_heat.set_title("Promedio Carreras ANOTADAS por Inning — Todos los Equipos de Hoy (Temporada Completa)",
                       fontsize=11, color=WHITE, pad=10, loc="left", fontweight="bold")
    ax_heat.tick_params(axis="x", labelsize=8); ax_heat.tick_params(axis="y", labelsize=8, rotation=0)

    # ══════════════════════════════════════════════════════════════════
    # PANEL B — Heatmap total H2H por inning (promedio de los 9 partidos)
    # ══════════════════════════════════════════════════════════════════
    ax_h2h_heat=fig.add_subplot(outer[1])
    h2h_heat=[]; h2h_labels=[]
    for gd in games_data:
        row=[gd["h2h_avg_inn"].get(i,0) for i in range(1,10)]
        h2h_heat.append(row)
        h2h_labels.append(gd["label"])

    h2h_arr=np.array(h2h_heat)
    cmap_h2h=LinearSegmentedColormap.from_list("h2h",["#161b22","#2a1a3a",PURPLE],N=256)
    sns.heatmap(h2h_arr, ax=ax_h2h_heat, xticklabels=innings_label,
                yticklabels=h2h_labels, cmap=cmap_h2h, annot=True, fmt=".1f",
                linewidths=0.8, linecolor=BG, cbar_kws={"shrink":0.6,"label":"R totales/j"},
                annot_kws={"size":7.5,"color":WHITE})
    ax_h2h_heat.set_title("Promedio Carreras TOTALES por Inning — H2H 2026 (visitante+local juntos)",
                            fontsize=11, color=WHITE, pad=10, loc="left", fontweight="bold")
    ax_h2h_heat.tick_params(axis="x", labelsize=8); ax_h2h_heat.tick_params(axis="y", labelsize=8, rotation=0)

    # ══════════════════════════════════════════════════════════════════
    # FILAS 2..N+1 — Un bloque por partido
    # ══════════════════════════════════════════════════════════════════
    for gi, gd in enumerate(games_data):
        row_idx = 2 + gi // COLS
        col_idx = gi % COLS

        # Crear sub-grid para esta fila si no existe
        if col_idx == 0:
            row_gs = gridspec.GridSpecFromSubplotSpec(
                1, COLS, subplot_spec=outer[row_idx], wspace=0.38)

        ax = fig.add_subplot(row_gs[col_idx])

        inn_x = np.arange(1, 10)
        proj  = [gd["proj_inn"].get(i, 0) for i in range(1, 10)]
        s_avg = [gd["a_s_avg"][i] + gd["h_s_avg"][i] for i in range(1, 10)]
        w_avg_a = [np.mean(gd["a_w_scored"].get(i,[])) if gd["a_w_scored"].get(i) else None for i in range(1,10)]
        w_avg_h = [np.mean(gd["h_w_scored"].get(i,[])) if gd["h_w_scored"].get(i) else None for i in range(1,10)]
        w_total = [((w_avg_a[j] or 0) + (w_avg_h[j] or 0)) for j in range(9)]
        h2h_avg = [gd["h2h_avg_inn"].get(i, 0) for i in range(1, 10)]

        # Barras de proyección por inning
        bar_col = [GREEN if p >= 0.6 else RED if p <= 0.25 else GREY for p in proj]
        bars = ax.bar(inn_x, proj, 0.55, color=bar_col, alpha=0.85, edgecolor=BG, zorder=3, label="Proyección")

        # Línea temporada
        ax.plot(inn_x, s_avg, "o--", color=BLUE, lw=1.8, ms=5, alpha=0.8, label="Temp (total)", zorder=4)
        # Línea semana
        w_valid = [w if w and w > 0 else None for w in w_total]
        if any(v is not None for v in w_valid):
            xv=[inn_x[j] for j,v in enumerate(w_valid) if v is not None]
            yv=[v for v in w_valid if v is not None]
            ax.plot(xv, yv, "s--", color=ORANGE, lw=1.8, ms=5, alpha=0.85, label="Semana", zorder=4)
        # H2H
        if any(v > 0 for v in h2h_avg):
            ax.plot(inn_x, h2h_avg, "^-", color=PURPLE, lw=2, ms=6, alpha=0.9, label=f"H2H({gd['h2h_n']}j)", zorder=5)

        # Anotar valores de proyección
        for xi, (p_val, bc) in enumerate(zip(proj, bar_col)):
            ax.text(xi+1, p_val+0.03, f"{p_val:.2f}", ha="center", fontsize=6.5,
                    color=WHITE, fontweight="bold", zorder=6)

        # Inning con más carreras esperado
        max_inn = proj.index(max(proj)) + 1
        ax.axvline(max_inn, color=GREEN, lw=1, linestyle=":", alpha=0.5)
        ax.text(max_inn + 0.12, max(proj) * 0.85, f"pico\ninn{max_inn}",
                fontsize=6, color=GREEN, alpha=0.8)

        ax.set_xticks(inn_x)
        ax.set_xticklabels([f"I{i}" for i in range(1, 10)], fontsize=8)
        ax.set_xlim(0.3, 9.7); ax.grid(axis="y", alpha=0.3)

        # Conclusión
        total_proj = sum(proj)
        hot_innings = [i for i, p in enumerate(proj, 1) if p >= 0.6]
        cold_innings = [i for i, p in enumerate(proj, 1) if p <= 0.22]

        an_short = gd["an"].split()[-1]; hn_short = gd["hn"].split()[-1]
        title_col = GREEN if total_proj >= gd["line_est"] else RED
        rec = "OVER" if total_proj >= gd["line_est"] else "UNDER"
        ax.set_title(
            f"{an_short} @ {hn_short}\n"
            f"{rec} {gd['line_est']}  ·  proj {total_proj:.1f} carr",
            fontsize=9, color=title_col, fontweight="bold", pad=5
        )

        # Anotación innings calientes/fríos
        note_parts=[]
        if hot_innings:  note_parts.append(f"Calientes: inn {hot_innings}")
        if cold_innings: note_parts.append(f"Fríos: inn {cold_innings}")
        if note_parts:
            ax.text(0.02, 0.97, "\n".join(note_parts), transform=ax.transAxes,
                    va="top", fontsize=6, color=ORANGE, family="monospace")

        ax.legend(fontsize=6, loc="upper right", framealpha=0.6)

    # ── Título global ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.978,
             f"MLB ANÁLISIS POR INNING  —  {TODAY}",
             ha="center", fontsize=17, color=WHITE, fontweight="bold")
    fig.text(0.5, 0.967,
             f"Temporada completa ({len(df_s)} juegos)  +  Esta semana ({len(df_w)} juegos)  +  H2H 2026  "
             f"|  Barras = proyección · línea azul = temporada · naranja = semana · morado = H2H",
             ha="center", fontsize=8.5, color=GREY)

    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_innings_dashboard.png"
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor=BG)
    print(f"\n  Imagen guardada: {out}")
    return out

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("📦 Cargando temporada con detalle de innings...")
    df_s = load_games_innings(SEASON_START, TODAY)
    print("📅 Cargando semana con innings...")
    df_w = load_games_innings(WEEK_START, TODAY, verbose=False)
    print(f"   {len(df_w)} juegos esta semana")
    print("📅 Partidos de hoy...")
    todays = load_today()
    print(f"   {len(todays)} partidos")

    LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]

    games_data = []
    print("\n" + "═"*80)
    print(f"  ANÁLISIS POR INNING — {TODAY}")
    print("═"*80)

    for g in todays:
        aid = g["teams"]["away"]["team"]["id"]; hid = g["teams"]["home"]["team"]["id"]
        an  = g["teams"]["away"]["team"]["name"]; hn = g["teams"]["home"]["team"]["name"]
        ap  = g["teams"]["away"].get("probablePitcher",{}).get("fullName","TBD")
        hp  = g["teams"]["home"].get("probablePitcher",{}).get("fullName","TBD")
        venue = g.get("venue",{}).get("name","")

        proj, a_s, h_s, a_w, h_w = proj_innings(df_s, df_w, aid, hid)
        h2h_df, h2h_inn, h2h_avg_inn = h2h_innings(df_s, aid, hid)

        # ajuste H2H
        total_proj = sum(proj.values())
        if len(h2h_df) >= 2:
            h2h_total = h2h_df["total"].mean()
            factor = 0.80
            for i in range(1,10):
                proj[i] = proj[i]*factor + (h2h_avg_inn.get(i,proj[i]))*(1-factor)
            total_proj = sum(proj.values())

        line_est = min(LINES, key=lambda x: abs(x - total_proj))
        rec = "OVER" if total_proj >= line_est else "UNDER"

        # innings más productivos
        sorted_inn = sorted(proj.items(), key=lambda x: -x[1])
        hot_inn = [f"#{i}" for i,v in sorted_inn[:3]]
        cold_inn = [f"#{i}" for i,v in sorted_inn[-3:]]

        print(f"\n  {an} @ {hn}")
        print(f"  {ap} vs {hp}  |  {venue}")
        print(f"  H2H 2026: {len(h2h_df)} juegos  |  avg total: {h2h_df['total'].mean():.1f}" if len(h2h_df)>0 else "  H2H 2026: sin datos")
        print(f"  Proyección total: {total_proj:.2f}  →  {'OVER' if rec=='OVER' else 'UNDER'} {line_est}")
        print(f"  Innings calientes: {', '.join(hot_inn)}  |  Fríos: {', '.join(cold_inn)}")
        print(f"  Desglose: " + "  ".join([f"I{i}:{proj[i]:.2f}" for i in range(1,10)]))

        games_data.append({
            "aid":aid,"hid":hid,"an":an,"hn":hn,"ap":ap,"hp":hp,"venue":venue,
            "label":f"{an.split()[-1]}@{hn.split()[-1]}",
            "proj_inn":proj,"total_proj":total_proj,"line_est":line_est,"rec":rec,
            "a_s_avg":a_s["avg_s"],"h_s_avg":h_s["avg_s"],
            "a_w_scored":a_w["scored"],"h_w_scored":h_w["scored"],
            "h2h_n":len(h2h_df),"h2h_avg_inn":h2h_avg_inn,"h2h_df":h2h_df,
        })

    print("\n" + "═"*80)
    print("  RESUMEN PICKS:")
    for gd in sorted(games_data, key=lambda x: -abs(x["total_proj"]-x["line_est"])):
        c = "\033[92m" if gd["rec"]=="OVER" else "\033[91m"
        print(f"  {c}{gd['rec']} {gd['line_est']}\033[0m  {gd['label']:<28}  "
              f"proj={gd['total_proj']:.2f}  dif={gd['total_proj']-gd['line_est']:+.2f}")

    print("\n🎨 Generando dashboard...")
    draw_dashboard(games_data, df_s, df_w)
    print("✅ Listo.")

if __name__ == "__main__":
    main()
