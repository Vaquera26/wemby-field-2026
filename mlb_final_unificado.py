#!/usr/bin/env python3
"""
MLB MODELO UNIFICADO — Un solo motor, una sola tabla.
El total de los innings = la proyección total = el pick O/U.
Ponderación idéntica en ambos niveles.
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd, numpy as np

TODAY="2026-06-04"; SEASON_START="2026-03-19"; WEEK_START="2026-05-28"
BG="#0d1117"; PANEL="#161b22"; BORDER="#30363d"
GREEN="#3fb950"; RED="#ff7b72"; BLUE="#58a6ff"
ORANGE="#ffa657"; WHITE="#e6edf3"; GREY="#8b949e"; DARK="#21262d"
LINES=[6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0,11.5,12.0]

def fetch(url):
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=25) as r: return json.loads(r.read().decode())
        except: time.sleep(1.5)
    return {}

# ── carga ─────────────────────────────────────────────────────────────────────
def load_games(start, end):
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
         f"&startDate={start}&endDate={end}&hydrate=linescore,team&gameType=R")
    rows=[]
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls=g.get("linescore",{}); inn_raw=ls.get("innings",[])
            inn={}
            for iv in inn_raw:
                n=iv.get("num",0)
                if 1<=n<=9: inn[n]=(iv.get("away",{}).get("runs",0) or 0,
                                    iv.get("home",{}).get("runs",0) or 0)
            if not inn: continue
            t=ls.get("teams",{})
            ar=t.get("away",{}).get("runs",0) or 0; hr=t.get("home",{}).get("runs",0) or 0
            row={"date":d["date"],
                 "away_id":g["teams"]["away"]["team"]["id"],
                 "home_id":g["teams"]["home"]["team"]["id"],
                 "away_name":g["teams"]["away"]["team"]["name"],
                 "home_name":g["teams"]["home"]["team"]["name"],
                 "away_total":ar,"home_total":hr,"total":ar+hr}
            for n in range(1,10):
                a,h=inn.get(n,(0,0))
                row[f"ai{n}"]=a; row[f"hi{n}"]=h; row[f"ti{n}"]=a+h
            rows.append(row)
    df=pd.DataFrame(rows)
    if not df.empty: df["date"]=pd.to_datetime(df["date"])
    return df

def load_today():
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={TODAY}"
         f"&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    return [g for d in fetch(url).get("dates",[]) for g in d.get("games",[])]

# ── MOTOR ÚNICO DE PROYECCIÓN ─────────────────────────────────────────────────
# Pesos idénticos para nivel de inning Y para nivel de juego completo
W_SEASON=0.25   # % de peso a promedio de temporada completa
W_LAST10=0.25   # % de peso a últimos 10 juegos del equipo en temporada
W_WEEK  =0.35   # % de peso a esta semana (may28-hoy)
W_H2H   =0.15   # % de peso a head-to-head 2026 (si hay ≥2 juegos)
# Si no hay H2H suficiente, su peso se redistribuye a semana+last10

def get_team_series(df, tid, field_away, field_home, sort=True):
    """Extrae la serie de valores de un campo para el equipo tid."""
    away=df[df["away_id"]==tid][field_away].tolist()
    home=df[df["home_id"]==tid][field_home].tolist()
    vals=away+home
    if sort:
        # mantener orden cronológico
        rows_a=df[df["away_id"]==tid][["date",field_away]].rename(columns={field_away:"v"})
        rows_h=df[df["home_id"]==tid][["date",field_home]].rename(columns={field_home:"v"})
        combined=pd.concat([rows_a,rows_h]).sort_values("date")
        vals=combined["v"].tolist()
    return vals

def blend(season_vals, week_vals, h2h_vals=None):
    """
    Aplica los pesos W_* a las tres fuentes de datos.
    Devuelve un float: la proyección ponderada.
    """
    sv = np.mean(season_vals)         if season_vals else 0
    l10= np.mean(season_vals[-10:])   if season_vals else 0
    wv = np.mean(week_vals)           if week_vals   else sv
    hv = np.mean(h2h_vals)            if h2h_vals and len(h2h_vals)>=2 else None

    if hv is not None:
        return sv*W_SEASON + l10*W_LAST10 + wv*W_WEEK + hv*W_H2H
    else:
        # redistribuir peso H2H
        ratio = 1/(W_SEASON+W_LAST10+W_WEEK)
        return (sv*W_SEASON + l10*W_LAST10 + wv*W_WEEK) * ratio

def project_game(df_s, df_w, aid, hid):
    """
    Devuelve:
      proj_total  — carreras totales proyectadas
      proj_inn    — dict {1..9: carreras totales proyectadas en ese inning}
      meta        — dict con desglose para mostrar
    Garantía: sum(proj_inn.values()) == proj_total  (misma fórmula)
    """
    # H2H directo
    h2h=df_s[((df_s["away_id"]==aid)&(df_s["home_id"]==hid))|
             ((df_s["away_id"]==hid)&(df_s["home_id"]==aid))].sort_values("date")

    proj_inn={}
    for i in range(1,10):
        # carreras que anota el visitante en inning i
        a_scored_s  = get_team_series(df_s, aid, f"ai{i}", f"hi{i}")
        a_scored_w  = get_team_series(df_w, aid, f"ai{i}", f"hi{i}")
        a_h2h       = (list(h2h[f"ai{i}"]) if aid in h2h["away_id"].values else []) + \
                      (list(h2h[f"hi{i}"]) if aid in h2h["home_id"].values else [])

        # carreras que anota el local en inning i
        h_scored_s  = get_team_series(df_s, hid, f"ai{i}", f"hi{i}")
        h_scored_w  = get_team_series(df_w, hid, f"ai{i}", f"hi{i}")
        h_h2h       = (list(h2h[f"hi{i}"]) if hid in h2h["home_id"].values else []) + \
                      (list(h2h[f"ai{i}"]) if hid in h2h["away_id"].values else [])

        # total del inning: visitante + local
        total_s  = [r["ti"+str(i)] for _,r in df_s.iterrows()
                    if r["away_id"] in (aid,hid) or r["home_id"] in (aid,hid)]
        # más preciso: total de inning en partidos donde juega cualquiera de los dos
        # Usamos la suma de las proyecciones individuales para mantener consistencia
        pa = blend(a_scored_s, a_scored_w, a_h2h if len(h2h)>=2 else None)
        ph = blend(h_scored_s, h_scored_w, h_h2h if len(h2h)>=2 else None)
        proj_inn[i] = max(0, pa + ph)

    proj_total = sum(proj_inn.values())

    # estado semana de cada equipo (solo carreras anotadas, no innings)
    a_s_total = get_team_series(df_s, aid, "away_total", "home_total")
    h_s_total = get_team_series(df_s, hid, "away_total", "home_total")
    a_w_total = get_team_series(df_w, aid, "away_total", "home_total")
    h_w_total = get_team_series(df_w, hid, "away_total", "home_total")

    a_avg_s = np.mean(a_s_total) if a_s_total else 0
    h_avg_s = np.mean(h_s_total) if h_s_total else 0
    a_avg_w = np.mean(a_w_total) if a_w_total else None
    h_avg_w = np.mean(h_w_total) if h_w_total else None

    def estado(avg_s, avg_w):
        if avg_w is None: return "—", 0
        d=avg_w-avg_s
        if d>=1.0: return "HOT", d
        if d<=-1.0: return "COLD", d
        return "OK", d

    a_est, a_dif = estado(a_avg_s, a_avg_w)
    h_est, h_dif = estado(h_avg_s, h_avg_w)

    meta={
        "h2h_n":   len(h2h),
        "h2h_avg": h2h["total"].mean() if len(h2h)>0 else None,
        "h2h_over_rate": None,
        "a_avg_s": a_avg_s, "a_avg_w": a_avg_w, "a_est": a_est, "a_dif": a_dif,
        "h_avg_s": h_avg_s, "h_avg_w": h_avg_w, "h_est": h_est, "h_dif": h_dif,
    }
    if len(h2h)>=2:
        line_est=min(LINES,key=lambda x:abs(x-proj_total))
        meta["h2h_over_rate"]=(h2h["total"]>line_est).mean()

    return proj_total, proj_inn, meta

# ── DIBUJO ────────────────────────────────────────────────────────────────────
def draw(picks):
    plt.rcParams.update({
        "figure.facecolor":BG,"font.family":"monospace","text.color":WHITE})
    fig=plt.figure(figsize=(26,8.5))
    fig.patch.set_facecolor(BG)
    ax=fig.add_axes([0.01,0.08,0.98,0.82])
    ax.set_facecolor(BG); ax.axis("off")

    # columnas
    col_defs=[
        ("PARTIDO",           0.130),
        ("PITCHERS",          0.090),
        ("I1",  0.046),("I2",0.046),("I3",0.046),("I4",0.046),("I5",0.046),
        ("I6",  0.046),("I7",0.046),("I8",0.046),("I9",0.046),
        ("TOTAL\nproyectado", 0.058),
        ("LÍNEA\nO/U",        0.052),
        ("PICK\nFINAL",       0.064),
        ("H2H\njuegos",       0.045),
        ("H2H\navg",          0.050),
        ("Visitante\nsemana", 0.058),
        ("Local\nsemana",     0.058),
        ("INN\n★",            0.044),
    ]
    labels=[c[0] for c in col_defs]
    widths=[c[1] for c in col_defs]
    xs=[0.005]; [xs.append(xs[-1]+w) for w in widths[:-1]]

    ROW_H=0.092; HEAD_Y=0.93; Y0=HEAD_Y-ROW_H

    cmap_inn=LinearSegmentedColormap.from_list("inn",
        ["#1a0a0a","#2d1a0a","#1a2d0a","#0a2d0a"],N=256)
    norm_inn=Normalize(vmin=0.0, vmax=2.6)

    def inn_fc(v):
        r,g,b,_=cmap_inn(norm_inn(v)); return (r,g,b)
    def inn_tc(v):
        if v>=1.3: return GREEN
        if v>=0.75: return ORANGE
        if v<=0.30: return "#bb3333"
        return GREY

    # ── header ────────────────────────────────────────────────────────────────
    for xi,(lbl,w) in enumerate(zip(labels,widths)):
        cx=xs[xi]+w/2
        ax.text(cx,HEAD_Y,lbl,ha="center",va="center",fontsize=7.2,
                color=BLUE,fontweight="bold",linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.22",facecolor=DARK,
                          edgecolor=BORDER,lw=0.8))

    # ── filas ─────────────────────────────────────────────────────────────────
    for ri,p in enumerate(picks):
        y=Y0-ri*ROW_H
        row_bg="#12171f" if ri%2==0 else PANEL
        ax.add_patch(plt.Rectangle((0.004,y-ROW_H*0.47),0.992,ROW_H*0.90,
            facecolor=row_bg,edgecolor=BORDER,lw=0.35,
            transform=ax.transAxes,clip_on=False))

        inn_vals=[p["proj_inn"][i] for i in range(1,10)]
        hot_inn =inn_vals.index(max(inn_vals))+1

        # estado visitante/local semana
        def fmt_estado(avg_s, avg_w, est, dif):
            if avg_w is None: return "—", GREY
            s=f"{avg_w:.2f}"
            if est=="HOT":  return f"↑{s}", GREEN
            if est=="COLD": return f"↓{s}", "#58a6ff"
            return f" {s}", GREY

        a_str,a_col=fmt_estado(p["a_avg_s"],p["a_avg_w"],p["a_est"],p["a_dif"])
        h_str,h_col=fmt_estado(p["h_avg_s"],p["h_avg_w"],p["h_est"],p["h_dif"])

        row_vals=[
            p["label"],
            p["pitchers"],
            *[f"{v:.2f}" for v in inn_vals],
            f"{p['proj_total']:.2f}",
            str(p["line"]),
            p["rec"],
            str(p["h2h_n"]),
            f"{p['h2h_avg']:.1f}" if p["h2h_avg"] else "—",
            a_str, h_str,
            f"#{hot_inn}",
        ]
        row_colors=[WHITE,GREY,
                    *[inn_tc(v) for v in inn_vals],
                    WHITE,GREY,
                    GREEN if p["rec"]=="OVER" else RED,
                    ORANGE if p["h2h_n"]>=3 else GREY,
                    WHITE if p["h2h_avg"] else GREY,
                    a_col, h_col, ORANGE]
        row_bolds=["bold","normal",
                   *["bold" if v>=1.2 else "normal" for v in inn_vals],
                   "bold","normal","bold",
                   "normal","normal","bold","bold","bold"]

        for ci,(val,w) in enumerate(zip(row_vals,widths)):
            cx=xs[ci]+w/2
            # Fondo especial para innings
            if 2<=ci<=10:
                ax.add_patch(plt.Rectangle(
                    (xs[ci]+0.002,y-ROW_H*0.40),w-0.004,ROW_H*0.78,
                    facecolor=inn_fc(float(val)),edgecolor="none",
                    transform=ax.transAxes,clip_on=False))
            # Fondo especial para PICK
            if ci==13:
                fc="#152515" if val=="OVER" else "#251515"
                ec=GREEN    if val=="OVER" else RED
                ax.add_patch(plt.Rectangle(
                    (xs[ci]+0.003,y-ROW_H*0.40),w-0.006,ROW_H*0.78,
                    facecolor=fc,edgecolor=ec,lw=1.1,
                    transform=ax.transAxes,clip_on=False))

            ax.text(cx,y,val,ha="center",va="center",
                    fontsize=7.3 if ci not in (0,13) else (7.8 if ci==0 else 8.5),
                    color=row_colors[ci],fontweight=row_bolds[ci])

    # ── nota de pesos ─────────────────────────────────────────────────────────
    nota=(f"Pesos del modelo:  Temporada {int(W_SEASON*100)}%  +  "
          f"Últimos 10j {int(W_LAST10*100)}%  +  "
          f"Esta semana {int(W_WEEK*100)}%  +  "
          f"H2H directo {int(W_H2H*100)}% (si ≥2 enfrentamientos)  "
          f"—  Misma fórmula para inning y total → suma exacta")
    ax.text(0.5,-0.04,nota,ha="center",va="top",fontsize=7.5,
            color=GREY,transform=ax.transAxes)

    # ── leyenda ───────────────────────────────────────────────────────────────
    patches=[
        mpatches.Patch(color=GREEN,        label="Inning caliente ≥1.3"),
        mpatches.Patch(color=ORANGE,       label="Inning tibio 0.75-1.3"),
        mpatches.Patch(color="#bb3333",    label="Inning frío ≤0.30"),
        mpatches.Patch(color="#152515",ec=GREEN,  label="OVER"),
        mpatches.Patch(color="#251515",ec=RED,    label="UNDER"),
        mpatches.Patch(color=GREEN,  alpha=0.7,   label="↑ HOT esta semana"),
        mpatches.Patch(color=BLUE,   alpha=0.7,   label="↓ COLD esta semana"),
    ]
    ax.legend(handles=patches,loc="lower left",fontsize=7.3,
              facecolor=PANEL,edgecolor=BORDER,labelcolor=WHITE,
              ncol=7,bbox_to_anchor=(0.01,-0.06))

    # ── colorbar ──────────────────────────────────────────────────────────────
    sm=ScalarMappable(cmap=cmap_inn,norm=norm_inn); sm.set_array([])
    cbar=fig.colorbar(sm,ax=ax,orientation="vertical",fraction=0.010,pad=0.005)
    cbar.set_label("Runs / inning",color=GREY,fontsize=7)
    cbar.ax.yaxis.set_tick_params(color=GREY,labelsize=6.5)
    plt.setp(cbar.ax.yaxis.get_ticklabels(),color=GREY)
    cbar.outline.set_edgecolor(BORDER)

    # ── título ────────────────────────────────────────────────────────────────
    fig.text(0.5,0.995,
             f"MLB  ·  PROYECCIÓN UNIFICADA POR INNING  ·  {TODAY}",
             ha="center",fontsize=15,color=WHITE,fontweight="bold",va="top")
    fig.text(0.5,0.972,
             "Modelo único: el TOTAL es exactamente la suma de I1–I9  "
             "|  Semana = May 28–Jun 4  |  Verde brillante = inning más peligroso",
             ha="center",fontsize=8.2,color=GREY,va="top")

    ax.set_xlim(0,1)
    ax.set_ylim(Y0-len(picks)*ROW_H-0.08, HEAD_Y+ROW_H)

    out="/Users/vaquera/Documents/NBA-SPURS/mlb_unificado_final.png"
    plt.savefig(out,dpi=185,bbox_inches="tight",facecolor=BG)
    print(f"  Imagen: {out}")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Cargando temporada..."); df_s=load_games(SEASON_START,TODAY)
    print(f"  {len(df_s)} juegos")
    print("Cargando semana...");    df_w=load_games(WEEK_START,TODAY)
    print(f"  {len(df_w)} juegos esta semana")
    print("Partidos de hoy...");    todays=load_today()
    print(f"  {len(todays)} partidos\n")

    picks=[]
    print(f"{'PARTIDO':<30} {'I1':>5}{'I2':>5}{'I3':>5}{'I4':>5}{'I5':>5}"
          f"{'I6':>5}{'I7':>5}{'I8':>5}{'I9':>5}  {'TOT':>6}  {'LÍNEA':>5}  PICK")
    print("─"*100)

    for g in todays:
        aid=g["teams"]["away"]["team"]["id"]; hid=g["teams"]["home"]["team"]["id"]
        an=g["teams"]["away"]["team"]["name"]; hn=g["teams"]["home"]["team"]["name"]
        ap=g["teams"]["away"].get("probablePitcher",{}).get("fullName","TBD")
        hp=g["teams"]["home"].get("probablePitcher",{}).get("fullName","TBD")
        venue=g.get("venue",{}).get("name","")

        proj_total, proj_inn, meta = project_game(df_s, df_w, aid, hid)
        line=min(LINES,key=lambda x:abs(x-proj_total))
        rec ="OVER" if proj_total>=line else "UNDER"
        diff=proj_total-line

        label=f"{an.split()[-1]} @ {hn.split()[-1]}"
        inn_str="".join([f"{proj_inn[i]:5.2f}" for i in range(1,10)])
        c="\033[92m" if rec=="OVER" else "\033[91m"; r="\033[0m"
        print(f"{label:<30}{inn_str}  {proj_total:6.2f}  {line:5.1f}  {c}{rec}{r} ({diff:+.2f})")

        picks.append({
            "label":label, "an":an,"hn":hn,
            "pitchers":f"{ap.split()[-1]} vs {hp.split()[-1]}",
            "proj_total":proj_total,"proj_inn":proj_inn,
            "line":line,"rec":rec,"diff":diff,
            **meta,
        })

    print("─"*100)
    print("\nPICKS por confianza (|diff| mayor = más confianza):")
    for p in sorted(picks,key=lambda x:-abs(x["diff"])):
        c="\033[92m" if p["rec"]=="OVER" else "\033[91m"; r="\033[0m"
        h2h_info=f"H2H {p['h2h_n']}j avg {p['h2h_avg']:.1f}" if p["h2h_avg"] else "sin H2H"
        a_w=f"{p['a_avg_w']:.1f}" if p["a_avg_w"] else "—"
        h_w=f"{p['h_avg_w']:.1f}" if p["h_avg_w"] else "—"
        print(f"  {c}{p['rec']} {p['line']}{r}  {p['label']:<28} "
              f"proj={p['proj_total']:.2f} dif={p['diff']:+.2f}  "
              f"{h2h_info}  "
              f"{p['an'].split()[-1]}:{p['a_est']}({a_w})  "
              f"{p['hn'].split()[-1]}:{p['h_est']}({h_w})")

    print("\nGenerando imagen unificada...")
    draw(picks)
    print("✅ Listo.")

if __name__=="__main__":
    main()
