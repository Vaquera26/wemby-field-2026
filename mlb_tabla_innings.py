#!/usr/bin/env python3
"""Tabla compacta: 9 partidos × 9 innings con proyección y pick."""
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import pandas as pd, numpy as np

TODAY="2026-06-04"; SEASON_START="2026-03-19"; WEEK_START="2026-05-28"
BG="#0d1117"; PANEL="#161b22"; BORDER="#30363d"
GREEN="#3fb950"; RED="#ff7b72"; BLUE="#58a6ff"
ORANGE="#ffa657"; WHITE="#e6edf3"; GREY="#8b949e"; DARK="#21262d"

def fetch(url):
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=25) as r: return json.loads(r.read().decode())
        except: time.sleep(1.5)
    return {}

def load_games(start, end):
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
         f"&startDate={start}&endDate={end}"
         f"&hydrate=linescore,team&gameType=R")
    rows=[]
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls=g.get("linescore",{}); inn_raw=ls.get("innings",[])
            inn={}
            for iv in inn_raw:
                n=iv.get("num",0)
                if 1<=n<=9:
                    inn[n]=(iv.get("away",{}).get("runs",0) or 0,
                            iv.get("home",{}).get("runs",0) or 0)
            if not inn: continue
            t=ls.get("teams",{})
            ar=t.get("away",{}).get("runs",0) or 0; hr=t.get("home",{}).get("runs",0) or 0
            row={"date":d["date"],"away_id":g["teams"]["away"]["team"]["id"],
                 "home_id":g["teams"]["home"]["team"]["id"],
                 "away_name":g["teams"]["away"]["team"]["name"],
                 "home_name":g["teams"]["home"]["team"]["name"],
                 "away_total":ar,"home_total":hr,"total":ar+hr}
            for n in range(1,10):
                a,h=inn.get(n,(0,0)); row[f"ai{n}"]=a; row[f"hi{n}"]=h; row[f"ti{n}"]=a+h
            rows.append(row)
    df=pd.DataFrame(rows)
    if not df.empty: df["date"]=pd.to_datetime(df["date"])
    return df

def load_today():
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={TODAY}"
         f"&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    return [g for d in fetch(url).get("dates",[]) for g in d.get("games",[])]

def team_avg_inn(df, tid):
    scored={i:[] for i in range(1,10)}; allowed={i:[] for i in range(1,10)}
    for _,r in df[(df["away_id"]==tid)|(df["home_id"]==tid)].iterrows():
        is_away=r["away_id"]==tid
        for i in range(1,10):
            scored[i].append(r[f"ai{i}"] if is_away else r[f"hi{i}"])
            allowed[i].append(r[f"hi{i}"] if is_away else r[f"ai{i}"])
    return {i:np.mean(v) if v else 0 for i,v in scored.items()}, \
           {i:np.mean(v) if v else 0 for i,v in allowed.items()}

def project(df_s, df_w, aid, hid):
    a_s_sc, a_s_al = team_avg_inn(df_s, aid)
    h_s_sc, h_s_al = team_avg_inn(df_s, hid)
    a_w_sc, a_w_al = team_avg_inn(df_w, aid)
    h_w_sc, h_w_al = team_avg_inn(df_w, hid)

    def bl(sv, wv): return sv*0.25 + wv*0.75 if wv>0 else sv

    proj={}
    for i in range(1,10):
        pa = (bl(a_s_sc[i], a_w_sc[i]) + bl(h_s_al[i], h_w_al[i])) / 2
        ph = (bl(h_s_sc[i], h_w_sc[i]) + bl(a_s_al[i], a_w_al[i])) / 2
        proj[i] = pa + ph

    # H2H
    h2h=df_s[((df_s["away_id"]==aid)&(df_s["home_id"]==hid))|
             ((df_s["away_id"]==hid)&(df_s["home_id"]==aid))]
    if len(h2h)>=2:
        for i in range(1,10):
            h_avg=h2h[f"ti{i}"].mean()
            proj[i]=proj[i]*0.75 + h_avg*0.25

    return proj, len(h2h), h2h["total"].mean() if len(h2h)>0 else None

LINES=[6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0]

def main():
    print("Cargando datos..."); df_s=load_games(SEASON_START,TODAY)
    df_w=load_games(WEEK_START,TODAY); todays=load_today()
    print(f"  {len(df_s)} juegos temporada | {len(df_w)} semana | {len(todays)} hoy")

    rows=[]
    for g in todays:
        aid=g["teams"]["away"]["team"]["id"]; hid=g["teams"]["home"]["team"]["id"]
        an=g["teams"]["away"]["team"]["name"]; hn=g["teams"]["home"]["team"]["name"]
        ap=g["teams"]["away"].get("probablePitcher",{}).get("fullName","TBD")
        hp=g["teams"]["home"].get("probablePitcher",{}).get("fullName","TBD")
        proj,h2h_n,h2h_avg=project(df_s,df_w,aid,hid)
        total=sum(proj.values())
        line=min(LINES,key=lambda x:abs(x-total))
        rec="OVER" if total>=line else "UNDER"
        hot_inn=max(proj,key=proj.get)
        rows.append({"partido":f"{an.split()[-1]} @ {hn.split()[-1]}",
                     "pitchers":f"{ap.split()[-1]} vs {hp.split()[-1]}",
                     **{f"I{i}":round(proj[i],2) for i in range(1,10)},
                     "TOTAL":round(total,2),"LÍNEA":line,"PICK":rec,
                     "H2H":h2h_n,"H2H avg":round(h2h_avg,1) if h2h_avg else "—",
                     "inn★":f"#{hot_inn}"})

    # ── FIGURA ────────────────────────────────────────────────────────────────
    plt.rcParams.update({"figure.facecolor":BG,"font.family":"monospace"})
    fig,ax=plt.subplots(figsize=(22,7))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")

    col_labels=["PARTIDO","PITCHERS",
                "I1","I2","I3","I4","I5","I6","I7","I8","I9",
                "TOTAL","LÍNEA","PICK","H2H\n(j)","H2H\navg","INN★"]
    col_w     =[0.130, 0.095,
                0.045,0.045,0.045,0.045,0.045,0.045,0.045,0.045,0.045,
                0.055,0.050,0.060,0.045,0.052,0.048]

    # acumular x
    xs=[0.005]; [xs.append(xs[-1]+w) for w in col_w[:-1]]
    ROW_H=0.088; HEAD_Y=0.91; Y0=HEAD_Y-ROW_H*0.9

    # colormap por valor de inning (0 → rojo oscuro, 1 → naranja, 2+ → verde)
    cmap_inn=LinearSegmentedColormap.from_list("inn",
        ["#2a1a1a","#3a2a1a","#2a3a1a","#1a3a1a","#0d2a0d"],N=256)
    norm_inn=Normalize(vmin=0.0, vmax=2.5)

    def inn_bg(val):
        r,g_,b,_=cmap_inn(norm_inn(val))
        return (r,g_,b)

    def inn_txt(val):
        if val>=1.3: return GREEN
        if val>=0.8: return ORANGE
        if val<=0.35: return "#cc4444"
        return GREY

    # ── Header ────────────────────────────────────────────────────────────────
    for xi,(lbl,w) in enumerate(zip(col_labels,col_w)):
        cx=xs[xi]+w/2
        ax.text(cx, HEAD_Y, lbl, ha="center", va="center", fontsize=7.5,
                color=BLUE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=DARK,
                          edgecolor=BORDER, lw=0.8))

    # ── Filas ─────────────────────────────────────────────────────────────────
    for ri, row in enumerate(rows):
        y = Y0 - ri*ROW_H
        row_bg = "#12171f" if ri%2==0 else PANEL

        # fondo fila
        ax.add_patch(plt.Rectangle((0.004, y-ROW_H*0.48), 0.992, ROW_H*0.92,
            facecolor=row_bg, edgecolor=BORDER, lw=0.4, transform=ax.transAxes, clip_on=False))

        vals=[row["partido"], row["pitchers"],
              row["I1"],row["I2"],row["I3"],row["I4"],row["I5"],
              row["I6"],row["I7"],row["I8"],row["I9"],
              row["TOTAL"],row["LÍNEA"],row["PICK"],
              row["H2H"],row["H2H avg"],row["inn★"]]

        for ci,(val,w) in enumerate(zip(vals,col_w)):
            cx=xs[ci]+w/2
            txt=str(val); color=WHITE; fw="normal"; fsize=7.5; bg=None

            # Innings I1-I9 (columnas 2-10)
            if 2<=ci<=10:
                fval=float(val)
                color=inn_txt(fval)
                bg=inn_bg(fval)
                fw="bold" if fval>=1.2 else "normal"
                ax.add_patch(plt.Rectangle(
                    (xs[ci]+0.002, y-ROW_H*0.40), w-0.004, ROW_H*0.78,
                    facecolor=bg, edgecolor="none",
                    transform=ax.transAxes, clip_on=False))

            # TOTAL
            elif ci==11:
                fval=float(val)
                color=WHITE; fw="bold"; fsize=8

            # LÍNEA
            elif ci==12:
                color=GREY; fsize=7.5

            # PICK
            elif ci==13:
                color=GREEN if val=="OVER" else RED
                fw="bold"; fsize=8.5
                ax.add_patch(plt.Rectangle(
                    (xs[ci]+0.003, y-ROW_H*0.40), w-0.006, ROW_H*0.78,
                    facecolor="#1a3a1a" if val=="OVER" else "#3a1a1a",
                    edgecolor=GREEN if val=="OVER" else RED, lw=1,
                    transform=ax.transAxes, clip_on=False))

            # H2H juegos
            elif ci==14:
                color=GREY if val==0 else ORANGE if val<=2 else WHITE

            # H2H avg
            elif ci==15:
                color=GREY if val=="—" else WHITE

            # inn★
            elif ci==16:
                color=ORANGE; fw="bold"

            ax.text(cx, y, txt, ha="center", va="center", fontsize=fsize,
                    color=color, fontweight=fw)

    # ── Leyenda de colores ────────────────────────────────────────────────────
    leg_y=Y0-len(rows)*ROW_H-0.06
    patches=[
        mpatches.Patch(color=GREEN,  label="Inning caliente (≥1.3 runs)"),
        mpatches.Patch(color=ORANGE, label="Inning tibio (0.8-1.3 runs)"),
        mpatches.Patch(color="#cc4444", label="Inning frío (≤0.35 runs)"),
        mpatches.Patch(color=GREEN, alpha=0.4, label="OVER"),
        mpatches.Patch(color=RED,   alpha=0.4, label="UNDER"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7.5,
              facecolor=PANEL, edgecolor=BORDER, labelcolor=WHITE,
              ncol=5, bbox_to_anchor=(0.01, -0.01))

    # ── Colorbar lateral para innings ─────────────────────────────────────────
    sm=ScalarMappable(cmap=cmap_inn, norm=norm_inn)
    sm.set_array([])
    cbar=fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.012, pad=0.01)
    cbar.set_label("Runs proyectados / inning", color=GREY, fontsize=7.5)
    cbar.ax.yaxis.set_tick_params(color=GREY, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=GREY)
    cbar.outline.set_edgecolor(BORDER)

    # ── Título ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.99,
             f"MLB  ·  CARRERAS PROYECTADAS POR INNING  ·  {TODAY}",
             ha="center", fontsize=15, color=WHITE, fontweight="bold")
    fig.text(0.5, 0.965,
             f"Ponderación: 75% esta semana + 25% temporada  |  Ajuste H2H cuando hay ≥2 enfrentamientos  "
             f"|  inn★ = inning con más carreras esperadas",
             ha="center", fontsize=8, color=GREY)

    ax.set_xlim(0,1); ax.set_ylim(Y0-len(rows)*ROW_H-0.10, HEAD_Y+ROW_H)

    out="/Users/vaquera/Documents/NBA-SPURS/mlb_tabla_innings.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    print(f"\n  Imagen: {out}")

if __name__=="__main__":
    main()
