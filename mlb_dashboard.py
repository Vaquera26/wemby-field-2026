#!/usr/bin/env python3
"""
MLB Dashboard — Una sola imagen con todo el análisis.
Tabla de picks + barras semanales + heatmap H2H + barras de confianza.
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
LINES=[6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0]

BG="#0d1117"; PANEL="#161b22"; BORDER="#30363d"
GREEN="#3fb950"; RED="#ff7b72"; BLUE="#58a6ff"
ORANGE="#ffa657"; PURPLE="#d2a8ff"; WHITE="#e6edf3"; GREY="#8b949e"

def fetch(url):
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=20) as r: return json.loads(r.read().decode())
        except: time.sleep(1)
    return {}

def load_games(start, end):
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1"
         f"&startDate={start}&endDate={end}&hydrate=linescore,team&gameType=R")
    rows=[]
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls=g.get("linescore",{}).get("teams",{})
            ar=ls.get("away",{}).get("runs",0) or 0; hr=ls.get("home",{}).get("runs",0) or 0
            rows.append({"date":d["date"],"away_id":g["teams"]["away"]["team"]["id"],
                         "home_id":g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "away_runs":ar,"home_runs":hr,"total":ar+hr})
    df=pd.DataFrame(rows); df["date"]=pd.to_datetime(df["date"]); return df

def load_today():
    url=(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={TODAY}"
         f"&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    return [g for d in fetch(url).get("dates",[]) for g in d.get("games",[])]

def rs(df,tid): return list(df[df["away_id"]==tid]["away_runs"])+list(df[df["home_id"]==tid]["home_runs"])
def ra(df,tid): return list(df[df["away_id"]==tid]["home_runs"])+list(df[df["home_id"]==tid]["away_runs"])

def analyze(df_s, df_w, todays):
    picks=[]
    for g in todays:
        aid=g["teams"]["away"]["team"]["id"]; hid=g["teams"]["home"]["team"]["id"]
        an=g["teams"]["away"]["team"]["name"];  hn=g["teams"]["home"]["team"]["name"]
        ap=g["teams"]["away"].get("probablePitcher",{}).get("fullName","TBD")
        hp=g["teams"]["home"].get("probablePitcher",{}).get("fullName","TBD")
        venue=g.get("venue",{}).get("name","")

        a_rs_s=rs(df_s,aid); h_rs_s=rs(df_s,hid)
        a_ra_s=ra(df_s,aid); h_ra_s=ra(df_s,hid)
        a_rs_w=rs(df_w,aid); h_rs_w=rs(df_w,hid)
        a_ra_w=ra(df_w,aid); h_ra_w=ra(df_w,hid)

        def bl(s_list, w_list):
            sv=np.mean(s_list) if s_list else 0
            l10=np.mean(s_list[-10:]) if s_list else 0
            if len(w_list)>=2: return sv*0.20+l10*0.30+np.mean(w_list)*0.50
            return sv*0.40+l10*0.60

        pa=(bl(a_rs_s,a_rs_w)+bl(h_ra_s,h_ra_w))/2
        ph=(bl(h_rs_s,h_rs_w)+bl(a_ra_s,a_ra_w))/2
        proj=pa+ph

        h2h=df_s[((df_s["away_id"]==aid)&(df_s["home_id"]==hid))|
                 ((df_s["away_id"]==hid)&(df_s["home_id"]==aid))].sort_values("date")
        h2h_avg=h2h["total"].mean() if len(h2h)>=1 else None
        if len(h2h)>=2: proj=proj*0.75+h2h_avg*0.25

        line=min(LINES,key=lambda x:abs(x-proj))
        diff=proj-line
        rec="OVER" if diff>=0 else "UNDER"

        # estado semana
        def status(s_list, w_list):
            if not s_list or len(w_list)<2: return "—", 0
            d=np.mean(w_list)-np.mean(s_list)
            if d>=1.2: return "HOT", d
            if d<=-1.2: return "COLD", d
            return "OK", d

        a_status, a_diff=status(a_rs_s, a_rs_w)
        h_status, h_diff=status(h_rs_s, h_rs_w)

        # confianza 1-5
        conf=min(5, max(1, int(abs(diff)*2.5+1)))
        if h2h_avg is not None and len(h2h)>=3:
            h2h_rate=(h2h["total"]>line).mean()
            if rec=="OVER" and h2h_rate>=0.67:   conf=min(5,conf+1)
            if rec=="UNDER" and h2h_rate<=0.33:  conf=min(5,conf+1)
            if rec=="OVER" and h2h_rate<=0.33:   conf=max(1,conf-1)
            if rec=="UNDER" and h2h_rate>=0.67:  conf=max(1,conf-1)

        picks.append({
            "label":   f"{an.split()[-1]} @ {hn.split()[-1]}",
            "away":an,"home":hn,"ap":ap,"hp":hp,"venue":venue,
            "proj":proj,"pa":pa,"ph":ph,"line":line,"diff":diff,"rec":rec,"conf":conf,
            "a_avg_s":np.mean(a_rs_s) if a_rs_s else 0,
            "h_avg_s":np.mean(h_rs_s) if h_rs_s else 0,
            "a_avg_w":np.mean(a_rs_w) if a_rs_w else None,
            "h_avg_w":np.mean(h_rs_w) if h_rs_w else None,
            "a_ra_s":np.mean(a_ra_s) if a_ra_s else 0,
            "h_ra_s":np.mean(h_ra_s) if h_ra_s else 0,
            "a_status":a_status,"h_status":h_status,
            "a_diff":a_diff,"h_diff":h_diff,
            "h2h":h2h,"h2h_avg":h2h_avg,
            "h2h_totals":list(h2h["total"]) if len(h2h)>0 else [],
            "h2h_dates":[d.strftime("%b%d") for d in h2h["date"]] if len(h2h)>0 else [],
        })
    return picks

# ─── DASHBOARD ────────────────────────────────────────────────────────────────
def draw(picks, df_s, df_w):
    plt.rcParams.update({
        "figure.facecolor":BG,"axes.facecolor":PANEL,"axes.edgecolor":BORDER,
        "text.color":WHITE,"axes.labelcolor":WHITE,"xtick.color":GREY,"ytick.color":GREY,
        "grid.color":BORDER,"font.size":9,
    })

    N=len(picks)
    # Layout: fila 0 = header tabla (1 fila alto), fila 1 = mini barras semana,
    #         fila 2 = heatmap H2H, fila 3 = barras confianza+pick
    fig=plt.figure(figsize=(26, 22))
    fig.patch.set_facecolor(BG)

    outer=gridspec.GridSpec(4,1,figure=fig,hspace=0.55,
                            top=0.93,bottom=0.03,left=0.04,right=0.97,
                            height_ratios=[2.2, 2.2, 2.8, 2.0])

    # ══════════════════════════════════════════════════════════════════
    # FILA 0 — Tabla de picks
    # ══════════════════════════════════════════════════════════════════
    ax_tbl=fig.add_subplot(outer[0])
    ax_tbl.set_facecolor(BG); ax_tbl.axis("off")

    col_w=[0.17,0.07,0.07,0.06,0.06,0.065,0.065,0.065,0.065,0.09,0.085,0.075]
    headers=["PARTIDO","Pitcher\nVisitante","Pitcher\nLocal","Proy\nCarreras",
             "Línea\nO/U","Vis R/G\nTemp","Vis R/G\nSemana","Loc R/G\nTemp","Loc R/G\nSemana",
             "H2H\nAvg","H2H\nRate","PICK"]
    assert len(headers)==len(col_w)

    # posiciones x acumuladas
    xs=[0]; [xs.append(xs[-1]+w) for w in col_w[:-1]]
    y_header=0.87; row_h=0.155; y0=y_header-row_h

    # header
    for xi,(h,w) in enumerate(zip(headers,col_w)):
        cx=xs[xi]+w/2
        ax_tbl.text(cx,y_header,h,ha="center",va="center",fontsize=7.5,
                    color=BLUE,fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="#21262d",edgecolor=BORDER,lw=0.8))

    # filas
    for ri,p in enumerate(picks):
        y=y0-ri*row_h
        bg_col="#1a1f27" if ri%2==0 else PANEL
        ax_tbl.add_patch(plt.Rectangle((0,y-row_h*0.45),1,row_h*0.92,
                         facecolor=bg_col,edgecolor=BORDER,lw=0.4,transform=ax_tbl.transAxes,
                         clip_on=False))

        row_vals=[
            p["label"],
            p["ap"][:18] if p["ap"]!="TBD" else "TBD",
            p["hp"][:18],
            f"{p['proj']:.2f}",
            f"{p['line']}",
            f"{p['a_avg_s']:.2f}",
            f"{p['a_avg_w']:.2f}" if p["a_avg_w"] is not None else "—",
            f"{p['h_avg_s']:.2f}",
            f"{p['h_avg_w']:.2f}" if p["h_avg_w"] is not None else "—",
            f"{p['h2h_avg']:.1f}" if p["h2h_avg"] else "—",
            f"{len(p['h2h'])}j",
            p["rec"],
        ]
        for ci,(val,w) in enumerate(zip(row_vals,col_w)):
            cx=xs[ci]+w/2
            color=WHITE
            fw="normal"
            if ci==11:  # PICK
                color=GREEN if val=="OVER" else RED
                fw="bold"
                ax_tbl.add_patch(plt.Rectangle((xs[ci]+0.003,y-row_h*0.38),w-0.006,row_h*0.76,
                    facecolor="#1a3a1a" if val=="OVER" else "#3a1a1a",
                    edgecolor=GREEN if val=="OVER" else RED,lw=1,
                    transform=ax_tbl.transAxes,clip_on=False))
            elif ci==0: fw="bold"
            elif ci in (6,8):  # semana
                if val!="—":
                    try:
                        seas=float(row_vals[ci-1]); wk=float(val)
                        color=GREEN if wk>seas+0.5 else RED if wk<seas-0.5 else GREY
                    except: pass
            ax_tbl.text(cx,y,val,ha="center",va="center",fontsize=7.2,
                        color=color,fontweight=fw)

    ax_tbl.set_xlim(0,1); ax_tbl.set_ylim(y0-N*row_h-0.05,y_header+row_h*0.7)
    ax_tbl.set_title("TABLA DE PICKS — Temporada + Semana + H2H",
                      fontsize=11,color=WHITE,pad=8,loc="left",fontweight="bold")

    # ══════════════════════════════════════════════════════════════════
    # FILA 1 — Mini barras: R/G semana vs temporada por equipo
    # ══════════════════════════════════════════════════════════════════
    ax_bar=fig.add_subplot(outer[1])
    teams_flat=[]; s_avgs=[]; w_avgs=[]; bar_colors=[]; statuses=[]
    seen=set()
    for p in picks:
        for nm,sa,wa,st,di in [
            (p["away"], p["a_avg_s"], p["a_avg_w"], p["a_status"], p["a_diff"]),
            (p["home"], p["h_avg_s"], p["h_avg_w"], p["h_status"], p["h_diff"]),
        ]:
            short=nm.split()[-1]
            if short in seen: continue
            seen.add(short)
            teams_flat.append(short)
            s_avgs.append(sa)
            w_avgs.append(wa if wa is not None else 0)
            if st=="HOT":   bar_colors.append(ORANGE); statuses.append("HOT")
            elif st=="COLD": bar_colors.append(BLUE);  statuses.append("COLD")
            else:           bar_colors.append(GREY);  statuses.append("OK")

    x=np.arange(len(teams_flat)); w2=0.35
    ax_bar.bar(x-w2/2, s_avgs, w2, color=GREY,   alpha=0.55, edgecolor=BG, label="R/G Temporada")
    ax_bar.bar(x+w2/2, w_avgs, w2, color=bar_colors, alpha=0.90, edgecolor=BG, label="R/G Esta Semana")

    for xi,(sa,wa,st) in enumerate(zip(s_avgs,w_avgs,statuses)):
        if wa>0:
            ax_bar.text(xi+w2/2, wa+0.1, f"{wa:.1f}", ha="center", fontsize=6.5, color=WHITE)
        if st in ("HOT","COLD"):
            c=ORANGE if st=="HOT" else BLUE
            ax_bar.text(xi, max(sa,wa)+0.5, st, ha="center", fontsize=6, color=c, fontweight="bold")

    ax_bar.set_xticks(x); ax_bar.set_xticklabels(teams_flat, fontsize=7.5, rotation=30, ha="right")
    ax_bar.set_ylabel("Carreras / juego"); ax_bar.grid(axis="y",alpha=0.3)
    ax_bar.set_title("Calor/Frío Esta Semana — Todos los Equipos de Hoy",
                      fontsize=11,color=WHITE,pad=8,loc="left",fontweight="bold")
    hot_p =mpatches.Patch(color=ORANGE, label="HOT (+1.2 sobre avg)")
    cold_p=mpatches.Patch(color=BLUE,   label="COLD (-1.2 bajo avg)")
    avg_p =mpatches.Patch(color=GREY,   alpha=0.55, label="R/G Temporada")
    ax_bar.legend(handles=[hot_p,cold_p,avg_p], fontsize=7.5, loc="upper right")

    # separadores entre partidos
    pos=0
    for p in picks:
        an=p["away"].split()[-1]; hn=p["home"].split()[-1]
        idxs=[i for i,t in enumerate(teams_flat) if t in (an,hn)]
        if len(idxs)==2:
            mid=(idxs[0]+idxs[1])/2
            ax_bar.axvline(mid+0.5, color=BORDER, lw=1, linestyle="--", alpha=0.6)

    # ══════════════════════════════════════════════════════════════════
    # FILA 2 — Heatmap H2H (9 mini-heatmaps, uno por partido)
    # ══════════════════════════════════════════════════════════════════
    ax_h2h_container=fig.add_subplot(outer[2])
    ax_h2h_container.set_facecolor(BG); ax_h2h_container.axis("off")
    ax_h2h_container.set_title("HISTORIAL H2H 2026 — Carreras por Enfrentamiento (verde=OVER, rojo=UNDER)",
                                 fontsize=11,color=WHITE,pad=8,loc="left",fontweight="bold")

    # subdividir fila 2 en 9 columnas
    inner2=gridspec.GridSpecFromSubplotSpec(1,N,subplot_spec=outer[2],wspace=0.35)

    cmap_ou=LinearSegmentedColormap.from_list("ou",["#3a1a1a",PANEL,"#1a3a1a"],N=256)

    for ci,p in enumerate(picks):
        ax=fig.add_subplot(inner2[ci])
        totals=p["h2h_totals"]; dates=p["h2h_dates"]
        line=p["line"]

        if totals:
            # Construir matriz 1×N para heatmap visual
            mat=np.array(totals).reshape(1,-1)
            sns.heatmap(mat, ax=ax, cmap=cmap_ou, annot=True, fmt=".0f",
                        cbar=False, linewidths=1.5, linecolor=BG,
                        annot_kws={"size":8.5,"weight":"bold","color":WHITE},
                        vmin=max(0,line-6), vmax=line+8)
            ax.set_xticklabels(dates, fontsize=6, rotation=35, ha="right")
            ax.set_yticklabels([])
            ax.set_yticks([])
            # colorear celdas manualmente: sobre línea = verde, bajo = rojo
            for xi,t in enumerate(totals):
                col=GREEN if t>line else RED
                ax.add_patch(plt.Rectangle((xi,0),1,1,fill=True,
                    color="#1a3a1a" if t>line else "#3a1a1a",
                    alpha=0.55,zorder=0))
                ax.text(xi+0.5,0.5,str(t),ha="center",va="center",
                        fontsize=9,fontweight="bold",color=col,zorder=3)
            # promedio
            ax.set_xlabel(f"avg {p['h2h_avg']:.1f} | {len(totals)}j", fontsize=7, color=GREY)
        else:
            ax.text(0.5,0.5,"Sin H2H",ha="center",va="center",
                    transform=ax.transAxes,fontsize=8,color=GREY)
            ax.set_xlabel("0 juegos", fontsize=7, color=GREY)

        short_a=p["away"].split()[-1]; short_h=p["home"].split()[-1]
        rec_col=GREEN if p["rec"]=="OVER" else RED
        ax.set_title(f"{short_a}@{short_h}\n{p['rec']} {line}",
                      fontsize=8, color=rec_col, fontweight="bold", pad=4)

    # ══════════════════════════════════════════════════════════════════
    # FILA 3 — Barras de proyección + confianza
    # ══════════════════════════════════════════════════════════════════
    ax_conf=fig.add_subplot(outer[3])
    labels=[p["label"] for p in picks]
    projs =[p["proj"]  for p in picks]
    lines_=[p["line"]  for p in picks]
    diffs =[p["diff"]  for p in picks]
    confs =[p["conf"]  for p in picks]
    recs  =[p["rec"]   for p in picks]
    bar_c =[GREEN if r=="OVER" else RED for r in recs]

    x3=np.arange(N); w3=0.45
    bars=ax_conf.bar(x3, projs, w3, color=bar_c, alpha=0.88, edgecolor=BG, zorder=3)
    ax_conf.scatter(x3, lines_, marker="D", s=110, color=WHITE, zorder=5, label="Línea O/U")

    for xi,(proj,line,diff,conf,rec) in enumerate(zip(projs,lines_,diffs,confs,recs)):
        # valor proyectado
        ax_conf.text(xi, proj+0.18, f"{proj:.1f}", ha="center", fontsize=8.5,
                     color=WHITE, fontweight="bold")
        # línea
        ax_conf.text(xi, line+(0.25 if line<proj else -0.55), f"◆{line}",
                     ha="center", fontsize=7.5, color=GREY)
        # rec + diferencia
        c=GREEN if rec=="OVER" else RED
        ax_conf.text(xi, -0.9, f"{rec}  ({diff:+.2f})",
                     ha="center", fontsize=8, color=c, fontweight="bold")
        # estrellas de confianza
        stars="★"*conf+"☆"*(5-conf)
        ax_conf.text(xi, -1.7, stars, ha="center", fontsize=8,
                     color=ORANGE if conf>=4 else GREY)

    ax_conf.set_xticks(x3)
    ax_conf.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax_conf.set_ylim(-2.5, max(projs)+2.5)
    ax_conf.set_ylabel("Carreras proyectadas"); ax_conf.grid(axis="y",alpha=0.3)
    ax_conf.set_title("Proyección Final + Confianza (★ = más confiable)",
                       fontsize=11,color=WHITE,pad=8,loc="left",fontweight="bold")
    over_p =mpatches.Patch(color=GREEN, label="OVER")
    under_p=mpatches.Patch(color=RED,   label="UNDER")
    lp=plt.Line2D([0],[0],marker="D",color="w",markerfacecolor=WHITE,
                  markersize=8,linestyle="None",label="Línea O/U")
    ax_conf.legend(handles=[over_p,under_p,lp], fontsize=8, loc="upper right")

    # ══════════════════════════════════════════════════════════════════
    # Título global
    # ══════════════════════════════════════════════════════════════════
    fig.text(0.5, 0.965,
             f"MLB OVER/UNDER DASHBOARD  —  {TODAY}",
             ha="center", fontsize=17, color=WHITE, fontweight="bold")
    fig.text(0.5, 0.952,
             f"Basado en {len(df_s)} juegos de temporada  |  {len(df_w)} juegos esta semana  "
             f"|  Ponderación: 50% semana + 30% últ10 + 20% temporada + H2H",
             ha="center", fontsize=9, color=GREY)

    out="/Users/vaquera/Documents/NBA-SPURS/mlb_dashboard_final.png"
    plt.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    print(f"Imagen guardada: {out}")
    return out

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Cargando temporada..."); df_s=load_games(SEASON_START, TODAY)
    print(f"  {len(df_s)} juegos")
    print("Cargando semana...");   df_w=load_games(WEEK_START, TODAY)
    print(f"  {len(df_w)} juegos esta semana")
    print("Partidos de hoy...");   todays=load_today()
    print(f"  {len(todays)} partidos")

    picks=analyze(df_s, df_w, todays)

    print("\nPICKS FINALES:")
    for p in sorted(picks, key=lambda x:-abs(x["diff"])):
        c="\033[92m" if p["rec"]=="OVER" else "\033[91m"
        print(f"  {c}{p['rec']} {p['line']}\033[0m  {p['label']:<28}  "
              f"proj={p['proj']:.2f}  dif={p['diff']:+.2f}  conf={'★'*p['conf']}")

    print("\nGenerando dashboard...")
    make_graphic=draw
    make_graphic(picks, df_s, df_w)

if __name__=="__main__":
    main()
