#!/usr/bin/env python3
"""H2H completo 2026 para los partidos de hoy — carreras totales por juego."""
import urllib.request, json, time, sys, importlib

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

pd = ensure("pandas"); np = ensure("numpy"); ensure("matplotlib")
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import pandas as pd, numpy as np

TODAY="2026-06-04"; SEASON_START="2026-03-19"
LINES=[6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,10.5,11.0]

def fetch(url):
    for _ in range(3):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req,timeout=20) as r: return json.loads(r.read().decode())
        except: time.sleep(1)
    return {}

def get_games():
    url=(f"https://statsapi.mlb.com/api/v1/schedule"
         f"?sportId=1&startDate={SEASON_START}&endDate={TODAY}"
         f"&hydrate=linescore,team&gameType=R")
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

def get_today():
    url=(f"https://statsapi.mlb.com/api/v1/schedule"
         f"?sportId=1&date={TODAY}&hydrate=probablePitcher,linescore,team,venue&gameType=R")
    return [g for d in fetch(url).get("dates",[]) for g in d.get("games",[])]

def main():
    print("Cargando datos..."); df=get_games(); todays=get_today()

    plt.rcParams.update({"figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
        "axes.edgecolor":"#30363d","text.color":"#e6edf3","axes.labelcolor":"#e6edf3",
        "xtick.color":"#8b949e","ytick.color":"#8b949e","grid.color":"#21262d"})

    n=len(todays)
    fig,axes=plt.subplots(3,3,figsize=(22,20))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle(f"RIVALIDAD H2H 2026 — Carreras por Partido  |  {TODAY}",
                 fontsize=16, color="#e6edf3", fontweight="bold", y=0.98)
    axes=axes.flatten()

    print("\n" + "="*80)
    print(f"  H2H HISTÓRICO 2026 — CADA PARTIDO DE HOY")
    print("="*80)

    for i, g in enumerate(todays):
        ax=axes[i]
        aid=g["teams"]["away"]["team"]["id"]; hid=g["teams"]["home"]["team"]["id"]
        aname=g["teams"]["away"]["team"]["name"]; hname=g["teams"]["home"]["team"]["name"]
        ap=g["teams"]["away"].get("probablePitcher",{}).get("fullName","TBD")
        hp=g["teams"]["home"].get("probablePitcher",{}).get("fullName","TBD")
        venue=g.get("venue",{}).get("name","")

        h2h=df[((df["away_id"]==aid)&(df["home_id"]==hid))|
               ((df["away_id"]==hid)&(df["home_id"]==aid))].sort_values("date")

        # season stats individuales
        def runs_scored(tid):
            a=df[df["away_id"]==tid]["away_runs"]; ho=df[df["home_id"]==tid]["home_runs"]
            return list(a)+list(ho)
        def runs_allowed(tid):
            a=df[df["away_id"]==tid]["home_runs"]; ho=df[df["home_id"]==tid]["away_runs"]
            return list(a)+list(ho)

        a_rs=runs_scored(aid); h_rs=runs_scored(hid)
        a_ra=runs_allowed(aid); h_ra=runs_allowed(hid)

        # proyección ponderada con semana
        WEEK_START="2026-05-28"
        df_w=df[df["date"]>=pd.to_datetime(WEEK_START)]
        def rs_week(tid):
            a=df_w[df_w["away_id"]==tid]["away_runs"]; ho=df_w[df_w["home_id"]==tid]["home_runs"]
            r=list(a)+list(ho); return (np.mean(r),len(r)) if r else (None,0)
        def ra_week(tid):
            a=df_w[df_w["away_id"]==tid]["home_runs"]; ho=df_w[df_w["home_id"]==tid]["away_runs"]
            r=list(a)+list(ho); return (np.mean(r),len(r)) if r else (None,0)

        aw_s,aw_n=rs_week(aid); hw_s,hw_n=rs_week(hid)
        aw_a,_=ra_week(aid);    hw_a,_=ra_week(hid)

        def blend(season_val, week_val, week_n, l10_list):
            l10=np.mean(l10_list[-10:]) if l10_list else season_val or 0
            if week_val is not None and week_n>=2:
                return season_val*0.20 + l10*0.30 + week_val*0.50
            return season_val*0.40 + l10*0.60 if season_val else 0

        pa_off=blend(np.mean(a_rs) if a_rs else 0, aw_s, aw_n, a_rs)
        ph_def=blend(np.mean(h_ra) if h_ra else 0, hw_a, _,    h_ra)
        ph_off=blend(np.mean(h_rs) if h_rs else 0, hw_s, hw_n, h_rs)
        pa_def=blend(np.mean(a_ra) if a_ra else 0, aw_a, _,    a_ra)
        pa_final=(pa_off+ph_def)/2
        ph_final=(ph_off+pa_def)/2
        proj=pa_final+ph_final
        if len(h2h)>=2: proj=proj*0.75+h2h["total"].mean()*0.25
        line=min(LINES,key=lambda x:abs(x-proj))
        rec="OVER" if proj>=line else "UNDER"
        diff=proj-line

        # ── consola ──────────────────────────────────────────────────────────
        print(f"\n  {aname} @ {hname}")
        print(f"  Pitchers: {ap} vs {hp}  |  {venue}")
        if not h2h.empty:
            for _,row in h2h.iterrows():
                wo="W" if (row["away_id"]==aid and row["away_runs"]>row["home_runs"]) or \
                          (row["home_id"]==aid and row["home_runs"]>row["away_runs"]) else "L"
                print(f"    {row['date'].strftime('%b %d')}  {aname.split()[-1]} {row['away_runs' if row['away_id']==aid else 'home_runs']} - "
                      f"{row['home_runs' if row['home_id']==hid else 'away_runs']} {hname.split()[-1]}  "
                      f"TOTAL={row['total']}  {'OVER' if row['total']>line else 'UNDER'} {line}")
            print(f"    H2H avg total: {h2h['total'].mean():.1f}  |  OVER {line} rate: {(h2h['total']>line).mean()*100:.0f}%")
        else:
            print("    Sin enfrentamientos previos en 2026")
        col="\033[92m" if rec=="OVER" else "\033[91m"
        r="\033[0m"
        print(f"  --> Proyeccion: {proj:.2f}  Linea: {line}  {col}{rec} ({diff:+.2f}){r}")

        # ── grafica individual ────────────────────────────────────────────────
        if not h2h.empty:
            dates_h2h=[d.strftime("%b%d") for d in h2h["date"]]
            totals_h2h=list(h2h["total"])
            bar_cols=["#3fb950" if t>line else "#ff7b72" for t in totals_h2h]
            ax.bar(range(len(totals_h2h)), totals_h2h, color=bar_cols, edgecolor="#0d1117", width=0.6)
            ax.set_xticks(range(len(totals_h2h)))
            ax.set_xticklabels(dates_h2h, fontsize=7.5, rotation=25)
            for xi,t in enumerate(totals_h2h):
                ax.text(xi,t+0.15,str(t),ha="center",fontsize=8.5,color="white",fontweight="bold")
        else:
            ax.text(0.5,0.5,"Sin H2H\nen 2026",ha="center",va="center",
                    transform=ax.transAxes,fontsize=12,color="#8b949e")

        # línea O/U y proyección
        ax.axhline(line,color="white",lw=1.5,linestyle="--",alpha=0.8,label=f"Línea {line}")
        ax.axhline(proj,color="#ffa657",lw=2,linestyle="-",alpha=0.9,label=f"Proy {proj:.1f}")

        col_r="#3fb950" if rec=="OVER" else "#ff7b72"
        short=f"{aname.split()[-1]} @ {hname.split()[-1]}"
        ax.set_title(f"{short}\n{rec} {line}  (proj {proj:.1f})", fontsize=9.5,
                     color=col_r, fontweight="bold", pad=6)
        ax.set_ylabel("Carreras totales", fontsize=7.5)

        # anotación de status semana
        tags=[]
        if aw_s is not None and aw_n>=2:
            diff_a=aw_s-(np.mean(a_rs) if a_rs else 0)
            tags.append(f"{aname.split()[-1]} {'HOT' if diff_a>0.8 else 'COLD' if diff_a<-0.8 else 'OK'} ({aw_s:.1f})")
        if hw_s is not None and hw_n>=2:
            diff_h=hw_s-(np.mean(h_rs) if h_rs else 0)
            tags.append(f"{hname.split()[-1]} {'HOT' if diff_h>0.8 else 'COLD' if diff_h<-0.8 else 'OK'} ({hw_s:.1f})")
        if tags:
            ax.text(0.02,0.97,"\n".join(tags),transform=ax.transAxes,va="top",
                    fontsize=6.5,color="#ffa657",family="monospace")

        ax.legend(fontsize=7,loc="lower right")
        ax.grid(axis="y",alpha=0.3)

    # ocultar axes sobrantes
    for j in range(len(todays),len(axes)): axes[j].set_visible(False)

    plt.tight_layout(rect=[0,0,1,0.97])
    out="/Users/vaquera/Documents/NBA-SPURS/mlb_h2h_hoy.png"
    plt.savefig(out,dpi=160,bbox_inches="tight",facecolor="#0d1117")
    print(f"\n  Imagen: {out}")

if __name__=="__main__":
    main()
