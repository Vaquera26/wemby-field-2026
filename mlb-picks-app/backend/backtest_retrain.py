#!/usr/bin/env python3
"""
Reentrenamiento con resultados del backtest — Stacking + Calibración.

Capas:
  Capa 1 (ya existente):
    XGBoost + Ridge entrenados con 2025 → predicen carreras totales

  Capa 2 (nueva — entrenada con errores del backtest 2026):
    A) Corrector de residuos: aprende el error sistemático y lo corrige
       Features: predicción_capa1, diferencia_modelos, h2h_n, mes, dia_semana, etc.
       Target: error real (predicho - actual)

    B) Calibrador de confianza: aprende qué pick es correcto vs incorrecto
       Features: mismos + delta_vs_linea
       Target: binario (1=correcto, 0=incorrecto)
       → filtra picks de baja confianza

Validación honesta: walk-forward dentro del backtest
  Entrena corrector en primeros 70% de días → prueba en últimos 30%
  Compara accuracy antes vs después de corrección
"""

import sys, importlib, json, statistics
from pathlib import Path

def ensure(pkg, imp=None):
    try: return importlib.import_module(imp or pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q","--break-system-packages"])
        return importlib.import_module(imp or pkg)

np  = ensure("numpy"); pd = ensure("pandas")
ensure("matplotlib"); ensure("seaborn"); ensure("xgboost"); ensure("scikit-learn","sklearn")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np, pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier

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
    for d in fetch(url).get("dates",[]):
        for g in d.get("games",[]):
            if g.get("status",{}).get("statusCode","") not in ("F","FR","FT","FO"): continue
            ls = g.get("linescore",{}).get("teams",{})
            ar = ls.get("away",{}).get("runs",0) or 0
            hr = ls.get("home",{}).get("runs",0) or 0
            if ar + hr == 0: continue
            rows.append({"date":d["date"],"away_id":g["teams"]["away"]["team"]["id"],
                         "home_id":g["teams"]["home"]["team"]["id"],
                         "away_name":g["teams"]["away"]["team"]["name"],
                         "home_name":g["teams"]["home"]["team"]["name"],
                         "away_runs":ar,"home_runs":hr,"total":ar+hr})
    df = pd.DataFrame(rows)
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    return df

def load_team_stats(season):
    hit = {}; pit = {}
    for group, store in [("hitting",hit),("pitching",pit)]:
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
    sc = [r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
    al = [r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]
    wins = sum(1 for s,a in zip(sc,al) if s>a)
    return {"rpg":np.mean(sc),"rpg_l10":np.mean(sc[-10:]),"ra_pg":np.mean(al),
            "win_pct":wins/len(sc),"scored":sc,"allowed":al}

def h2h_before(df, aid, hid, before_date):
    mask=(((df["away_id"]==aid)&(df["home_id"]==hid))|
          ((df["away_id"]==hid)&(df["home_id"]==aid))) & (df["date"]<before_date)
    return df[mask]["total"].tolist()

def formula_predict(a, h, h2h):
    def proj(att, def_):
        sv=att["rpg"]; l10=att["rpg_l10"]
        wv=np.mean(att["scored"][-7:]) if len(att["scored"])>=3 else sv
        return sv*0.25+l10*0.25+wv*0.35+def_["ra_pg"]*0.15
    p=proj(a,h)+proj(h,a)
    if len(h2h)>=2: p=p*0.85+np.mean(h2h)*0.15
    line=nearest_line(p); return p,line,"OVER" if p>=line else "UNDER"

from ml_model import MLPredictor, build_features

def train_2025(df_2025, hit25, pit25):
    agg={}
    for tid in set(df_2025["away_id"])|set(df_2025["home_id"]):
        sub=df_2025[(df_2025["away_id"]==tid)|(df_2025["home_id"]==tid)]
        sc=[r["away_runs"] if r["away_id"]==tid else r["home_runs"] for _,r in sub.iterrows()]
        al=[r["home_runs"] if r["away_id"]==tid else r["away_runs"] for _,r in sub.iterrows()]
        wins=sum(1 for s,a in zip(sc,al) if s>a)
        agg[tid]={"rpg":np.mean(sc) if sc else 4.,"rpg_l10":np.mean(sc[-10:]) if sc else 4.,
                  "ra_pg":np.mean(al) if al else 4.,"win_pct":wins/len(sc) if sc else .5}
    rows=[]
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
    return MLPredictor().fit(rows)

# ── Genera dataset completo del backtest ──────────────────────────────────────
def generate_backtest_dataset(df_2026, hit26, pit26, ml_layer1):
    print("Generando dataset backtest (walk-forward)...")
    cutoff = pd.Timestamp("2026-04-01")
    game_dates = sorted(df_2026[df_2026["date"]>=cutoff]["date"].unique())
    records = []
    for di, gdate in enumerate(game_dates):
        for _, g in df_2026[df_2026["date"]==gdate].iterrows():
            aid=g["away_id"]; hid=g["home_id"]
            a_st=rolling_stats(df_2026,aid,gdate)
            h_st=rolling_stats(df_2026,hid,gdate)
            if not a_st or not h_st: continue
            h2h=h2h_before(df_2026,aid,hid,gdate)

            f_proj,f_line,f_rec=formula_predict(a_st,h_st,h2h)

            aps=pit26.get(aid,{}); hps=pit26.get(hid,{})
            ahs=hit26.get(aid,{}); hhs=hit26.get(hid,{})
            feats=build_features(a_st["rpg"],a_st["rpg_l10"],a_st["win_pct"],
                                  aps.get("era","4.20"),aps.get("whip","1.30"),ahs.get("ops",".720"),
                                  "",a_st["ra_pg"],h_st["rpg"],h_st["rpg_l10"],h_st["win_pct"],
                                  hps.get("era","4.20"),hps.get("whip","1.30"),hhs.get("ops",".720"),
                                  "",h_st["ra_pg"],
                                  np.mean(h2h) if len(h2h)>=2 else None,len(h2h),gdate.month)
            ml_r=ml_layer1.predict(feats)
            ml_tot=ml_r["total"] or f_proj

            ens=ml_tot*0.55+f_proj*0.45
            actual=float(g["total"])
            error=ens-actual
            line=nearest_line(ens)
            rec="OVER" if ens>=line else "UNDER"
            correct=(rec=="OVER" and actual>line) or (rec=="UNDER" and actual<line)

            records.append({
                "date":gdate, "month":gdate.month, "dow":gdate.dayofweek,
                "away_id":aid, "home_id":hid,
                "away":g["away_name"].split()[-1], "home":g["home_name"].split()[-1],
                # Predicciones de capa 1
                "f_proj":f_proj, "ml_proj":ml_tot, "ens_proj":ens,
                "disagreement":abs(ml_tot-f_proj),    # cuan diferentes son los modelos
                "delta_line":ens-line,                 # diferencia vs linea
                "abs_delta":abs(ens-line),
                "h2h_n":len(h2h),
                "h2h_avg":np.mean(h2h) if h2h else ens,
                # Stats de equipo
                "a_rpg":a_st["rpg"], "h_rpg":h_st["rpg"],
                "a_ra":a_st["ra_pg"], "h_ra":h_st["ra_pg"],
                "a_wpct":a_st["win_pct"], "h_wpct":h_st["win_pct"],
                # Resultado
                "line":line, "rec":rec, "actual":actual,
                "error":error, "abs_error":abs(error),
                "correct":correct,
                "is_over_pick": rec=="OVER",
            })
        if (di+1)%10==0: print(f"  {di+1}/{len(game_dates)} dias...", end="\r")
    print(f"\n  {len(records)} juegos en el dataset")
    return pd.DataFrame(records)

# ── Features para capa 2 ──────────────────────────────────────────────────────
FEAT_COLS_L2 = [
    "f_proj","ml_proj","ens_proj","disagreement","delta_line","abs_delta",
    "h2h_n","h2h_avg","a_rpg","h_rpg","a_ra","h_ra","a_wpct","h_wpct",
    "month","dow","is_over_pick",
]

def make_X(df): return df[FEAT_COLS_L2].values.astype(float)

# ── Capa 2A: Corrector de residuos ────────────────────────────────────────────
def train_residual_corrector(df_train):
    """Aprende a predecir el ERROR del modelo para corregirlo."""
    X = make_X(df_train)
    y = df_train["error"].values          # target = predicho - real

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("m",  XGBRegressor(n_estimators=300, learning_rate=0.04,
                             max_depth=3, subsample=0.75,
                             reg_alpha=1.0, reg_lambda=2.0,
                             random_state=42, verbosity=0)),
    ])
    cv = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
    pipe.fit(X, y)
    return pipe, float(-cv.mean())

# ── Capa 2B: Calibrador de confianza ─────────────────────────────────────────
def train_confidence_calibrator(df_train):
    """Aprende a predecir si un pick sera correcto (1) o incorrecto (0)."""
    X = make_X(df_train)
    y = df_train["correct"].astype(int).values

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("m",  XGBClassifier(n_estimators=300, learning_rate=0.04,
                              max_depth=3, subsample=0.75,
                              reg_alpha=1.0, reg_lambda=2.0,
                              use_label_encoder=False,
                              eval_metric="logloss",
                              random_state=42, verbosity=0)),
    ])
    cv = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    pipe.fit(X, y)
    return pipe, float(cv.mean())

# ── Evalúa con corrección aplicada ───────────────────────────────────────────
def evaluate_with_correction(df_test, corrector, calibrator, threshold=0.54):
    X = make_X(df_test)

    # Corregir predicción
    correction = corrector.predict(X)
    corrected_proj = df_test["ens_proj"].values - correction   # quita el error sistemático
    corrected_line = np.array([nearest_line(p) for p in corrected_proj])
    corrected_rec  = np.where(corrected_proj >= corrected_line, "OVER", "UNDER")
    corrected_correct = np.array([
        (r=="OVER" and a>l) or (r=="UNDER" and a<l)
        for r,a,l in zip(corrected_rec, df_test["actual"].values, corrected_line)
    ])

    # Probabilidad de acierto del calibrador
    proba_correct = calibrator.predict_proba(X)[:,1]   # P(correcto)

    results = df_test.copy()
    results["corr_proj"]    = corrected_proj
    results["corr_line"]    = corrected_line
    results["corr_rec"]     = corrected_rec
    results["corr_correct"] = corrected_correct
    results["proba"]        = proba_correct
    # Solo jugar cuando el calibrador dice que hay >= threshold de prob
    results["play"]         = proba_correct >= threshold

    return results

# ── Print comparativa ─────────────────────────────────────────────────────────
def print_comparison(df_before, df_after, threshold=0.54):
    n = len(df_after)
    plays = df_after["play"].sum()

    acc_before = df_before["correct"].mean()*100
    acc_after  = df_after["corr_correct"].mean()*100
    acc_filtered = df_after.loc[df_after["play"],"corr_correct"].mean()*100 if plays>0 else 0
    pnl_before   = df_before["correct"].map({True:100.,False:-110.}).sum()
    pnl_after    = df_after["corr_correct"].map({True:100.,False:-110.}).sum()
    pnl_filtered = df_after.loc[df_after["play"],"corr_correct"].map({True:100.,False:-110.}).sum() if plays>0 else 0

    print("\n" + "="*60)
    print(f"  COMPARATIVA — TEST SET ({n} juegos)")
    print("="*60)
    print(f"  {'':30s} {'Acc':>6}  {'P&L':>8}")
    print(f"  {'─'*30} {'─'*6}  {'─'*8}")
    print(f"  {'Antes (Capa 1 original)':30s} {acc_before:5.1f}%  ${pnl_before:+.0f}")
    print(f"  {'Despues (+ corrector residuos)':30s} {acc_after:5.1f}%  ${pnl_after:+.0f}")
    print(f"  {'Solo picks confiables (p>={threshold})':30s} {acc_filtered:5.1f}%  ${pnl_filtered:+.0f}  ({plays}/{n} juegos)")
    print()

    # By pick type after correction
    print("  Despues de correccion, por tipo:")
    for rec in ["OVER","UNDER"]:
        sub = df_after[df_after["corr_rec"]==rec]
        if len(sub): print(f"    {rec}: {sub['corr_correct'].mean()*100:.1f}%  ({len(sub)} juegos)")

    print()
    print("  Solo picks confiables (p>=0.54), por tipo:")
    for rec in ["OVER","UNDER"]:
        sub = df_after[(df_after["corr_rec"]==rec) & df_after["play"]]
        if len(sub): print(f"    {rec}: {sub['corr_correct'].mean()*100:.1f}%  ({len(sub)} juegos)")

# ── Imagen comparativa ────────────────────────────────────────────────────────
def plot_comparison(df_train, df_test, df_test_eval):
    plt.rcParams.update({
        "figure.facecolor":"#ffffff","axes.facecolor":"#fafafa",
        "axes.edgecolor":"#d1d5db","text.color":"#111",
        "xtick.color":"#666","ytick.color":"#666","grid.color":"#e5e7eb","font.size":9,
    })

    fig = plt.figure(figsize=(22, 20))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38,
                            top=0.93, bottom=0.05, left=0.07, right=0.97)

    # ── 1. Error antes vs después ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df_train["error"],    bins=30, alpha=0.65, color="#9ca3af",  label="Train error dist.")
    ax1.hist(df_test["error"],     bins=30, alpha=0.65, color="#dc2626",  label="Test: antes corrección")
    residuals_after = df_test_eval["corr_proj"] - df_test_eval["actual"]
    ax1.hist(residuals_after,      bins=30, alpha=0.65, color="#166534",  label="Test: despues corrección")
    ax1.axvline(0, color="#111", lw=1.5)
    ax1.axvline(df_test["error"].mean(),   color="#dc2626", lw=1.5, linestyle="--",
                label=f"Media antes: {df_test['error'].mean():+.2f}")
    ax1.axvline(residuals_after.mean(),   color="#166534", lw=1.5, linestyle="--",
                label=f"Media despues: {residuals_after.mean():+.2f}")
    ax1.set_xlabel("Error (proyectado − real)"); ax1.set_ylabel("# juegos")
    ax1.set_title("Distribución del error — antes vs después de corrección",
                  fontsize=11, pad=8, loc="left")
    ax1.legend(fontsize=7.5)
    ax1.grid(axis="y", alpha=0.4)

    # ── 2. Accuracy por cuartil de confianza del calibrador ───────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    df_test_eval["proba_bin"] = pd.cut(df_test_eval["proba"],
                                        bins=[0,.45,.50,.54,.60,.70,1.0],
                                        labels=["<45","45-50","50-54","54-60","60-70",">70"])
    grouped = df_test_eval.groupby("proba_bin").agg(
        acc=("corr_correct","mean"), n=("corr_correct","count")).reset_index()
    bar_c = ["#166534" if a>=0.524 else "#dc2626" for a in grouped["acc"]]
    bars  = ax2.bar(range(len(grouped)), grouped["acc"]*100, color=bar_c,
                    alpha=0.85, edgecolor="#fff", width=0.65)
    ax2.axhline(52.4, color="#374151", lw=1.2, linestyle="--", label="52.4% break-even")
    ax2.axhline(50,   color="#aaa", lw=0.8, linestyle=":")
    for xi,(_, row) in enumerate(grouped.iterrows()):
        ax2.text(xi, row["acc"]*100+0.5, f"{row['acc']*100:.1f}%\n({int(row['n'])}j)",
                 ha="center", fontsize=8)
    ax2.set_xticks(range(len(grouped)))
    ax2.set_xticklabels([f"P={l}" for l in grouped["proba_bin"]], fontsize=8)
    ax2.set_ylim(0, 80); ax2.set_ylabel("Accuracy %")
    ax2.set_title("Accuracy por nivel de confianza del calibrador\n(filtrar P<0.54 elimina picks malos)",
                  fontsize=11, pad=8, loc="left")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.4)

    # ── 3. P&L acumulado: antes vs después vs filtrado ────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    df_sorted = df_test_eval.sort_values("date").reset_index(drop=True)
    cum_before   = df_test["correct"].map({True:100.,False:-110.}).cumsum().values
    cum_after    = df_sorted["corr_correct"].map({True:100.,False:-110.}).cumsum().values
    # Filtrado: solo cuando play=True, de lo contrario 0
    filtered_pnl = df_sorted.apply(
        lambda r: 100. if r["play"] and r["corr_correct"] else
                 (-110. if r["play"] else 0.), axis=1).cumsum().values

    x_idx = range(len(df_sorted))
    ax3.plot(x_idx, cum_before,   color="#9ca3af", lw=1.5, label="Antes (capa 1)")
    ax3.plot(x_idx, cum_after,    color="#374151", lw=2,   label="Después (+ corrector)")
    ax3.plot(x_idx, filtered_pnl, color="#166534", lw=2.5, label="Filtrado (P≥0.54)")
    ax3.axhline(0, color="#dc2626", lw=1, linestyle="--")
    ax3.fill_between(x_idx, filtered_pnl, 0,
                     where=np.array(filtered_pnl)>=0, alpha=0.12, color="#166534")
    ax3.fill_between(x_idx, filtered_pnl, 0,
                     where=np.array(filtered_pnl)<0,  alpha=0.12, color="#dc2626")
    ax3.set_xlabel("Juego # (test set, orden cronológico)")
    ax3.set_ylabel("P&L acumulado USD")
    ax3.set_title("P&L acumulado — test set  |  Antes vs Después vs Filtrado por confianza",
                  fontsize=12, pad=10, loc="left", fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.4)

    # ── 4. Accuracy diaria después de corrección ──────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    daily = df_sorted.groupby("date").agg(
        acc_before=("correct","mean"),
        acc_after =("corr_correct","mean"),
    ).reset_index()
    dates = pd.to_datetime(daily["date"])
    ax4.bar(dates, daily["acc_after"]*100,
            color=["#166534" if a>=52.4 else "#dc2626" for a in daily["acc_after"]],
            alpha=0.75, width=0.8, label="Accuracy después de corrección")
    roll7 = pd.Series(daily["acc_after"].values).rolling(5, min_periods=2).mean()*100
    ax4.plot(dates, roll7, color="#111", lw=2)
    ax4.axhline(52.4, color="#374151", lw=1.2, linestyle="--")
    ax4.set_ylim(0, 105); ax4.set_ylabel("Accuracy %")
    ax4.set_title("Accuracy diaria — test set (después de corrección)",
                  fontsize=11, pad=8, loc="left")
    ax4.grid(axis="y", alpha=0.4)
    ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %d"))

    # ── 5. Top features del corrector de residuos ─────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    # (pasamos el corrector como global)
    try:
        imp = _corrector_model.named_steps["m"].feature_importances_
        feat_imp = sorted(zip(FEAT_COLS_L2, imp), key=lambda x: -x[1])[:12]
        names_f = [f[0] for f in feat_imp]; vals_f = [f[1]*100 for f in feat_imp]
        ax5.barh(range(len(names_f)), vals_f, color="#374151", alpha=0.85)
        ax5.set_yticks(range(len(names_f)))
        ax5.set_yticklabels(names_f, fontsize=9)
        ax5.set_xlabel("Importancia %")
        ax5.set_title("Features más importantes — corrector de residuos",
                      fontsize=11, pad=8, loc="left")
        ax5.grid(axis="x", alpha=0.4)
    except: pass

    n_play = df_test_eval["play"].sum()
    acc_filt = df_test_eval.loc[df_test_eval["play"],"corr_correct"].mean()*100 if n_play>0 else 0
    pnl_filt = df_test_eval.loc[df_test_eval["play"],"corr_correct"].map({True:100.,False:-110.}).sum()

    fig.suptitle(
        f"Reentrenamiento con resultados del backtest — Stacking\n"
        f"Train: {len(df_train)} juegos (70%)  |  Test: {len(df_test)} juegos (30%)  |  "
        f"Accuracy test: {df_test_eval['corr_correct'].mean()*100:.1f}%  |  "
        f"Filtrado (P≥0.54): {acc_filt:.1f}% en {n_play} picks  |  P&L ${pnl_filt:+.0f}",
        fontsize=12, fontweight="bold", y=0.975
    )
    out = "/Users/vaquera/Documents/NBA-SPURS/mlb_stacking.png"
    plt.savefig(out, dpi=155, bbox_inches="tight", facecolor="#ffffff")
    print(f"\nImagen: {out}")

_corrector_model = None

def main():
    global _corrector_model
    print("Cargando datos...")
    df_2025 = load_games("2025-03-20","2025-09-28")
    df_2026 = load_games("2026-03-19","2026-06-04")
    hit25,pit25 = load_team_stats(2025)
    hit26,pit26 = load_team_stats(2026)

    print("Entrenando capa 1 con 2025...")
    ml_l1 = train_2025(df_2025, hit25, pit25)
    print(f"  CV MAE capa 1: {ml_l1.cv_mae:.3f} runs\n")

    # Generar dataset completo del backtest
    df_bt = generate_backtest_dataset(df_2026, hit26, pit26, ml_l1)

    # Split temporal 70/30 — importante: NO shuffle, es walk-forward
    dates_sorted = sorted(df_bt["date"].unique())
    split_idx    = int(len(dates_sorted) * 0.70)
    split_date   = dates_sorted[split_idx]
    df_train = df_bt[df_bt["date"] <  split_date].copy()
    df_test  = df_bt[df_bt["date"] >= split_date].copy()

    print(f"\nSplit temporal 70/30:")
    print(f"  Train: {len(df_train)} juegos hasta {split_date.strftime('%b %d')}")
    print(f"  Test:  {len(df_test)} juegos desde {split_date.strftime('%b %d')}")
    print(f"  Acc train (capa 1): {df_train['correct'].mean()*100:.1f}%")
    print(f"  Acc test  (capa 1): {df_test['correct'].mean()*100:.1f}%")

    # Entrenar capa 2
    print("\nEntrenando capa 2A — corrector de residuos...")
    corrector, corr_mae = train_residual_corrector(df_train)
    _corrector_model = corrector
    print(f"  CV MAE del corrector: {corr_mae:.3f} runs")

    print("Entrenando capa 2B — calibrador de confianza...")
    calibrator, calib_acc = train_confidence_calibrator(df_train)
    print(f"  CV Accuracy calibrador: {calib_acc*100:.1f}%")

    # Evaluar en test set
    print("\nEvaluando en test set...")
    df_eval = evaluate_with_correction(df_test, corrector, calibrator, threshold=0.54)
    print_comparison(df_test, df_eval, threshold=0.54)

    # ── Guardar modelos ───────────────────────────────────────────────────────
    import pickle
    cache_dir = Path(__file__).parent / "cache"
    with open(cache_dir / "corrector.pkl","wb") as f: pickle.dump(corrector, f)
    with open(cache_dir / "calibrator.pkl","wb") as f: pickle.dump(calibrator, f)
    print("\nModelos de capa 2 guardados en cache/")

    # ── Imagen ────────────────────────────────────────────────────────────────
    print("\nGenerando imagen...")
    plot_comparison(df_train, df_test, df_eval)
    print("Listo.")

if __name__ == "__main__":
    main()
