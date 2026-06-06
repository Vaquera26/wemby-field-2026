"""
ML model para predecir carreras totales de un partido MLB.

Pipeline:
  1. Features extraídas de los stats acumulados de cada equipo hasta antes del partido.
  2. Target: total de carreras reales del partido.
  3. Modelo: XGBoost Regressor (mejor para tabular con pocos datos) + Ridge como fallback.
  4. Ensemble final: 55% XGB + 45% formula existente.
  5. Se entrena con datos de 2025 (completa) + 2026 (YTD).
     Para evitar data leakage se usan rolling stats hasta ANTES de cada juego,
     pero como simplificación práctica usamos los promedios de temporada
     (válido desde mediados de temporada en adelante).

Feature vector (16 features):
  away_rpg, away_rpg_l10, away_win_pct, away_era, away_whip, away_ops,
  away_streak, away_ra_pg,
  home_rpg, home_rpg_l10, home_win_pct, home_era, home_whip, home_ops,
  home_streak, home_ra_pg,
  h2h_avg, h2h_n, month
"""

import pickle, statistics
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBRegressor, XGBClassifier

CACHE_PATH     = Path(__file__).parent / "cache" / "ml_model.pkl"
WIN_CACHE_PATH = Path(__file__).parent / "cache" / "win_model.pkl"

FEATURE_NAMES = [
    "away_rpg",     "away_rpg_l10", "away_win_pct",
    "away_era",     "away_whip",    "away_ops",
    "away_streak",  "away_ra_pg",
    "home_rpg",     "home_rpg_l10", "home_win_pct",
    "home_era",     "home_whip",    "home_ops",
    "home_streak",  "home_ra_pg",
    "h2h_avg",      "h2h_n",        "month",
    # v3 extras
    "away_rpg_away",  "home_rpg_home",   # splits home/away
    "away_std",       "home_std",        # consistencia
    "park_factor",                        # factor del estadio
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _streak_num(code: str) -> float:
    """'W3' → +3.0  |  'L2' → -2.0  |  '' → 0.0"""
    if not code or len(code) < 2:
        return 0.0
    sign = 1.0 if code[0].upper() == "W" else -1.0
    try:
        return sign * float(code[1:])
    except ValueError:
        return 0.0


def _safe(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _ops_float(ops_str) -> float:
    """'.712' → 0.712"""
    try:
        s = str(ops_str).replace(",", ".")
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _era_float(era_str) -> float:
    try:
        return float(str(era_str))
    except (TypeError, ValueError):
        return 4.5   # league average fallback


def build_features(
    away_rpg: float,
    away_rpg_l10: float,
    away_win_pct: float,
    away_era: str,
    away_whip: str,
    away_ops: str,
    away_streak: str,
    away_ra_pg: float,
    home_rpg: float,
    home_rpg_l10: float,
    home_win_pct: float,
    home_era: str,
    home_whip: str,
    home_ops: str,
    home_streak: str,
    home_ra_pg: float,
    h2h_avg: Optional[float],
    h2h_n: int,
    month: int,
    # v3 extras — default a valores neutros para compatibilidad hacia atras
    away_rpg_away: float = 0.0,
    home_rpg_home: float = 0.0,
    away_std: float = 2.5,
    home_std: float = 2.5,
    park_factor: float = 1.0,
) -> list[float]:
    return [
        _safe(away_rpg),
        _safe(away_rpg_l10),
        _safe(away_win_pct),
        _era_float(away_era),
        _safe(away_whip, 1.30),
        _ops_float(away_ops),
        _streak_num(away_streak),
        _safe(away_ra_pg),
        _safe(home_rpg),
        _safe(home_rpg_l10),
        _safe(home_win_pct),
        _era_float(home_era),
        _safe(home_whip, 1.30),
        _ops_float(home_ops),
        _streak_num(home_streak),
        _safe(home_ra_pg),
        _safe(h2h_avg, 8.5),
        float(h2h_n),
        float(month),
        # v3 extras
        _safe(away_rpg_away, away_rpg),
        _safe(home_rpg_home, home_rpg),
        _safe(away_std, 2.5),
        _safe(home_std, 2.5),
        _safe(park_factor, 1.0),
    ]


# ── Model ─────────────────────────────────────────────────────────────────────

class MLPredictor:
    def __init__(self):
        self.xgb_pipe: Optional[Pipeline] = None
        self.ridge_pipe: Optional[Pipeline] = None
        self.cv_mae: float = 0.0
        self.train_n: int = 0
        self.feature_importances: dict = {}
        self.trained: bool = False

    def fit(self, rows: list[dict]) -> "MLPredictor":
        """
        rows: [{"features": [...], "target": float, "date": "YYYY-MM-DD"}, ...]
        """
        if len(rows) < 50:
            return self   # not enough data

        X = np.array([r["features"] for r in rows], dtype=float)
        y = np.array([r["target"]   for r in rows], dtype=float)

        # XGBoost
        xgb = XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.5,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0,
        )
        self.xgb_pipe = Pipeline([("scaler", StandardScaler()), ("model", xgb)])

        # Ridge (linear, stable baseline)
        self.ridge_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ])

        # Cross-validation (5-fold) on XGB
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv = cross_val_score(self.xgb_pipe, X, y,
                             cv=kf, scoring="neg_mean_absolute_error")
        self.cv_mae = round(float(-cv.mean()), 3)

        # Final fit on all data
        self.xgb_pipe.fit(X, y)
        self.ridge_pipe.fit(X, y)

        # Feature importances from XGB
        imp = self.xgb_pipe.named_steps["model"].feature_importances_
        self.feature_importances = {
            name: round(float(v), 4)
            for name, v in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
        }
        self.train_n = len(rows)
        self.trained = True
        return self

    def predict(self, features: list[float]) -> dict:
        """Returns {total, conf_width, xgb, ridge}"""
        if not self.trained:
            return {"total": None, "xgb": None, "ridge": None, "conf_width": None}
        X = np.array([features], dtype=float)
        xgb_pred   = float(self.xgb_pipe.predict(X)[0])
        ridge_pred = float(self.ridge_pipe.predict(X)[0])
        # Weighted ensemble: 65% XGB + 35% Ridge
        ensemble   = xgb_pred * 0.65 + ridge_pred * 0.35
        # Confidence: inverse of distance from 8.5 midpoint (rough heuristic)
        conf_width = abs(ensemble - 8.5)
        return {
            "total":      round(ensemble, 3),
            "xgb":        round(xgb_pred, 3),
            "ridge":      round(ridge_pred, 3),
            "conf_width": round(conf_width, 3),
        }

    def save(self):
        CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> Optional["MLPredictor"]:
        if CACHE_PATH.exists():
            try:
                with open(CACHE_PATH, "rb") as f:
                    obj = pickle.load(f)
                if obj.trained:
                    return obj
            except Exception:
                pass
        return None


# ── Model B: Winner classifier (calibrado) ────────────────────────────────────

def build_win_features(base_feats: list[float],
                        away_win_pct: float, home_win_pct: float,
                        away_rpg: float,     home_rpg: float,
                        away_era_f: float,   home_era_f: float,
                        away_ra_pg: float,   home_ra_pg: float,
                        h2h_home_win_rate: float = 0.5) -> list[float]:
    """Base features (24) + 8 diferenciales = 32 features para el clasificador de victorias."""
    win_diff = home_win_pct - away_win_pct
    rpg_diff = home_rpg    - away_rpg
    era_adv  = away_era_f  - home_era_f    # + = local tiene mejor pitcher
    ra_diff  = away_ra_pg  - home_ra_pg    # + = local tiene mejor defensa
    h2h_hw   = h2h_home_win_rate - 0.5
    return list(base_feats) + [
        win_diff, rpg_diff, era_adv, ra_diff,
        h2h_hw, 0.04,              # home_adv constante ~54%
        home_win_pct, away_win_pct,
    ]


class WinPredictor:
    """
    Clasificador calibrado: predice P(home_team_wins).
    Igual que Model B en backtest_v5 — CalibratedClassifierCV isotónico.
    """
    def __init__(self):
        self.model:    Optional[Pipeline] = None
        self.trained:  bool  = False
        self.train_n:  int   = 0
        self.cv_acc:   float = 0.0

    def fit(self, rows: list[dict]) -> "WinPredictor":
        if len(rows) < 100:
            return self
        X = np.array([r["features"] for r in rows], dtype=float)
        y = np.array([r["target"]   for r in rows], dtype=int)

        xgb = XGBClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=4,
            subsample=0.80, reg_alpha=0.5, reg_lambda=1.5,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        self.model = Pipeline([
            ("sc", StandardScaler()),
            ("m",  CalibratedClassifierCV(xgb, method="isotonic", cv=3)),
        ])
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv = cross_val_score(self.model, X, y, cv=kf, scoring="accuracy")
        self.cv_acc = round(float(cv.mean()), 3)
        self.model.fit(X, y)
        self.trained = True
        self.train_n = len(rows)
        return self

    def predict(self, features: list[float]) -> dict:
        """Devuelve {"pHome": float, "pAway": float} en porcentaje (0-100)."""
        if not self.trained or self.model is None:
            return {"pHome": 50.0, "pAway": 50.0}
        X = np.array([features], dtype=float)
        proba  = self.model.predict_proba(X)[0]
        p_home = round(float(proba[1]) * 100, 1)
        return {"pHome": p_home, "pAway": round(100 - p_home, 1)}

    def save(self):
        WIN_CACHE_PATH.parent.mkdir(exist_ok=True)
        with open(WIN_CACHE_PATH, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load() -> Optional["WinPredictor"]:
        if WIN_CACHE_PATH.exists():
            try:
                with open(WIN_CACHE_PATH, "rb") as f:
                    obj = pickle.load(f)
                if obj.trained:
                    return obj
            except Exception:
                pass
        return None
