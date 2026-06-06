"""
Motor de prediccion unificado.
Misma formula para total del juego e innings individuales.

Pesos:
  season   20%  - promedio completo de temporada actual
  last10   25%  - ultimos 10 juegos del equipo
  week     30%  - juegos de los ultimos 7 dias
  h2h      10%  - head-to-head directo 2026 (si hay >=2)
  prev     10%  - temporada anterior (2025)
  defense   5%  - carreras permitidas por el rival (esta semana / temporada)
  streak    -   - modificador +/-3% por juego en racha, tope 15%

"""
import statistics
from typing import Optional

LINES = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]

W_SEASON = 0.20
W_LAST10 = 0.25
W_WEEK   = 0.30
W_H2H    = 0.10
W_PREV   = 0.10
W_DEF    = 0.05


def nearest_line(total: float) -> float:
    return min(LINES, key=lambda x: abs(x - total))


def streak_mod(streak_code: str) -> float:
    """'W3' -> +0.09  |  'L2' -> -0.06  |  '' -> 0"""
    if not streak_code or len(streak_code) < 2:
        return 0.0
    direction = 1.0 if streak_code[0].upper() == "W" else -1.0
    try:
        n = int(streak_code[1:])
    except ValueError:
        return 0.0
    return min(direction * n * 0.03, 0.15) if direction > 0 else max(direction * n * 0.03, -0.15)


def _blend(season: Optional[float],
           last10: Optional[float],
           week: Optional[float],
           prev: Optional[float],
           h2h: Optional[float],
           defense: Optional[float]) -> float:
    """
    Devuelve la proyeccion de carreras para un equipo/inning.
    Si alguna fuente no esta disponible, redistribuye su peso.
    """
    sources = {
        "season": (season, W_SEASON),
        "last10": (last10, W_LAST10),
        "week":   (week,   W_WEEK),
        "h2h":    (h2h,    W_H2H),
        "prev":   (prev,   W_PREV),
        "def":    (defense, W_DEF),
    }
    total_w = 0.0
    total_v = 0.0
    for k, (v, w) in sources.items():
        if v is not None and v > 0:
            total_v += v * w
            total_w += w
    if total_w == 0:
        return 0.0
    return total_v / total_w


def _safe_mean(lst: list) -> Optional[float]:
    return statistics.mean(lst) if lst else None


# ─────────────────────────────────────────────────────────────────────────────
# Estructuras de datos del equipo para el motor
# ─────────────────────────────────────────────────────────────────────────────

class TeamData:
    def __init__(self, tid: int):
        self.tid = tid
        # carreras anotadas por juego (temporada)
        self.scored_season: list[float] = []
        # carreras permitidas por juego (temporada)
        self.allowed_season: list[float] = []
        # log cronológico para W/L y gráfica: [{date, scored, allowed, opp_name}]
        self.game_log: list[dict] = []
        # carreras por inning (temporada): {1: [r1,r2,...], ...}
        self.scored_inn_season: dict[int, list[float]] = {i: [] for i in range(1, 10)}
        self.allowed_inn_season: dict[int, list[float]] = {i: [] for i in range(1, 10)}
        # esta semana
        self.scored_week: list[float] = []
        self.allowed_week: list[float] = []
        self.scored_inn_week: dict[int, list[float]] = {i: [] for i in range(1, 10)}
        self.allowed_inn_week: dict[int, list[float]] = {i: [] for i in range(1, 10)}
        # temporada anterior
        self.scored_prev: list[float] = []
        # standings
        self.wins: int = 0
        self.losses: int = 0
        self.win_pct: float = 0.0
        self.streak_code: str = ""

    def add_game(self, is_away: bool, scored: int, allowed: int,
                 inn: dict[int, tuple[int, int]], is_week: bool,
                 date: str = "", opp_name: str = ""):
        self.scored_season.append(scored)
        self.allowed_season.append(allowed)
        self.game_log.append({
            "date": date, "scored": scored, "allowed": allowed,
            "result": "W" if scored > allowed else "L",
            "opp": opp_name,
            "isAway": is_away,
            "innings": {str(k): [ar, hr] for k, (ar, hr) in inn.items()} if inn else {},
        })
        if is_week:
            self.scored_week.append(scored)
            self.allowed_week.append(allowed)
        for n in range(1, 10):
            ar, hr = inn.get(n, (0, 0))
            s = ar if is_away else hr
            a = hr if is_away else ar
            self.scored_inn_season[n].append(s)
            self.allowed_inn_season[n].append(a)
            if is_week:
                self.scored_inn_week[n].append(s)
                self.allowed_inn_week[n].append(a)

    def add_prev_game(self, scored: int):
        self.scored_prev.append(scored)

    # helpers
    def season_rpg(self)  -> Optional[float]: return _safe_mean(self.scored_season)
    def last10_rpg(self)  -> Optional[float]: return _safe_mean(self.scored_season[-10:])
    def week_rpg(self)    -> Optional[float]: return _safe_mean(self.scored_week) if self.scored_week else None
    def prev_rpg(self)    -> Optional[float]: return _safe_mean(self.scored_prev)
    def season_ra(self)   -> Optional[float]: return _safe_mean(self.allowed_season)
    def week_ra(self)     -> Optional[float]: return _safe_mean(self.allowed_week) if self.allowed_week else None

    def inn_season(self, n: int)  -> Optional[float]: return _safe_mean(self.scored_inn_season[n])
    def inn_last10(self, n: int)  -> Optional[float]: return _safe_mean(self.scored_inn_season[n][-10:])
    def inn_week(self, n: int)    -> Optional[float]: return _safe_mean(self.scored_inn_week[n]) if self.scored_inn_week[n] else None
    def inn_allowed_season(self, n: int) -> Optional[float]: return _safe_mean(self.allowed_inn_season[n])
    def inn_allowed_week(self, n: int)   -> Optional[float]: return _safe_mean(self.allowed_inn_week[n]) if self.allowed_inn_week[n] else None


# ─────────────────────────────────────────────────────────────────────────────
# Motor principal
# ─────────────────────────────────────────────────────────────────────────────

def predict_game(away: TeamData, home: TeamData,
                 h2h_games: list[dict]) -> dict:
    """
    Devuelve el diccionario completo de prediccion para un partido.
    h2h_games: lista de juegos terminados entre estos dos equipos esta temporada.
    """
    h2h_totals   = [g["total"]     for g in h2h_games]
    h2h_avg_total = _safe_mean(h2h_totals)

    # ── proyeccion total ──────────────────────────────────────────────────────
    # Carreras visitante: su ofensiva vs defensa del local
    def proj_team(att: TeamData, def_: TeamData, h2h_scored: list[float]) -> float:
        off = _blend(
            season  = att.season_rpg(),
            last10  = att.last10_rpg(),
            week    = att.week_rpg(),
            prev    = att.prev_rpg(),
            h2h     = _safe_mean(h2h_scored),
            defense = def_.season_ra(),
        )
        mod = streak_mod(att.streak_code)
        return off * (1 + mod)

    h2h_away_scored = [g["away_scored"] for g in h2h_games]
    h2h_home_scored = [g["home_scored"] for g in h2h_games]

    pa = proj_team(away, home, h2h_away_scored)
    ph = proj_team(home, away, h2h_home_scored)
    proj_total = pa + ph

    # ajuste H2H directo sobre el total
    if len(h2h_games) >= 2 and h2h_avg_total:
        proj_total = proj_total * 0.90 + h2h_avg_total * 0.10

    line = nearest_line(proj_total)
    diff = proj_total - line
    rec  = "OVER" if diff >= 0 else "UNDER"
    conf = 3 if abs(diff) > 1.5 else 2 if abs(diff) > 0.6 else 1

    # ── proyeccion por inning (misma formula) ─────────────────────────────────
    innings_proj = []
    for n in range(1, 10):
        h2h_inn_totals = [g["innings"].get(n, (0, 0))[0] + g["innings"].get(n, (0, 0))[1]
                          for g in h2h_games]

        pa_inn = _blend(
            season  = away.inn_season(n),
            last10  = away.inn_last10(n),
            week    = away.inn_week(n),
            prev    = None,
            h2h     = _safe_mean([g["innings"].get(n, (0, 0))[0] for g in h2h_games]),
            defense = home.inn_allowed_season(n),
        )
        ph_inn = _blend(
            season  = home.inn_season(n),
            last10  = home.inn_last10(n),
            week    = home.inn_week(n),
            prev    = None,
            h2h     = _safe_mean([g["innings"].get(n, (0, 0))[1] for g in h2h_games]),
            defense = away.inn_allowed_season(n),
        )
        inn_proj = pa_inn + ph_inn
        if len(h2h_games) >= 2:
            inn_proj = inn_proj * 0.90 + (_safe_mean(h2h_inn_totals) or inn_proj) * 0.10

        innings_proj.append({
            "inning":       n,
            "proj":         round(inn_proj, 3),
            "seasonAvg":    round((away.inn_season(n) or 0) + (home.inn_season(n) or 0), 3),
            "weekAvg":      round((away.inn_week(n) or 0)   + (home.inn_week(n) or 0), 3)
                            if (away.inn_week(n) or home.inn_week(n)) else None,
            "h2hAvg":       round(_safe_mean(h2h_inn_totals), 3) if h2h_inn_totals else None,
        })

    # H2H over rate
    h2h_over_pct = None
    if len(h2h_games) >= 2:
        h2h_over_pct = round(sum(1 for g in h2h_games if g["total"] > line) / len(h2h_games), 3)

    return {
        "awayRuns":    round(pa, 3),
        "homeRuns":    round(ph, 3),
        "total":       round(proj_total, 3),
        "line":        line,
        "rec":         rec,
        "diff":        round(diff, 3),
        "conf":        conf,
        "h2hN":        len(h2h_games),
        "h2hAvgTotal": round(h2h_avg_total, 2) if h2h_avg_total else None,
        "h2hOverPct":  h2h_over_pct,
        "innings":     innings_proj,
    }
