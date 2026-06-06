import { useState, useEffect, useCallback } from "react";

const yesterday = () => {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return d.toISOString().slice(0, 10);
};

const fmt = (iso) => {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("es-MX", {
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  });
};

const fmtShort = (iso) => {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("es-MX", { day: "numeric", month: "short" });
};

const addDays = (iso, n) => {
  const d = new Date(iso + "T12:00:00");
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
};

export default function ResultsPage({ setPage }) {
  const [date, setDate]       = useState(yesterday());
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const load = useCallback(async (d) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/results?date=${d}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(date); }, [date, load]);

  const s = data?.summary;

  return (
    <div style={{ maxWidth: 1320, margin: "0 auto", padding: "0 16px 48px" }}>

      {/* ── Header ── */}
      <header style={{
        borderBottom: "2px solid #111",
        padding: "20px 0 12px",
        marginBottom: 24,
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: 12,
      }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: "#666", marginBottom: 2 }}>
            MLB · Resultados del modelo
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: -0.5 }}>
            {fmt(date)}
          </div>
          {data && (
            <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
              {data.games.length} partidos&nbsp;·&nbsp;
              <span style={{ color: "#166534" }}>
                {data.games.filter(g => g.pred.ouCorrect).length} O/U HIT
              </span>
              {" / "}
              <span style={{ color: "#991b1b" }}>
                {data.games.filter(g => !g.pred.ouCorrect).length} O/U MISS
              </span>
              {data.cached && <span style={{ marginLeft: 8 }}>· cached</span>}
            </div>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <NavBtn onClick={() => setDate(d => addDays(d, -1))}>Prev</NavBtn>
          <input
            type="date"
            value={date}
            onChange={e => setDate(e.target.value)}
            style={inputStyle}
          />
          <NavBtn onClick={() => setDate(d => addDays(d, 1))}>Next</NavBtn>
        </div>
      </header>

      {/* ── Tabs ── */}
      {setPage && (
        <div style={{ display: "flex", borderBottom: "2px solid #e5e7eb", marginBottom: 20 }}>
          {[
            { key: "picks",   label: "Picks de hoy" },
            { key: "parlays", label: "Parlays" },
            { key: "results", label: "Resultados del modelo" },
          ].map(t => (
            <button
              key={t.key}
              onClick={() => setPage(t.key)}
              style={{
                padding: "8px 22px", border: "none",
                borderBottom: t.key === "results" ? "3px solid #111" : "3px solid transparent",
                background: "none", cursor: "pointer", fontSize: 12,
                fontWeight: t.key === "results" ? 700 : 400,
                color: t.key === "results" ? "#111" : "#888",
                fontFamily: "inherit", letterSpacing: 0.3, marginBottom: -2,
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      {/* ── 30-day accuracy panel ── */}
      {s && s.games > 0 && (
        <AccuracyPanel summary={s} />
      )}

      {/* ── States ── */}
      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: "#888", fontSize: 13 }}>
          Cargando resultados...
        </div>
      )}
      {error && (
        <div style={{ textAlign: "center", padding: 40, color: "#991b1b", fontSize: 13 }}>
          Error: {error}
        </div>
      )}

      {!loading && data && data.games.length === 0 && (
        <div style={{ textAlign: "center", padding: 60, color: "#888", fontSize: 13 }}>
          No hay partidos terminados para esta fecha.
        </div>
      )}

      {/* ── Result cards ── */}
      {!loading && data && data.games.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {data.games.map(g => (
            <ResultCard key={g.gamePk} game={g} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── 30-day accuracy summary panel ──────────────────────────────────────── */
function AccuracyPanel({ summary: s }) {
  return (
    <div style={{
      border: "1px solid #e0e0e0",
      background: "#fafafa",
      padding: "14px 20px",
      marginBottom: 20,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
        textTransform: "uppercase", color: "#555", marginBottom: 10,
      }}>
        Precisión últimos 30 días &nbsp;·&nbsp;
        {fmtShort(s.periodStart)} – {fmtShort(s.periodEnd)} &nbsp;·&nbsp;
        {s.games} partidos
      </div>

      <div style={{ display: "flex", gap: 32, flexWrap: "wrap" }}>
        <AccuracyRow
          label="O/U"
          correct={s.ouCorrect}
          total={s.games}
          pct={s.ouPct}
          streak={s.ouStreak}
        />
        <AccuracyRow
          label="WIN"
          correct={s.winCorrect}
          total={s.games}
          pct={s.winPct}
          streak={s.winStreak}
        />
      </div>
    </div>
  );
}

function AccuracyRow({ label, correct, total, pct, streak }) {
  const pctNum  = pct ?? 0;
  const barPct  = Math.max(0, Math.min(100, pctNum));
  const good    = pctNum >= 54;
  const color   = pctNum >= 57 ? "#166534" : pctNum >= 52 ? "#374151" : "#991b1b";
  const streakAbs = Math.abs(streak ?? 0);
  const streakPos = (streak ?? 0) > 0;
  const streakColor = streakPos ? "#166534" : "#991b1b";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 260 }}>
      <span style={{
        fontSize: 10, fontWeight: 800, letterSpacing: 1,
        color: "#444", width: 30, flexShrink: 0,
      }}>
        {label}
      </span>

      {/* Progress bar */}
      <div style={{
        flex: 1, height: 8, background: "#e5e7eb",
        position: "relative", maxWidth: 180,
      }}>
        <div style={{
          position: "absolute", left: 0, top: 0, bottom: 0,
          width: `${barPct}%`,
          background: color,
          transition: "width 0.3s",
        }} />
        {/* 50% marker */}
        <div style={{
          position: "absolute", left: "50%", top: -2, bottom: -2,
          width: 1, background: "#9ca3af",
        }} />
      </div>

      <span style={{ fontSize: 14, fontWeight: 900, color, minWidth: 46 }}>
        {pct}%
      </span>

      <span style={{ fontSize: 11, color: "#888" }}>
        {correct}/{total}
      </span>

      {streakAbs > 0 && (
        <span style={{
          fontSize: 10, fontWeight: 700,
          color: streakColor,
          background: streakColor + "15",
          border: `1px solid ${streakColor}40`,
          padding: "1px 7px",
          letterSpacing: 0.3,
        }}>
          {streakPos ? "+" : ""}{streak} {streakPos ? "✓" : "✗"}
        </span>
      )}
    </div>
  );
}

/* ── Individual result card ─────────────────────────────────────────────── */
function ResultCard({ game }) {
  const { away, home, pred, date } = game;
  const bothCorrect = pred.ouCorrect && pred.winCorrect;
  const noneCorrect = !pred.ouCorrect && !pred.winCorrect;

  const borderColor = bothCorrect ? "#166534"
                    : noneCorrect ? "#991b1b"
                    : "#b45309";

  const awayWin = away.score > home.score;

  // Who was predicted to win?
  const predHomeWin = pred.predictedWinner === "home";
  const actualHomeWin = home.score > away.score;

  return (
    <div style={{
      background: "#fff",
      border: "1px solid #e0e0e0",
      borderLeft: `4px solid ${borderColor}`,
      marginBottom: 1,
    }}>
      <div style={{
        display: "grid",
        gridTemplateColumns: "1fr 160px 1fr",
        alignItems: "center",
        padding: "14px 16px",
        gap: 8,
      }}>
        {/* Away team */}
        <TeamResult team={away} align="left" won={awayWin} winProb={pred.awayWinProb} />

        {/* Center: score + prediction results */}
        <div style={{ textAlign: "center" }}>
          {/* Score */}
          <div style={{
            fontSize: 28, fontWeight: 900, letterSpacing: -1,
            color: "#111", lineHeight: 1, marginBottom: 4,
          }}>
            <span style={{ color: awayWin ? "#166534" : "#9ca3af" }}>{away.score}</span>
            <span style={{ color: "#ccc", fontSize: 18, margin: "0 6px" }}>-</span>
            <span style={{ color: !awayWin ? "#166534" : "#9ca3af" }}>{home.score}</span>
          </div>
          <div style={{ fontSize: 9, color: "#aaa", letterSpacing: 1, marginBottom: 10 }}>FINAL</div>

          {/* O/U result */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: 5, marginBottom: 5,
          }}>
            <span style={{
              fontSize: 9, fontWeight: 800, letterSpacing: 0.5, color: "#666",
            }}>O/U</span>
            <span style={{
              fontSize: 11, fontWeight: 700,
              color: pred.rec === "OVER" ? "#b45309" : "#1d4ed8",
            }}>
              {pred.rec} {pred.line}
            </span>
            <span style={{ fontSize: 10, color: "#999" }}>→ {pred.actualTotal}</span>
            <HitMiss correct={pred.ouCorrect} />
          </div>

          {/* WIN result */}
          <div style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: 5,
          }}>
            <span style={{ fontSize: 9, fontWeight: 800, letterSpacing: 0.5, color: "#666" }}>WIN</span>
            <span style={{ fontSize: 10, fontWeight: 700, color: "#374151" }}>
              {predHomeWin ? home.abbr : away.abbr}
            </span>
            <span style={{ fontSize: 9, color: "#aaa" }}>
              {predHomeWin ? pred.homeWinProb : pred.awayWinProb}%
            </span>
            <HitMiss correct={pred.winCorrect} />
          </div>
        </div>

        {/* Home team */}
        <TeamResult team={home} align="right" won={!awayWin} winProb={pred.homeWinProb} />
      </div>

      {/* Bottom strip — predicted vs actual total */}
      <div style={{
        borderTop: "1px solid #f3f4f6",
        padding: "5px 16px",
        display: "flex", alignItems: "center", gap: 16,
        fontSize: 10, color: "#888",
      }}>
        <span>Predicción: <strong>{pred.predictedTotal}</strong> runs</span>
        <span>Real: <strong style={{ color: pred.actualTotal > pred.line ? "#b45309" : "#1d4ed8" }}>
          {pred.actualTotal}
        </strong></span>
        <span style={{ color: "#bbb" }}>Línea: {pred.line}</span>
        <span style={{ color: "#bbb" }}>Error: {Math.abs(pred.predictedTotal - pred.actualTotal).toFixed(1)} runs</span>
      </div>
    </div>
  );
}

function HitMiss({ correct }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", justifyContent: "center",
      width: 16, height: 16,
      background: correct ? "#166534" : "#dc2626",
      color: "#fff",
      fontSize: 9, fontWeight: 900,
      borderRadius: "50%",
    }}>
      {correct ? "✓" : "✗"}
    </span>
  );
}

function TeamResult({ team, align, won, winProb }) {
  const isLeft = align === "left";
  const probColor = winProb >= 58 ? "#166534" : winProb >= 52 ? "#374151" : "#991b1b";

  return (
    <div style={{
      display: "flex",
      flexDirection: isLeft ? "row" : "row-reverse",
      alignItems: "center",
      gap: 12,
    }}>
      <img
        src={team.logoUrl}
        alt={team.abbr}
        width={42} height={42}
        style={{ objectFit: "contain", flexShrink: 0, opacity: won ? 1 : 0.45 }}
        onError={e => { e.target.style.display = "none"; }}
      />
      <div style={{ textAlign: isLeft ? "left" : "right" }}>
        <div style={{
          fontWeight: won ? 800 : 500,
          fontSize: 15,
          color: won ? "#111" : "#999",
        }}>
          {team.name}
        </div>
        <div style={{
          fontSize: 12, fontWeight: 700, marginTop: 2,
          color: probColor,
        }}>
          {winProb}% <span style={{ fontSize: 9, fontWeight: 500, color: "#aaa" }}>WIN pred</span>
        </div>
      </div>
    </div>
  );
}

function NavBtn({ onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding: "6px 14px",
      border: "1px solid #bbb",
      background: "#fff",
      cursor: "pointer",
      fontSize: 12,
      fontFamily: "inherit",
    }}>
      {children}
    </button>
  );
}

const inputStyle = {
  padding: "6px 10px",
  border: "1px solid #bbb",
  background: "#fff",
  fontSize: 12,
  fontFamily: "inherit",
  cursor: "pointer",
};
