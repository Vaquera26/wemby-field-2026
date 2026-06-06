import { useState, useEffect, useCallback } from "react";
import GameCard from "./components/GameCard.jsx";
import ResultsPage from "./components/ResultsPage.jsx";
import ParlaysPage from "./components/ParlaysPage.jsx";

const today = () => new Date().toISOString().slice(0, 10);

const fmt = (iso) => {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
};

const addDays = (iso, n) => {
  const d = new Date(iso + "T12:00:00");
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
};

export default function App() {
  const [page, setPage]   = useState("picks");   // "picks" | "results" | "parlays"
  const [date, setDate]   = useState(today());
  const [data, setData]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async (d, refresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const url = refresh ? `/api/picks/refresh?date=${d}` : `/api/picks?date=${d}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(date); }, [date, load]);

  const overCount  = data?.games?.filter(g => g.projection.rec === "OVER").length  ?? 0;
  const underCount = data?.games?.filter(g => g.projection.rec === "UNDER").length ?? 0;

  if (page === "results") {
    return (
      <div style={{ maxWidth: 1320, margin: "0 auto", padding: "0 16px 48px" }}>
        <ResultsPage setPage={setPage} />
      </div>
    );
  }

  if (page === "parlays") {
    return <ParlaysPage setPage={setPage} />;
  }

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
            MLB Over / Under
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: -0.5 }}>
            {fmt(date)}
          </div>
          {data && (
            <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
              {data.games.length} games &nbsp;·&nbsp;
              <span style={{ color: "#166534" }}>{overCount} OVER</span>
              {" / "}
              <span style={{ color: "#991b1b" }}>{underCount} UNDER</span>
              {data.mlMeta?.trained && (
                <span style={{ marginLeft: 10 }}>
                  · ML trained on {data.mlMeta.trainN?.toLocaleString()} games
                  · CV MAE {Number(data.mlMeta.cvMae).toFixed(2)} runs
                </span>
              )}
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
          <NavBtn onClick={() => load(date, true)}>Refresh</NavBtn>
        </div>
      </header>

      <TabBar page={page} setPage={setPage} />

      {/* ── Algorithm key ── */}
      <div style={{
        fontSize: 11, color: "#666",
        borderBottom: "1px solid #ddd",
        paddingBottom: 10, marginBottom: 24,
        display: "flex", gap: 24, flexWrap: "wrap",
      }}>
        <span>Model weights:</span>
        <span>Season 20%</span>
        <span>Last 10 25%</span>
        <span>This week 30%</span>
        <span>H2H 10%</span>
        <span>2025 season 10%</span>
        <span>Defense 5%</span>
        <span style={{ marginLeft: "auto" }}>Win streak modifier +/-3% per game (cap 15%)</span>
      </div>

      {/* ── State ── */}
      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: "#888", fontSize: 13 }}>
          Loading...
        </div>
      )}
      {error && (
        <div style={{ textAlign: "center", padding: 40, color: "#991b1b", fontSize: 13 }}>
          Error: {error}
        </div>
      )}

      {/* ── Games ── */}
      {!loading && data && data.games.length === 0 && (
        <div style={{ textAlign: "center", padding: 60, color: "#888", fontSize: 13 }}>
          No games scheduled for this date.
        </div>
      )}

      {!loading && data && (
        <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
          {data.games.map(g => (
            <GameCard key={g.gamePk} game={g} mlMeta={data.mlMeta} />
          ))}
        </div>
      )}

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

function TabBar({ page, setPage }) {
  const tabs = [
    { key: "picks",   label: "Picks de hoy" },
    { key: "parlays", label: "Parlays" },
    { key: "results", label: "Resultados del modelo" },
  ];
  return (
    <div style={{
      display: "flex",
      gap: 0,
      borderBottom: "2px solid #e5e7eb",
      marginBottom: 20,
    }}>
      {tabs.map(t => (
        <button
          key={t.key}
          onClick={() => setPage(t.key)}
          style={{
            padding: "8px 22px",
            border: "none",
            borderBottom: page === t.key ? "3px solid #111" : "3px solid transparent",
            background: "none",
            cursor: "pointer",
            fontSize: 12,
            fontWeight: page === t.key ? 700 : 400,
            color: page === t.key ? "#111" : "#888",
            fontFamily: "inherit",
            letterSpacing: 0.3,
            marginBottom: -2,
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
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
