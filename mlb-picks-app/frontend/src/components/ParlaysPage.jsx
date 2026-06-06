import { useState, useEffect, useCallback } from "react";

const today = () => new Date().toISOString().slice(0, 10);

const fmt = (iso) => {
  const d = new Date(iso + "T12:00:00");
  return d.toLocaleDateString("es-MX", {
    weekday: "long", year: "numeric", month: "long", day: "numeric",
  });
};

const addDays = (iso, n) => {
  const d = new Date(iso + "T12:00:00");
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
};

export default function ParlaysPage({ setPage }) {
  const [date, setDate]       = useState(today());
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const load = useCallback(async (d) => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`/api/parlays?date=${d}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData(await res.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(date); }, [date, load]);

  return (
    <div style={{ maxWidth: 1320, margin: "0 auto", padding: "0 16px 48px" }}>

      {/* ── Tabs ── */}
      {setPage && <TabBar page="parlays" setPage={setPage} />}

      {/* ── Header ── */}
      <header style={{
        borderBottom: "2px solid #111",
        padding: "20px 0 12px", marginBottom: 24,
        display: "flex", alignItems: "flex-end",
        justifyContent: "space-between", flexWrap: "wrap", gap: 12,
      }}>
        <div>
          <div style={{ fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: "#666", marginBottom: 2 }}>
            MLB · Parlays recomendados
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: -0.5 }}>
            {fmt(date)}
          </div>
          {data && (
            <div style={{ fontSize: 11, color: "#888", marginTop: 4 }}>
              {data.totalGames} partidos · {data.parlays?.length ?? 0} parlays generados
              <span style={{ marginLeft: 10 }}>
                · Precisión 30d: O/U {data.histOuAcc}% · WIN {data.histWinAcc}%
              </span>
            </div>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <NavBtn onClick={() => setDate(d => addDays(d, -1))}>Prev</NavBtn>
          <input type="date" value={date} onChange={e => setDate(e.target.value)} style={inputStyle} />
          <NavBtn onClick={() => setDate(d => addDays(d, 1))}>Next</NavBtn>
          <NavBtn onClick={() => load(date)}>↻</NavBtn>
        </div>
      </header>

      {/* ── Contrarian warning banner ── */}
      {data?.contrarianMode && (
        <div style={{
          background: "#fffbeb", border: "1px solid #f59e0b",
          borderLeft: "4px solid #d97706",
          padding: "10px 16px", marginBottom: 16,
          fontSize: 12, color: "#92400e", lineHeight: 1.6,
        }}>
          <strong>Modelo O/U en racha negativa ({data.histOuAcc}% ultimos 30 dias)</strong>
          {" — "}el modelo acierta menos que el azar en totales. Los parlays
          {" "}<strong>O/U invertido</strong> y <strong>Stack invertido</strong> apuestan
          lo contrario del modelo y han tenido mejor rendimiento historico en este contexto.
        </div>
      )}

      {/* ── How parlays work ── */}
      <div style={{
        fontSize: 11, color: "#666",
        borderBottom: "1px solid #ddd", paddingBottom: 10, marginBottom: 24,
        display: "flex", gap: 24, flexWrap: "wrap",
      }}>
        <span>Cómo funciona:</span>
        <span><b>O/U</b> — combinaciones de totales</span>
        <span><b>Ganadores</b> — equipo favorito en cada partido</span>
        <span><b>Game Stack</b> — OVER + ganador mismo partido (correlacionado)</span>
        <span><b>Mixto</b> — los mejores picks del día</span>
        <span style={{ marginLeft: "auto", color: "#aaa" }}>
          Pago típico: 2-leg 2.64x · 3-leg 6x · 4-leg 12x · 5-leg 24x
        </span>
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: "#888", fontSize: 13 }}>
          Generando parlays...
        </div>
      )}
      {error && (
        <div style={{ textAlign: "center", padding: 40, color: "#991b1b", fontSize: 13 }}>
          Error: {error}
        </div>
      )}

      {!loading && data && data.parlays?.length === 0 && (
        <div style={{ textAlign: "center", padding: 60, color: "#888", fontSize: 13 }}>
          No hay partidos programados para esta fecha.
        </div>
      )}

      {/* ── Parlay grid ── */}
      {!loading && data && data.parlays?.length > 0 && (
        <>
          {/* Top parlay highlighted */}
          {data.parlays[0] && (
            <div style={{ marginBottom: 20 }}>
              <SectionLabel>Mejor parlay del día</SectionLabel>
              <ParlayCard parlay={data.parlays[0]} featured histOuAcc={data.histOuAcc} histWinAcc={data.histWinAcc} />
            </div>
          )}

          {/* Grid of remaining parlays */}
          <SectionLabel>Todos los parlays recomendados</SectionLabel>
          <div style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
            gap: 12,
          }}>
            {data.parlays.map((p, i) => (
              <ParlayCard key={p.id} parlay={p} histOuAcc={data.histOuAcc} histWinAcc={data.histWinAcc} />
            ))}
          </div>

          {/* Disclaimer */}
          <div style={{
            marginTop: 32, padding: "12px 16px",
            background: "#f9fafb", border: "1px solid #e5e7eb",
            fontSize: 10, color: "#9ca3af", lineHeight: 1.6,
          }}>
            <b>Nota metodológica:</b> Los parlays se construyen con los modelos ML entrenados sobre 2025+2026.
            La probabilidad combinada asume independencia entre legs de distintos partidos.
            Game Stacks incluyen un ajuste de +10% por correlación positiva (OVER + favorito del mismo partido).
            La "Tasa histórica estimada" usa la precisión real de los modelos en los últimos 30 días
            ajustada por el nivel de confianza de cada pick individual.
            Los resultados pasados no garantizan resultados futuros. Apostá con responsabilidad.
          </div>
        </>
      )}
    </div>
  );
}

/* ── Parlay card ─────────────────────────────────────────────────────────── */
function ParlayCard({ parlay: p, featured = false }) {
  const evPositive  = p.ev > 0;
  const borderColor = p.contrarian  ? "#7c3aed"
                    : evPositive    ? "#166534"
                    : p.histHitPct >= 25 ? "#b45309" : "#6b7280";

  return (
    <div style={{
      border: "1px solid #e0e0e0",
      borderLeft: `4px solid ${borderColor}`,
      background: "#fff",
      padding: featured ? "18px 20px" : "14px 16px",
    }}>
      {/* Header row */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        marginBottom: 10,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: 0.5, color: "#374151" }}>
              {p.category}
            </div>
            {p.contrarian && (
              <div style={{ fontSize: 9, color: "#7c3aed", fontWeight: 700, letterSpacing: 0.3 }}>
                CONTRARIO AL MODELO
              </div>
            )}
            {p.correlated && !p.contrarian && (
              <div style={{ fontSize: 9, color: "#b45309", fontWeight: 600, letterSpacing: 0.3 }}>
                CORRELACIONADO
              </div>
            )}
          </div>
        </div>

        {/* Combined prob */}
        <div style={{ textAlign: "right" }}>
          <div style={{
            fontSize: featured ? 28 : 22,
            fontWeight: 900, letterSpacing: -1,
            color: borderColor, lineHeight: 1,
          }}>
            {p.combinedProb}%
          </div>
          <div style={{ fontSize: 9, color: "#aaa" }}>prob combinada</div>
        </div>
      </div>

      {/* Legs */}
      <div style={{
        borderTop: "1px solid #f3f4f6",
        borderBottom: "1px solid #f3f4f6",
        padding: "8px 0", marginBottom: 10,
      }}>
        {p.legs.map((leg, i) => (
          <LegRow key={i} leg={leg} />
        ))}
      </div>

      {/* Stats row */}
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <Stat label="Tasa hist. est." value={`${p.histHitPct}%`}
              color={p.histHitPct >= 35 ? "#166534" : p.histHitPct >= 25 ? "#b45309" : "#991b1b"} />
        <Stat label="Pago estimado" value={`${p.payoutEstimate}x`} color="#374151" />
        <Stat label="EV estimado"
              value={`${p.ev > 0 ? "+" : ""}${p.ev}%`}
              color={evPositive ? "#166534" : "#991b1b"} />
        <div style={{
          marginLeft: "auto",
          fontSize: 9, color: "#9ca3af",
          maxWidth: 160, textAlign: "right", lineHeight: 1.4,
        }}>
          {p.note}
        </div>
      </div>
    </div>
  );
}

function LegRow({ leg }) {
  const isOU    = leg.type === "ou" || leg.type === "ou_inv";
  const isInv   = leg.type === "ou_inv";
  const probColor = leg.prob >= 60 ? "#166534" : leg.prob >= 55 ? "#b45309" : "#374151";
  const barW = Math.max(0, Math.min(100, (leg.prob - 50) * 5));

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 8,
      padding: "4px 0",
    }}>
      {/* Type badge */}
      <span style={{
        fontSize: 8, fontWeight: 800, letterSpacing: 1,
        padding: "1px 5px",
        background: isInv ? "#f5f3ff" : isOU ? "#eff6ff" : "#f0fdf4",
        color:      isInv ? "#7c3aed" : isOU ? "#1d4ed8" : "#166534",
        border:     `1px solid ${isInv ? "#ddd6fe" : isOU ? "#bfdbfe" : "#bbf7d0"}`,
        flexShrink: 0, width: 24, textAlign: "center",
      }}>
        {isInv ? "INV" : isOU ? "O/U" : "WIN"}
      </span>

      {/* Team logo (win picks) */}
      {!isOU && leg.pickLogoUrl && (
        <img src={leg.pickLogoUrl} alt="" width={18} height={18}
             style={{ objectFit: "contain", flexShrink: 0 }}
             onError={e => { e.target.style.display = "none"; }} />
      )}

      {/* Pick text */}
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, fontWeight: 700, color: "#111", letterSpacing: 0.2 }}>
          {leg.pick}
        </div>
        <div style={{ fontSize: 9, color: "#aaa" }}>{leg.game}</div>
      </div>

      {/* Probability bar */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ width: 50, height: 4, background: "#f3f4f6", position: "relative" }}>
          <div style={{
            position: "absolute", left: 0, top: 0, bottom: 0,
            width: `${barW}%`, background: probColor,
          }} />
        </div>
        <span style={{ fontSize: 11, fontWeight: 800, color: probColor, minWidth: 36, textAlign: "right" }}>
          {leg.prob}%
        </span>
      </div>
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 13, fontWeight: 900, color: color ?? "#374151" }}>{value}</div>
      <div style={{ fontSize: 8, color: "#aaa", letterSpacing: 0.5, textTransform: "uppercase" }}>{label}</div>
    </div>
  );
}

/* ── Tab bar ─────────────────────────────────────────────────────────────── */
function TabBar({ page, setPage }) {
  const tabs = [
    { key: "picks",   label: "Picks de hoy" },
    { key: "parlays", label: "Parlays" },
    { key: "results", label: "Resultados" },
  ];
  return (
    <div style={{ display: "flex", borderBottom: "2px solid #e5e7eb", marginBottom: 20 }}>
      {tabs.map(t => (
        <button key={t.key} onClick={() => setPage(t.key)} style={{
          padding: "8px 22px", border: "none",
          borderBottom: t.key === page ? "3px solid #111" : "3px solid transparent",
          background: "none", cursor: "pointer", fontSize: 12,
          fontWeight: t.key === page ? 700 : 400,
          color: t.key === page ? "#111" : "#888",
          fontFamily: "inherit", letterSpacing: 0.3, marginBottom: -2,
        }}>
          {t.label}
        </button>
      ))}
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
      textTransform: "uppercase", color: "#555", marginBottom: 10,
    }}>
      {children}
    </div>
  );
}

function NavBtn({ onClick, children }) {
  return (
    <button onClick={onClick} style={{
      padding: "6px 14px", border: "1px solid #bbb",
      background: "#fff", cursor: "pointer", fontSize: 12, fontFamily: "inherit",
    }}>
      {children}
    </button>
  );
}

const inputStyle = {
  padding: "6px 10px", border: "1px solid #bbb",
  background: "#fff", fontSize: 12, fontFamily: "inherit", cursor: "pointer",
};
