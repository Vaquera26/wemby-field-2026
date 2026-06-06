/**
 * Shows ML vs formula breakdown for one game.
 * Light, plain, no decorative elements.
 */

const NUM = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));

export default function MLPanel({ proj, mlMeta }) {
  if (!proj.mlTotal) {
    return <div style={{ fontSize: 12, color: "#888" }}>ML model not available.</div>;
  }

  const agree = proj.agreement;
  const formulaRec = proj.formulaRec;
  const mlRec      = proj.mlRec;
  const finalRec   = proj.rec;

  return (
    <div>
      {/* Agreement banner */}
      <div style={{
        padding: "8px 12px",
        background: agree ? "#f0fdf4" : "#fef9ec",
        border: `1px solid ${agree ? "#86efac" : "#fde68a"}`,
        fontSize: 12,
        marginBottom: 16,
        color: agree ? "#166534" : "#92400e",
      }}>
        {agree
          ? `Both models agree: ${finalRec} ${proj.line}`
          : `Models disagree — Formula: ${formulaRec}  /  ML: ${mlRec}  →  Ensemble: ${finalRec} ${proj.line}`
        }
      </div>

      {/* Side-by-side comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16, marginBottom: 20 }}>
        <ScoreCard
          label="Formula (weighted avg)"
          total={proj.formulaTotal}
          rec={formulaRec}
          line={proj.line}
          sublabel="Season · Last10 · Week · H2H"
        />
        <ScoreCard
          label="ML — XGBoost"
          total={proj.mlXgb}
          rec={proj.mlXgb >= proj.line ? "OVER" : "UNDER"}
          line={proj.line}
          sublabel="Trained on 2025+2026 games"
        />
        <ScoreCard
          label="ML — Ridge"
          total={proj.mlRidge}
          rec={proj.mlRidge >= proj.line ? "OVER" : "UNDER"}
          line={proj.line}
          sublabel="Linear baseline"
        />
      </div>

      {/* Ensemble explanation */}
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12, marginBottom: 20 }}>
        <tbody>
          <Row label="Formula total"    value={NUM(proj.formulaTotal)} />
          <Row label="ML ensemble total" value={NUM(proj.mlTotal)} />
          <Row label="Final (55% ML + 45% formula)" value={NUM(proj.total)} bold />
          <Row label="vs line"          value={`${proj.diff > 0 ? "+" : ""}${NUM(proj.diff)}  →  ${proj.rec} ${proj.line}`} />
        </tbody>
      </table>

      {/* Model accuracy */}
      {mlMeta && (
        <div style={{ marginBottom: 16 }}>
          <SL>Model accuracy (cross-validation, 5-fold)</SL>
          <div style={{ fontSize: 12, color: "#444" }}>
            MAE: <strong>{NUM(mlMeta.cvMae, 3)} runs</strong>
            <span style={{ color: "#888", marginLeft: 16 }}>
              Trained on {mlMeta.trainN?.toLocaleString()} games
              (2025 full season + 2026 YTD)
            </span>
          </div>
        </div>
      )}

      {/* Feature importances */}
      {mlMeta?.featureImportances && (
        <div>
          <SL>Top feature importances (XGBoost)</SL>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {Object.entries(mlMeta.featureImportances)
              .slice(0, 10)
              .map(([name, val]) => (
                <div key={name} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
                  <span style={{ width: 160, color: "#555", fontFamily: "monospace" }}>{name}</span>
                  <div style={{
                    width: Math.round(val * 600),
                    height: 8,
                    background: "#374151",
                    flexShrink: 0,
                  }} />
                  <span style={{ color: "#888", fontFamily: "monospace" }}>{(val * 100).toFixed(1)}%</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ScoreCard({ label, total, rec, line, sublabel }) {
  const isOver = rec === "OVER";
  const diff = total - line;
  return (
    <div style={{
      border: "1px solid #e0e0e0",
      padding: "10px 14px",
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: "uppercase", color: "#666", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, fontFamily: "monospace", marginBottom: 4 }}>
        {Number(total).toFixed(1)}
      </div>
      <div style={{
        fontSize: 12, fontWeight: 700,
        color: isOver ? "#166534" : "#991b1b",
        marginBottom: 4,
      }}>
        {rec} {line}
        <span style={{ fontWeight: 400, color: "#888", marginLeft: 6 }}>
          ({diff > 0 ? "+" : ""}{diff.toFixed(2)})
        </span>
      </div>
      <div style={{ fontSize: 10, color: "#aaa" }}>{sublabel}</div>
    </div>
  );
}

function Row({ label, value, bold }) {
  return (
    <tr style={{ borderBottom: "1px solid #f0f0f0" }}>
      <td style={{ padding: "4px 8px 4px 0", color: "#666", width: "60%" }}>{label}</td>
      <td style={{ padding: "4px 0", fontFamily: "monospace", fontWeight: bold ? 700 : 400 }}>{value}</td>
    </tr>
  );
}

function SL({ children }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
      textTransform: "uppercase", color: "#555", marginBottom: 8,
    }}>
      {children}
    </div>
  );
}
