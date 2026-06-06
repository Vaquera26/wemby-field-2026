import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell, Line, ComposedChart,
} from "recharts";

const HOT_COLOR   = "#166534";
const COLD_COLOR  = "#d1d5db";
const WARM_COLOR  = "#6b7280";

function barColor(proj) {
  if (proj >= 1.3) return HOT_COLOR;
  if (proj >= 0.7) return WARM_COLOR;
  return COLD_COLOR;
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div style={{
      background: "#fff",
      border: "1px solid #ddd",
      padding: "8px 12px",
      fontSize: 11,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>Inning {label}</div>
      <div>Projected: <strong>{d.proj.toFixed(3)}</strong></div>
      <div style={{ color: "#888" }}>Season avg: {d.seasonAvg?.toFixed(3) ?? "—"}</div>
      {d.weekAvg != null && <div style={{ color: "#888" }}>Week avg: {d.weekAvg.toFixed(3)}</div>}
      {d.h2hAvg != null && <div style={{ color: "#888" }}>H2H avg: {d.h2hAvg.toFixed(3)}</div>}
    </div>
  );
};

export default function InningChart({ innings }) {
  if (!innings?.length) return null;

  const data = innings.map(d => ({
    ...d,
    name: `I${d.inning}`,
  }));

  const maxVal = Math.max(...data.map(d => Math.max(
    d.proj,
    d.seasonAvg ?? 0,
    d.weekAvg ?? 0,
    d.h2hAvg ?? 0,
  ))) + 0.3;

  return (
    <div style={{ height: 180 }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis
            dataKey="name"
            tick={{ fontSize: 11, fill: "#666" }}
            axisLine={{ stroke: "#ccc" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#999" }}
            axisLine={false}
            tickLine={false}
            domain={[0, maxVal]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "#f5f5f5" }} />

          {/* Season avg line */}
          <Line
            type="monotone"
            dataKey="seasonAvg"
            stroke="#9ca3af"
            strokeWidth={1}
            dot={false}
            strokeDasharray="4 2"
          />

          {/* Week avg line */}
          <Line
            type="monotone"
            dataKey="weekAvg"
            stroke="#f59e0b"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="2 2"
          />

          {/* H2H line */}
          <Line
            type="monotone"
            dataKey="h2hAvg"
            stroke="#6366f1"
            strokeWidth={1.5}
            dot={false}
          />

          {/* Projection bars */}
          <Bar dataKey="proj" maxBarSize={28} radius={[2, 2, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={barColor(d.proj)} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div style={{
        display: "flex", gap: 16, marginTop: 4,
        fontSize: 10, color: "#888",
      }}>
        <LegendItem color={HOT_COLOR} label="Hot inning (>=1.3)" />
        <LegendItem color={WARM_COLOR} label="Neutral" />
        <LegendItem color={COLD_COLOR} label="Cold (<=0.7)" />
        <LegendItem color="#9ca3af" line label="Season avg" />
        <LegendItem color="#f59e0b" line label="Week avg" />
        <LegendItem color="#6366f1" line label="H2H avg" />
      </div>
    </div>
  );
}

function LegendItem({ color, label, line }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
      {line
        ? <span style={{ display: "inline-block", width: 18, height: 2, background: color }} />
        : <span style={{ display: "inline-block", width: 10, height: 10, background: color }} />
      }
      {label}
    </span>
  );
}
