import {
  ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer, Cell,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: "#fff", border: "1px solid #ddd",
      padding: "7px 11px", fontSize: 11,
    }}>
      <div style={{ fontWeight: 700, marginBottom: 2 }}>{d?.date}</div>
      <div>vs {d?.opp}</div>
      <div style={{ marginTop: 3 }}>
        <span style={{ color: "#166534" }}>Scored: {d?.scored}</span>
        {"  "}
        <span style={{ color: "#991b1b" }}>Allowed: {d?.allowed}</span>
      </div>
    </div>
  );
};

export default function RunsChart({ data, teamName, avgScored }) {
  if (!data?.length) return <div style={{ fontSize: 11, color: "#aaa" }}>No data</div>;

  const chartData = data.map((g, i) => ({
    ...g,
    idx: i + 1,
    label: g.date.slice(5),   // "MM-DD"
  }));

  return (
    <div>
      <div style={{ height: 160 }}>
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#ececec" vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 9, fill: "#aaa" }}
              axisLine={{ stroke: "#ddd" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fontSize: 9, fill: "#aaa" }}
              axisLine={false}
              tickLine={false}
              domain={[0, "auto"]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "#f9f9f9" }} />

            {/* Average line */}
            {avgScored != null && (
              <ReferenceLine
                y={avgScored}
                stroke="#9ca3af"
                strokeDasharray="4 2"
                label={{
                  value: `avg ${Number(avgScored).toFixed(1)}`,
                  position: "insideTopRight",
                  fontSize: 9,
                  fill: "#9ca3af",
                }}
              />
            )}

            {/* Allowed (light red area) */}
            <Bar dataKey="allowed" maxBarSize={14} radius={[1, 1, 0, 0]} opacity={0.35}>
              {chartData.map((d, i) => (
                <Cell key={i} fill="#991b1b" />
              ))}
            </Bar>

            {/* Scored (main bars) */}
            <Bar dataKey="scored" maxBarSize={14} radius={[2, 2, 0, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={d.result === "W" ? "#166534" : "#dc2626"} />
              ))}
            </Bar>

            {/* Runs scored line */}
            <Line
              type="monotone"
              dataKey="scored"
              stroke="#111"
              strokeWidth={1.2}
              dot={{ r: 2, fill: "#111" }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div style={{ display: "flex", gap: 16, fontSize: 10, color: "#888", marginTop: 3 }}>
        <LegendItem color="#166534" label="Win (scored)" />
        <LegendItem color="#dc2626" label="Loss (scored)" />
        <LegendItem color="#991b1b" opacity={0.35} label="Allowed" />
        <LegendItem color="#9ca3af" line label="Season avg" />
      </div>
    </div>
  );
}

function LegendItem({ color, label, line, opacity = 1 }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
      {line
        ? <span style={{ display: "inline-block", width: 18, height: 2, background: color }} />
        : <span style={{ display: "inline-block", width: 10, height: 10, background: color, opacity }} />
      }
      {label}
    </span>
  );
}
