import { useState } from "react";
import InningChart from "./InningChart.jsx";
import RunsChart from "./RunsChart.jsx";
import MLPanel from "./MLPanel.jsx";

const NUM = (v, dec = 2) => v == null ? "—" : Number(v).toFixed(dec);
const PCT = (v) => v == null ? "—" : (Number(v) * 100).toFixed(1) + "%";

// Aproximación de erf para calcular probabilidad normal
function calcProb(absDiff, agreement) {
  const z = absDiff / 3.5;
  const t = 1 / (1 + 0.3275911 * z);
  const erf = 1 - (0.254829592*t - 0.284496736*t**2 + 1.421413741*t**3
                   - 1.453152027*t**4 + 1.061405429*t**5) * Math.exp(-z*z);
  let p = 50 + 50 * erf;
  if (agreement) p = Math.min(p + 2, 99);
  else           p = Math.max(p - 2, 50);
  return Math.round(p * 10) / 10;
}

export default function GameCard({ game, mlMeta }) {
  const [open, setOpen]     = useState(false);
  const [tab, setTab]       = useState("stats");
  const [adjLine, setAdjLine] = useState(null);   // null = usa la línea del modelo

  const { away, home, projection: proj, h2hGames, venue, gameTime, status } = game;

  // Valores reactivos al slider
  const activeLine = adjLine ?? proj.line;
  const activeDiff = proj.total - activeLine;
  const activeRec  = activeDiff >= 0 ? "OVER" : "UNDER";
  const activeProb = calcProb(Math.abs(activeDiff), proj.agreement);
  const isOver     = activeRec === "OVER";

  const timeStr = gameTime
    ? new Date(gameTime).toLocaleTimeString("en-US", {
        hour: "numeric", minute: "2-digit", timeZoneName: "short",
      })
    : status;

  return (
    <div style={{
      background: "#fff",
      border: "1px solid #e0e0e0",
      borderLeft: `4px solid ${activeRec === "OVER" ? "#166534" : "#991b1b"}`,
      marginBottom: 1,
    }}>

      {/* ── Summary row ── */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 140px 1fr 20px",
          alignItems: "center",
          padding: "14px 16px",
          cursor: "pointer",
          gap: 8,
        }}
      >
        <TeamSide team={away} align="left"  winProb={proj.awayWinProb} />

        <div style={{ textAlign: "center" }}>
          <div style={{ fontSize: 10, color: "#999", marginBottom: 4 }}>{timeStr}</div>

          <PickBadge rec={activeRec} line={activeLine} diff={activeDiff} conf={proj.conf} prob={activeProb} />

          {/* Slider de línea */}
          <div style={{ padding: "6px 4px 0" }}>
            <input
              type="range"
              min={6.5} max={12.0} step={0.5}
              value={activeLine}
              onChange={e => setAdjLine(parseFloat(e.target.value))}
              style={{
                width: "100%", cursor: "pointer",
                accentColor: activeRec === "OVER" ? "#166534" : "#991b1b",
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: "#ccc", marginTop: -2 }}>
              <span>6.5</span>
              {adjLine !== null && (
                <span
                  onClick={() => setAdjLine(null)}
                  style={{ cursor: "pointer", color: "#aaa", textDecoration: "underline" }}
                >reset</span>
              )}
              <span>12</span>
            </div>
          </div>

          <div style={{ fontSize: 10, color: "#999", marginTop: 3 }}>
            proj {NUM(proj.total, 1)} runs
          </div>

          {/* ML pick */}
          <div style={{
            marginTop: 5,
            display: "inline-flex", alignItems: "center", gap: 4,
            padding: "2px 8px",
            background: proj.mlRec === "OVER" ? "#f0fdf4" : "#fef2f2",
            border: `1px solid ${proj.mlRec === "OVER" ? "#bbf7d0" : "#fecaca"}`,
          }}>
            <span style={{ fontSize: 9, color: "#aaa", fontWeight: 600, letterSpacing: 0.5 }}>ML</span>
            <span style={{ fontSize: 11, fontWeight: 800, color: proj.mlRec === "OVER" ? "#166534" : "#991b1b" }}>
              {proj.mlRec} {proj.line}
            </span>
            <span style={{ fontSize: 9, color: "#888" }}>({NUM(proj.mlTotal, 1)})</span>
          </div>
        </div>

        <TeamSide team={home} align="right" winProb={proj.homeWinProb} />
        <div style={{ fontSize: 10, color: "#bbb", userSelect: "none", textAlign: "right" }}>
          {open ? "▲" : "▼"}
        </div>
      </div>

      {/* ── Detail panel ── */}
      {open && (
        <div style={{ borderTop: "1px solid #ebebeb" }}>

          {/* Tab bar */}
          <div style={{
            display: "flex",
            borderBottom: "1px solid #e0e0e0",
            background: "#fafafa",
          }}>
            {[
              { key: "stats",   label: "Team Stats" },
              { key: "record",  label: "Last 10 + Runs" },
              { key: "innings", label: "Innings" },
              { key: "h2h",     label: `H2H (${h2hGames.length})` },
              { key: "ml",      label: proj.agreement ? "ML — agree" : "ML — split" },
            ].map(t => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                style={{
                  padding: "8px 16px",
                  border: "none",
                  borderBottom: tab === t.key ? "2px solid #111" : "2px solid transparent",
                  background: "none",
                  cursor: "pointer",
                  fontSize: 11,
                  fontWeight: tab === t.key ? 700 : 400,
                  color: tab === t.key ? "#111" : "#666",
                  fontFamily: "inherit",
                  letterSpacing: 0.3,
                }}
              >
                {t.label}
              </button>
            ))}
            <div style={{ flex: 1, textAlign: "right", padding: "8px 14px", fontSize: 10, color: "#bbb", alignSelf: "center" }}>
              {venue}
            </div>
          </div>

          <div style={{ padding: "16px 20px" }}>

            {/* ── TAB: Team Stats ── */}
            {tab === "stats" && (
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                <StatsBlock team={away} label="Visitor" />
                <StatsBlock team={home} label="Home" />
              </div>
            )}

            {/* ── TAB: Last 10 + Runs history ── */}
            {tab === "record" && (
              <div>
                {/* W/L strip */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 24 }}>
                  <Last10Block team={away} />
                  <Last10Block team={home} />
                </div>

                {/* Runs history charts */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
                  <div>
                    <SectionLabel>{away.name} — last {away.runsHistory?.length} games (runs scored/allowed)</SectionLabel>
                    <RunsChart
                      data={away.runsHistory}
                      teamName={away.name}
                      avgScored={away.rpgSeason}
                    />
                  </div>
                  <div>
                    <SectionLabel>{home.name} — last {home.runsHistory?.length} games (runs scored/allowed)</SectionLabel>
                    <RunsChart
                      data={home.runsHistory}
                      teamName={home.name}
                      avgScored={home.rpgSeason}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* ── TAB: Innings ── */}
            {tab === "innings" && (
              <div>
                <SectionLabel>Projected runs per inning (same formula as total)</SectionLabel>
                <InningChart innings={proj.innings} />
                <div style={{ marginTop: 14 }}>
                  <InningTable innings={proj.innings} />
                </div>
              </div>
            )}

            {/* ── TAB: ML ── */}
            {tab === "ml" && (
              <MLPanel proj={proj} mlMeta={mlMeta} />
            )}

            {/* ── TAB: H2H ── */}
            {tab === "h2h" && (
              <div>
                {h2hGames.length === 0 ? (
                  <div style={{ fontSize: 12, color: "#888" }}>No head-to-head games in 2026 yet.</div>
                ) : (
                  <>
                    <SectionLabel>
                      {away.abbr} vs {home.abbr} in 2026 — {h2hGames.length} games
                      {" · "}avg {NUM(proj.h2hAvgTotal, 1)} runs
                      {proj.h2hOverPct != null && ` · OVER ${(proj.h2hOverPct * 100).toFixed(0)}%`}
                    </SectionLabel>

                    {/* H2H runs bar chart */}
                    <H2HChart games={h2hGames} line={proj.line} awayAbbr={away.abbr} homeAbbr={home.abbr} />

                    {/* H2H detail table */}
                    <table style={{ ...tblStyle, marginTop: 12 }}>
                      <thead>
                        <tr style={{ background: "#f4f4f4" }}>
                          <Th>Date</Th>
                          <Th>{away.abbr}</Th>
                          <Th>{home.abbr}</Th>
                          <Th>Total</Th>
                          <Th>vs {proj.line}</Th>
                        </tr>
                      </thead>
                      <tbody>
                        {h2hGames.map((hg, i) => {
                          const tot = hg.awayScored + hg.homeScored;
                          const isO = tot > proj.line;
                          return (
                            <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#f9f9f9" }}>
                              <Td>{hg.date}</Td>
                              <Td mono>{hg.awayScored}</Td>
                              <Td mono>{hg.homeScored}</Td>
                              <Td mono bold>{tot}</Td>
                              <Td style={{ color: isO ? "#166534" : "#991b1b", fontWeight: 700 }}>
                                {isO ? "OVER" : "UNDER"}
                              </Td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Last 10 W/L block ───────────────────────────────────────────────────── */
function Last10Block({ team }) {
  const log = team.last10Log ?? [];
  const wins   = log.filter(g => g.result === "W").length;
  const losses = log.filter(g => g.result === "L").length;
  const overs  = log.filter(g => g.ouResult === "OVER").length;
  const unders = log.length - overs;
  const avgScored  = log.length ? log.reduce((s, g) => s + g.scored,  0) / log.length : null;
  const avgAllowed = log.length ? log.reduce((s, g) => s + g.allowed, 0) / log.length : null;
  const avgTotal   = log.length ? log.reduce((s, g) => s + (g.total ?? g.scored + g.allowed), 0) / log.length : null;

  // Inning frecuency: where does this team score most often?
  const innFreq = {};
  log.forEach(g => {
    const inn = g.teamHotInning;
    if (inn) innFreq[inn] = (innFreq[inn] || 0) + 1;
  });
  const topInn = Object.entries(innFreq).sort((a,b) => b[1]-a[1])[0];

  return (
    <div>
      <SectionLabel>{team.name} — últimos {log.length} juegos</SectionLabel>

      {/* W/L boxes */}
      <div style={{ display: "flex", gap: 3, marginBottom: 8, flexWrap: "wrap" }}>
        {log.map((g, i) => (
          <div
            key={i}
            title={`${g.date}  vs ${g.opp}  ${g.scored}-${g.allowed}  Total:${g.total ?? g.scored+g.allowed}`}
            style={{
              width: 28, height: 28,
              display: "flex", alignItems: "center", justifyContent: "center",
              background: g.result === "W" ? "#166534" : "#dc2626",
              color: "#fff", fontSize: 11, fontWeight: 700, cursor: "default",
            }}
          >
            {g.result}
          </div>
        ))}
      </div>

      {/* Summary chips */}
      <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
        <Chip color="#374151">{wins}W-{losses}L</Chip>
        <Chip color="#374151">Anota: {NUM(avgScored,1)} · Permite: {NUM(avgAllowed,1)}</Chip>
        <Chip color="#374151">Total avg: {NUM(avgTotal,1)}</Chip>
        <Chip color={overs >= 5 ? "#b45309" : "#1d4ed8"}>
          {overs}↑ OVER · {unders}↓ UNDER (ref 8.5)
        </Chip>
        {topInn && (
          <Chip color="#6b21a8">Inning favorito: I{topInn[0]} ({topInn[1]}/{log.length} juegos)</Chip>
        )}
      </div>

      {/* Detailed game-by-game table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ ...tblStyle, fontSize: 11, minWidth: 480 }}>
          <thead>
            <tr style={{ background: "#f4f4f4" }}>
              <Th>Fecha</Th>
              <Th>vs</Th>
              <Th>Score</Th>
              <Th>Total</Th>
              <Th>O/U 8.5</Th>
              <Th>Inn. caliente</Th>
              <Th>Inn. del equipo</Th>
            </tr>
          </thead>
          <tbody>
            {[...log].reverse().map((g, i) => {
              const total = g.total ?? (g.scored + g.allowed);
              const opp   = g.opp?.split(" ").slice(-1)[0] ?? g.opp ?? "—";
              return (
                <tr key={i} style={{ background: i % 2 === 0 ? "#fff" : "#f9f9f9" }}>
                  <Td>{g.date?.slice(5) ?? "—"}</Td>
                  <Td>{opp}</Td>
                  <Td>
                    <span style={{ color: g.result === "W" ? "#166534" : "#dc2626", fontWeight: 700 }}>
                      {g.scored}-{g.allowed} {g.result}
                    </span>
                  </Td>
                  <Td mono bold style={{ color: total >= 9 ? "#b45309" : total <= 7 ? "#1d4ed8" : "#374151" }}>
                    {total}
                  </Td>
                  <Td style={{ fontWeight: 700, color: g.ouResult === "OVER" ? "#b45309" : "#1d4ed8" }}>
                    {g.ouResult ?? "—"}
                  </Td>
                  <Td mono>
                    {g.hotInning
                      ? <span>I{g.hotInning} <span style={{ color: "#888", fontSize: 10 }}>({g.hotInningRuns}R)</span></span>
                      : "—"}
                  </Td>
                  <Td mono>
                    {g.teamHotInning
                      ? <span>I{g.teamHotInning} <span style={{ color: "#888", fontSize: 10 }}>({g.teamHotRuns}R)</span></span>
                      : "—"}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Chip({ children, color }) {
  return (
    <span style={{
      display: "inline-block", padding: "2px 8px",
      background: color + "15", border: `1px solid ${color}40`,
      color, fontSize: 10, fontWeight: 600, borderRadius: 2,
    }}>
      {children}
    </span>
  );
}

/* ── H2H bar chart ───────────────────────────────────────────────────────── */
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Cell, Legend,
} from "recharts";

function H2HChart({ games, line, awayAbbr, homeAbbr }) {
  const data = games.map(g => ({
    date:   g.date.slice(5),
    away:   g.awayScored,
    home:   g.homeScored,
    total:  g.awayScored + g.homeScored,
  }));

  return (
    <div style={{ height: 150 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#ececec" vertical={false} />
          <XAxis dataKey="date" tick={{ fontSize: 9, fill: "#aaa" }} axisLine={{ stroke: "#ddd" }} tickLine={false} />
          <YAxis tick={{ fontSize: 9, fill: "#aaa" }} axisLine={false} tickLine={false} />
          <Tooltip
            formatter={(v, name) => [v, name === "away" ? awayAbbr : homeAbbr]}
            contentStyle={{ fontSize: 11 }}
          />
          <ReferenceLine y={line} stroke="#111" strokeDasharray="4 2"
            label={{ value: `O/U ${line}`, position: "insideTopRight", fontSize: 9, fill: "#666" }} />
          <Bar dataKey="away" stackId="a" fill="#374151" maxBarSize={30} />
          <Bar dataKey="home" stackId="a" fill="#9ca3af" maxBarSize={30} radius={[2, 2, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.total > line ? "#166534" : "#991b1b"} opacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Inning table ────────────────────────────────────────────────────────── */
function InningTable({ innings }) {
  return (
    <table style={{ ...tblStyle, fontSize: 11 }}>
      <thead>
        <tr style={{ background: "#f4f4f4" }}>
          {innings.map(d => <Th key={d.inning}>I{d.inning}</Th>)}
          <Th>Total</Th>
        </tr>
      </thead>
      <tbody>
        <tr>
          {innings.map(d => (
            <Td key={d.inning} mono style={{
              color: d.proj >= 1.3 ? "#166534" : d.proj <= 0.35 ? "#991b1b" : "#374151",
              fontWeight: d.proj >= 1.3 ? 700 : 400,
            }}>
              {d.proj.toFixed(2)}
            </Td>
          ))}
          <Td mono bold>{innings.reduce((s, d) => s + d.proj, 0).toFixed(2)}</Td>
        </tr>
        <tr style={{ background: "#f9f9f9", color: "#888" }}>
          {innings.map(d => <Td key={d.inning} mono style={{ fontSize: 10 }}>{d.seasonAvg?.toFixed(2) ?? "—"}</Td>)}
          <Td style={{ fontSize: 10, color: "#aaa" }}>season</Td>
        </tr>
      </tbody>
    </table>
  );
}

/* ── Team side summary ───────────────────────────────────────────────────── */
function TeamSide({ team, align, winProb }) {
  const isLeft = align === "left";
  const probColor = (winProb ?? 0) >= 58 ? "#166534"
                  : (winProb ?? 0) >= 52 ? "#374151"
                  : "#991b1b";
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
        width={44} height={44}
        style={{ objectFit: "contain", flexShrink: 0 }}
        onError={e => { e.target.style.display = "none"; }}
      />
      <div style={{ textAlign: isLeft ? "left" : "right" }}>
        <div style={{ fontWeight: 700, fontSize: 15 }}>{team.name}</div>
        <div style={{ fontSize: 11, color: "#555", marginTop: 1 }}>
          {team.record} · .{Math.round(team.winPct * 1000)} WP
        </div>
        <div style={{ fontSize: 11, color: "#888", marginTop: 1 }}>
          {team.probablePitcher}{team.era !== "—" && ` · ERA ${team.era}`}
        </div>
        <StreakBadge code={team.streak} />
        {winProb != null && (
          <div style={{
            marginTop: 4,
            fontSize: 16,
            fontWeight: 900,
            color: probColor,
            letterSpacing: -0.5,
          }}>
            {winProb}% <span style={{ fontSize: 9, fontWeight: 500, color: "#aaa", letterSpacing: 0 }}>WIN</span>
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Stats block ─────────────────────────────────────────────────────────── */
function StatsBlock({ team, label }) {
  return (
    <div>
      <SectionLabel>{label} — {team.name}</SectionLabel>
      <table style={{ width: "100%" }}>
        <tbody>
          <SR label="Record" value={`${team.record}  (${PCT(team.winPct)} WP)`} />
          <SR label="Streak" value={team.streak} />
          <SR label="R/G season"        value={team.rpgSeason}  mono />
          <SR label="R/G last 10"       value={team.rpgLast10}  mono />
          <SR label={`R/G this week (${team.weekGames}g)`} value={team.rpgWeek ?? "—"} mono />
          <SR label="OPS"               value={team.ops}  mono />
          <SR label="ERA (staff)"       value={team.era}  mono />
          <SR label="WHIP"              value={team.whip} mono />
          <SR label="Probable pitcher"  value={team.probablePitcher} />
        </tbody>
      </table>
    </div>
  );
}

/* ── Inning heat strip ───────────────────────────────────────────────────── */
function InningHeatStrip({ innings }) {
  if (!innings?.length) return null;

  const max = Math.max(...innings.map(d => d.proj));

  return (
    <div style={{ marginTop: 6, marginBottom: 2 }}>
      {/* Mini bar per inning */}
      <div style={{ display: "flex", gap: 2, justifyContent: "center", alignItems: "flex-end", height: 22 }}>
        {innings.map(d => {
          const pct = max > 0 ? d.proj / max : 0;
          const hot = pct >= 0.85;
          const warm = pct >= 0.65;
          const color = hot ? "#b45309" : warm ? "#374151" : "#d1d5db";
          return (
            <div key={d.inning} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1 }}>
              <div style={{
                width: 10,
                height: Math.max(4, Math.round(pct * 20)),
                background: color,
                borderRadius: 1,
              }} />
              <span style={{ fontSize: 7, color: hot ? "#b45309" : "#bbb", fontWeight: hot ? 700 : 400 }}>
                {d.inning}
              </span>
            </div>
          );
        })}
      </div>
      {/* Top 2 innings label */}
      {(() => {
        const sorted = [...innings].sort((a, b) => b.proj - a.proj).slice(0, 2);
        return (
          <div style={{ fontSize: 9, color: "#888", marginTop: 2, textAlign: "center" }}>
            {sorted.map(d => (
              <span key={d.inning} style={{ marginRight: 6 }}>
                <span style={{ fontWeight: 700, color: "#b45309" }}>I{d.inning}</span>
                {" "}{d.proj.toFixed(1)}R
              </span>
            ))}
          </div>
        );
      })()}
    </div>
  );
}

/* ── Pick badge ──────────────────────────────────────────────────────────── */
function PickBadge({ rec, line, diff, conf, prob }) {
  const isOver = rec === "OVER";
  const color  = isOver ? "#166534" : "#991b1b";
  const probHigh = (prob ?? 0) >= 62;
  const probMid  = (prob ?? 0) >= 57;
  return (
    <div style={{ textAlign: "center" }}>
      {/* Porcentaje — visible de entrada */}
      {prob != null && (
        <div style={{
          fontSize: probHigh ? 18 : 15,
          fontWeight: 900,
          color,
          letterSpacing: -0.5,
          lineHeight: 1,
          marginBottom: 3,
        }}>
          {prob}%
        </div>
      )}
      <span style={{
        display: "inline-block",
        padding: "3px 12px",
        border: `2px solid ${color}`,
        color,
        fontWeight: 700, fontSize: 13, letterSpacing: 1,
      }}>
        {rec} {line}
      </span>
      <div style={{ fontSize: 10, color: "#888", marginTop: 3, display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
        {diff > 0 ? "+" : ""}{Number(diff).toFixed(2)}
        <span>
          {[1,2,3].map(i => (
            <span key={i} style={{ color: i <= conf ? "#333" : "#ccc" }}>●</span>
          ))}
        </span>
      </div>
    </div>
  );
}

/* ── Streak badge ────────────────────────────────────────────────────────── */
function StreakBadge({ code }) {
  if (!code || code === "—") return null;
  const isW = code.startsWith("W");
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, marginTop: 2, display: "inline-block",
      color: isW ? "#166534" : "#991b1b",
    }}>
      {code}
    </span>
  );
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */
function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, fontWeight: 700, letterSpacing: 1.5,
      textTransform: "uppercase", color: "#555",
      marginBottom: 7,
    }}>
      {children}
    </div>
  );
}

function SR({ label, value, mono }) {
  return (
    <tr>
      <td style={{ padding: "3px 8px 3px 0", fontSize: 11, color: "#666", whiteSpace: "nowrap" }}>{label}</td>
      <td style={{ padding: "3px 0", fontSize: 11, fontFamily: mono ? "monospace" : "inherit", fontWeight: 500 }}>
        {value ?? "—"}
      </td>
    </tr>
  );
}

const tblStyle = { borderCollapse: "collapse", width: "100%", fontSize: 12 };
function Th({ children }) {
  return <th style={{ padding: "5px 10px", textAlign: "left", fontSize: 11, fontWeight: 600, color: "#444", whiteSpace: "nowrap" }}>{children}</th>;
}
function Td({ children, mono, bold, style: s }) {
  return (
    <td style={{ padding: "4px 10px", fontFamily: mono ? "monospace" : "inherit", fontWeight: bold ? 700 : 400, ...s }}>
      {children}
    </td>
  );
}
