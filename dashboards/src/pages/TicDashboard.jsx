import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell
} from "recharts";

// ── Data ──────────────────────────────────────────────────────────────────
const historicalData = [
  { year: "Dec-15", Japan: 1122, China: 1246, UK: 375, Belgium: 354, Ireland: 264, Total: 6135 },
  { year: "Dec-16", Japan: 1107, China: 1058, UK: 394, Belgium: 317, Ireland: 288, Total: 5941 },
  { year: "Dec-17", Japan: 1062, China: 1185, UK: 327, Belgium: 178, Ireland: 326, Total: 6291 },
  { year: "Dec-18", Japan: 1041, China: 1124, UK: 336, Belgium: 187, Ireland: 297, Total: 6218 },
  { year: "Dec-19", Japan: 1154, China: 1070, UK: 410, Belgium: 239, Ireland: 334, Total: 6779 },
  { year: "Dec-20", Japan: 1260, China: 1063, UK: 480, Belgium: 290, Ireland: 354, Total: 7446 },
  { year: "Dec-21", Japan: 1301, China: 1068, UK: 604, Belgium: 366, Ireland: 328, Total: 7720 },
  { year: "Dec-22", Japan: 1076, China: 867, UK: 645, Belgium: 341, Ireland: 330, Total: 7302 },
  { year: "Dec-23", Japan: 1150, China: 816, UK: 716, Belgium: 370, Ireland: 305, Total: 7996 },
  { year: "Dec-24", Japan: 1062, China: 759, UK: 723, Belgium: 375, Ireland: 339, Total: 8619 },
  { year: "Dec-25", Japan: 1186, China: 684, UK: 866, Belgium: 477, Ireland: 341, Total: 9271 },
];

const totalDebt = {
  "Dec-15": 18100, "Dec-16": 19800, "Dec-17": 20200, "Dec-18": 21500,
  "Dec-19": 23000, "Dec-20": 27700, "Dec-21": 29600, "Dec-22": 30900,
  "Dec-23": 33200, "Dec-24": 36200, "Dec-25": 38900,
};

const COLORS = {
  Japan:   "#3b82f6",
  China:   "#ef4444",
  UK:      "#10b981",
  Belgium: "#f59e0b",
  Ireland: "#8b5cf6",
};

const COUNTRIES = ["Japan", "China", "UK", "Belgium", "Ireland"];

const latestYear = historicalData[historicalData.length - 1];
const prevYear   = historicalData[historicalData.length - 2];

// ── Helpers ───────────────────────────────────────────────────────────────
const fmtB  = (v) => `$${v}B`;
const fmtPct = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
const delta  = (c) => latestYear[c] - prevYear[c];
const pctChg = (c) => ((latestYear[c] - prevYear[c]) / prevYear[c]) * 100;
const pctOfDebt = (c, row) =>
  ((row[c] / totalDebt[row.year]) * 100).toFixed(1);

// ── Components ────────────────────────────────────────────────────────────
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0f172a", border: "1px solid #1e293b",
      borderRadius: 6, padding: "10px 14px", fontSize: 12,
    }}>
      <p style={{ color: "#94a3b8", marginBottom: 6, fontWeight: 600 }}>{label}</p>
      {payload.map((p) => (
        <p key={p.dataKey} style={{ color: p.color, margin: "2px 0" }}>
          {p.dataKey}: <strong>${p.value}B</strong>
        </p>
      ))}
    </div>
  );
};

export default function TicDashboard() {
  const [activeTab, setActiveTab] = useState("trend");
  const [visibleCountries, setVisibleCountries] = useState(
    Object.fromEntries(COUNTRIES.map((c) => [c, true]))
  );

  const toggleCountry = (c) =>
    setVisibleCountries((prev) => ({ ...prev, [c]: !prev[c] }));

  const latest = historicalData[historicalData.length - 1];
  const barData = COUNTRIES.map((c) => ({
    name: c,
    value: latest[c],
    color: COLORS[c],
  })).sort((a, b) => b.value - a.value);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0f1e",
      color: "#e2e8f0",
      fontFamily: "'Inter', 'Helvetica Neue', sans-serif",
      padding: "32px 24px",
    }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 6 }}>
          <span style={{
            fontSize: 11, fontWeight: 700, letterSpacing: "0.12em",
            color: "#3b82f6", textTransform: "uppercase",
          }}>
            US Treasury International Capital
          </span>
          <span style={{ fontSize: 11, color: "#475569" }}>· Source: TIC / US Treasury</span>
        </div>
        <h1 style={{
          fontSize: 28, fontWeight: 700, color: "#f1f5f9",
          margin: 0, letterSpacing: "-0.02em",
        }}>
          Foreign Holdings of US Treasuries
        </h1>
        <p style={{ color: "#64748b", fontSize: 13, marginTop: 6 }}>
          Top 5 foreign holders — Dec 2015 to Dec 2025 · Billions USD
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: 12, marginBottom: 28,
      }}>
        {COUNTRIES.map((c) => {
          const d = delta(c);
          const p = pctChg(c);
          return (
            <div key={c} style={{
              background: "#111827",
              border: `1px solid ${COLORS[c]}33`,
              borderRadius: 10, padding: "14px 16px",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                <span style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600 }}>{c}</span>
                <span style={{
                  fontSize: 11, fontWeight: 700, padding: "2px 7px", borderRadius: 20,
                  background: d >= 0 ? "#14532d" : "#7f1d1d",
                  color: d >= 0 ? "#4ade80" : "#fca5a5",
                }}>
                  {fmtPct(p)}
                </span>
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: COLORS[c] }}>
                ${latestYear[c]}B
              </div>
              <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
                {d >= 0 ? "▲" : "▼"} ${Math.abs(d)}B vs prior year
              </div>
            </div>
          );
        })}
        <div style={{
          background: "#111827", border: "1px solid #1e3a5f",
          borderRadius: 10, padding: "14px 16px",
        }}>
          <div style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, marginBottom: 8 }}>
            Total Foreign
          </div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#60a5fa" }}>
            ${latestYear.Total.toLocaleString()}B
          </div>
          <div style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>
            ~{((latestYear.Total / totalDebt["Dec-25"]) * 100).toFixed(1)}% of public debt
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 20 }}>
        {[
          { key: "trend", label: "Trend (2015–2025)" },
          { key: "snapshot", label: "Latest Snapshot" },
          { key: "share", label: "% of Total Debt" },
        ].map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            style={{
              padding: "7px 16px", borderRadius: 6, fontSize: 12,
              fontWeight: 600, cursor: "pointer", border: "none",
              background: activeTab === t.key ? "#3b82f6" : "#1e293b",
              color: activeTab === t.key ? "#fff" : "#94a3b8",
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Chart Area */}
      <div style={{
        background: "#111827", borderRadius: 12,
        border: "1px solid #1e293b", padding: "24px 16px",
        marginBottom: 24,
      }}>

        {/* Country toggles — trend only */}
        {activeTab === "trend" && (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
            {COUNTRIES.map((c) => (
              <button
                key={c}
                onClick={() => toggleCountry(c)}
                style={{
                  padding: "4px 12px", borderRadius: 20, fontSize: 11,
                  fontWeight: 600, cursor: "pointer", border: "none",
                  background: visibleCountries[c] ? COLORS[c] + "22" : "#1e293b",
                  color: visibleCountries[c] ? COLORS[c] : "#475569",
                  outline: visibleCountries[c] ? `1px solid ${COLORS[c]}` : "1px solid #334155",
                }}
              >
                {c}
              </button>
            ))}
          </div>
        )}

        {activeTab === "trend" && (
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={historicalData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="year" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis
                tickFormatter={(v) => `$${v}B`}
                tick={{ fill: "#64748b", fontSize: 11 }}
                width={60}
              />
              <Tooltip content={<CustomTooltip />} />
              {COUNTRIES.filter((c) => visibleCountries[c]).map((c) => (
                <Line
                  key={c} type="monotone" dataKey={c}
                  stroke={COLORS[c]} strokeWidth={2}
                  dot={{ r: 3, fill: COLORS[c] }}
                  activeDot={{ r: 5 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}

        {activeTab === "snapshot" && (
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={barData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 12 }} />
              <YAxis tickFormatter={fmtB} tick={{ fill: "#64748b", fontSize: 11 }} width={60} />
              <Tooltip
                formatter={(v) => [`$${v}B`, "Holdings"]}
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {barData.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}

        {activeTab === "share" && (
          <ResponsiveContainer width="100%" height={360}>
            <LineChart
              data={historicalData.map((row) => ({
                year: row.year,
                ...Object.fromEntries(
                  COUNTRIES.map((c) => [c, parseFloat(pctOfDebt(c, row))])
                ),
              }))}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="year" tick={{ fill: "#64748b", fontSize: 11 }} />
              <YAxis
                tickFormatter={(v) => `${v}%`}
                tick={{ fill: "#64748b", fontSize: 11 }}
                width={44}
              />
              <Tooltip
                formatter={(v) => [`${v}%`, ""]}
                contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 6 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              {COUNTRIES.map((c) => (
                <Line
                  key={c} type="monotone" dataKey={c}
                  stroke={COLORS[c]} strokeWidth={2}
                  dot={{ r: 3, fill: COLORS[c] }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Footer note */}
      <p style={{ fontSize: 11, color: "#334155", textAlign: "center" }}>
        Data: TIC Table 5 (major holders). Belgium & Ireland include Euroclear/Clearstream custodial flows on behalf of third parties.
        Dec-25 figures are preliminary estimates.
      </p>
    </div>
  );
}
