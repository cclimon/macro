

import { useState } from "react";

const PARTIES = {
  Likud: { color: "#1a3a6b", light: "#d6e4f7" },
  "Religious Zionist": { color: "#7b3a00", light: "#fde8cc" },
  "Otzma Yehudit": { color: "#5a0a0a", light: "#fdd9d9" },
  "New Hope": { color: "#1a5c2a", light: "#d4f0dc" },
  UTJ: { color: "#3a3060", light: "#e0dcf8" },
};

const cabinet = [
  { role: "Prime Minister", name: "Benjamin Netanyahu", party: "Likud", tier: 0 },
  { role: "Deputy PM · Minister of Interior", name: "Yariv Levin", party: "Likud", tier: 1, note: "Also: Justice, Religious Affairs, Jerusalem & Tradition" },
  { role: "Minister of Defense", name: "Israel Katz", party: "Likud", tier: 1 },
  { role: "Minister of Foreign Affairs", name: "Gideon Sa'ar", party: "New Hope", tier: 1 },
  { role: "Minister of Finance", name: "Bezalel Smotrich", party: "Religious Zionist", tier: 1 },
  { role: "Minister of Justice", name: "Yariv Levin", party: "Likud", tier: 2 },
  { role: "Minister of Health", name: "Haim Katz", party: "Likud", tier: 2, note: "Also: Welfare, Tourism, Galilee & National Resilience" },
  { role: "Minister of Education", name: "Yoav Kisch", party: "Likud", tier: 2 },
  { role: "Minister of Transportation", name: "Miri Regev", party: "Likud", tier: 2 },
  { role: "Minister of National Security", name: "Itamar Ben Gvir", party: "Otzma Yehudit", tier: 2 },
  { role: "Minister of Energy & Infrastructure", name: "Eli Cohen", party: "Likud", tier: 2 },
  { role: "Minister of Economy & Industry", name: "Nir Barkat", party: "Likud", tier: 2 },
  { role: "Minister of Environmental Protection", name: "Idit Silman", party: "Likud", tier: 2 },
  { role: "Minister of Communications", name: "Shlomo Karhi", party: "Likud", tier: 2 },
  { role: "Minister of Agriculture", name: "Avi Dichter", party: "Likud", tier: 2 },
  { role: "Minister of Housing & Construction", name: "Israel Eichler", party: "UTJ", tier: 2 },
  { role: "Minister of Strategic Affairs", name: "Ron Dermer", party: "Likud", tier: 2, note: "Non-MK" },
  { role: "Minister of Innovation, Science & Technology", name: "Gila Gamliel", party: "Likud", tier: 2, note: "Also: Intelligence" },
  { role: "Minister of Culture & Sports", name: "Miki Zohar", party: "Likud", tier: 2 },
  { role: "Minister of Social Equality & Women", name: "May Golan", party: "Likud", tier: 2 },
  { role: "Minister of Diaspora Affairs", name: "Amichai Chikli", party: "Likud", tier: 2 },
  { role: "Minister of Immigration & Absorption", name: "Ofir Sofer", party: "Religious Zionist", tier: 2 },
  { role: "Minister of National Missions", name: "Orit Strock", party: "Religious Zionist", tier: 2 },
  { role: "Minister for Negev & Galilee", name: "Yitzhak Wasserlauf", party: "Otzma Yehudit", tier: 2 },
  { role: "Minister of Heritage", name: "Amichai Eliyahu", party: "Otzma Yehudit", tier: 2 },
  { role: "Minister within Finance Ministry", name: "Ze'ev Elkin", party: "New Hope", tier: 2 },
];

function Card({ member, isActive, onClick }) {
  const party = PARTIES[member.party] || { color: "#555", light: "#eee" };
  return (
    <div
      onClick={onClick}
      style={{
        background: isActive ? party.color : "#fff",
        color: isActive ? "#fff" : "#1a1a2e",
        border: `2px solid ${party.color}`,
        borderRadius: 10,
        padding: "10px 14px",
        cursor: "pointer",
        boxShadow: isActive ? `0 4px 18px ${party.color}55` : "0 2px 8px rgba(0,0,0,0.08)",
        transition: "all 0.2s",
        minWidth: 180,
        maxWidth: 230,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.5, color: isActive ? "#ffffffcc" : party.color, marginBottom: 4, textTransform: "uppercase" }}>
        {member.party || "Independent"}
      </div>
      <div style={{ fontWeight: 700, fontSize: 14, lineHeight: 1.3 }}>{member.name}</div>
      <div style={{ fontSize: 11, marginTop: 4, color: isActive ? "#ffffffcc" : "#555", lineHeight: 1.4 }}>{member.role}</div>
      {member.note && (
        <div style={{ fontSize: 10, marginTop: 5, background: isActive ? "rgba(255,255,255,0.15)" : party.light, color: isActive ? "#fff" : party.color, borderRadius: 4, padding: "2px 5px" }}>
          {member.note}
        </div>
      )}
    </div>
  );
}

export default function IsraelGov() {
  const [activeParty, setActiveParty] = useState(null);
  const [activeCard, setActiveCard] = useState(null);

  const tier1 = cabinet.filter((m) => m.tier === 1);
  const tier2 = cabinet.filter((m) => m.tier === 2);
  const pm = cabinet.find((m) => m.tier === 0);
  const filteredTier2 = activeParty ? tier2.filter((m) => m.party === activeParty) : tier2;

  return (
    <div style={{ fontFamily: "'Segoe UI', system-ui, sans-serif", background: "linear-gradient(135deg, #f0f4ff 0%, #faf8ff 100%)", minHeight: "100vh", padding: "24px 16px" }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <div style={{ display: "inline-block", background: "#1a3a6b", color: "#fff", borderRadius: 8, padding: "4px 16px", fontSize: 11, fontWeight: 700, letterSpacing: 1, marginBottom: 8 }}>
          37th GOVERNMENT OF ISRAEL
        </div>
        <h1 style={{ fontSize: 26, fontWeight: 800, color: "#1a1a2e", margin: "0 0 4px", letterSpacing: -0.5 }}>Government Organigram</h1>
        <p style={{ color: "#666", fontSize: 13, margin: 0 }}>Formed December 29, 2022 · Led by PM Benjamin Netanyahu</p>
      </div>

      {/* Party Filter */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", flexWrap: "wrap", marginBottom: 28 }}>
        <button onClick={() => setActiveParty(null)} style={{ background: activeParty === null ? "#1a1a2e" : "#fff", color: activeParty === null ? "#fff" : "#1a1a2e", border: "2px solid #1a1a2e", borderRadius: 20, padding: "5px 14px", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
          All Parties
        </button>
        {Object.entries(PARTIES).map(([party, { color }]) => (
          <button key={party} onClick={() => setActiveParty(activeParty === party ? null : party)} style={{ background: activeParty === party ? color : "#fff", color: activeParty === party ? "#fff" : color, border: `2px solid ${color}`, borderRadius: 20, padding: "5px 14px", cursor: "pointer", fontSize: 12, fontWeight: 600 }}>
            {party}
          </button>
        ))}
      </div>

      {/* President */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 8 }}>
        <div style={{ background: "#f0f0f0", border: "2px dashed #aaa", borderRadius: 10, padding: "10px 20px", textAlign: "center" }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, color: "#999", textTransform: "uppercase" }}>President (Ceremonial)</div>
          <div style={{ fontWeight: 700, fontSize: 14, color: "#333" }}>Isaac Herzog</div>
          <div style={{ fontSize: 11, color: "#888" }}>Head of State</div>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "center" }}><div style={{ width: 2, height: 20, background: "#1a3a6b55" }} /></div>

      {/* PM */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: 8 }}>
        <Card member={pm} isActive={activeCard === pm.name} onClick={() => setActiveCard(activeCard === pm.name ? null : pm.name)} />
      </div>

      <div style={{ display: "flex", justifyContent: "center" }}><div style={{ width: 2, height: 20, background: "#1a3a6b55" }} /></div>

      {/* Tier 1 */}
      <div style={{ textAlign: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, color: "#999", textTransform: "uppercase" }}>Senior Security & Core Ministers</span>
      </div>
      <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap", marginBottom: 8, padding: "0 8px" }}>
        {tier1.filter((m) => !activeParty || m.party === activeParty).map((m) => (
          <Card key={m.name + m.role} member={m} isActive={activeCard === m.name + m.role} onClick={() => setActiveCard(activeCard === m.name + m.role ? null : m.name + m.role)} />
        ))}
      </div>

      <div style={{ display: "flex", justifyContent: "center" }}><div style={{ width: 2, height: 20, background: "#1a3a6b44" }} /></div>

      {/* Tier 2 */}
      <div style={{ textAlign: "center", marginBottom: 10 }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, color: "#999", textTransform: "uppercase" }}>Full Cabinet Ministers</span>
      </div>
      <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap", padding: "0 8px", marginBottom: 24 }}>
        {filteredTier2.map((m) => (
          <Card key={m.name + m.role} member={m} isActive={activeCard === m.name + m.role} onClick={() => setActiveCard(activeCard === m.name + m.role ? null : m.name + m.role)} />
        ))}
      </div>

      {/* Legend */}
      <div style={{ background: "#fff", borderRadius: 10, padding: "14px 18px", maxWidth: 600, margin: "0 auto", boxShadow: "0 2px 10px rgba(0,0,0,0.06)" }}>
        <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1, color: "#999", textTransform: "uppercase", marginBottom: 10 }}>Coalition Composition</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
          {Object.entries(PARTIES).map(([party, { color, light }]) => {
            const count = cabinet.filter((m) => m.party === party).length;
            return (
              <div key={party} style={{ display: "flex", alignItems: "center", gap: 6, background: light, borderRadius: 6, padding: "4px 10px" }}>
                <div style={{ width: 10, height: 10, borderRadius: 2, background: color }} />
                <span style={{ fontSize: 12, fontWeight: 600, color }}>{party}</span>
                <span style={{ fontSize: 11, color: "#888" }}>({count} posts)</span>
              </div>
            );
          })}
        </div>
        <div style={{ marginTop: 10, fontSize: 11, color: "#aaa" }}>
          Note: UTJ and Shas left the coalition in July 2025 over conscription law disputes. Coalition currently consists of Likud, Religious Zionist, Otzma Yehudit, and New Hope.
          Click any card to highlight · Use party filters above to focus by faction.
        </div>
      </div>
    </div>
  );
}
