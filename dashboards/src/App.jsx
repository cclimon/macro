import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import IsraelGov from './pages/IsraelGov'
import TicDashboard from './pages/TicDashboard'
import LiquidityDashboard from './pages/LiquidityDashboard'
import centileLogo from './assets/centileLogo.png'

const internalDashboards = [
  { path: '/israel-gov', label: 'Israel Government Set-up' },
  { path: '/tic-dashboard', label: 'TIC Dashboard' },
  { path: '/liquidity', label: 'US Liquidity — Fed Plumbing' },
]

const externalDashboards = [
  {
    href: 'https://cclimon-macro-dashboard-app.streamlit.app',
    label: 'G10 FX Signal Dashboard',
    tag: 'Live · Streamlit',
  },
  {
    href: 'https://cclimon-margin-debt.streamlit.app',
    label: 'FINRA — Margin Debt Analysis',
    tag: 'Live · Streamlit',
  },
]

const linkStyle = { fontSize: 18, color: '#4fa3e8', textDecoration: 'none' }
const tagStyle = {
  fontSize: 10, fontWeight: 700, letterSpacing: 1,
  color: '#4fa3e8', border: '1px solid #4fa3e8',
  borderRadius: 4, padding: '1px 6px', marginLeft: 10,
  verticalAlign: 'middle', opacity: 0.7,
}

function Home() {
  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 800, margin: '60px auto', padding: '0 24px' }}>
      <img src={centileLogo} alt="Centile" style={{ height: 72, display: 'block', margin: '0 auto 24px' }} />
      <h1 style={{ color: '#fff' }}>Centile Macro Dashboards</h1>

      <ul style={{ listStyle: 'none', padding: 0, marginTop: 32 }}>
        {internalDashboards.map(({ path, label }) => (
          <li key={path} style={{ marginBottom: 16 }}>
            <Link to={path} style={linkStyle}>{label}</Link>
          </li>
        ))}
        {externalDashboards.map(({ href, label, tag }) => (
          <li key={href} style={{ marginBottom: 16 }}>
            <a href={href} target="_blank" rel="noopener noreferrer" style={linkStyle}>
              {label}
            </a>
            <span style={tagStyle}>{tag}</span>
          </li>
        ))}
      </ul>

      <p style={{ marginTop: 48, fontSize: 11, color: '#666', borderTop: '1px solid #333', paddingTop: 16 }}>
        Centile Partners is a trading name of Kepler Cheuvreux S.A. (FCA Firm Reference Number 958983) and Kepler Cheuvreux Agency Inc (NFA ID: 0560205)
      </p>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter basename="/macro">
      <img
        src={centileLogo}
        alt="Centile"
        style={{ position: 'fixed', top: 16, right: 20, height: 36, zIndex: 1000, opacity: 0.9 }}
      />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/israel-gov" element={<IsraelGov />} />
        <Route path="/tic-dashboard" element={<TicDashboard />} />
        <Route path="/liquidity" element={<LiquidityDashboard />} />
      </Routes>
    </BrowserRouter>
  )
}
