import { BrowserRouter, Routes, Route, Link } from 'react-router-dom'
import IsraelGov from './pages/IsraelGov'
import centileLogo from './assets/centileLogo.png'

const dashboards = [
  { path: '/israel-gov', label: 'Israel Government Set-up' },
]

function Home() {
  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 800, margin: '60px auto', padding: '0 24px' }}>
      <img src={centileLogo} alt="Centile" style={{ height: 48, marginBottom: 24, display: 'block' }} />
      <h1>Centile Macro Dashboards</h1>
      <ul style={{ listStyle: 'none', padding: 0, marginTop: 32 }}>
        {dashboards.map(({ path, label }) => (
          <li key={path} style={{ marginBottom: 16 }}>
            <Link
              to={path}
              style={{ fontSize: 18, color: '#1a73e8', textDecoration: 'none' }}
            >
              {label}
            </Link>
          </li>
        ))}
      </ul>
      <p style={{ marginTop: 48, fontSize: 11, color: '#999', borderTop: '1px solid #e0e0e0', paddingTop: 16 }}>
        Centile Partners is a trading name of Kepler Cheuvreux S.A. (FCA Firm Reference Number 958983) and Kepler Cheuvreux Agency Inc (NFA ID: 0560205)
      </p>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter basename="/macro">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/israel-gov" element={<IsraelGov />} />
      </Routes>
    </BrowserRouter>
  )
}
