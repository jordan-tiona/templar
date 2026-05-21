import { useEffect, useState, useCallback } from 'react'
import { api } from './api'
import type { LatestSignals, Portfolio, Position, ExplainData, HistoryEntry } from './types'
import PortfolioBar from './components/PortfolioBar'
import SignalsTable from './components/SignalsTable'
import ExplainPanel from './components/ExplainPanel'
import PositionsTable from './components/PositionsTable'
import HistoryView from './components/HistoryView'

type Tab = 'dashboard' | 'history'

export default function App() {
  const [tab, setTab] = useState<Tab>('dashboard')
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [positions, setPositions] = useState<Position[]>([])
  const [signals, setSignals] = useState<LatestSignals | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [explainData, setExplainData] = useState<ExplainData | null>(null)
  const [explainLoading, setExplainLoading] = useState(false)
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)

  const refresh = useCallback(async () => {
    const [p, pos, sig] = await Promise.all([
      api.portfolio().catch(() => null),
      api.positions().catch(() => []),
      api.latestSignals().catch(() => ({ generated_at: null, signals: [] })),
    ])
    setPortfolio(p)
    setPositions(pos)
    setSignals(sig)
    setLastRefresh(new Date())
  }, [])

  useEffect(() => { refresh() }, [refresh])

  useEffect(() => {
    if (tab === 'history') {
      api.history().then(setHistory).catch(() => setHistory([]))
    }
  }, [tab])

  useEffect(() => {
    if (!selectedSymbol) { setExplainData(null); return }
    setExplainLoading(true)
    setExplainData(null)
    api.explain(selectedSymbol)
      .then(setExplainData)
      .catch(() => setExplainData(null))
      .finally(() => setExplainLoading(false))
  }, [selectedSymbol])

  const handleSelectSymbol = (sym: string) => {
    setSelectedSymbol(prev => prev === sym ? null : sym)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      {/* Header */}
      <header style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 24px', height: 52, flexShrink: 0,
        background: 'var(--bg-card)', borderBottom: '1px solid var(--border)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          <span style={{
            fontWeight: 800, fontSize: 16, letterSpacing: 3,
            color: 'var(--accent)', fontVariant: 'all-small-caps',
          }}>
            TEMPLAR
          </span>
          <div style={{ display: 'flex', gap: 2 }}>
            {(['dashboard', 'history'] as Tab[]).map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                style={{
                  padding: '4px 14px', borderRadius: 5, border: 'none', cursor: 'pointer',
                  background: tab === t ? 'var(--accent)' : 'transparent',
                  color: tab === t ? '#fff' : 'var(--text-muted)',
                  fontWeight: tab === t ? 600 : 400,
                  fontSize: 13, textTransform: 'capitalize',
                  transition: 'background 0.15s',
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          {lastRefresh && (
            <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              {lastRefresh.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={refresh}
            style={{
              padding: '4px 14px', borderRadius: 5, border: '1px solid var(--border)',
              background: 'transparent', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 13,
            }}
          >
            Refresh
          </button>
        </div>
      </header>

      {/* Portfolio bar */}
      {portfolio && <PortfolioBar portfolio={portfolio} />}

      {/* Body */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {tab === 'dashboard' && (
          <>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'auto', minWidth: 0 }}>
              <SignalsTable
                signals={signals}
                selectedSymbol={selectedSymbol}
                onSelect={handleSelectSymbol}
              />
              <PositionsTable positions={positions} />
            </div>
            {selectedSymbol && (
              <ExplainPanel
                symbol={selectedSymbol}
                data={explainData}
                loading={explainLoading}
                onClose={() => setSelectedSymbol(null)}
              />
            )}
          </>
        )}
        {tab === 'history' && <HistoryView history={history} />}
      </div>
    </div>
  )
}
