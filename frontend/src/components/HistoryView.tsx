import { useState } from 'react'
import type { HistoryEntry } from '../types'

function fmtPred(v: number | null) {
  if (v === null) return '—'
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%'
}

export default function HistoryView({ history }: { history: HistoryEntry[] }) {
  const [selected, setSelected] = useState(0)

  if (history.length === 0) {
    return (
      <div style={{ padding: 40, color: 'var(--text-muted)', fontSize: 13 }}>
        No signal history yet. Run <code>python main.py run</code> to generate signals.
      </div>
    )
  }

  const entry = history[selected]
  const sorted = [...entry.signals].sort((a, b) => (b.ensemble_score ?? 0) - (a.ensemble_score ?? 0))

  return (
    <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
      {/* Run list */}
      <div style={{
        width: 200, flexShrink: 0, borderRight: '1px solid var(--border)', overflow: 'auto',
      }}>
        <div style={{
          padding: '12px 16px 6px', fontSize: 11, fontWeight: 600,
          color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5,
        }}>
          Past Runs
        </div>
        {history.map((h, i) => {
          const dt = new Date(h.generated_at)
          const buys = h.signals.filter(s => s.signal === 'BUY').length
          const shorts = h.signals.filter(s => s.signal === 'SHORT').length
          return (
            <div
              key={h.generated_at + i}
              onClick={() => setSelected(i)}
              style={{
                padding: '9px 16px', cursor: 'pointer',
                background: selected === i ? 'var(--bg-selected)' : undefined,
                borderBottom: '1px solid var(--border)',
              }}
              onMouseEnter={e => {
                if (selected !== i) (e.currentTarget as HTMLElement).style.background = 'var(--bg-row-hover)'
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = selected === i ? 'var(--bg-selected)' : ''
              }}
            >
              <div style={{ fontWeight: 600, fontSize: 12 }}>{dt.toLocaleDateString()}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                {dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                {'  '}
                <span style={{ color: 'var(--green)' }}>{buys}↑</span>
                {' '}
                <span style={{ color: 'var(--red)' }}>{shorts}↓</span>
              </div>
            </div>
          )
        })}
      </div>

      {/* Signal table for selected run */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ padding: '14px 24px 8px' }}>
          <h2 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
            {new Date(entry.generated_at).toLocaleString()}
          </h2>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Symbol', 'Signal', 'Score', 'TFT Pred', 'XGB Pred'].map(h => (
                <th key={h} style={{
                  padding: '5px 24px', textAlign: 'left', fontWeight: 500,
                  color: 'var(--text-dim)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.5,
                }}>
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map(row => {
              const sigColor = row.signal === 'BUY' ? 'var(--green)' : row.signal === 'SHORT' ? 'var(--red)' : 'var(--text-muted)'
              return (
                <tr key={row.symbol} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px 24px', fontWeight: 700, fontSize: 13 }}>{row.symbol}</td>
                  <td style={{ padding: '8px 24px', color: sigColor, fontWeight: 700, fontSize: 11 }}>
                    {row.signal}
                  </td>
                  <td style={{ padding: '8px 24px', fontFamily: 'monospace', fontSize: 13 }}>
                    {row.ensemble_score != null ? `${(row.ensemble_score * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td style={{
                    padding: '8px 24px', fontFamily: 'monospace', fontSize: 13,
                    color: (row.tft_pred ?? 0) >= 0 ? 'var(--green)' : 'var(--red)',
                  }}>
                    {fmtPred(row.tft_pred)}
                  </td>
                  <td style={{
                    padding: '8px 24px', fontFamily: 'monospace', fontSize: 13,
                    color: (row.xgb_pred ?? 0) >= 0 ? 'var(--green)' : 'var(--red)',
                  }}>
                    {fmtPred(row.xgb_pred)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
