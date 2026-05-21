import type { LatestSignals, SignalRow } from '../types'

function SignalBadge({ signal }: { signal: string }) {
  const style: Record<string, React.CSSProperties> = {
    BUY:   { background: 'var(--green-dim)', color: 'var(--green)' },
    SHORT: { background: 'var(--red-dim)',   color: 'var(--red)' },
    HOLD:  { background: 'transparent',      color: 'var(--text-muted)' },
  }
  return (
    <span style={{
      ...(style[signal] ?? style.HOLD),
      padding: '2px 10px', borderRadius: 4,
      fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
    }}>
      {signal}
    </span>
  )
}

function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return <span style={{ color: 'var(--text-dim)' }}>—</span>
  const pct = Math.round(score * 100)
  const color = score >= 0.6 ? 'var(--green)' : score <= 0.4 ? 'var(--red)' : 'var(--text-muted)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ width: 72, height: 5, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3 }} />
      </div>
      <span style={{ color, fontSize: 12, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
        {pct}%
      </span>
    </div>
  )
}

function fmtPred(v: number | null) {
  if (v === null) return '—'
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(2) + '%'
}

interface Props {
  signals: LatestSignals | null
  selectedSymbol: string | null
  onSelect: (sym: string) => void
}

export default function SignalsTable({ signals, selectedSymbol, onSelect }: Props) {
  const rows: SignalRow[] = signals?.signals
    ? [...signals.signals].sort((a, b) => (b.ensemble_score ?? 0) - (a.ensemble_score ?? 0))
    : []

  return (
    <div>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '14px 24px 8px',
      }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          Signals
        </h2>
        {signals?.generated_at && (
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            {new Date(signals.generated_at).toLocaleString()}
          </span>
        )}
      </div>

      {rows.length === 0 ? (
        <div style={{ padding: '12px 24px 20px', color: 'var(--text-muted)', fontSize: 13 }}>
          No signals yet — run <code>python main.py run</code> to generate them.
        </div>
      ) : (
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
            {rows.map(row => {
              const selected = row.symbol === selectedSymbol
              return (
                <tr
                  key={row.symbol}
                  onClick={() => onSelect(row.symbol)}
                  style={{
                    background: selected ? 'var(--bg-selected)' : undefined,
                    borderBottom: '1px solid var(--border)',
                    cursor: 'pointer',
                  }}
                  onMouseEnter={e => {
                    if (!selected) (e.currentTarget as HTMLElement).style.background = 'var(--bg-row-hover)'
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLElement).style.background = selected ? 'var(--bg-selected)' : ''
                  }}
                >
                  <td style={{ padding: '9px 24px', fontWeight: 700, fontSize: 13 }}>{row.symbol}</td>
                  <td style={{ padding: '9px 24px' }}><SignalBadge signal={row.signal} /></td>
                  <td style={{ padding: '9px 24px' }}><ScoreBar score={row.ensemble_score} /></td>
                  <td style={{
                    padding: '9px 24px', fontFamily: 'monospace', fontSize: 13,
                    color: (row.tft_pred ?? 0) >= 0 ? 'var(--green)' : 'var(--red)',
                  }}>
                    {fmtPred(row.tft_pred)}
                  </td>
                  <td style={{
                    padding: '9px 24px', fontFamily: 'monospace', fontSize: 13,
                    color: (row.xgb_pred ?? 0) >= 0 ? 'var(--green)' : 'var(--red)',
                  }}>
                    {fmtPred(row.xgb_pred)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      )}
    </div>
  )
}
