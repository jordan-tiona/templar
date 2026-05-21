import type { Position } from '../types'

function fmt$(n: number) {
  return '$' + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export default function PositionsTable({ positions }: { positions: Position[] }) {
  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      <div style={{ padding: '14px 24px 8px' }}>
        <h2 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
          Open Positions
        </h2>
      </div>

      {positions.length === 0 ? (
        <div style={{ padding: '4px 24px 20px', color: 'var(--text-muted)', fontSize: 13 }}>
          No open positions.
        </div>
      ) : (
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {['Symbol', 'Side', 'Qty', 'Entry', 'Current', 'Unrealized P&L'].map(h => (
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
            {positions.map(p => {
              const plColor = p.unrealized_pl >= 0 ? 'var(--green)' : 'var(--red)'
              return (
                <tr key={p.symbol} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '8px 24px', fontWeight: 700, fontSize: 13 }}>{p.symbol}</td>
                  <td style={{ padding: '8px 24px' }}>
                    <span style={{
                      color: p.side === 'long' ? 'var(--green)' : 'var(--red)',
                      fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                    }}>
                      {p.side}
                    </span>
                  </td>
                  <td style={{ padding: '8px 24px', fontFamily: 'monospace', fontSize: 13 }}>
                    {Math.abs(p.qty)}
                  </td>
                  <td style={{ padding: '8px 24px', fontFamily: 'monospace', fontSize: 13 }}>
                    {fmt$(p.avg_entry)}
                  </td>
                  <td style={{ padding: '8px 24px', fontFamily: 'monospace', fontSize: 13 }}>
                    {fmt$(p.current_price)}
                  </td>
                  <td style={{ padding: '8px 24px', fontFamily: 'monospace', fontSize: 13, color: plColor }}>
                    {p.unrealized_pl >= 0 ? '+' : '−'}{fmt$(p.unrealized_pl)}
                    <span style={{ fontSize: 11, marginLeft: 6 }}>
                      ({(p.unrealized_plpc * 100).toFixed(2)}%)
                    </span>
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
