import type { Portfolio } from '../types'

function fmt$(n: number) {
  return '$' + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function fmtPct(n: number) {
  return (n >= 0 ? '+' : '') + (n * 100).toFixed(2) + '%'
}

interface CardProps {
  label: string
  value: string
  sub?: string
  color?: string
}

function Card({ label, value, sub, color }: CardProps) {
  return (
    <div style={{
      background: 'var(--bg-card)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '10px 18px', minWidth: 148,
    }}>
      <div style={{
        fontSize: 10, fontWeight: 600, textTransform: 'uppercase',
        letterSpacing: 1, color: 'var(--text-muted)', marginBottom: 4,
      }}>
        {label}
      </div>
      <div style={{ fontSize: 19, fontWeight: 700, color: color ?? 'var(--text)', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 12, color: color ?? 'var(--text-muted)', marginTop: 1 }}>{sub}</div>
      )}
    </div>
  )
}

export default function PortfolioBar({ portfolio: p }: { portfolio: Portfolio }) {
  const plColor = p.day_pl >= 0 ? 'var(--green)' : 'var(--red)'
  return (
    <div style={{
      display: 'flex', gap: 10, padding: '10px 24px',
      borderBottom: '1px solid var(--border)', flexShrink: 0, flexWrap: 'wrap',
    }}>
      <Card label="Portfolio Equity" value={fmt$(p.equity)} />
      <Card label="Cash" value={fmt$(p.cash)} />
      <Card label="Buying Power" value={fmt$(p.buying_power)} />
      <Card
        label="Day P&L"
        value={(p.day_pl >= 0 ? '+' : '−') + fmt$(p.day_pl)}
        sub={fmtPct(p.day_pl_pct)}
        color={plColor}
      />
    </div>
  )
}
