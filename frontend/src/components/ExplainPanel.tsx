import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Cell, ReferenceLine, ResponsiveContainer,
  AreaChart, Area,
} from 'recharts'
import type { ExplainData } from '../types'

// Shorten verbose feature names for the chart axis
const SHORT_NAMES: Record<string, string> = {
  MACD_12_26_9: 'MACD',
  MACDh_12_26_9: 'MACD hist',
  MACDs_12_26_9: 'MACD signal',
  RSI_14: 'RSI 14',
  STOCHk_14_3_3: 'Stoch %K',
  STOCHd_14_3_3: 'Stoch %D',
  BBP_20_2_0_2_0: 'BB %B',
  BBB_20_2_0_2_0: 'BB Width',
  BBL_20_2_0_2_0: 'BB Lower',
  BBM_20_2_0_2_0: 'BB Mid',
  BBU_20_2_0_2_0: 'BB Upper',
  ATRr_14: 'ATR 14',
  EMA_9: 'EMA 9',
  EMA_21: 'EMA 21',
  EMA_50: 'EMA 50',
  return_1d: 'Return 1d',
  return_5d: 'Return 5d',
  return_20d: 'Return 20d',
  log_volume: 'Log Vol',
  sentiment_mean: 'Sentiment',
  sentiment_count: 'News count',
  VWAP_D: 'VWAP',
}

function shortName(f: string) { return SHORT_NAMES[f] ?? f }

function fmtPct(v: number | null | undefined, dp = 2) {
  if (v == null) return '—'
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(dp) + '%'
}

function fmtNum(v: number | null | undefined, dp = 2) {
  if (v == null) return '—'
  return v.toFixed(dp)
}

function fmtDollar(v: number | null | undefined) {
  if (v == null) return '—'
  return '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

function rsiLabel(v: number | null | undefined) {
  if (v == null) return ''
  if (v >= 70) return 'overbought'
  if (v <= 30) return 'oversold'
  if (v >= 55) return 'bullish'
  if (v <= 45) return 'bearish'
  return 'neutral'
}

function emaLabel(s: Record<string, number | null>) {
  const [e9, e21, e50] = [s['EMA_9'], s['EMA_21'], s['EMA_50']]
  if (e9 == null || e21 == null || e50 == null) return '—'
  if (e9 > e21 && e21 > e50) return 'bullish (9>21>50)'
  if (e9 < e21 && e21 < e50) return 'bearish (9<21<50)'
  return 'mixed'
}

function signalColor(s: string) {
  if (s === 'BUY') return 'var(--green)'
  if (s === 'SHORT') return 'var(--red)'
  return 'var(--text-muted)'
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function ShapTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={{
      background: '#1a1e38', border: '1px solid var(--border)',
      borderRadius: 6, padding: '8px 12px', fontSize: 12,
    }}>
      <div style={{ fontWeight: 600, marginBottom: 3 }}>{d.fullName}</div>
      <div style={{ color: d.value >= 0 ? 'var(--green)' : 'var(--red)' }}>
        {d.value >= 0 ? '+' : ''}{(d.value * 100).toFixed(4)}%
      </div>
    </div>
  )
}

interface Props {
  symbol: string
  data: ExplainData | null
  loading: boolean
  onClose: () => void
}

export default function ExplainPanel({ symbol, data, loading, onClose }: Props) {
  const snap = data?.snapshot ?? {}

  const shapData = (data?.shap ?? []).slice(0, 14).map(s => ({
    fullName: s.feature,
    name: shortName(s.feature),
    value: s.value,
  }))

  const chartHeight = Math.max(180, shapData.length * 26 + 20)

  const snapshotRows = [
    ['Close',       fmtDollar(snap['close'])],
    ['RSI 14',      `${fmtNum(snap['RSI_14'], 1)} (${rsiLabel(snap['RSI_14'])})`],
    ['Return 1d',   fmtPct(snap['return_1d'])],
    ['Stoch %K',    fmtNum(snap['STOCHk_14_3_3'], 1)],
    ['Return 5d',   fmtPct(snap['return_5d'])],
    ['Stoch %D',    fmtNum(snap['STOCHd_14_3_3'], 1)],
    ['Return 20d',  fmtPct(snap['return_20d'])],
    ['MACD',        fmtNum(snap['MACD_12_26_9'], 3)],
    ['EMA 9',       fmtDollar(snap['EMA_9'])],
    ['MACD hist',   fmtNum(snap['MACDh_12_26_9'], 3)],
    ['EMA 21',      fmtDollar(snap['EMA_21'])],
    ['BB %B',       `${fmtNum(snap['BBP_20_2_0_2_0'], 2)} (0=lo, 1=hi)`],
    ['EMA 50',      fmtDollar(snap['EMA_50'])],
    ['BB Width',    fmtNum(snap['BBB_20_2_0_2_0'], 2)],
    ['EMA trend',   emaLabel(snap as Record<string, number | null>)],
    ['ATR 14',      fmtNum(snap['ATRr_14'], 3)],
    ['Sentiment',   `${fmtNum(snap['sentiment_mean'], 3)} (${snap['sentiment_count'] ?? 0} articles)`],
  ]

  return (
    <div style={{
      width: 440, flexShrink: 0, display: 'flex', flexDirection: 'column',
      background: 'var(--bg-card)', borderLeft: '1px solid var(--border)',
      overflow: 'hidden',
    }}>
      {/* Sticky header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '13px 20px', borderBottom: '1px solid var(--border)',
        background: 'var(--bg-card)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontWeight: 800, fontSize: 15 }}>{symbol}</span>
          {data && (
            <span style={{ color: signalColor(data.signal), fontWeight: 700, fontSize: 13 }}>
              {data.signal}
            </span>
          )}
          {data?.ensemble_score != null && (
            <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
              {Math.round(data.ensemble_score * 100)}%
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'var(--text-muted)', fontSize: 16, lineHeight: 1, padding: 4,
          }}
        >
          ✕
        </button>
      </div>

      {/* Scrollable body */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {loading && (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            Loading…
          </div>
        )}

        {!loading && !data && (
          <div style={{ padding: 32, color: 'var(--text-muted)', fontSize: 13 }}>
            No explanation data for {symbol}.
            <br />Run the pipeline to generate it.
          </div>
        )}

        {!loading && data && (
          <>
            {/* Plain-English summary */}
            {data.summary && (
              <div style={{
                margin: '14px 20px 0',
                padding: '12px 14px',
                background: 'rgba(99,102,241,0.08)',
                border: '1px solid rgba(99,102,241,0.25)',
                borderRadius: 8,
                fontSize: 13,
                lineHeight: 1.6,
                color: 'var(--text)',
              }}>
                {data.summary}
              </div>
            )}

            {/* Model predictions */}
            <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
              {[
                { label: 'TFT Pred',  value: fmtPct(data.tft_pred) },
                { label: 'XGB Pred',  value: fmtPct(data.xgb_pred) },
                { label: 'Ensemble',  value: fmtNum(data.ensemble_score) },
              ].map(({ label, value }, i) => (
                <div key={label} style={{
                  flex: 1, padding: '10px 16px',
                  borderRight: i < 2 ? '1px solid var(--border)' : undefined,
                }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 3 }}>
                    {label}
                  </div>
                  <div style={{ fontWeight: 700, fontSize: 14, fontFamily: 'monospace' }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            {/* SHAP chart */}
            <div style={{ padding: '16px 20px 4px' }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 12 }}>
                Feature Contributions (SHAP)
              </div>
              <ResponsiveContainer width="100%" height={chartHeight}>
                <BarChart
                  layout="vertical"
                  data={shapData}
                  margin={{ top: 0, right: 16, left: 4, bottom: 0 }}
                >
                  <CartesianGrid horizontal={false} stroke="var(--border)" strokeOpacity={0.6} />
                  <XAxis
                    type="number"
                    tickFormatter={v => `${(v * 100).toFixed(2)}%`}
                    tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={88}
                    tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip content={<ShapTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                  <ReferenceLine x={0} stroke="var(--border)" />
                  <Bar dataKey="value" radius={[0, 2, 2, 0]} maxBarSize={14}>
                    {shapData.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={entry.value >= 0 ? 'var(--green)' : 'var(--red)'}
                        fillOpacity={0.75}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4 }}>
                Model base: {(data.xgb_bias * 100).toFixed(4)}%
              </div>
            </div>

            {/* TFT variable importance */}
            {data.tft_interpretation && data.tft_interpretation.variable_importance.length > 0 && (() => {
              const impData = data.tft_interpretation!.variable_importance.slice(0, 12).map(d => ({
                name: shortName(d.feature),
                fullName: d.feature,
                value: d.importance,
              }))
              const impHeight = Math.max(160, impData.length * 24 + 20)
              return (
                <div style={{ padding: '12px 20px 4px', borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 12 }}>
                    TFT Variable Importance
                  </div>
                  <ResponsiveContainer width="100%" height={impHeight}>
                    <BarChart
                      layout="vertical"
                      data={impData}
                      margin={{ top: 0, right: 16, left: 4, bottom: 0 }}
                    >
                      <CartesianGrid horizontal={false} stroke="var(--border)" strokeOpacity={0.6} />
                      <XAxis
                        type="number"
                        tickFormatter={v => v.toFixed(3)}
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        type="category"
                        dataKey="name"
                        width={88}
                        tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        content={({ active, payload }: any) => {
                          if (!active || !payload?.length) return null
                          const d = payload[0].payload
                          return (
                            <div style={{ background: '#1a1e38', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 12px', fontSize: 12 }}>
                              <div style={{ fontWeight: 600, marginBottom: 3 }}>{d.fullName}</div>
                              <div style={{ color: 'var(--accent)' }}>{d.value.toFixed(4)}</div>
                            </div>
                          )
                        }}
                        cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                      />
                      <Bar dataKey="value" radius={[0, 2, 2, 0]} maxBarSize={14} fill="var(--accent)" fillOpacity={0.7} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )
            })()}

            {/* TFT attention weights */}
            {data.tft_interpretation && data.tft_interpretation.attention.length > 0 && (() => {
              const att = data.tft_interpretation!.attention
              const attData = att.map((v, i) => ({ t: i - att.length + 1, weight: v }))
              return (
                <div style={{ padding: '12px 20px 4px', borderTop: '1px solid var(--border)' }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 12 }}>
                    TFT Encoder Attention
                    <span style={{ fontWeight: 400, marginLeft: 6, textTransform: 'none', letterSpacing: 0 }}>
                      (days before signal)
                    </span>
                  </div>
                  <ResponsiveContainer width="100%" height={120}>
                    <AreaChart data={attData} margin={{ top: 4, right: 16, left: 4, bottom: 0 }}>
                      <defs>
                        <linearGradient id="attGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--accent)" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="var(--accent)" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid stroke="var(--border)" strokeOpacity={0.5} />
                      <XAxis
                        dataKey="t"
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        interval={Math.floor(att.length / 6)}
                      />
                      <YAxis
                        tick={{ fill: 'var(--text-muted)', fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        width={36}
                        tickFormatter={v => v.toFixed(2)}
                      />
                      <Tooltip
                        formatter={(v: number) => [v.toFixed(4), 'Attention']}
                        labelFormatter={(t: number) => `${Math.abs(t)}d ago`}
                        contentStyle={{ background: '#1a1e38', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
                      />
                      <Area type="monotone" dataKey="weight" stroke="var(--accent)" strokeWidth={1.5} fill="url(#attGrad)" dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                  <div style={{ fontSize: 11, color: 'var(--text-dim)', marginTop: 4, marginBottom: 8 }}>
                    Peak at {(() => { const pk = attData.reduce((a, b) => b.weight > a.weight ? b : a); return `${Math.abs(pk.t)}d ago` })()}
                  </div>
                </div>
              )
            })()}

            {/* Market snapshot */}
            <div style={{ padding: '12px 20px 24px', borderTop: '1px solid var(--border)', marginTop: 8 }}>
              <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>
                Market Snapshot · {data.date}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 12px' }}>
                {snapshotRows.map(([label, value]) => (
                  <div
                    key={label}
                    style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      padding: '4px 0', borderBottom: '1px solid var(--border)',
                    }}
                  >
                    <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{label}</span>
                    <span style={{ fontSize: 11, fontFamily: 'monospace', color: 'var(--text)' }}>{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
