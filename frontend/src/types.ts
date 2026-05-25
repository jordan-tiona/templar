export interface SignalRow {
  symbol: string
  signal: 'BUY' | 'SHORT' | 'HOLD'
  ensemble_score: number | null
  tft_pred: number | null
  xgb_pred: number | null
}

export interface LatestSignals {
  generated_at: string | null
  signals: SignalRow[]
}

export interface Position {
  symbol: string
  qty: number
  side: 'long' | 'short'
  avg_entry: number
  current_price: number
  market_value: number
  unrealized_pl: number
  unrealized_plpc: number
}

export interface Portfolio {
  equity: number
  cash: number
  buying_power: number
  last_equity: number
  day_pl: number
  day_pl_pct: number
}

export interface ShapEntry {
  feature: string
  value: number
}

export interface TftVarImportance {
  feature: string
  importance: number
}

export interface TftInterpretation {
  variable_importance: TftVarImportance[]
  attention: number[]
}

export interface ExplainData {
  symbol: string
  date: string
  signal: 'BUY' | 'SHORT' | 'HOLD'
  ensemble_score: number | null
  tft_pred: number | null
  xgb_pred: number | null
  shap: ShapEntry[]
  xgb_bias: number
  snapshot: Record<string, number | null>
  tft_interpretation?: TftInterpretation | null
  summary?: string | null
}

export interface HistoryEntry {
  generated_at: string
  signals: SignalRow[]
}
