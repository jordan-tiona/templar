import type { LatestSignals, Portfolio, Position, ExplainData, HistoryEntry } from './types'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

export const api = {
  portfolio: () => get<Portfolio>('/portfolio'),
  positions: () => get<Position[]>('/positions'),
  latestSignals: () => get<LatestSignals>('/signals/latest'),
  history: () => get<HistoryEntry[]>('/signals/history'),
  explain: (symbol: string) => get<ExplainData>(`/signals/${symbol}/explain`),
}
