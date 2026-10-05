export const fmtPrice = (v) => (v == null ? '—' : v.toFixed(5))

// The ensemble is precision-tuned and emits small probabilities, so plain
// percentages collapse to "0.0%". Scientific notation keeps the movement legible.
export const fmtProb = (v) => {
  if (v == null) return '—'
  if (v === 0) return '0'
  return v.toExponential(2).replace('e-', ' e-')
}

export const fmtNum = (v, d = 2) =>
  v == null || Number.isNaN(v) ? '—' : Number(v).toLocaleString('en-US',
    { minimumFractionDigits: d, maximumFractionDigits: d })

export const fmtCompact = (v) => {
  if (v == null) return '—'
  const a = Math.abs(v)
  if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B'
  if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M'
  if (a >= 1e3) return (v / 1e3).toFixed(1) + 'k'
  return v.toFixed(2)
}

export const simClock = (minute) => {
  const h = Math.floor(minute / 60)
  const m = minute % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
}

// Attention tier drives the visual emphasis. Thresholds are on |z| — how
// abnormal the metric is against its own 4h robust baseline.
export const attnTier = (z) => {
  const a = Math.abs(z || 0)
  if (a >= 4.0) return 4
  if (a >= 2.5) return 3
  if (a >= 1.5) return 2
  if (a >= 0.8) return 1
  return 0
}
