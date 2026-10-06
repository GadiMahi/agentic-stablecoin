import React, { useMemo } from 'react'

/**
 * Price and probability over simulated time.
 *
 * Hand-rolled SVG rather than a charting library: the dataset is a simple
 * series, and this keeps the dependency list to react + react-dom.
 *
 * The probability track is log-scaled. The ensemble is precision-tuned with a
 * threshold of 1.8e-3 and moves over roughly one and a half decades, so a
 * linear axis would render the entire run as a flat line on the floor.
 */

const ACTION_COLOR = {
  HOLD: '#8A8171', WATCH: '#9A7F46', WARN: '#8A5A22', CRITICAL: '#7A2E1C',
}

export default function Chart({ series, threshold }) {
  const W = 760, H = 150, PH = 92, GAP = 16, PAD = 38

  const view = useMemo(() => {
    if (!series || series.length < 2) return null
    const pts = series.slice(-420)
    const n = pts.length

    const closes = pts.map(p => p.close)
    let lo = Math.min(...closes), hi = Math.max(...closes)
    const span = Math.max(hi - lo, 1e-4)
    lo -= span * 0.12; hi += span * 0.12

    const probs = pts.map(p => Math.max(p.p ?? 1e-6, 1e-6))
    const logLo = Math.log10(Math.min(...probs, threshold || 1e-3)) - 0.15
    const logHi = Math.log10(Math.max(...probs, threshold || 1e-3)) + 0.15

    const x = i => PAD + (i / Math.max(n - 1, 1)) * (W - PAD - 10)
    const yP = v => 8 + (1 - (v - lo) / (hi - lo)) * (PH - 16)
    const yQ = v => {
      const t = (Math.log10(Math.max(v, 1e-6)) - logLo) / Math.max(logHi - logLo, 1e-9)
      return PH + GAP + (1 - t) * (H - PH - GAP - 14)
    }

    const pricePath = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yP(p.close).toFixed(1)}`).join('')
    const probPath = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yQ(p.p ?? 1e-6).toFixed(1)}`).join('')

    // Contiguous runs of the same action, drawn as background bands.
    const bands = []
    let start = 0
    for (let i = 1; i <= n; i++) {
      if (i === n || pts[i].action !== pts[start].action) {
        if (pts[start].action !== 'HOLD') {
          bands.push({ x0: x(start), x1: x(i - 1), action: pts[start].action })
        }
        start = i
      }
    }

    return { pts, x, yP, yQ, pricePath, probPath, bands, lo, hi, logLo, logHi,
             pegY: (lo <= 1 && hi >= 1) ? yP(1) : null,
             thrY: threshold ? yQ(threshold) : null }
  }, [series, threshold])

  if (!view) {
    return <div className="empty">Waiting for the feed to warm up — the models need
      24h of history before the first complete feature vector.</div>
  }

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {view.bands.map((b, i) => (
          <rect key={i} x={b.x0} y={0} width={Math.max(b.x1 - b.x0, 1)} height={H}
                fill={ACTION_COLOR[b.action]} opacity={b.action === 'CRITICAL' ? 0.14 : 0.08} />
        ))}

        {/* price panel */}
        {view.pegY != null && (
          <>
            <line x1={PAD} y1={view.pegY} x2={W - 10} y2={view.pegY}
                  stroke="#C9BAA2" strokeWidth="1" strokeDasharray="4 3" />
            <text x={4} y={view.pegY + 3} fontSize="8.5" fill="#7C7161"
                  fontFamily="JetBrains Mono, monospace">$1.00</text>
          </>
        )}
        <path d={view.pricePath} fill="none" stroke="#14120F" strokeWidth="1.5"
              strokeLinejoin="round" strokeLinecap="round" />
        <text x={W - 10} y={12} fontSize="8.5" fill="#7C7161" textAnchor="end"
              fontFamily="JetBrains Mono, monospace">{view.hi.toFixed(4)}</text>
        <text x={W - 10} y={PH - 4} fontSize="8.5" fill="#7C7161" textAnchor="end"
              fontFamily="JetBrains Mono, monospace">{view.lo.toFixed(4)}</text>

        {/* probability panel */}
        <line x1={PAD} y1={PH + GAP - 7} x2={W - 10} y2={PH + GAP - 7}
              stroke="#DED2BE" strokeWidth="1" />
        {view.thrY != null && (
          <>
            <line x1={PAD} y1={view.thrY} x2={W - 10} y2={view.thrY}
                  stroke="#7A2E1C" strokeWidth="1" strokeDasharray="3 3" opacity=".7" />
            <text x={4} y={view.thrY + 3} fontSize="8" fill="#7A2E1C"
                  fontFamily="JetBrains Mono, monospace">thr</text>
          </>
        )}
        <path d={view.probPath} fill="none" stroke="#8A5A22" strokeWidth="1.5"
              strokeLinejoin="round" strokeLinecap="round" />
      </svg>

      <div className="legend" style={{ marginTop: 8 }}>
        <span><i style={{ background: '#14120F' }} />price</span>
        <span><i style={{ background: '#8A5A22' }} />p(depeg), log scale</span>
        <span><i style={{ background: '#7A2E1C' }} />decision threshold</span>
        <span><i style={{ background: 'rgba(138,90,34,.35)' }} />shaded = agent elevated</span>
      </div>
    </>
  )
}
