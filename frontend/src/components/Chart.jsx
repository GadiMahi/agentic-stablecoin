import React, { useMemo, useRef, useState } from 'react'
import { fmtProb, simClock } from '../lib/format.js'

/**
 * Price and depeg probability over simulated time.
 *
 * Two measures of very different scale, so they get two stacked panels sharing
 * one x-axis — never a dual y-axis on one plot.
 *
 * The probability panel is log-scaled: the ensemble is precision-tuned around
 * 1.8e-3 and moves over roughly a decade and a half, which a linear axis would
 * render as a flat line on the floor.
 *
 * Hover gives a crosshair and a tooltip reading both series at that minute.
 */

const SEV = {
  HOLD: '#A49C8B', WATCH: '#C58A2E', WARN: '#B2541A', CRITICAL: '#7E2112',
}
const INK = '#141210'
const PROB = '#A1561C'

const W = 860, H = 232
const PRICE_H = 126, GAP = 20, PAD_L = 8, PAD_R = 58, TOP = 10

export default function Chart({ series, threshold, startMinute = 0 }) {
  const [hover, setHover] = useState(null)
  const wrapRef = useRef(null)

  const v = useMemo(() => {
    if (!series || series.length < 2) return null
    const pts = series.slice(-480)
    const n = pts.length

    const closes = pts.map(p => p.close)
    let lo = Math.min(...closes), hi = Math.max(...closes)
    const span = Math.max(hi - lo, 2e-4)
    lo -= span * 0.14; hi += span * 0.14

    const probs = pts.map(p => Math.max(p.p ?? 1e-6, 1e-6))
    const lLo = Math.log10(Math.min(...probs, threshold || 1e-3)) - 0.18
    const lHi = Math.log10(Math.max(...probs, threshold || 1e-3)) + 0.18

    const x = i => PAD_L + (i / Math.max(n - 1, 1)) * (W - PAD_L - PAD_R)
    const yP = val => TOP + (1 - (val - lo) / (hi - lo)) * (PRICE_H - TOP - 6)
    const yQ = val => {
      const t = (Math.log10(Math.max(val, 1e-6)) - lLo) / Math.max(lHi - lLo, 1e-9)
      return PRICE_H + GAP + (1 - t) * (H - PRICE_H - GAP - 16)
    }

    const line = (sel, yf) =>
      pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yf(sel(p)).toFixed(1)}`).join('')

    // Contiguous runs of the same action -> background bands.
    const bands = []
    let s = 0
    for (let i = 1; i <= n; i++) {
      if (i === n || pts[i].action !== pts[s].action) {
        if (pts[s].action && pts[s].action !== 'HOLD') {
          bands.push({ x0: x(s), x1: x(i - 1), action: pts[s].action })
        }
        s = i
      }
    }

    return {
      pts, n, x, yP, yQ, lo, hi,
      pricePath: line(p => p.close, yP),
      probPath: line(p => Math.max(p.p ?? 1e-6, 1e-6), yQ),
      bands,
      pegY: lo <= 1 && hi >= 1 ? yP(1) : null,
      thrY: threshold ? yQ(threshold) : null,
    }
  }, [series, threshold])

  if (!v) {
    return (
      <div className="empty">
        Waiting for the feed — the models need 24h of history before the first
        complete feature vector.
      </div>
    )
  }

  const onMove = e => {
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    const frac = (px - PAD_L) / (W - PAD_L - PAD_R)
    const i = Math.round(frac * (v.n - 1))
    if (i < 0 || i >= v.n) { setHover(null); return }
    setHover({ i, left: (v.x(i) / W) * rect.width, pxRight: v.x(i) > W * 0.62 })
  }

  const hp = hover ? v.pts[hover.i] : null

  return (
    <div className="chart-wrap" ref={wrapRef}>
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={onMove}
           onMouseLeave={() => setHover(null)} role="img"
           aria-label="Price and depeg probability over simulated time">

        {v.bands.map((b, i) => (
          <rect key={i} x={b.x0} y={0} width={Math.max(b.x1 - b.x0, 1)} height={H}
                fill={SEV[b.action]} opacity={b.action === 'CRITICAL' ? 0.13 : 0.075} />
        ))}

        {/* ---- price ---- */}
        {v.pegY != null && (
          <>
            <line x1={PAD_L} y1={v.pegY} x2={W - PAD_R} y2={v.pegY}
                  stroke="#D3C5AE" strokeWidth="1" strokeDasharray="3 3" />
            <text x={W - PAD_R + 7} y={v.pegY + 3.5} fontSize="9.5" fill="#867B6B"
                  fontFamily="JetBrains Mono, monospace">$1.0000</text>
          </>
        )}
        <path d={v.pricePath} fill="none" stroke={INK} strokeWidth="1.7"
              strokeLinejoin="round" strokeLinecap="round" />
        <text x={W - PAD_R + 7} y={TOP + 4} fontSize="9.5" fill="#867B6B"
              fontFamily="JetBrains Mono, monospace">{v.hi.toFixed(4)}</text>
        <text x={W - PAD_R + 7} y={PRICE_H - 4} fontSize="9.5" fill="#867B6B"
              fontFamily="JetBrains Mono, monospace">{v.lo.toFixed(4)}</text>

        {/* ---- probability ---- */}
        <line x1={PAD_L} y1={PRICE_H + GAP - 9} x2={W - PAD_R} y2={PRICE_H + GAP - 9}
              stroke="#E4DACA" strokeWidth="1" />
        {v.thrY != null && (
          <>
            <line x1={PAD_L} y1={v.thrY} x2={W - PAD_R} y2={v.thrY}
                  stroke="#7E2112" strokeWidth="1" strokeDasharray="3 3" opacity=".65" />
            <text x={W - PAD_R + 7} y={v.thrY + 3.5} fontSize="9" fill="#7E2112"
                  fontFamily="JetBrains Mono, monospace">threshold</text>
          </>
        )}
        <path d={v.probPath} fill="none" stroke={PROB} strokeWidth="1.7"
              strokeLinejoin="round" strokeLinecap="round" />

        {/* ---- hover crosshair ---- */}
        {hp && (
          <g pointerEvents="none">
            <line x1={v.x(hover.i)} y1={0} x2={v.x(hover.i)} y2={H}
                  stroke={INK} strokeWidth="1" opacity=".26" />
            <circle cx={v.x(hover.i)} cy={v.yP(hp.close)} r="4"
                    fill={INK} stroke="#FDFAF4" strokeWidth="2" />
            <circle cx={v.x(hover.i)} cy={v.yQ(Math.max(hp.p ?? 1e-6, 1e-6))} r="4"
                    fill={PROB} stroke="#FDFAF4" strokeWidth="2" />
          </g>
        )}
      </svg>

      {hp && (
        <div className="tooltip"
             style={{
               left: hover.pxRight ? undefined : hover.left + 14,
               right: hover.pxRight ? `calc(100% - ${hover.left - 14}px)` : undefined,
               top: 6,
             }}>
          <div className="t-head">{simClock(startMinute + hover.i)} sim</div>
          <div className="t-row"><span>price</span><b>{hp.close.toFixed(5)}</b></div>
          <div className="t-row"><span>p(depeg)</span><b>{fmtProb(hp.p)}</b></div>
          <div className="t-row"><span>stress</span><b>{(hp.stress ?? 0).toFixed(1)}</b></div>
          <div className="t-row"><span>agent</span><b>{hp.action}</b></div>
        </div>
      )}

      <div className="legend">
        <span><i style={{ background: INK }} />price</span>
        <span><i style={{ background: PROB }} />p(depeg), log scale</span>
        <span><i style={{ background: '#7E2112' }} />decision threshold</span>
        <span><i style={{ background: '#C58A2E', opacity: .5 }} />shaded = agent elevated</span>
      </div>
    </div>
  )
}
