import React from 'react'
import { attnTier, fmtNum } from '../lib/format.js'

/**
 * The agent's attention map.
 *
 * Each tick the agent publishes, per watched metric, a robust z-score against
 * that metric's own 4-hour baseline plus a normalised attention weight. The
 * emphasis below is driven straight from that z-score — it is an output of the
 * reasoning layer, not a styling decision taken here.
 *
 * Only the metrics the agent is actually reacting to are shown by default; the
 * quiet ones collapse into a single line so the panel stays readable.
 */

const FORMAT = {
  peg_deviation: v => `$${v.toFixed(5)}`,
  volume_vs_24h: v => `${v.toFixed(2)}x`,
  volume_imbalance: v => v.toFixed(3),
  taker_buy_ratio: v => v.toFixed(3),
  trade_size_vs_24h: v => `${v.toFixed(2)}x`,
  peg_deviation_volatility: v => v.toExponential(1),
  close_std_1440m: v => v.toExponential(1),
  circulating_supply_percent_change_1d: v => `${v.toFixed(3)}%`,
}

const VISIBLE = 5

export default function Attention({ attention, intensity, pollInterval }) {
  const all = Object.entries(attention || {})
  if (!all.length) {
    return <div className="empty">No attention signal yet.</div>
  }

  all.sort((a, b) => Math.abs(b[1].z) - Math.abs(a[1].z))
  const shown = all.slice(0, VISIBLE)
  const rest = all.length - shown.length
  const pct = Math.round((intensity || 0) * 100)

  return (
    <>
      <div className="focus-head">
        <div>
          <span className="focus-pct">{pct}%</span>
          <span style={{ fontSize: 11.5, color: 'var(--muted)', marginLeft: 7 }}>
            focus intensity
          </span>
        </div>
        <div className="note">every {pollInterval ?? '—'}s</div>
      </div>
      <div className="track"><i style={{ width: `${pct}%` }} /></div>

      <div className="metrics">
        {shown.map(([key, m]) => {
          const tier = attnTier(m.z)
          const fmt = FORMAT[key] || (v => fmtNum(v, 3))
          const w = Math.min(100, Math.round((m.weight || 0) * 220))
          return (
            <div key={key} className={`metric${tier ? ` t${tier}` : ''}`}>
              <div className="m-name">{m.label || key}</div>
              <div className="m-val">{fmt(m.value ?? 0)}</div>
              <div className="m-track"><i style={{ width: `${w}%` }} /></div>
              <div className="m-z">{m.z > 0 ? '+' : ''}{(m.z ?? 0).toFixed(1)}&#963;</div>
            </div>
          )
        })}
      </div>

      {rest > 0 && (
        <div style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 10, paddingLeft: 10 }}>
          {rest} further {rest === 1 ? 'metric' : 'metrics'} within normal range
        </div>
      )}
    </>
  )
}
