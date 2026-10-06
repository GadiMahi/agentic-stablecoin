import React from 'react'
import { attnTier, fmtNum } from '../lib/format.js'

/**
 * The agent's attention map.
 *
 * Every tick the agent publishes, per watched metric, a robust z-score against
 * that metric's own 4-hour baseline plus a normalised attention weight. This
 * panel is the direct visual expression of that output: the more abnormal a
 * number is, the larger and louder its row becomes.
 *
 * The emphasis is not decoration chosen in the browser — the tier comes
 * straight from the z-score the agent computed.
 */

const VALUE_FORMAT = {
  peg_deviation: v => `$${v.toFixed(5)}`,
  volume_vs_24h: v => `${v.toFixed(2)}x`,
  volume_imbalance: v => v.toFixed(3),
  taker_buy_ratio: v => v.toFixed(3),
  trade_size_vs_24h: v => `${v.toFixed(2)}x`,
  peg_deviation_volatility: v => v.toExponential(1),
  close_std_1440m: v => v.toExponential(1),
  circulating_supply_percent_change_1d: v => `${v.toFixed(3)}%`,
}

export default function Attention({ attention, intensity, pollInterval }) {
  const entries = Object.entries(attention || {})
  if (!entries.length) {
    return <div className="empty">No attention signal yet.</div>
  }

  // Order by how hard the agent is looking, so the loudest row is always on top.
  entries.sort((a, b) => Math.abs(b[1].z) - Math.abs(a[1].z))

  const pct = Math.round((intensity || 0) * 100)

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'baseline', marginBottom: 10 }}>
        <div>
          <span className="stat-value" style={{ fontSize: 22 }}>{pct}%</span>
          <span className="stat-sub" style={{ display: 'inline', marginLeft: 7 }}>
            focus intensity
          </span>
        </div>
        <div className="card-note">re-checking every {pollInterval ?? '—'}s</div>
      </div>

      <div className="warmbar" style={{ marginBottom: 12 }}>
        <i style={{ width: `${pct}%` }} />
      </div>

      {entries.map(([key, m]) => {
        const tier = attnTier(m.z)
        const fmt = VALUE_FORMAT[key] || (v => fmtNum(v, 3))
        const weight = Math.min(100, Math.round((m.weight || 0) * 100 * 2.2))
        return (
          <div key={key} className={`metric${tier ? ` attn-${tier}` : ''}`}>
            <div className="m-name">{m.label || key}</div>
            <div className="m-val">{fmt(m.value ?? 0)}</div>
            <div className="m-bar"><i style={{ width: `${weight}%` }} /></div>
            <div className="m-z">
              {m.z > 0 ? '+' : ''}{(m.z ?? 0).toFixed(1)}&#963;
            </div>
          </div>
        )
      })}
    </>
  )
}
