import React from 'react'
import { fmtNum } from '../lib/format.js'

const SEV = { HOLD: 'var(--hold)', WATCH: 'var(--watch)',
              WARN: 'var(--warn)', CRITICAL: 'var(--critical)' }

/**
 * The reasoning trace.
 *
 * The agent scores every available action by expected utility and takes the
 * argmax. All four scores are shown, not just the winner, so the ranking can be
 * watched flipping as probability and stress move — which is the difference
 * between a decision and a threshold.
 */
export default function Utility({ decision }) {
  if (!decision) return <div className="empty">No decision yet.</div>

  const scores = decision.scores || []
  const eus = scores.map(s => s.expected_utility)
  const best = Math.max(...eus), worst = Math.min(...eus)
  const span = Math.max(best - worst, 1e-6)

  return (
    <>
      <div className="ladder">
        {scores.map(s => {
          const on = s.action === decision.action
          const w = ((s.expected_utility - worst) / span) * 100
          return (
            <div key={s.action} className={`rung${on ? ' on' : ''}`}>
              <div className="r-name">
                <i style={{ background: SEV[s.action] }} />{s.action}
              </div>
              <div className="r-track"><i style={{ width: `${w.toFixed(1)}%` }} /></div>
              <div className="r-eu">{fmtNum(s.expected_utility, 1)}</div>
            </div>
          )
        })}
      </div>
      <div className="quote">{decision.rationale}</div>
    </>
  )
}
