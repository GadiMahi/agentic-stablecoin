import React from 'react'
import { fmtNum } from '../lib/format.js'

/**
 * The reasoning trace.
 *
 * The agent scores every available action by expected utility and takes the
 * argmax. This panel shows all four scores, not just the winner, so a reviewer
 * can see *why* the chosen action beat the alternatives — and watch the ranking
 * flip as the probability and stress move.
 */
export default function Utility({ decision }) {
  if (!decision) return <div className="empty">No decision yet.</div>

  const scores = decision.scores || []
  const eus = scores.map(s => s.expected_utility)
  const best = Math.max(...eus)
  const worst = Math.min(...eus)
  const span = Math.max(best - worst, 1e-6)

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span className={`action-chip ${decision.action}`}>{decision.action}</span>
        <span className="card-note">
          chosen by {fmtNum(decision.margin, 2)} utility over the runner-up
        </span>
      </div>

      <div className="ladder">
        {scores.map(s => {
          const chosen = s.action === decision.action
          // Normalised so the best action fills the track and the worst is empty.
          const w = ((s.expected_utility - worst) / span) * 100
          return (
            <div key={s.action} className={`rung${chosen ? ' chosen' : ''}`}>
              <div className="r-name">{s.action}</div>
              <div className="r-track"><i style={{ width: `${w.toFixed(1)}%` }} /></div>
              <div className="r-eu">{fmtNum(s.expected_utility, 1)}</div>
            </div>
          )
        })}
      </div>

      <div className="rationale">{decision.rationale}</div>
    </>
  )
}
