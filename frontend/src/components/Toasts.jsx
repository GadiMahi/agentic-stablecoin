import React, { useEffect } from 'react'

/**
 * In-page alert toast.
 *
 * The OS notification is a nice-to-have: it depends on a per-origin browser
 * permission, a separate OS-level permission, and the user not being in Focus
 * or presentation mode — any of which silently swallows it, and all of which
 * are outside the application's control.
 *
 * This is the dependable surface. It renders inside the page, so it is visible
 * on a shared screen, needs no permission, and cannot be suppressed by the
 * operating system. The OS notification remains as an additional channel for
 * when the dashboard is not in the foreground.
 */

const AUTO_DISMISS_MS = 14000

export default function Toasts({ toasts, onDismiss }) {
  if (!toasts.length) return null
  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {toasts.map(t => (
        <Toast key={t.key} toast={t} onDismiss={() => onDismiss(t.key)} />
      ))}
    </div>
  )
}

function Toast({ toast, onDismiss }) {
  useEffect(() => {
    const id = setTimeout(onDismiss, AUTO_DISMISS_MS)
    return () => clearTimeout(id)
  }, [onDismiss])

  const n = toast.data
  return (
    <div className={`toast sev-${n.level}`}>
      <button className="toast-x" onClick={onDismiss} aria-label="Dismiss">&times;</button>
      <div className="toast-head">
        <i className="mark" />
        <span className="toast-level">{n.level}</span>
        <span className="toast-tag">
          {toast.test ? 'test delivery' : `alert #${n.alert_id}`}
        </span>
      </div>
      <div className="toast-title">Stablecoin depeg risk</div>
      <dl className="toast-kv">
        <dt>Price</dt>
        <dd>${Number(n.price).toFixed(5)} <span>({Number(n.deviation_pct).toFixed(2)}%)</span></dd>
        <dt>p(depeg)</dt>
        <dd>{Number(n.probability).toExponential(2)} <span>({n.ratio}&#215; thr)</span></dd>
        <dt>Stress</dt>
        <dd>{Number(n.stress).toFixed(1)}/100 <span>({n.stress_band})</span></dd>
      </dl>
      <div className="toast-why">{n.rationale}</div>
      {n.analogue && (
        <div className="toast-foot">
          closest episode: <b>{n.analogue.incident}</b> ({n.analogue.date})
        </div>
      )}
    </div>
  )
}
