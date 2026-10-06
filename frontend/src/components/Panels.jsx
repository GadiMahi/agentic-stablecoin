import React from 'react'
import { fmtNum, fmtProb } from '../lib/format.js'

/** Scenario picker + transport controls. */
export function Controls({ scenarios, selected, onSelect, onStart, onPause,
                           onSpeed, running, paused, speed, busy }) {
  return (
    <>
      <div className="scenario-grid">
        {scenarios.map(s => (
          <button key={s.key}
                  className={`scenario${selected === s.key ? ' sel' : ''}`}
                  onClick={() => onSelect(s.key)}>
            <b>{s.name}</b>
            <span>{s.description}</span>
          </button>
        ))}
      </div>

      <div className="controls" style={{ marginTop: 12 }}>
        <button className="btn primary" onClick={onStart} disabled={busy}>
          {busy ? 'Warming up…' : running ? 'Restart simulation' : 'Start simulation'}
        </button>
        <button className="btn" onClick={onPause} disabled={!running || busy}>
          {paused ? 'Resume' : 'Pause'}
        </button>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8,
                        fontSize: 12, color: 'var(--muted)' }}>
          speed
          <input type="range" min="2" max="240" step="2" value={speed}
                 onChange={e => onSpeed(Number(e.target.value))}
                 style={{ width: 120 }} />
          <span className="stat-value" style={{ fontSize: 12, minWidth: 54 }}>
            {speed}&#215; min/s
          </span>
        </label>
      </div>
    </>
  )
}

/** Alerts raised during this run, newest first. */
export function Alerts({ alerts }) {
  if (!alerts || !alerts.length) {
    return <div className="empty">No alerts raised. On a calm market this is the
      correct outcome — the agent is supposed to stay quiet.</div>
  }
  return (
    <div className="rows">
      {alerts.map(a => {
        const tag = a.outcome === 'TRUE_POSITIVE' ? 'tp'
                  : a.outcome === 'FALSE_POSITIVE' ? 'fp' : 'open'
        const label = a.outcome ? a.outcome.replace('_', ' ') : 'open'
        return (
          <div key={a.alert_id ?? a.id} className="row">
            <div className="r-l" style={{ flexDirection: 'column', gap: 3 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'baseline' }}>
                <strong style={{ fontSize: 12 }}>#{a.alert_id ?? a.id} {a.level}</strong>
                <code>p={fmtProb(a.probability)}</code>
                <code>stress {fmtNum(a.stress, 1)}</code>
              </div>
              {a.peak_dev != null && a.peak_dev > 0 && (
                <code style={{ fontSize: 10.5 }}>
                  peak deviation ${fmtNum(a.peak_dev, 5)}
                </code>
              )}
            </div>
            <span className={`tag ${tag}`}>{label}</span>
          </div>
        )
      })}
    </div>
  )
}

/** The agent's tool-call audit trail. */
export function ToolLog({ tools }) {
  if (!tools || !tools.length) return <div className="empty">No tool calls yet.</div>
  return (
    <div className="rows">
      {[...tools].reverse().map((t, i) => (
        <div key={i} className="row">
          <div className="r-l">
            <code style={{ fontWeight: 600 }}>{t.name}()</code>
            <span style={{ color: 'var(--muted)', fontSize: 11.5 }}>{t.summary}</span>
          </div>
          <code style={{ fontSize: 10.5 }}>{fmtNum(t.ms, 2)}ms</code>
        </div>
      ))}
    </div>
  )
}

/** Cost weights and the track record that moves them. */
export function Learning({ learning, status }) {
  const cm = learning?.cost_model || {}
  const st = learning?.stats || {}
  return (
    <>
      <dl className="kv">
        <dt>Cost of a miss</dt><dd>{fmtNum(cm.miss_base, 0)}</dd>
        <dt>Cost of a false alarm</dt><dd>{fmtNum(cm.false_base, 1)}</dd>
        <dt>True positives</dt><dd>{st.true_positives ?? 0}</dd>
        <dt>False positives</dt><dd>{st.false_positives ?? 0}</dd>
        <dt>Missed breaches</dt><dd>{st.missed_breaches ?? 0}</dd>
        <dt>Precision</dt>
        <dd>{st.precision != null ? fmtNum(st.precision * 100, 1) + '%' : '—'}</dd>
        <dt>Weight updates</dt><dd>{st.updates ?? 0}</dd>
      </dl>
      {status && (
        <>
          <div style={{ height: 1, background: 'var(--line)', margin: '12px 0' }} />
          <dl className="kv">
            <dt>Bars perceived</dt><dd>{status.ticks ?? 0}</dd>
            <dt>Model invocations</dt><dd>{status.model_invocations ?? 0}</dd>
            <dt>Inference saved</dt>
            <dd>{status.sensing_efficiency != null
              ? fmtNum(status.sensing_efficiency * 100, 0) + '%' : '—'}</dd>
          </dl>
          <p className="stat-sub" style={{ marginTop: 8 }}>
            Perception runs every bar; the models are only invoked when the
            agent&rsquo;s own poll schedule calls for it. Looking less often while
            calm is a decision the agent makes, not an optimisation.
          </p>
        </>
      )}
    </>
  )
}

/** Closest historical episode to the current market state. */
export function Analogue({ analogue }) {
  if (!analogue) {
    return <div className="empty">The agent queries incident memory only once it
      escalates to WARN or above.</div>
  }
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'baseline', gap: 10 }}>
        <strong style={{ fontSize: 14, color: 'var(--ink)' }}>{analogue.incident}</strong>
        <code style={{ fontSize: 11 }}>{analogue.date}</code>
      </div>
      <div className="stat-sub" style={{ marginTop: 2 }}>
        similarity {fmtNum((analogue.similarity ?? 0) * 100, 0)}% &middot;
        trough ${fmtNum(analogue.trough, 2)}
      </div>
      <div className="rationale">{analogue.note}</div>
    </>
  )
}
