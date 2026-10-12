import React from 'react'
import { fmtCompact, fmtNum, fmtProb } from '../lib/format.js'

/** Alerts raised during this run, newest first. */
export function Alerts({ alerts }) {
  if (!alerts || !alerts.length) {
    return (
      <div className="empty">
        No alerts raised. On a calm market that is the correct outcome — the
        agent is supposed to stay quiet.
      </div>
    )
  }
  return (
    <div className="rows">
      {alerts.map(a => {
        const cls = a.outcome === 'TRUE_POSITIVE' ? 'tp'
                  : a.outcome === 'FALSE_POSITIVE' ? 'fp' : ''
        const label = a.outcome ? a.outcome.replace('_', ' ') : 'open'
        return (
          <div key={a.alert_id ?? a.id} className="row">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 0 }}>
              <div style={{ display: 'flex', gap: 9, alignItems: 'baseline', flexWrap: 'wrap' }}>
                <strong>#{a.alert_id ?? a.id} {a.level}</strong>
                <code>p={fmtProb(a.probability)}</code>
                <code>stress {fmtNum(a.stress, 1)}</code>
              </div>
              {a.peak_dev > 0 && (
                <code style={{ fontSize: 10.5 }}>peak deviation ${fmtNum(a.peak_dev, 5)}</code>
              )}
            </div>
            <span className={`badge ${cls}`}>{label}</span>
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
          <div style={{ display: 'flex', gap: 9, alignItems: 'baseline', minWidth: 0 }}>
            <code style={{ fontWeight: 600 }}>{t.name}()</code>
            <span style={{ color: 'var(--muted)', fontSize: 11.5 }}>{t.summary}</span>
          </div>
          <code style={{ fontSize: 10.5 }}>{fmtNum(t.ms, 2)}ms</code>
        </div>
      ))}
    </div>
  )
}

/** Ensemble internals + the raw bar the models were given. */
export function Ensemble({ pred, bar }) {
  return (
    <div className="split">
      <div>
        <p className="eyebrow" style={{ marginBottom: 10 }}>Model outputs</p>
        <dl className="kv">
          <dt>Random Forest</dt><dd>{pred ? fmtNum(pred.rf_proba, 4) : '—'}</dd>
          <dt>XGBoost</dt><dd>{pred ? fmtNum(pred.xgb_proba, 4) : '—'}</dd>
          <dt>Meta-learner</dt><dd>{pred ? fmtProb(pred.depeg_proba) : '—'}</dd>
          <dt>Agreement (min)</dt><dd>{pred ? fmtNum(pred.agreement, 4) : '—'}</dd>
          <dt>Disagreement</dt><dd>{pred ? fmtNum(pred.disagreement, 4) : '—'}</dd>
          <dt>Past threshold</dt><dd>{pred ? (pred.breach ? 'yes' : 'no') : '—'}</dd>
        </dl>
      </div>
      <div>
        <p className="eyebrow" style={{ marginBottom: 10 }}>Raw bar in</p>
        <dl className="kv">
          <dt>Close</dt><dd>{bar ? bar.close.toFixed(5) : '—'}</dd>
          <dt>Volume (1m)</dt><dd>{bar ? fmtCompact(bar.volume) : '—'}</dd>
          <dt>Trades (1m)</dt><dd>{bar ? fmtNum(bar.trades, 0) : '—'}</dd>
          <dt>Taker buy</dt><dd>{bar ? fmtCompact(bar.taker_base) : '—'}</dd>
          <dt>Market cap</dt><dd>{bar ? '$' + fmtCompact(bar.market_cap) : '—'}</dd>
          <dt>Supply</dt><dd>{bar ? fmtCompact(bar.circulating_supply) : '—'}</dd>
        </dl>
      </div>
    </div>
  )
}

/** Cost weights, track record, and the sensing economy. */
export function Learning({ learning, status }) {
  const cm = learning?.cost_model || {}
  const st = learning?.stats || {}
  return (
    <div className="split">
      <div>
        <p className="eyebrow" style={{ marginBottom: 10 }}>Cost weights &amp; outcomes</p>
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
      </div>
      <div>
        <p className="eyebrow" style={{ marginBottom: 10 }}>Adaptive sensing</p>
        <dl className="kv">
          <dt>Bars perceived</dt><dd>{status?.ticks ?? 0}</dd>
          <dt>Model invocations</dt><dd>{status?.model_invocations ?? 0}</dd>
          <dt>Inference saved</dt>
          <dd>{status?.sensing_efficiency != null
            ? fmtNum(status.sensing_efficiency * 100, 0) + '%' : '—'}</dd>
        </dl>
        <p style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 12, lineHeight: 1.55 }}>
          Perception runs every bar; the models are invoked only when the
          agent&rsquo;s own poll schedule calls for it. Looking less often while
          calm is a decision the agent makes, not an optimisation applied to it.
        </p>
      </div>
    </div>
  )
}

/** Closest historical episode to the current market state. */
export function Analogue({ analogue }) {
  if (!analogue) {
    return (
      <div className="empty">
        Incident memory is queried only once the agent escalates to WARN or above.
      </div>
    )
  }
  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'space-between',
                    alignItems: 'baseline', gap: 12 }}>
        <strong style={{ fontSize: 14.5, color: 'var(--ink)' }}>{analogue.incident}</strong>
        <code style={{ fontSize: 11, color: 'var(--muted)' }}>{analogue.date}</code>
      </div>
      <div style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 3 }}>
        similarity {fmtNum((analogue.similarity ?? 0) * 100, 0)}% &middot;
        trough ${fmtNum(analogue.trough, 2)}
      </div>
      <div className="quote">{analogue.note}</div>
    </>
  )
}
