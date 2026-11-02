import React, { useCallback, useEffect, useState } from 'react'
import * as api from '../lib/api.js'
import { fmtNum } from '../lib/format.js'

/**
 * Automated response.
 *
 * Shows which delivery channels are live, lets the operator grant browser
 * notification permission, and fires a labelled test alert through every
 * configured channel — so the response layer can be demonstrated on demand
 * instead of by waiting for the market to break.
 *
 * Delivery receipts come from the server and include failures. A channel that
 * could not deliver says so; nothing is reported as sent that was not.
 */
export default function Response({ notifyState, onRefresh }) {
  const [perm, setPerm] = useState(
    typeof Notification === 'undefined' ? 'unsupported' : Notification.permission)
  const [testing, setTesting] = useState(false)
  const [lastTest, setLastTest] = useState(null)

  useEffect(() => { onRefresh?.() }, [onRefresh])

  const askPermission = useCallback(async () => {
    if (typeof Notification === 'undefined') return
    const p = await Notification.requestPermission()
    setPerm(p)
  }, [])

  const sendTest = useCallback(async () => {
    setTesting(true); setLastTest(null)
    try {
      const r = await api.testNotify()
      setLastTest(r)
      onRefresh?.()
    } catch (e) {
      setLastTest({ ok: false, receipts: [{ channel: 'request', ok: false,
                                            detail: String(e), ms: 0 }] })
    } finally { setTesting(false) }
  }, [onRefresh])

  const channels = notifyState?.channels || []
  const receipts = notifyState?.receipts || []
  const active = notifyState?.active || []

  return (
    <>
      <div className="split">
        <div>
          <p className="eyebrow" style={{ marginBottom: 10 }}>Delivery channels</p>
          <div className="chan-list">
            {channels.map(c => (
              <div key={c.name} className={`chan${c.configured ? ' on' : ''}`}>
                <div className="chan-top">
                  <span className="chan-name">
                    <i className="chan-dot" />{c.name}
                  </span>
                  <span className="badge">{c.configured ? 'live' : 'not configured'}</span>
                </div>
                <div className="chan-hint">{c.hint || c.blurb}</div>
              </div>
            ))}
            {!channels.length && <div className="empty">Channel status unavailable.</div>}
          </div>

          <p style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 14, lineHeight: 1.55 }}>
            Escalations at <b>{notifyState?.min_level || 'WARN'}</b> or above are
            dispatched to every live channel. Unconfigured channels are skipped —
            never reported as delivered.
          </p>
        </div>

        <div>
          <p className="eyebrow" style={{ marginBottom: 10 }}>Verify the pipeline</p>

          <div className="chan-actions">
            {perm !== 'granted' && perm !== 'unsupported' && (
              <button className="btn" onClick={askPermission}>
                Enable desktop notifications
              </button>
            )}
            {perm === 'granted' && (
              <span className="badge tp">desktop notifications allowed</span>
            )}
            {perm === 'denied' && (
              <span className="badge fp">blocked in browser settings</span>
            )}
            <button className="btn primary" onClick={sendTest} disabled={testing}>
              {testing ? 'Sending…' : 'Send test alert'}
            </button>
          </div>

          <p style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 10, lineHeight: 1.55 }}>
            Sends a clearly-labelled test message through {active.length || 'no'}
            {active.length === 1 ? ' live channel' : ' live channels'}
            {active.length ? ` (${active.join(', ')})` : ''}.
          </p>

          {lastTest && (
            <div className="rows" style={{ marginTop: 12, maxHeight: 150 }}>
              {lastTest.receipts.map((r, i) => (
                <div key={i} className="row">
                  <div style={{ display: 'flex', gap: 9, alignItems: 'baseline' }}>
                    <code style={{ fontWeight: 600 }}>{r.channel}</code>
                    <span style={{ color: 'var(--muted)', fontSize: 11.5 }}>{r.detail}</span>
                  </div>
                  <span className={`badge ${r.ok ? 'tp' : 'fp'}`}>
                    {r.ok ? 'sent' : 'failed'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div style={{ height: 1, background: 'var(--line)', margin: '18px 0 14px' }} />
      <p className="eyebrow" style={{ marginBottom: 10 }}>Delivery log</p>
      {receipts.length ? (
        <div className="rows">
          {receipts.map((r, i) => (
            <div key={i} className="row">
              <div style={{ display: 'flex', gap: 9, alignItems: 'baseline', minWidth: 0 }}>
                <code style={{ fontWeight: 600 }}>{r.channel}</code>
                <code style={{ fontSize: 10.5 }}>
                  {r.alert_id ? `alert #${r.alert_id}` : 'test'}
                </code>
                <span style={{ color: 'var(--muted)', fontSize: 11.5 }}>{r.detail}</span>
              </div>
              <div style={{ display: 'flex', gap: 9, alignItems: 'baseline' }}>
                <code style={{ fontSize: 10.5 }}>{fmtNum(r.ms, 0)}ms</code>
                <span className={`badge ${r.ok ? 'tp' : 'fp'}`}>{r.ok ? 'sent' : 'failed'}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="empty">
          No deliveries yet. Escalate past {notifyState?.min_level || 'WARN'} or
          send a test alert.
        </div>
      )}
    </>
  )
}
