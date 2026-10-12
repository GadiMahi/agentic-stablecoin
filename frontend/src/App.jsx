import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Chart from './components/Chart.jsx'
import Attention from './components/Attention.jsx'
import Utility from './components/Utility.jsx'
import { Alerts, Analogue, Ensemble, Learning, ToolLog } from './components/Panels.jsx'
import * as api from './lib/api.js'
import { fmtNum, fmtProb, simClock } from './lib/format.js'

const TABS = [
  ['ensemble', 'Ensemble'],
  ['alerts', 'Alerts'],
  ['tools', 'Tool calls'],
  ['learning', 'Learning'],
  ['analogue', 'Incident memory'],
]

export default function App() {
  const [health, setHealth] = useState(null)
  const [scenarios, setScenarios] = useState([])
  const [selected, setSelected] = useState('svb_bleed')
  const [tick, setTick] = useState(null)
  const [series, setSeries] = useState([])
  const [status, setStatus] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [connected, setConnected] = useState(false)
  const [running, setRunning] = useState(false)
  const [paused, setPaused] = useState(false)
  const [speed, setSpeed] = useState(30)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState('ensemble')
  const wsRef = useRef(null)

  /* ------------------------------------------------------------ bootstrap */
  useEffect(() => {
    api.getHealth().then(setHealth).catch(() => {})
    api.getScenarios().then(d => setScenarios(d.scenarios)).catch(() => {})
    api.getStatus().then(s => {
      if (s.active) {
        setRunning(true); setPaused(s.paused); setSpeed(s.speed)
        setSelected(s.scenario); setStatus(s.agent)
        setSeries(s.series || []); setTick(s.last_tick || null)
      }
    }).catch(() => {})
  }, [])

  /* ---------------------------------------------------------------- socket */
  useEffect(() => {
    let closed = false, retry
    const connect = () => {
      const ws = new WebSocket(api.wsUrl())
      wsRef.current = ws
      ws.onopen = () => setConnected(true)
      ws.onclose = () => { setConnected(false); if (!closed) retry = setTimeout(connect, 1500) }
      ws.onerror = () => ws.close()
      ws.onmessage = ev => {
        let msg
        try { msg = JSON.parse(ev.data) } catch { return }
        if (msg.type === 'tick') {
          const d = msg.data
          setTick(d)
          if (d.scored && d.prediction) {
            setSeries(prev => {
              const next = [...prev, {
                ts: d.ts, close: d.bar.close, p: d.prediction.depeg_proba,
                stress: d.prediction.stress,
                action: d.decision?.action || 'HOLD', focus: d.focus_intensity,
              }]
              return next.length > 700 ? next.slice(next.length - 700) : next
            })
          }
          if (d.resolved?.length) api.getAlerts().then(r => setAlerts(r.alerts)).catch(() => {})
        } else if (msg.type === 'finished') {
          setRunning(false); setStatus(msg.status)
          api.getAlerts().then(r => setAlerts(r.alerts)).catch(() => {})
        }
      }
    }
    connect()
    return () => { closed = true; clearTimeout(retry); wsRef.current?.close() }
  }, [])

  /* Slower aggregates reconciled over REST; the socket carries the fast path. */
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => {
      api.getStatus().then(s => { if (s.active) setStatus(s.agent) }).catch(() => {})
      api.getAlerts().then(r => setAlerts(r.alerts)).catch(() => {})
    }, 4000)
    return () => clearInterval(id)
  }, [running])

  /* --------------------------------------------------------------- actions */
  const start = useCallback(async () => {
    setBusy(true); setSeries([]); setTick(null); setAlerts([])
    try {
      const r = await api.startSession(selected, speed)
      setRunning(true); setPaused(false); setStatus(r.status)
    } catch (e) {
      console.error('failed to start session', e)
    } finally { setBusy(false) }
  }, [selected, speed])

  const togglePause = useCallback(async () => {
    const next = !paused
    setPaused(next)
    await api.control(next ? 'pause' : 'resume').catch(() => {})
  }, [paused])

  const changeSpeed = useCallback(v => {
    setSpeed(v)
    if (running) api.control('speed', v).catch(() => {})
  }, [running])

  /* ------------------------------------------------------------------ view */
  const pred = tick?.prediction
  const dec = tick?.decision
  const feed = tick?.feed
  const bar = tick?.bar
  const action = dec?.action || 'HOLD'
  const ratio = pred && health ? pred.depeg_proba / health.model.threshold : null
  const blurb = useMemo(
    () => scenarios.find(s => s.key === selected)?.description || '',
    [scenarios, selected])

  const warming = tick && !tick.warm

  return (
    <>
      <header className="topbar">
        <div className="topbar-in">
          <h1 className="wordmark">SthiraSense <span>/ depeg early warning</span></h1>
          <div className="topbar-meta">
            <span className={`chip ${connected ? 'live' : 'off'}`}>
              <i className="dot" />{connected ? 'live' : 'offline'}
            </span>
            {health && (
              <>
                <span className="chip">{health.model.n_features} features</span>
                <span className="chip">
                  {health.model.rf_trees} trees + XGB &rarr; meta
                </span>
                <span className="chip">thr {health.model.threshold.toExponential(1)}</span>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="page">
        {/* ------------------------------------------------------ controls */}
        <div className="controlbar">
          <div className="segmented" role="group" aria-label="Scenario">
            {scenarios.map(s => (
              <button key={s.key} aria-pressed={selected === s.key}
                      onClick={() => setSelected(s.key)}>{s.name}</button>
            ))}
          </div>
          <div className="transport">
            <button className="btn primary" onClick={start} disabled={busy}>
              {busy ? 'Warming up…' : running ? 'Restart' : 'Start simulation'}
            </button>
            <button className="btn" onClick={togglePause} disabled={!running || busy}>
              {paused ? 'Resume' : 'Pause'}
            </button>
            <label className="speed">
              speed
              <input type="range" min="2" max="240" step="2" value={speed}
                     onChange={e => changeSpeed(Number(e.target.value))} />
              <b>{speed}&#215; min/s</b>
            </label>
          </div>
        </div>
        {blurb && <p className="scenario-blurb">{blurb}</p>}

        {/* --------------------------------------------------------- stats */}
        <section className="section">
          <div className="stats">
            <div className="stat">
              <div className="stat-label">Price</div>
              <div className="stat-value">{bar ? bar.close.toFixed(5) : '—'}</div>
              <div className="stat-foot">
                {bar ? <>deviation <b>{fmtNum(Math.abs(bar.close - 1) * 100, 3)}%</b></> : '—'}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">p(depeg, 24h)</div>
              <div className="stat-value">{pred ? fmtProb(pred.depeg_proba) : '—'}</div>
              <div className="stat-foot">
                {ratio != null ? <><b>{fmtNum(ratio, 1)}&#215;</b> threshold</> : '—'}
              </div>
            </div>
            <div className="stat">
              <div className="stat-label">Liquidity stress</div>
              <div className="stat-value">{pred ? fmtNum(pred.stress, 1) : '—'}</div>
              <div className="stat-foot">{pred ? <b>{pred.stress_band}</b> : '—'}</div>
            </div>
            <div className="stat">
              <div className="stat-label">Agent</div>
              <div className={`stat-value sev sev-${action}`}>
                <i className="mark" /><span className="txt">{action}</span>
              </div>
              <div className="stat-foot">
                {tick ? <>re-checking every <b>{tick.poll_interval_s}s</b></> : '—'}
              </div>
            </div>
          </div>
        </section>

        {/* --------------------------------------------------------- chart */}
        <section className="section">
          <div className="panel">
            <div className="section-head">
              <p className="eyebrow">Market &amp; model response</p>
              <span className="note">
                {feed ? `sim ${simClock(feed.minute)} · ${feed.phase}` : 'idle'}
              </span>
            </div>
            {warming ? (
              <>
                <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 7 }}>
                  warming up feature windows — {Math.round((tick.warmup_progress || 0) * 100)}%
                </div>
                <div className="track">
                  <i style={{ width: `${(tick.warmup_progress || 0) * 100}%` }} />
                </div>
              </>
            ) : (
              <Chart series={series} threshold={health?.model.threshold} />
            )}
          </div>
        </section>

        {/* ------------------------------------------------ decision + focus */}
        <section className="section">
          <div className="cols">
            <div className="panel">
              <div className="section-head">
                <p className="eyebrow">Decision — expected utility</p>
                <span className="note">
                  {dec ? `+${fmtNum(dec.margin, 1)} over runner-up` : ''}
                </span>
              </div>
              <Utility decision={dec} />
            </div>
            <div className="panel">
              <div className="section-head">
                <p className="eyebrow">Agent attention</p>
                <span className="note">robust z vs 4h</span>
              </div>
              <Attention attention={tick?.attention} intensity={tick?.focus_intensity}
                         pollInterval={tick?.poll_interval_s} />
            </div>
          </div>
        </section>

        {/* ---------------------------------------------------------- tabs */}
        <section className="section">
          <div className="tabs" role="tablist">
            {TABS.map(([k, label]) => (
              <button key={k} role="tab" aria-selected={tab === k}
                      onClick={() => setTab(k)}>
                {label}
                {k === 'alerts' && alerts.length > 0 ? ` (${alerts.length})` : ''}
              </button>
            ))}
          </div>
          <div className="tabpanel" role="tabpanel">
            {tab === 'ensemble' && <Ensemble pred={pred} bar={bar} />}
            {tab === 'alerts' && <Alerts alerts={alerts} />}
            {tab === 'tools' && <ToolLog tools={tick?.tools} />}
            {tab === 'learning' && <Learning learning={tick?.learning} status={status} />}
            {tab === 'analogue' && <Analogue analogue={tick?.analogue} />}
          </div>
        </section>

        <footer className="foot">
          <span>RF + XGBoost &rarr; logistic meta-learner · 106 features · 24h warm-up</span>
          <span>Threshold tuned for precision, not accuracy</span>
          <span>Simulated market data — not investment advice</span>
        </footer>
      </main>
    </>
  )
}
