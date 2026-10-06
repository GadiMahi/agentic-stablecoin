import React, { useCallback, useEffect, useRef, useState } from 'react'
import Chart from './components/Chart.jsx'
import Attention from './components/Attention.jsx'
import Utility from './components/Utility.jsx'
import { Alerts, Analogue, Controls, Learning, ToolLog } from './components/Panels.jsx'
import * as api from './lib/api.js'
import { fmtCompact, fmtNum, fmtPrice, fmtProb, simClock } from './lib/format.js'

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
  const wsRef = useRef(null)

  // ---------------------------------------------------------------- bootstrap
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

  // ------------------------------------------------------------------ socket
  useEffect(() => {
    let closed = false
    let retry

    const connect = () => {
      const ws = new WebSocket(api.wsUrl())
      wsRef.current = ws
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = setTimeout(connect, 1500)
      }
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
                ts: d.ts,
                close: d.bar.close,
                p: d.prediction.depeg_proba,
                stress: d.prediction.stress,
                action: d.decision?.action || 'HOLD',
                focus: d.focus_intensity,
              }]
              return next.length > 600 ? next.slice(next.length - 600) : next
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

  // Periodic reconciliation — the socket carries the fast path, this keeps the
  // slower aggregates (agent status, alert table) honest.
  useEffect(() => {
    if (!running) return
    const id = setInterval(() => {
      api.getStatus().then(s => { if (s.active) setStatus(s.agent) }).catch(() => {})
      api.getAlerts().then(r => setAlerts(r.alerts)).catch(() => {})
    }, 4000)
    return () => clearInterval(id)
  }, [running])

  // ----------------------------------------------------------------- actions
  const start = useCallback(async () => {
    setBusy(true); setSeries([]); setTick(null); setAlerts([])
    try {
      const r = await api.startSession(selected, speed)
      setRunning(true); setPaused(false); setStatus(r.status)
    } catch (e) {
      console.error('failed to start session', e)
    } finally {
      setBusy(false)
    }
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

  // -------------------------------------------------------------------- view
  const pred = tick?.prediction
  const dec = tick?.decision
  const feed = tick?.feed
  const bar = tick?.bar
  const warm = tick?.warm ?? false
  const ratio = pred && health
    ? pred.depeg_proba / health.model.threshold : null

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>SthiraSense</h1>
          <p>
            Autonomous early warning for stablecoin depeg risk. The simulator
            emits raw market bars only — every probability, stress score and
            decision below is computed live by the trained ensemble and the agent.
          </p>
        </div>
        <div className="masthead-right">
          <span className={`pill ${connected ? 'live' : 'down'}`}>
            <i className="dot" />{connected ? 'stream live' : 'disconnected'}
          </span>
          {health && (
            <>
              <span className="pill">{health.model.n_features} features</span>
              <span className="pill">{health.model.rf_trees} trees + XGB &rarr; meta</span>
              <span className="pill">thr {health.model.threshold.toExponential(1)}</span>
            </>
          )}
        </div>
      </header>

      <div className="grid main">
        {/* ------------------------------------------------- left column */}
        <div className="stack">
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Market &amp; model response</h2>
              <span className="card-note">
                {feed ? `${feed.name} · sim ${simClock(feed.minute)} · ${feed.phase}` : 'idle'}
              </span>
            </div>

            {!warm && tick && (
              <div style={{ marginBottom: 12 }}>
                <div className="stat-sub" style={{ marginBottom: 5 }}>
                  warming up feature windows — {Math.round((tick.warmup_progress || 0) * 100)}%
                </div>
                <div className="warmbar">
                  <i style={{ width: `${(tick.warmup_progress || 0) * 100}%` }} />
                </div>
              </div>
            )}

            <div className="grid trio" style={{ marginBottom: 14 }}>
              <div>
                <div className="stat-value big">{bar ? fmtPrice(bar.close) : '—'}</div>
                <div className="stat-sub">
                  price {bar && `· dev ${fmtNum(Math.abs(bar.close - 1) * 100, 3)}%`}
                </div>
              </div>
              <div>
                <div className="stat-value big">{pred ? fmtProb(pred.depeg_proba) : '—'}</div>
                <div className="stat-sub">
                  p(depeg 24h){ratio != null && ` · ${fmtNum(ratio, 1)}× threshold`}
                </div>
              </div>
              <div>
                <div className="stat-value big">{pred ? fmtNum(pred.stress, 1) : '—'}</div>
                <div className="stat-sub">
                  liquidity stress {pred && `· ${pred.stress_band}`}
                </div>
              </div>
            </div>

            <Chart series={series} threshold={health?.model.threshold} />
          </section>

          <div className="grid duo">
            <section className="card">
              <div className="card-head">
                <h2 className="card-title">Decision — expected utility</h2>
              </div>
              <Utility decision={dec} />
            </section>

            <section className="card">
              <div className="card-head">
                <h2 className="card-title">Ensemble detail</h2>
              </div>
              <dl className="kv">
                <dt>Random Forest</dt><dd>{pred ? fmtNum(pred.rf_proba, 4) : '—'}</dd>
                <dt>XGBoost</dt><dd>{pred ? fmtNum(pred.xgb_proba, 4) : '—'}</dd>
                <dt>Meta-learner output</dt><dd>{pred ? fmtProb(pred.depeg_proba) : '—'}</dd>
                <dt>Agreement (min)</dt><dd>{pred ? fmtNum(pred.agreement, 4) : '—'}</dd>
                <dt>Disagreement</dt><dd>{pred ? fmtNum(pred.disagreement, 4) : '—'}</dd>
                <dt>Past threshold</dt><dd>{pred ? (pred.breach ? 'yes' : 'no') : '—'}</dd>
              </dl>
              <div style={{ height: 1, background: 'var(--line)', margin: '12px 0' }} />
              <dl className="kv">
                <dt>Volume (1m)</dt><dd>{bar ? fmtCompact(bar.volume) : '—'}</dd>
                <dt>Trades (1m)</dt><dd>{bar ? fmtNum(bar.trades, 0) : '—'}</dd>
                <dt>Market cap</dt><dd>{bar ? '$' + fmtCompact(bar.market_cap) : '—'}</dd>
              </dl>
            </section>
          </div>

          <div className="grid duo">
            <section className="card">
              <div className="card-head">
                <h2 className="card-title">Alerts this run</h2>
                <span className="card-note">{alerts.length} total</span>
              </div>
              <Alerts alerts={alerts} />
            </section>
            <section className="card">
              <div className="card-head">
                <h2 className="card-title">Tool calls</h2>
              </div>
              <ToolLog tools={tick?.tools} />
            </section>
          </div>
        </div>

        {/* ------------------------------------------------ right column */}
        <div className="stack">
          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Simulation</h2>
            </div>
            <Controls scenarios={scenarios} selected={selected} onSelect={setSelected}
                      onStart={start} onPause={togglePause} onSpeed={changeSpeed}
                      running={running} paused={paused} speed={speed} busy={busy} />
          </section>

          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Agent attention</h2>
              <span className="card-note">robust z vs 4h baseline</span>
            </div>
            <Attention attention={tick?.attention} intensity={tick?.focus_intensity}
                       pollInterval={tick?.poll_interval_s} />
          </section>

          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Closest historical episode</h2>
            </div>
            <Analogue analogue={tick?.analogue} />
          </section>

          <section className="card">
            <div className="card-head">
              <h2 className="card-title">Learning element</h2>
            </div>
            <Learning learning={tick?.learning} status={status} />
          </section>
        </div>
      </div>

      <footer className="foot">
        <span>RF + XGBoost &rarr; logistic meta-learner · 106 features · 24h warm-up</span>
        <span>Threshold tuned for precision, not accuracy</span>
        <span>Simulated market data — not investment advice</span>
      </footer>
    </div>
  )
}
