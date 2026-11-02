# SthiraSense

**Autonomous early-warning system for stablecoin depeg and liquidity-stress risk.**

A stablecoin is a digital token engineered to always be worth exactly one US
dollar. Roughly $287 billion of them are in circulation, and they are the
settlement layer the rest of the crypto market runs on. Occasionally one breaks
that promise — it *depegs* — and holders lose money within hours.

Every monitoring tool in common use reports the deviation **after** the price has
already moved. SthiraSense answers the forward-looking question instead: given
the last 24 hours of minute-level market behaviour, how likely is this coin to
breach its peg band in the next 24 hours — and what, if anything, should be done
about it right now.

The second half of that sentence is the point. A probability is not a decision.
On top of the models sits a **utility-based agent with a learning element** that
weighs the cost of a missed break against the cost of a false alarm, decides for
itself whether to stay quiet, watch harder, or escalate, controls how often it
looks at the market, and re-tunes its own cost weights from how its past alerts
turned out.

---

## What is real here

This repository runs the genuine pipeline end to end. Specifically:

| Claim | How it is kept honest |
|---|---|
| The models are the real trained artefacts | `models/*.pkl` as published, loaded and scored at runtime — never mocked |
| Live features match training features | `tests/test_feature_parity.py` drives the original batch pandas pipeline and the live engine over the same series and asserts all **106** columns agree |
| The simulator cannot fake a result | Scenarios emit **raw OHLCV + market-cap bars only**. They never touch a probability, a stress score or agent state — `test_agent_only_sees_raw_bars` pins the schema |
| The agent genuinely decides | Every action is scored by expected utility and the full ranking is published to the UI, not just the winner |
| The learning element genuinely learns | Cost weights move on resolved outcomes; every update is written to SQLite and shown in the dashboard |

Two ways to verify this yourself:

```bash
pytest tests/ -q          # 57 tests, no network, ~1.5 minutes
python verify_not_faked.py   # adversarial check, prints its reasoning
```

`verify_not_faked.py` is short enough to read aloud: it asserts the simulator's
output schema contains no model-side field, scores a calm and a collapsing market
through the same artefacts, and confirms the 106-column contract is met exactly.

---

## Quick start

```bash
# 1. Python side
pip install -r requirements.txt

# 2. Dashboard (one-time build)
cd frontend && npm install && npm run build && cd ..

# 3. Run
python run.py
# -> http://127.0.0.1:8000
```

Pick a scenario, press **Start simulation**. The feed first replays 24 hours of
calm history to fill the rolling windows (fast — perception only, no inference),
then begins the scenario.

For frontend development with hot reload, run `python run.py` in one terminal and
`cd frontend && npm run dev` in another; Vite proxies `/api` and `/ws` to the
Python service.

---

## Architecture

```
LAYER 1  PERCEPTION      raw bars: OHLCV, trades, taker flow, market cap, supply
LAYER 2  FEATURE ENGINE  106 features across 3h / 6h / 12h / 24h rolling windows
LAYER 3  MODELS          RandomForest + XGBoost -> logistic meta-learner
                         + 0-100 liquidity-stress index
LAYER 4  AGENT           state, attention, expected-utility choice, learning
LAYER 5  RESPONSE        tiered alerts, escalation, evidence, audit log
                         ^ feedback: outcomes retune costs
                         ^ adaptive sensing: the agent sets its own poll rate
```

```
sthirasense/
  config.py        paths, rolling windows, warm-up size
  features.py      incremental 106-feature engine (numpy ring buffers)
  models.py        DepegEnsemble + liquidity-stress index
  store.py         SQLite audit log
  api.py           FastAPI + WebSocket service
  agent/
    utility.py     cost model, action scoring, argmax over expected utility
    state.py       memory, robust-z attention map, alert lifecycle
    learning.py    cost-weight updates from resolved alerts
    tools.py       named tool registry + incident memory
    agent.py       perceive -> believe -> attend -> reason -> act -> learn
  sim/
    scenarios.py   four market trajectories (raw quantities only)
    engine.py      bar generator + warm-up
ml/                original research pipeline, vendored unchanged
models/            the four trained artefacts
frontend/          React + Vite dashboard
tests/             parity, model contract, agent behaviour
```

### The decision rule

Each minute the agent holds a depeg probability `p` and a stress score, and picks
the action maximising expected utility:

```
EU(a) = − p · C_miss(a) − (1 − p) · C_false(a) − C_op(a)
```

over `{HOLD, WATCH, WARN, CRITICAL}`. `C_miss` falls as the action covers more of
the loss; `C_false` rises with operator fatigue — `WATCH` is nearly free because
no human is interrupted, `CRITICAL` is expensive because crying wolf at that
level is what gets an alert system switched off. Stress amplifies `C_miss`: a
break from an already-stressed book does more damage than one from a calm one.

This is why the agent is **utility-based** and not goal-based. "Never miss a
depeg" is a goal perfectly satisfied by alerting constantly, which destroys the
system's usefulness. The two ways of being wrong are asymmetric and both real, so
the decision is a trade-off, not a goal test.

### Attention

Each tick the agent computes a **robust z-score** (median/MAD, so one spike can't
poison the baseline) for every watched metric against its own 4-hour history, and
publishes the result. The dashboard renders that directly: the more abnormal a
number is, the larger and louder its row becomes. The emphasis is an output of
the reasoning layer, not a styling choice.

### Adaptive sensing

Perception runs every bar; **inference does not**. The agent's chosen action sets
its own poll interval — 300s at HOLD down to 5s at CRITICAL — so on a calm market
it deliberately looks less often, skipping ~95% of model invocations while still
keeping every rolling window complete. The dashboard reports this as
*inference saved*.

---

## Real market data (Binance + CoinGecko)

Besides the four simulated scenarios, the dashboard can run the agent on real
market data. Both sources are **free**: Binance's public market-data endpoints
need no account or key, and CoinGecko works on its free public tier.

| Source | What it provides | Endpoint |
|---|---|---|
| Binance | 1-minute OHLC, volume, quote volume, trades, taker buy volume | `GET /api/v3/klines` (falls back to `data-api.binance.vision`) |
| CoinGecko | Circulating supply and market cap | `GET /coins/markets`, `GET /coins/{id}/market_chart/range` |

Two real sources appear next to the simulations:

- **FDUSD depeg, April 2025 (real data)** replays Binance FDUSD/USDT 1-minute
  bars from 1 to 5 April 2025 (UTC), after 24 hours of real warm-up. On
  2 April FDUSD fell to about $0.87 after its issuer was publicly accused of
  insolvency, and took about 36 hours to recover. The first run downloads about
  7,200 bars and caches them in `data_cache/`; after that the replay works
  offline. CoinGecko's free plans only serve the last 365 days of history, so
  supply is held at the issuer's stated size (about $2.5B) while the market cap
  still moves with the price.
  The USDC/SVB depeg of March 2023 cannot be replayed from Binance: its USDC/USDT
  pair was only listed at 14:00 UTC on 11 March 2023, mid-crisis, so there is no
  pre-event data to warm up on.
- **USDC live (Binance + CoinGecko)** warms up on the last 24 hours, then
  hands the agent each new minute as it closes. Supply refreshes every 10 minutes.

```bash
python fetch_real_data.py check     # are Binance and CoinGecko reachable right now?
python fetch_real_data.py replay    # download and cache the FDUSD April 2025 replay
python fetch_real_data.py run       # play it through the agent and print its timeline
```

Optional: set `COINGECKO_API_KEY` to a free CoinGecko Demo key for a higher
rate limit. Real feeds emit exactly the same 12 raw fields as the simulator, so
nothing downstream changes, and they never compute a probability or a decision.

---

## Scenarios

| Scenario | What it tests |
|---|---|
| **Normal operation** | The contrast case. The agent must stay silent — no alerts, HOLD throughout. |
| **USDC / SVB-style slow bleed** | The realistic early-warning case: hours of deterioration with rising volume and one-sided selling before the trough, then recovery. |
| **Flash crash** | Reaction speed when there is no lead time to find. |
| **Terminal collapse (UST-style)** | Algorithmic death spiral with expanding supply; the peg never returns. |

Observed behaviour, same seed, with the published artefacts:

| Scenario | p(depeg) range | Peak stress | Actions | Alerts |
|---|---|---|---|---|
| normal | 1.66e-3 (flat) | 18 | HOLD only | 0 |
| svb_bleed | 1.66e-3 → 3.7e-2 | 86 | HOLD → WARN → CRITICAL | 1 |
| flash_crash | 1.66e-3 → 3.7e-2 | 79 | HOLD → WARN → CRITICAL | 1 |
| terminal_collapse | 1.66e-3 → 4.2e-2 | 100 | HOLD → WARN → CRITICAL | 1 |

Note the calm case sits at **0.9× the decision threshold** — just below the line,
which is the whole reason it stays quiet.

---

## Notes on the trained models

Findings from working with the published artefacts, recorded because they are
easy to get wrong:

- **The meta-learner takes 7 inputs, not 4.** The stack is
  `[rf, xgb, rf·xgb, |rf−xgb|, min(rf,xgb), mean(rf,xgb), both>0.5]` and the
  coefficients are positional. Getting the order wrong silently corrupts output.
- **The longest rolling window actually required is 1440m.** The 7d/30d windows
  from the research pipeline were dropped before training (`DROP_PATTERNS` in
  `ml/train.py`), so warm-up is 24 hours of bars, not 30 days.
- **`close_min` is absent at 720m/1440m**, and `peg_deviation_max_1440m` is absent
  entirely — these were dropped as leakage.
- **The forest is 300 trees at depth 4**, not the 100 trees the original README
  states.
- **`n_jobs=-1` is a pessimisation for single-row serving** — it spawns a joblib
  pool per call and costs ~3.3× more wall-clock for bit-identical output.
- **The decision threshold is 0.0018**, tuned for ≥90% precision at ≥30% recall.
  Accuracy is meaningless at this class imbalance.
- The liquidity-stress **regressor** was never published with the source
  repository, so the 0-100 index is computed from the component formula in
  `ml/liquidity_stress_regression.py` rather than substituting an invented model.

---

## Automated response

An escalation at WARN or above is dispatched to every configured channel. All
four are free; unconfigured ones report themselves unconfigured and are skipped,
never counted as delivered.

| Channel | Setup | Notes |
|---|---|---|
| **console** | none | Writes to the service log. Always on. |
| **browser** | none | Pushed over the open WebSocket; the dashboard raises a real OS notification. Allow notifications when asked. |
| **webhook** | one env var | Discord or Slack incoming webhook. One payload carries both Discord's `content` and Slack's `text`, so either works. |
| **email** | four env vars | SMTP. Gmail requires an App Password, not your account password. |

```bash
export STHIRA_WEBHOOK_URL="https://discord.com/api/webhooks/..."
export STHIRA_SMTP_HOST=smtp.gmail.com STHIRA_SMTP_PORT=587
export STHIRA_SMTP_USER=you@gmail.com STHIRA_SMTP_PASS="app password"
export STHIRA_SMTP_TO=you@gmail.com
export STHIRA_MIN_LEVEL=WARN          # or CRITICAL
python run.py
```

`.env.example` documents every variable. No secret is committed.

The **Response** tab shows which channels are live, a delivery log including
failures, and a **Send test alert** button that fires a clearly-labelled test
through every live channel — so the response path can be demonstrated without
waiting for the market to break.

Delivery is best-effort and runs off the agent's thread. A channel that times
out or throws records the failure and never propagates into the agent loop: an
early-warning system that dies because SMTP was slow is worse than one that
misses an email.

---

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + model contract summary |
| GET | `/api/scenarios` | available simulations |
| POST | `/api/session` | start / restart a run |
| POST | `/api/session/control` | pause, resume, speed |
| GET | `/api/status` | agent status + recent series |
| GET | `/api/alerts` | alerts for the active run |
| GET | `/api/history` | alert track record across runs |
| WS | `/ws` | live tick stream |

---

## Limitations

Stated plainly, because they matter more than the feature list.

- **The training data covers an unusually calm period.** The published models
  were fitted on roughly a year of minute data in which the peg barely moved.
  They have learned what normal looks like; they have never seen a real crisis.
  Extending the dataset backwards through the March 2023 USDC/SVB window is the
  single highest-value next step.
- **Scenario behaviour is not a performance claim.** The simulator produces
  plausible trajectories, not historical replays. That the models react sensibly
  to them demonstrates the pipeline is wired correctly end to end — it is not a
  measured precision or recall figure.
- **Alert outcomes use a fixed 0.5% breach band** for auditability, whereas the
  models' own training target used the volume-adaptive band of Carey (2023).
- **Real-data caveats.** Prices come from Binance's USDC/USDT pair, so a
  wobble in USDT also shows up. The FDUSD replay holds supply constant, so the
  supply-change feature reads zero during that replay.
- Reserve quality and redemption queues are off-chain and invisible to this
  system. It can only ever see the market's *reaction*.

---

## Credits

The ML research pipeline — feature engineering, hybrid ensemble training,
liquidity-stress study, importance and temporal analysis — originates from
[RK-NerdyBirdy/SthiraSense-ML](https://github.com/RK-NerdyBirdy/SthiraSense-ML)
by **Team NazarBattu**, and is vendored unchanged under `ml/`.

The depeg definition follows Riyad Carey, *Defining Depegs: A New Metric for
Stablecoin Stability*, Kaiko Research (2023). Class imbalance handling follows
Chawla et al., *SMOTE* (JAIR 16, 2002). The agent taxonomy and PEAS framing
follow Russell & Norvig, *Artificial Intelligence: A Modern Approach*.

Simulated and public market data. Not investment advice.
