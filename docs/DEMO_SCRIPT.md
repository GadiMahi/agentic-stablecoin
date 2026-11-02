# Demo script — simulation mode

**Read this once, then present from the five bold lines.** Everything else is
support for when you get asked something.

---

## 1. Before you click anything (20 seconds)

Say this *first*, because it answers the question the whole room is silently
asking — "is any of this real?"

> "Real depegs happen maybe twice a decade, so I can't wait for one to show you
> this. What I have instead is a market simulator. It generates raw market bars —
> price, volume, trade counts, who's buying versus selling, market cap — exactly
> the shape an exchange feed delivers. That's **all** it produces. It has no way
> to set the probability, the stress score, or the agent's decision, because
> there's no field in its output for any of those. Everything you're about to see
> on this screen is computed live, from those bars, by the trained models and the
> agent."

If you want the knockout version, add:

> "There's a script in the repo called `verify_not_faked.py` that asserts exactly
> that, and fails the build if it's ever untrue."

**Why this works:** you've pre-empted the only real objection before they form it.

---

## 2. The order to demo in

Always **Normal first**. A quiet system looks broken unless you've set it up as
the control case.

| # | Scenario | Point you're making | Time |
|---|---|---|---|
| 1 | Normal operation | It stays quiet. Silence is the feature. | ~45s |
| 2 | USDC / SVB slow bleed | It catches a build-up and escalates. | ~90s |
| 3 | Terminal collapse | It goes to CRITICAL and names the analogue. | ~45s |

Skip Flash crash unless you have time — it makes the same point as the bleed,
faster. Keep it in reserve for "what if there's no warning?"

---

## 3. Scenario 1 — Normal operation

**Setup line, before you press Start:**

> "This is a calm market, holding its peg. Nothing is wrong. Watch what the
> system does with that."

**While it runs, point at three things in this order:**

1. **p(depeg) reads about `1.66e-3`, labelled `0.9× threshold`.**
   > "The model's output is just *below* its own decision line. Not zero — it's
   > always thinking — but under the bar."

2. **Agent reads HOLD, and 're-checking every 300s'.**
   > "The agent has decided this doesn't deserve attention. It's deliberately
   > looking only once every five minutes."

3. **Alerts tab shows zero.**
   > "No alerts. On a calm market, that is the correct output. An early-warning
   > system that fires here is worthless, because people switch it off."

**The money line to close on:**

> **"That's the system working. The hardest thing to build isn't the alarm — it's
> the silence."**

Then open the **Learning tab** and point at *Inference saved ≈ 95%*:

> "And it saved 95% of its model runs by choosing to look less often. That's the
> agent managing its own cost."

---

## 4. Scenario 2 — USDC / SVB-style slow bleed

**Setup line:**

> "Now the realistic case. In March 2023 USDC — the audited, fully-backed,
> blue-chip one — fell to 87 cents because $3.3 billion of its reserves were
> stuck in a bank that had just failed. The warning signs were in the trading
> data before the price broke. This scenario has that shape: hours of
> deterioration with rising volume and one-sided selling."

**Watch for these, in the order they happen:**

1. **Price starts drifting below $1.0000** on the chart. Volume climbs.

2. **p(depeg) jumps roughly ten-fold**, to around `2.4e-2`, labelled
   **`13.6× threshold`**.
   > "The probability is now about thirteen times its decision line."

3. **Agent flips HOLD → WARN.** The poll interval drops **300s → 15s**.
   > "It just decided to look twenty times more often. Nobody told it to —
   > that's the agent reacting to its own belief."

4. **In the Attention panel, a metric grows larger and darker** — usually Peg
   deviation or Peg volatility, with a sigma value beside it.
   > "That's not styling. The agent publishes how abnormal each input is against
   > its own four-hour baseline, and the panel renders whatever it's reacting to."

5. **An alert appears in the Alerts tab, resolving TRUE POSITIVE.**

6. **In the Learning tab, 'Cost of a miss' has moved** from 4,000 to ~4,320.
   > "It just learned. That alert was correct, so missing one became more
   > expensive in its cost model — it'll be marginally quicker to escalate next
   > time."

**The money line:**

> **"Nothing in the simulator told it to escalate. It only ever sent price and
> volume. The escalation is the agent's own arithmetic."**

---

## 5. Scenario 3 — Terminal collapse

**Setup line:**

> "Last one. This is a Terra/UST-style death spiral — the peg never comes back."

**Point at:**

1. **Price collapsing hard** — down toward $0.19 and falling.
2. **Stress hits 100, band reads CRITICAL or HIGH.**
3. **Agent reads CRITICAL, polling every 5 seconds.**
4. **Open the Incident memory tab** → it has matched **Terra UST, 2022-05-09**.
   > "It searched its incident memory and found the closest historical analogue
   > on its own. It's a real death spiral, and it found the real one."

**The money line:**

> **"Full escalation, five-second polling, and it named the historical episode
> this most resembles."**

---

## 6. Reading the screen — what each number means

Keep this in your head; you'll get pointed at one of these.

| On screen | What it is | What to say if asked |
|---|---|---|
| **p(depeg, 24h)** | Probability the coin breaches its peg band in the next 24h | "Output of the stacked ensemble — Random Forest and XGBoost feeding a logistic meta-learner." |
| **`N× threshold`** | How far past its tuned decision line | "The threshold is 0.0018, tuned for 90% precision. Raw probability is small by design — the ratio is the readable number." |
| **Liquidity stress 0-100** | Weighted index: peg deviation 40%, volume 25%, supply 20%, volatility 15% | "A continuous risk score, so we're not limited to a yes/no flag." |
| **Agent HOLD/WATCH/WARN/CRITICAL** | The action chosen by expected utility | "Not a threshold — it scores all four actions and takes the best." |
| **re-checking every Ns** | The agent's self-chosen poll rate | "Adaptive sensing. How often it looks is itself one of its decisions." |
| **Focus intensity %** | How abnormal its most abnormal input is | "Driven by the largest sigma across the metrics it watches." |
| **σ values in Attention** | Robust z-score vs that metric's own 4h baseline | "Median and MAD, so one spike can't poison the baseline." |
| **Decision panel, four bars** | Expected utility of every action | "The winner is just the argmax. You can watch the ranking flip." |
| **Inference saved %** | Model runs skipped by polling lazily | "Perception runs every bar; inference only when the agent decides to look." |

---

## 7. If something unexpected happens

Presentations go sideways. These are the real cases and what to say.

**Nothing happens for a while after you press Start.**
> "It's replaying 24 hours of history first — the models need a full day of
> rolling windows before they'll produce anything."
Normal. Takes a few seconds. The warm-up bar is on screen.

**It escalates earlier or later than you expected.**
Don't fight it. Say:
> "Exact timing shifts with the market noise — what's consistent is the ordering:
> the flow turns, then volume, then the probability, then the agent."

**It sits on WARN and never reaches CRITICAL.**
This is correct behaviour, not a bug.
> "CRITICAL carries a much higher false-alarm cost, so it's reserved for peak
> dislocation. The agent is telling you WARN is the better-value action right
> now."

**It goes CRITICAL almost immediately on flash crash.**
> "That's the point of this scenario — there's no build-up to find, so the only
> thing you can measure is reaction speed."

**The numbers freeze / stream stops.**
Check the top-right chip. If it says *offline*, the browser lost the socket.
Refresh the page — the session keeps running server-side and the dashboard
re-attaches.

**Someone says "the probabilities look tiny."**
> "They are, and deliberately. The model is tuned for precision at a 1-in-100,000
> event rate, so its threshold sits at 0.0018. Judge it against its own line —
> that's why the ratio is displayed."

**Someone says "this is just a simulation, prove the ML is real."**
Drop to the Ensemble tab and point at Random Forest and XGBoost moving
independently (e.g. 0.14 and 0.56), then:
> "Two different models disagreeing, with the meta-learner combining them. If
> this were scripted, they'd move together. And there's a verification script in
> the repo that asserts the simulator can't touch any of it."

---

## 8. The five questions you will actually get

**"Why simulate instead of using real data?"**
> "The models *are* trained on real market data — a year of minute bars. The
> simulator is only how I show you behaviour across regimes in three minutes
> instead of waiting years for the next crisis."

**"How do I know the agent isn't just a threshold?"**
> "The Decision panel shows all four actions scored. Watch CRITICAL overtake WARN
> as stress rises while the probability barely moves — that's the cost arithmetic
> flipping. A threshold can't do that."

**"What's your accuracy?"**
Be straight.
> "I'm not quoting one yet. The training year was unusually calm, so the models
> have never seen a real crisis. Quoting a precision figure off that would be
> misleading. The next step is extending the data through the March 2023 window
> so a genuine depeg is in the test set."

**"Is the agent an LLM?"**
> "No. It's a utility-based agent — it holds state, scores actions by expected
> cost, acts, and re-tunes its weights from outcomes. No language model is
> involved, and it doesn't need one."

**"What happens with a real feed?"**
> "The simulator is one implementation of the feed interface. Point it at the
> Binance minute endpoint instead and nothing downstream changes — same bars,
> same features, same models, same agent."

---

## 9. One-line safety net

If you lose your thread entirely, this sentence rescues any moment of the demo:

> **"The simulator only sends price and volume. Everything else on this screen is
> the models and the agent working it out."**
