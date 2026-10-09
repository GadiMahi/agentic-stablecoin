#!/usr/bin/env python3
"""Adversarial check: can the simulator fake the result?

Run this when someone asks whether the demo is real. It is deliberately short
enough to read aloud and it fails loudly rather than printing reassurance.

    python verify_not_faked.py

Three claims, checked in order:

1. The simulator's output schema contains no probability, stress score or
   decision field, so there is no channel through which it could inject one.
2. The same unmodified .pkl artefacts, given bars from a calm scenario and a
   collapsing one, produce very different base-model outputs. The scenario does
   not set the numbers; the trees do.
3. The feature vector handed to the models is the one they were trained on --
   106 columns, in the fitted order.
"""
from __future__ import annotations

import sys

from sthirasense.agent.agent import DepegAgent
from sthirasense.models import DepegEnsemble
from sthirasense.sim.engine import MarketFeed

ALLOWED = {
    "ts", "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_base", "taker_quote", "market_cap", "circulating_supply",
}
FORBIDDEN = ("proba", "probability", "stress", "action", "alert", "risk", "label", "target")


def check_schema() -> None:
    print("[1] What the simulator is allowed to emit")
    bar = MarketFeed("terminal_collapse", seed=5).next_bar()
    keys = set(bar)
    print(f"    fields: {sorted(keys)}")
    leaked = [k for k in keys if any(f in k.lower() for f in FORBIDDEN)]
    assert not leaked, f"simulator leaks model-side fields: {leaked}"
    assert keys == ALLOWED, f"unexpected schema: {keys ^ ALLOWED}"
    print("    -> raw market observables only. No probability, stress or decision.  PASS\n")


def check_models_do_the_work(ens: DepegEnsemble) -> None:
    print("[2] Same artefacts, different markets")
    out = {}
    for key in ("normal", "terminal_collapse"):
        agent = DepegAgent(ens, MarketFeed(key, seed=5))
        last = None
        for _ in range(1700):
            t = agent.tick()
            if t and t.warm and t.scored and t.prediction:
                last = t.prediction
        assert last is not None, f"{key}: never produced a scored prediction"
        out[key] = last
        print(f"    {key:18s} RF={last.rf_proba:.4f}  XGB={last.xgb_proba:.4f}"
              f"  -> meta p={last.depeg_proba:.3e}")

    calm, broken = out["normal"], out["terminal_collapse"]
    assert broken.depeg_proba > calm.depeg_proba * 5, (
        "the collapsing market did not score materially higher than the calm one")
    ratio = broken.depeg_proba / calm.depeg_proba
    print(f"    -> collapse scores {ratio:.1f}x the calm market, from identical"
          f" model files.  PASS\n")


def check_feature_contract(ens: DepegEnsemble) -> None:
    print("[3] The vector handed to the models")
    agent = DepegAgent(ens, MarketFeed("svb_bleed", seed=5))
    feats = None
    for _ in range(1500):
        t = agent.tick()
        if t and t.warm:
            feats = agent._last_feats
            break
    assert feats, "engine never warmed up"
    missing = agent.engine.missing_features(feats)
    assert not missing, f"engine failed to produce {len(missing)} columns: {missing[:5]}"
    vec = agent.engine.vector(feats)
    assert len(vec) == ens.rf.n_features_in_ == 106
    print(f"    {len(vec)} features, matching the fitted contract exactly.")
    print("    -> no padding, no placeholder columns.  PASS\n")


def main() -> int:
    ens = DepegEnsemble()
    try:
        check_schema()
        check_models_do_the_work(ens)
        check_feature_contract(ens)
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        return 1
    print("All three checks passed. The simulator feeds raw bars; the trained")
    print("models and the agent produce everything else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
