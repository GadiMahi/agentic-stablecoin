"""HTTP + WebSocket service.

One process holds the loaded ensemble (expensive, loaded once) and at most one
live session. A session owns a market feed and an agent, and a background task
advances it at a configurable speed, broadcasting each tick to connected
WebSocket clients.

Endpoints
    GET  /api/health          liveness + model contract summary
    GET  /api/scenarios       available simulations
    POST /api/session         start (or restart) a run
    POST /api/session/control pause / resume / speed
    GET  /api/status          current agent status
    GET  /api/alerts          alerts for the active run
    GET  /api/history         alert track record across all runs
    WS   /ws                  live tick stream
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent.agent import DepegAgent
from .config import ROOT, WARMUP_BARS
from .models import DepegEnsemble
from .sim.engine import MarketFeed
from .sim.scenarios import SCENARIOS, list_scenarios
from .store import AuditStore

log = logging.getLogger("sthirasense")

app = FastAPI(title="SthiraSense", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

ENSEMBLE: Optional[DepegEnsemble] = None
STORE: Optional[AuditStore] = None
SESSION: Optional["Session"] = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class Session:
    """A running simulation + agent, broadcasting ticks to subscribers."""

    def __init__(self, scenario: str, seed: int, speed: float):
        assert ENSEMBLE is not None and STORE is not None
        self.scenario = scenario
        self.seed = seed
        self.speed = speed                   # simulated minutes per real second
        self.paused = False
        self.feed = MarketFeed(scenario, seed=seed)
        self.run_id = STORE.start_run(scenario, seed, ENSEMBLE.threshold)
        self.agent = DepegAgent(ENSEMBLE, self.feed, on_dispatch=self._on_dispatch)
        self.clients: Set[WebSocket] = set()
        self.task: Optional[asyncio.Task] = None
        self.last_tick: Optional[Dict[str, Any]] = None
        self.series: List[Dict[str, Any]] = []     # downsampled chart history
        self.finished = False
        self._seen_learning = 0

    # ------------------------------------------------------------- callbacks
    def _on_dispatch(self, alert: Dict[str, Any]) -> None:
        if STORE:
            STORE.record_alert(self.run_id, alert, alert.get("analogue"))

    # ----------------------------------------------------------------- warmup
    def warm_up(self) -> None:
        """Push 24h of history so the first scored bar has complete windows.

        Cheap: perception only, no inference — the agent's gate keeps the models
        idle until the engine is warm.
        """
        while not self.agent.engine.warm:
            if self.agent.tick() is None:
                break

    # ----------------------------------------------------------------- stream
    async def run(self) -> None:
        try:
            while not self.finished:
                if self.paused:
                    await asyncio.sleep(0.1)
                    continue

                tick = await asyncio.to_thread(self.agent.tick)
                if tick is None:
                    self.finished = True
                    await self.broadcast({"type": "finished",
                                          "status": self.agent.status()})
                    break

                payload = tick.to_dict()
                payload["feed"] = self.feed.info()
                self.last_tick = payload

                # Persist resolutions and any new learning updates.
                for r in tick.resolved:
                    if STORE:
                        STORE.resolve_alert(self.run_id, r)
                hist = tick.learning.get("stats", {}).get("history", [])
                if len(hist) > self._seen_learning and STORE:
                    for u in hist[self._seen_learning:]:
                        STORE.record_learning(self.run_id, u)
                    self._seen_learning = len(hist)

                if tick.scored and tick.prediction:
                    self.series.append({
                        "ts": tick.ts,
                        "close": round(tick.bar["close"], 6),
                        "p": tick.prediction.depeg_proba,
                        "stress": round(tick.prediction.stress, 2),
                        "action": tick.decision.action.label if tick.decision else "HOLD",
                        "focus": round(tick.focus_intensity, 3),
                    })
                    if len(self.series) > 1500:
                        del self.series[: len(self.series) - 1500]

                await self.broadcast({"type": "tick", "data": payload})
                await asyncio.sleep(max(0.004, 1.0 / max(self.speed, 0.1)))
        except asyncio.CancelledError:
            raise
        except Exception:                      # pragma: no cover
            log.exception("session loop failed")
            await self.broadcast({"type": "error", "message": "session loop failed"})

    async def broadcast(self, msg: Dict[str, Any]) -> None:
        if not self.clients:
            return
        text = json.dumps(msg, default=float)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def stop(self) -> None:
        self.finished = True
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StartRequest(BaseModel):
    scenario: str = Field(default="normal")
    seed: int = Field(default=42)
    speed: float = Field(default=20.0, ge=0.5, le=240.0)


class ControlRequest(BaseModel):
    action: str                      # pause | resume | speed
    speed: Optional[float] = Field(default=None, ge=0.5, le=240.0)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    global ENSEMBLE, STORE
    ENSEMBLE = DepegEnsemble()
    STORE = AuditStore()
    log.info("loaded ensemble: %d features, threshold %.6f",
             len(ENSEMBLE.feature_cols), ENSEMBLE.threshold)


@app.on_event("shutdown")
async def _shutdown() -> None:
    if SESSION:
        await SESSION.stop()
    if STORE:
        STORE.close()


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> Dict[str, Any]:
    assert ENSEMBLE is not None
    rf = ENSEMBLE.rf
    return {
        "ok": True,
        "model": {
            "n_features": len(ENSEMBLE.feature_cols),
            "threshold": ENSEMBLE.threshold,
            "target_precision": ENSEMBLE.target_precision,
            "min_recall": ENSEMBLE.min_recall,
            "rf_trees": int(getattr(rf, "n_estimators", 0)),
            "meta_inputs": int(ENSEMBLE.meta.n_features_in_),
        },
        "warmup_bars": WARMUP_BARS,
        "lifetime": STORE.lifetime_stats() if STORE else None,
    }


@app.get("/api/scenarios")
def scenarios() -> Dict[str, Any]:
    return {"scenarios": list_scenarios()}


@app.post("/api/session")
async def start_session(req: StartRequest) -> Dict[str, Any]:
    global SESSION
    if req.scenario not in SCENARIOS:
        return JSONResponse({"error": f"unknown scenario {req.scenario!r}"}, status_code=400)

    clients: Set[WebSocket] = set()
    if SESSION:
        clients = SESSION.clients
        await SESSION.stop()

    s = Session(req.scenario, req.seed, req.speed)
    s.clients = clients
    await asyncio.to_thread(s.warm_up)
    s.task = asyncio.create_task(s.run())
    SESSION = s

    await s.broadcast({"type": "session", "data": {
        "scenario": s.scenario, "seed": s.seed, "speed": s.speed,
        "run_id": s.run_id, "warmed": True,
    }})
    return {"ok": True, "run_id": s.run_id, "scenario": s.scenario,
            "speed": s.speed, "status": s.agent.status()}


@app.post("/api/session/control")
async def control(req: ControlRequest) -> Dict[str, Any]:
    if SESSION is None:
        return JSONResponse({"error": "no active session"}, status_code=409)
    if req.action == "pause":
        SESSION.paused = True
    elif req.action == "resume":
        SESSION.paused = False
    elif req.action == "speed" and req.speed:
        SESSION.speed = req.speed
    else:
        return JSONResponse({"error": "bad action"}, status_code=400)
    return {"ok": True, "paused": SESSION.paused, "speed": SESSION.speed}


@app.get("/api/status")
def status() -> Dict[str, Any]:
    if SESSION is None:
        return {"active": False}
    return {
        "active": True, "run_id": SESSION.run_id, "scenario": SESSION.scenario,
        "paused": SESSION.paused, "speed": SESSION.speed,
        "finished": SESSION.finished, "feed": SESSION.feed.info(),
        "agent": SESSION.agent.status(), "last_tick": SESSION.last_tick,
        "series": SESSION.series[-400:],
    }


@app.get("/api/alerts")
def alerts() -> Dict[str, Any]:
    if SESSION is None or STORE is None:
        return {"alerts": []}
    return {"alerts": STORE.alerts_for(SESSION.run_id)}


@app.get("/api/history")
def history() -> Dict[str, Any]:
    if STORE is None:
        return {"history": [], "lifetime": None}
    return {"history": STORE.history(60), "lifetime": STORE.lifetime_stats()}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    await socket.accept()
    if SESSION:
        SESSION.clients.add(socket)
        if SESSION.last_tick:
            await socket.send_text(json.dumps(
                {"type": "tick", "data": SESSION.last_tick}, default=float))
    try:
        while True:
            await socket.receive_text()          # keep-alive / client pings
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if SESSION:
            SESSION.clients.discard(socket)


# ---------------------------------------------------------------------------
# Static frontend (served only once the React app has been built)
# ---------------------------------------------------------------------------
DIST = ROOT / "frontend" / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(DIST / "index.html"))
