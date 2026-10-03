"""Audit store.

Every alert, resolution and cost-model update is written to SQLite. Two reasons
this is not optional: the learning element's claims about its own precision have
to be checkable against a record rather than taken on trust, and an agent that
acts autonomously needs its decisions to be reconstructable after the fact.

Deliberately plain SQL with no ORM — the schema is small and the point is that a
reviewer can open the file and read it.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario    TEXT    NOT NULL,
    seed        INTEGER NOT NULL,
    started_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    threshold   REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    alert_id    INTEGER NOT NULL,
    opened_ts   INTEGER NOT NULL,
    level       TEXT    NOT NULL,
    probability REAL    NOT NULL,
    stress      REAL    NOT NULL,
    rationale   TEXT,
    analogue    TEXT,
    outcome     TEXT,
    resolved_ts INTEGER,
    peak_dev    REAL
);
CREATE TABLE IF NOT EXISTS learning_updates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    ts          INTEGER NOT NULL,
    reason      TEXT    NOT NULL,
    miss_base   REAL    NOT NULL,
    false_base  REAL    NOT NULL,
    precision   REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_run ON alerts(run_id);
CREATE INDEX IF NOT EXISTS idx_learn_run ON learning_updates(run_id);
"""


class AuditStore:
    """Thread-safe, single-file audit log."""

    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # --------------------------------------------------------------- lifecycle
    def start_run(self, scenario: str, seed: int, threshold: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs (scenario, seed, threshold) VALUES (?,?,?)",
                (scenario, seed, threshold),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ writes
    def record_alert(self, run_id: int, alert: Dict[str, Any],
                     analogue: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO alerts (run_id, alert_id, opened_ts, level, probability,"
                " stress, rationale, analogue) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, alert["id"], alert["opened_ts"], alert["level"],
                 alert["probability"], alert["stress"], alert.get("rationale"),
                 json.dumps(analogue) if analogue else None),
            )
            self._conn.commit()

    def resolve_alert(self, run_id: int, alert: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE alerts SET outcome=?, resolved_ts=?, peak_dev=?"
                " WHERE run_id=? AND alert_id=?",
                (alert.get("outcome"), alert.get("resolved_ts"),
                 alert.get("peak_deviation"), run_id, alert["id"]),
            )
            self._conn.commit()

    def record_learning(self, run_id: int, update: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO learning_updates (run_id, ts, reason, miss_base,"
                " false_base, precision) VALUES (?,?,?,?,?,?)",
                (run_id, update.get("ts", 0), update.get("reason", ""),
                 update.get("miss_base", 0.0), update.get("false_base", 0.0),
                 update.get("precision")),
            )
            self._conn.commit()

    # ------------------------------------------------------------------- reads
    def alerts_for(self, run_id: int) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE run_id=? ORDER BY opened_ts DESC", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Alert outcomes across all runs — the agent's track record."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.scenario, a.* FROM alerts a JOIN runs r ON r.id = a.run_id"
                " ORDER BY a.id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def lifetime_stats(self) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) n,"
                " SUM(outcome='TRUE_POSITIVE') tp,"
                " SUM(outcome='FALSE_POSITIVE') fp,"
                " SUM(outcome IS NULL) open FROM alerts"
            ).fetchone()
        n, tp, fp, op = (row["n"] or 0, row["tp"] or 0, row["fp"] or 0, row["open"] or 0)
        resolved = tp + fp
        return {"alerts": n, "true_positives": tp, "false_positives": fp,
                "open": op, "precision": round(tp / resolved, 4) if resolved else None}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
