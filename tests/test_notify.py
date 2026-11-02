"""Automated-response tests.

The response layer is the part most tempting to fake, so these pin the
properties that make it real:

* a channel that is not configured reports itself unconfigured and is skipped —
  never counted as delivered;
* a channel that fails records the failure rather than swallowing it;
* a failing channel never propagates an exception into the agent loop;
* the level gate actually gates.

The webhook test runs against a real local HTTP server, so the POST path is
genuinely exercised rather than mocked at the library boundary.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sthirasense.notify import (                                  # noqa: E402
    AlertMessage, Channel, ConsoleChannel, EmailChannel,
    NotificationRouter, WebhookChannel, build_router,
)


def _msg(level: str = "WARN", alert_id: int = 1) -> AlertMessage:
    return AlertMessage(
        alert_id=alert_id, level=level, probability=2.46e-2, threshold=1.8126e-3,
        stress=70.0, stress_band="HIGH", price=0.91446, deviation_pct=-8.554,
        rationale="expected cost of missing outweighs a false alarm",
        scenario="svb_bleed", ts=1760000000,
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------- message
def test_message_renders_the_numbers_that_matter():
    t = _msg().text()
    assert "0.91446" in t
    assert "2.460e-02" in t
    assert "13.6x threshold" in t          # ratio, not just the raw probability
    assert "70.0/100" in t
    assert "WARN" in t


def test_ratio_is_relative_to_the_tuned_threshold():
    m = _msg()
    assert m.ratio == pytest.approx(2.46e-2 / 1.8126e-3, rel=1e-6)


# --------------------------------------------------------------------- gating
@pytest.mark.parametrize("min_level,level,expected", [
    ("WARN", "HOLD", False), ("WARN", "WATCH", False),
    ("WARN", "WARN", True), ("WARN", "CRITICAL", True),
    ("CRITICAL", "WARN", False), ("CRITICAL", "CRITICAL", True),
])
def test_level_gate(min_level, level, expected):
    r = NotificationRouter([ConsoleChannel()], min_level=min_level)
    assert r.should_notify(level) is expected


# -------------------------------------------------------------- configuration
def test_unconfigured_channel_is_skipped_not_faked(monkeypatch):
    for var in ("STHIRA_SMTP_HOST", "STHIRA_SMTP_USER",
                "STHIRA_SMTP_PASS", "STHIRA_SMTP_TO"):
        monkeypatch.delenv(var, raising=False)
    email = EmailChannel()
    assert email.configured is False
    assert email.setup_hint                      # tells the operator what to set

    r = NotificationRouter([email])
    receipts = r.dispatch(_msg(), blocking=True)
    assert receipts == [], "an unconfigured channel must not produce a receipt"


def test_webhook_unconfigured_without_env(monkeypatch):
    monkeypatch.delenv("STHIRA_WEBHOOK_URL", raising=False)
    assert WebhookChannel().configured is False


# ------------------------------------------------------------------- delivery
def test_webhook_posts_to_a_real_server():
    received = []
    port = _free_port()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            received.append(json.loads(self.rfile.read(n)))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ch = WebhookChannel(f"http://127.0.0.1:{port}/hook")
        assert ch.configured
        receipts = NotificationRouter([ch]).dispatch(_msg(), blocking=True)
        assert len(receipts) == 1 and receipts[0].ok, receipts[0].detail
        assert received, "server never received the POST"
        # One payload that works on both Discord (`content`) and Slack (`text`).
        assert "content" in received[0] and "text" in received[0]
        assert "WARN" in received[0]["content"]
    finally:
        srv.shutdown()


def test_failed_delivery_is_recorded_not_swallowed():
    ch = WebhookChannel(f"http://127.0.0.1:{_free_port()}/nothing-listening")
    r = NotificationRouter([ch])
    receipts = r.dispatch(_msg(), blocking=True)
    assert len(receipts) == 1
    assert receipts[0].ok is False
    assert receipts[0].detail                    # says *why*
    assert r.status()["receipts"][0]["ok"] is False


def test_a_broken_channel_never_raises_into_the_caller():
    """The agent loop must survive any channel misbehaving."""
    class Exploding(Channel):
        name = "exploding"
        def deliver(self, msg):
            raise RuntimeError("boom")

    r = NotificationRouter([Exploding(), ConsoleChannel()])
    receipts = r.dispatch(_msg(), blocking=True)       # must not raise
    by = {x.channel: x for x in receipts}
    assert by["exploding"].ok is False
    assert "boom" in by["exploding"].detail
    assert by["console"].ok is True, "one bad channel must not block the others"


def test_browser_channel_pushes_through_bound_transport():
    sent = []
    r = build_router(push=lambda payload: sent.append(payload))
    receipts = {x.channel: x for x in r.dispatch(_msg(), blocking=True)}
    assert receipts["browser"].ok is True
    assert len(sent) == 1
    assert sent[0]["level"] == "WARN"
    assert "subject" in sent[0] and "text" in sent[0]


def test_status_reports_only_genuinely_live_channels(monkeypatch):
    monkeypatch.delenv("STHIRA_WEBHOOK_URL", raising=False)
    for var in ("STHIRA_SMTP_HOST", "STHIRA_SMTP_USER",
                "STHIRA_SMTP_PASS", "STHIRA_SMTP_TO"):
        monkeypatch.delenv(var, raising=False)
    st = build_router(push=lambda p: None).status()
    assert "console" in st["active"]
    assert "browser" in st["active"]
    assert "webhook" not in st["active"]
    assert "email" not in st["active"]
