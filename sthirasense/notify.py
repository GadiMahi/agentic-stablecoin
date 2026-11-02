"""Automated response — notification channels.

This is the actuator end of the architecture: the point where the agent stops
reasoning and changes something outside itself.

Four channels, all free, all optional. Whatever is not configured simply
reports itself as unconfigured and is skipped — the system never pretends to
have delivered something it did not.

    console   always on; writes to the service log
    browser   pushed over the open WebSocket; the dashboard raises a real
              OS notification. Zero setup.
    webhook   generic JSON POST. Discord and Slack incoming webhooks both
              accept the payload shape used here.
    email     SMTP. Works with a Gmail App Password, Outlook, or any relay.

Configuration is by environment variable, so no secret is ever committed:

    STHIRA_WEBHOOK_URL      https://discord.com/api/webhooks/...
    STHIRA_SMTP_HOST        smtp.gmail.com
    STHIRA_SMTP_PORT        587
    STHIRA_SMTP_USER        you@gmail.com
    STHIRA_SMTP_PASS        an app password, never your account password
    STHIRA_SMTP_FROM        optional, defaults to SMTP_USER
    STHIRA_SMTP_TO          comma-separated recipients
    STHIRA_MIN_LEVEL        WARN (default) or CRITICAL

Delivery is best-effort and non-blocking: a channel that times out or errors
records the failure and never propagates an exception into the agent loop. An
early-warning system that crashes because an SMTP server was slow is worse than
one that misses an email.
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("sthirasense.notify")

LEVEL_ORDER = {"HOLD": 0, "WATCH": 1, "WARN": 2, "CRITICAL": 3}
SEND_TIMEOUT = 8.0


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------
@dataclass
class AlertMessage:
    """What gets delivered. Built once, rendered per channel."""
    alert_id: int
    level: str
    probability: float
    threshold: float
    stress: float
    stress_band: str
    price: float
    deviation_pct: float
    rationale: str
    scenario: str
    ts: int
    analogue: Optional[Dict[str, Any]] = None

    # ---- renderings -------------------------------------------------------
    @property
    def ratio(self) -> float:
        return self.probability / self.threshold if self.threshold else 0.0

    def subject(self) -> str:
        return f"[{self.level}] Stablecoin depeg risk — stress {self.stress:.0f}/100"

    def text(self) -> str:
        lines = [
            f"{self.level}: depeg risk elevated",
            "",
            f"  Price              ${self.price:.5f}  ({self.deviation_pct:+.3f}% from peg)",
            f"  p(depeg, 24h)      {self.probability:.3e}   ({self.ratio:.1f}x threshold)",
            f"  Liquidity stress   {self.stress:.1f}/100  ({self.stress_band})",
            "",
            f"  Why: {self.rationale}",
        ]
        if self.analogue:
            lines += ["", f"  Closest historical episode: {self.analogue.get('incident')} "
                          f"({self.analogue.get('date')})"]
        lines += ["", f"  Source: SthiraSense · scenario '{self.scenario}' · alert #{self.alert_id}"]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id, "level": self.level,
            "probability": self.probability, "threshold": self.threshold,
            "ratio": round(self.ratio, 2), "stress": round(self.stress, 1),
            "stress_band": self.stress_band, "price": round(self.price, 6),
            "deviation_pct": round(self.deviation_pct, 4),
            "rationale": self.rationale, "scenario": self.scenario,
            "ts": self.ts, "analogue": self.analogue,
            "subject": self.subject(), "text": self.text(),
        }


@dataclass
class Receipt:
    """One delivery attempt. Kept whether it succeeded or not."""
    channel: str
    alert_id: int
    ok: bool
    detail: str
    ms: float
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"channel": self.channel, "alert_id": self.alert_id, "ok": self.ok,
                "detail": self.detail, "ms": round(self.ms, 1), "ts": self.ts}


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
class Channel:
    name = "channel"
    blurb = ""

    @property
    def configured(self) -> bool:
        return True

    @property
    def setup_hint(self) -> str:
        return ""

    def deliver(self, msg: AlertMessage) -> str:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"name": self.name, "configured": self.configured,
                "blurb": self.blurb, "hint": "" if self.configured else self.setup_hint}


class ConsoleChannel(Channel):
    name = "console"
    blurb = "Service log — always on"

    def deliver(self, msg: AlertMessage) -> str:
        log.warning("ALERT %s #%d  p=%.3e (%.1fx)  stress=%.1f",
                    msg.level, msg.alert_id, msg.probability, msg.ratio, msg.stress)
        return "written to service log"


class BrowserChannel(Channel):
    """Pushes to connected dashboards, which raise a real OS notification.

    Zero configuration — the transport is the WebSocket that is already open.
    """
    name = "browser"
    blurb = "Desktop notification in the dashboard"

    def __init__(self, push: Optional[Callable[[Dict[str, Any]], None]] = None):
        self._push = push
        self.clients = 0

    @property
    def configured(self) -> bool:
        return self._push is not None

    @property
    def setup_hint(self) -> str:
        return "no setup needed — allow notifications when the browser asks"

    def deliver(self, msg: AlertMessage) -> str:
        if not self._push:
            raise RuntimeError("no push transport bound")
        self._push(msg.to_dict())
        return f"pushed to {self.clients} connected dashboard(s)"


class WebhookChannel(Channel):
    """Generic JSON POST. Discord and Slack incoming webhooks both accept this."""
    name = "webhook"
    blurb = "Discord / Slack incoming webhook"

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("STHIRA_WEBHOOK_URL", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @property
    def setup_hint(self) -> str:
        return "set STHIRA_WEBHOOK_URL to a Discord or Slack webhook URL"

    def _payload(self, msg: AlertMessage) -> Dict[str, Any]:
        body = f"**{msg.subject()}**\n```\n{msg.text()}\n```"
        # `content` is Discord's field, `text` is Slack's. Sending both means one
        # payload works on either without sniffing the URL.
        return {"content": body, "text": body, "username": "SthiraSense"}

    def deliver(self, msg: AlertMessage) -> str:
        data = json.dumps(self._payload(msg)).encode()
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json", "User-Agent": "SthiraSense/1.0"},
            method="POST")
        with urllib.request.urlopen(req, timeout=SEND_TIMEOUT) as r:
            return f"HTTP {r.status}"


class EmailChannel(Channel):
    """SMTP. Tested shape works with Gmail App Passwords and Outlook."""
    name = "email"
    blurb = "Email over SMTP"

    def __init__(self) -> None:
        e = os.environ
        self.host = e.get("STHIRA_SMTP_HOST", "").strip()
        self.port = int(e.get("STHIRA_SMTP_PORT", "587") or 587)
        self.user = e.get("STHIRA_SMTP_USER", "").strip()
        self.password = e.get("STHIRA_SMTP_PASS", "").strip()
        self.sender = e.get("STHIRA_SMTP_FROM", "").strip() or self.user
        self.to = [a.strip() for a in e.get("STHIRA_SMTP_TO", "").split(",") if a.strip()]

    @property
    def configured(self) -> bool:
        return bool(self.host and self.user and self.password and self.to)

    @property
    def setup_hint(self) -> str:
        return ("set STHIRA_SMTP_HOST / _USER / _PASS / _TO "
                "(Gmail: use an App Password, not your account password)")

    def deliver(self, msg: AlertMessage) -> str:
        em = EmailMessage()
        em["Subject"] = msg.subject()
        em["From"] = self.sender
        em["To"] = ", ".join(self.to)
        em.set_content(msg.text())

        if self.port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(self.host, self.port, timeout=SEND_TIMEOUT, context=ctx) as s:
                s.login(self.user, self.password)
                s.send_message(em)
        else:
            with smtplib.SMTP(self.host, self.port, timeout=SEND_TIMEOUT) as s:
                s.ehlo()
                s.starttls(context=ssl.create_default_context())
                s.login(self.user, self.password)
                s.send_message(em)
        return f"sent to {len(self.to)} recipient(s)"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class NotificationRouter:
    """Fans an alert out to every configured channel, off the agent's thread."""

    def __init__(self, channels: Optional[List[Channel]] = None,
                 min_level: Optional[str] = None, max_receipts: int = 200):
        self.channels: List[Channel] = channels or []
        self.min_level = (min_level or os.environ.get("STHIRA_MIN_LEVEL", "WARN")).upper()
        self.receipts: List[Receipt] = []
        self.max_receipts = max_receipts
        self._lock = threading.Lock()

    # ------------------------------------------------------------- inspection
    def status(self) -> Dict[str, Any]:
        chans = [c.status() for c in self.channels]
        return {
            "min_level": self.min_level,
            "channels": chans,
            "active": [c["name"] for c in chans if c["configured"]],
            "receipts": [r.to_dict() for r in self.receipts[-40:]][::-1],
        }

    def should_notify(self, level: str) -> bool:
        return LEVEL_ORDER.get(level, 0) >= LEVEL_ORDER.get(self.min_level, 2)

    # ---------------------------------------------------------------- sending
    def _record(self, r: Receipt) -> None:
        with self._lock:
            self.receipts.append(r)
            if len(self.receipts) > self.max_receipts:
                del self.receipts[: len(self.receipts) - self.max_receipts]

    def _deliver_one(self, ch: Channel, msg: AlertMessage) -> Receipt:
        t0 = time.perf_counter()
        try:
            detail = ch.deliver(msg)
            ok = True
        except urllib.error.HTTPError as e:
            ok, detail = False, f"HTTP {e.code}"
        except urllib.error.URLError as e:
            ok, detail = False, f"unreachable: {e.reason}"
        except smtplib.SMTPAuthenticationError:
            ok, detail = False, "SMTP auth rejected — check the app password"
        except Exception as e:                       # never escapes into the agent
            ok, detail = False, f"{type(e).__name__}: {e}"
        r = Receipt(ch.name, msg.alert_id, ok, detail, (time.perf_counter() - t0) * 1000)
        self._record(r)
        if not ok:
            log.warning("notify %s failed for alert #%d: %s", ch.name, msg.alert_id, detail)
        return r

    def dispatch(self, msg: AlertMessage, blocking: bool = False) -> List[Receipt]:
        """Send to every configured channel. Returns receipts when blocking."""
        targets = [c for c in self.channels if c.configured]
        if not targets:
            return []
        if blocking:
            return [self._deliver_one(c, msg) for c in targets]

        for c in targets:
            threading.Thread(target=self._deliver_one, args=(c, msg),
                             daemon=True, name=f"notify-{c.name}").start()
        return []


def build_router(push: Optional[Callable[[Dict[str, Any]], None]] = None) -> NotificationRouter:
    """Standard channel set, read from the environment."""
    return NotificationRouter([
        ConsoleChannel(),
        BrowserChannel(push),
        WebhookChannel(),
        EmailChannel(),
    ])
