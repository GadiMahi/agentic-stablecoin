"""Minimal JSON-over-HTTPS helper for the market data sources.

Standard library only (urllib), so connecting real data adds no dependency.
Every call goes through ``get_json``; tests replace it with a fake server.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

USER_AGENT = "SthiraSense/0.1 (+https://github.com/GadiMahi/agentic-stablecoin)"


class SourceError(RuntimeError):
    """A data source could not be reached or returned unusable data."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


def get_json(url: str, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> Any:
    """GET ``url`` with query ``params`` and decode the JSON body.

    Raises ``SourceError`` (with the HTTP status when there is one) on any
    network, HTTP or decoding failure, so callers handle one exception type.
    """
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json",
                                               **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:200]
        except Exception:
            pass
        raise SourceError(f"HTTP {e.code} from {url.split('?')[0]}: {body}", status=e.code) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise SourceError(f"could not reach {url.split('?')[0]}: {e}") from e
    except json.JSONDecodeError as e:
        raise SourceError(f"invalid JSON from {url.split('?')[0]}") from e
