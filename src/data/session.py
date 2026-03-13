"""
Shared yfinance session configuration.

Handles corporate proxy / SSL interception environments by creating a
curl_cffi session with SSL verification disabled and browser impersonation
enabled (required by Yahoo Finance). All data fetchers should use
get_session() to obtain a properly configured session.
"""

from __future__ import annotations

from curl_cffi.requests import Session
from yfinance import data as _yfdata

_SESSION: Session | None = None


def get_session() -> Session:
    """Return a singleton curl_cffi session with SSL verify disabled.

    Also clears yfinance's internal singleton cache to prevent stale
    rate-limit state from blocking fresh requests.
    """
    global _SESSION
    if _SESSION is None:
        # Clear any stale yfinance singleton state (cached rate-limit, bad crumbs)
        if hasattr(_yfdata, "YfData") and hasattr(_yfdata.YfData, "_instances"):
            _yfdata.YfData._instances = {}
        _SESSION = Session(verify=False, impersonate="chrome")
    return _SESSION
