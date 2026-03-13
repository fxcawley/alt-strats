"""
Price data fetcher using yfinance.

Handles daily OHLCV data for individual tickers and ETF benchmarks,
with a local parquet cache so repeated backtests don't hammer the API.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.data.session import get_session

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _cache_path(ticker: str, start: str, end: str, adjusted: bool = True) -> Path:
    key = hashlib.md5(f"{ticker}_{start}_{end}_adj={adjusted}".encode()).hexdigest()
    return CACHE_DIR / f"{ticker}_{key}.parquet"


def fetch_prices(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch daily OHLCV data for *ticker* (dividend-adjusted).

    Uses auto_adjust=True so all prices (Open, High, Low, Close) are
    adjusted for dividends and splits.  This means Close reflects total
    return, not just price return.

    Returns a DataFrame indexed by date with columns:
        Open, High, Low, Close, Volume
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(ticker, start, end)

    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, session=get_session())
    if df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    # Flatten MultiIndex columns if present (yfinance quirk with single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    if use_cache:
        df.to_parquet(cp)
    return df


def fetch_benchmark(
    benchmark: str = "SPY",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Convenience wrapper: fetch ETF/benchmark prices."""
    return fetch_prices(benchmark, start=start, end=end)


# Common benchmarks for quick reference
BENCHMARKS = {
    "sp500": "SPY",
    "nasdaq": "QQQ",
    "russell2000": "IWM",
    "total_market": "VTI",
    "bonds": "AGG",
    "gold": "GLD",
}
