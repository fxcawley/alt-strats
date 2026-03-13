"""
Satellite / macro geospatial signal.

Combines multiple macro-level data sources into a country/sector
rotation signal:
  1. VIIRS nighttime light luminosity changes
  2. Google Trends economic stress index
  3. FRED macro surprises

This is a macro-level signal (country/sector, not stock-level).
Monthly rebalancing with the ETF universe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def combine_geo_signals(
    luminosity_signals: dict[str, float] | None = None,
    stress_index: float | None = None,
    fred_surprises: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Combine multiple macro signals into a single score per ETF.

    Parameters
    ----------
    luminosity_signals : dict
        {ticker: luminosity_change_signal}
    stress_index : float
        Google Trends stress index (higher = more bearish for equities).
    fred_surprises : dict
        {macro_series_name: surprise_zscore}
    weights : dict
        Relative weights for each signal component.
        Defaults to equal weight.

    Returns a dict of {ticker: combined_signal}.
    """
    if weights is None:
        weights = {
            "luminosity": 0.33,
            "stress": 0.33,
            "fred": 0.34,
        }

    combined = {}

    # Luminosity signals (bullish for associated ETFs)
    if luminosity_signals:
        for ticker, sig in luminosity_signals.items():
            combined[ticker] = combined.get(ticker, 0.0) + sig * weights["luminosity"]

    # Stress index (bearish for all equities when high)
    if stress_index is not None:
        equity_tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VGK", "EWJ", "FXI",
                          "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU"]
        bond_tickers = ["TLT", "IEF", "SHY", "LQD", "HYG"]

        stress_signal = -stress_index * weights["stress"]
        for t in equity_tickers:
            combined[t] = combined.get(t, 0.0) + stress_signal
        # Bonds benefit from stress (flight to quality)
        for t in bond_tickers:
            combined[t] = combined.get(t, 0.0) - stress_signal * 0.5

    # FRED surprises (positive surprise = bullish for risk assets)
    if fred_surprises:
        # Average of macro surprises as a broad signal
        avg_surprise = np.mean([v for v in fred_surprises.values() if not np.isnan(v)])
        if not np.isnan(avg_surprise):
            risk_tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLK", "XLF", "XLE", "XLY"]
            for t in risk_tickers:
                combined[t] = combined.get(t, 0.0) + avg_surprise * weights["fred"]

    return combined


def geo_signal_to_weights(
    signals: dict[str, float],
    universe: list[str],
    top_n: int | None = None,
    long_only: bool = True,
) -> dict[str, float]:
    """Convert geo signals to portfolio weights.

    Ranks ETFs by combined signal and goes long the top N.
    """
    # Filter to universe
    filtered = {t: s for t, s in signals.items() if t in universe}
    if not filtered:
        return {}

    ranked = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    n = len(ranked)

    if top_n is None:
        top_n = max(1, n // 3)

    long_tickers = [t for t, _ in ranked[:top_n]]
    weights = {}
    if long_only:
        w = 1.0 / len(long_tickers)
        for t in long_tickers:
            weights[t] = w
    else:
        short_tickers = [t for t, _ in ranked[-top_n:]]
        w_long = 0.5 / len(long_tickers)
        w_short = -0.5 / len(short_tickers)
        for t in long_tickers:
            weights[t] = w_long
        for t in short_tickers:
            weights[t] = w_short

    return weights
