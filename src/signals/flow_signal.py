"""
Order flow / positioning signal.

Combines CFTC COT positioning data with other flow indicators to
produce a contrarian signal: extreme bullish positioning -> expect
mean reversion (bearish), and vice versa.

Signal components:
  1. COT z-score: z-score of large speculator net positioning
  2. Put/call ratio (placeholder for CBOE data)
  3. ETF flow momentum (placeholder for flow data)

The signal is primarily contrarian: when everyone is positioned one way,
the crowded trade is vulnerable to unwind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.cot import compute_positioning_zscore


def compute_flow_signal(
    positioning_zscore: pd.DataFrame,
    date: pd.Timestamp,
    contrarian: bool = True,
) -> pd.Series:
    """Compute the flow signal for a single date.

    Parameters
    ----------
    positioning_zscore : pd.DataFrame
        Z-scores of COT positioning, index=date, columns=tickers.
    date : pd.Timestamp
        The signal date (must use publication_date, not report_date).
    contrarian : bool
        If True (default), extreme long = bearish signal (and vice versa).
        The contrarian approach is well-documented in practitioner literature.

    Returns a Series indexed by ticker with signal values.
    Negative values = bearish, positive = bullish.
    """
    # Get the most recent data on or before this date
    available = positioning_zscore.loc[:date]
    if available.empty:
        return pd.Series(dtype=float)

    latest = available.iloc[-1]

    if contrarian:
        # Flip sign: extreme long positioning -> bearish signal
        signal = -latest
    else:
        signal = latest

    return signal.dropna()


def flow_signal_to_weights(
    signal: pd.Series,
    universe: list[str],
    n_long: int | None = None,
    n_short: int | None = None,
    long_only: bool = False,
) -> dict[str, float]:
    """Convert flow signal to portfolio weights.

    Parameters
    ----------
    signal : pd.Series
        Signal values indexed by ticker. Higher = more bullish.
    universe : list[str]
        Tickers we're allowed to trade.
    n_long, n_short : int or None
        Number of long/short positions. Defaults to top/bottom third.
    long_only : bool
        If True, only take long positions.

    Returns portfolio weights dict.
    """
    # Filter to universe
    signal = signal[signal.index.isin(universe)]
    if signal.empty:
        return {}

    ranked = signal.sort_values(ascending=False)
    n = len(ranked)

    if n_long is None:
        n_long = max(1, n // 3)
    if n_short is None:
        n_short = max(1, n // 3) if not long_only else 0

    long_tickers = ranked.head(n_long).index.tolist()
    short_tickers = ranked.tail(n_short).index.tolist() if n_short > 0 else []

    weights = {}
    if long_tickers:
        w = 1.0 / len(long_tickers) if long_only else 0.5 / len(long_tickers)
        for t in long_tickers:
            weights[t] = w

    if short_tickers and not long_only:
        w = -0.5 / len(short_tickers)
        for t in short_tickers:
            weights[t] = w

    return weights


class FlowSignalComputer:
    """Stateful flow signal computer for backtesting.

    Pre-computes positioning z-scores and serves them to the strategy
    on each rebalance date.
    """

    def __init__(
        self,
        positioning_zscore: pd.DataFrame,
        contrarian: bool = True,
    ):
        self.positioning_zscore = positioning_zscore
        self.contrarian = contrarian

    def get_signal(self, date: pd.Timestamp) -> pd.Series:
        """Get the flow signal for a date."""
        return compute_flow_signal(
            self.positioning_zscore, date, self.contrarian
        )
