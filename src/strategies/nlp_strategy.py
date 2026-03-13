"""
NLP Filing Strategy -- wraps the filing signal for the backtesting engine.

This strategy:
  1. Maintains a cache of parsed filing features per ticker.
  2. On each rebalance date, builds the signal from the most recent
     filings available as of that date (point-in-time).
  3. Returns portfolio weights based on quintile ranking.
  4. Returns None (keep existing positions) when no new filings are
     available since the last rebalance.

Designed for monthly rebalancing with the ETF universe. For
stock-level trading, the transaction cost model needs to be adjusted
(5-15bps per trade on individual stocks vs 3bps on ETFs).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.signals.filing_signal import (
    FilingFeatures,
    build_filing_signal_df,
    signal_to_weights,
)


class NLPFilingStrategy:
    """Strategy that trades on SEC filing NLP signals.

    Parameters
    ----------
    filing_features : list[FilingFeatures]
        Pre-computed filing features for all tickers in the universe.
        In production, these would be computed incrementally as new
        filings arrive. For backtesting, pre-compute all features
        and pass them in.
    long_only : bool
        If True, only take long positions (top quintile).
    n_long : int or None
        Number of long positions. Defaults to top quintile of universe.
    n_short : int or None
        Number of short positions. Ignored if long_only.
    stale_days : int
        Maximum age (in days) for a filing signal to be considered
        valid. After this many days, the signal decays to neutral.
    """

    def __init__(
        self,
        filing_features: list[FilingFeatures],
        long_only: bool = True,
        n_long: int | None = None,
        n_short: int | None = None,
        stale_days: int = 120,
    ):
        self.filing_features = filing_features
        self.long_only = long_only
        self.n_long = n_long
        self.n_short = n_short
        self.stale_days = stale_days

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float] | None:
        """Generate portfolio weights from NLP filing signals.

        Returns None if no filing data is available (keep existing positions).
        Returns {} if signal says go to cash (no valid filings in universe).
        """
        # Filter features to those available as of this date and in universe
        relevant = [
            ff for ff in self.filing_features
            if ff.ticker in universe
            and pd.Timestamp(ff.filing_date) <= date
        ]

        if not relevant:
            # No filings have ever been available -- no opinion yet
            return None

        # Filter out stale filings
        cutoff = date - pd.Timedelta(days=self.stale_days)
        fresh = [
            ff for ff in relevant
            if pd.Timestamp(ff.filing_date) >= cutoff
        ]

        if not fresh:
            # All filings are stale -- signal says go to cash
            return {}

        # Build signal DataFrame
        signal_df = build_filing_signal_df(fresh, as_of_date=date)
        if signal_df.empty:
            return {}

        # Only use tickers in the current universe
        signal_df = signal_df[signal_df.index.isin(universe)]
        if signal_df.empty:
            return {}

        # Convert to weights
        weights = signal_to_weights(
            signal_df,
            n_long=self.n_long,
            n_short=self.n_short,
            long_only=self.long_only,
        )

        # signal_to_weights returns {} if no positions pass the filter
        return weights
