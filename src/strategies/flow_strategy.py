"""
Order Flow Strategy -- wraps the flow signal for the backtesting engine.

Uses CFTC COT positioning z-scores as a contrarian signal.
Weekly rebalancing with 3bps costs (ETF-level trading).

The signal is primarily contrarian:
  - Extreme long speculator positioning -> expect mean reversion (underweight)
  - Extreme short speculator positioning -> expect mean reversion (overweight)
"""

from __future__ import annotations

import pandas as pd

from src.signals.flow_signal import FlowSignalComputer, flow_signal_to_weights


class FlowStrategy:
    """Strategy that trades on order flow / positioning signals.

    Parameters
    ----------
    positioning_zscore : pd.DataFrame
        Z-scores of COT positioning, index=publication_date, columns=tickers.
    long_only : bool
        If True, only take long positions.
    contrarian : bool
        If True (default), trade against extreme positioning.
    n_long : int or None
        Number of long positions.
    n_short : int or None
        Number of short positions.
    """

    def __init__(
        self,
        positioning_zscore: pd.DataFrame,
        long_only: bool = False,
        contrarian: bool = True,
        n_long: int | None = None,
        n_short: int | None = None,
    ):
        self.signal_computer = FlowSignalComputer(positioning_zscore, contrarian)
        self.long_only = long_only
        self.n_long = n_long
        self.n_short = n_short
        self._last_weights: dict[str, float] | None = None

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float] | None:
        """Generate portfolio weights from flow signals."""
        signal = self.signal_computer.get_signal(date)

        if signal.empty:
            return self._last_weights if self._last_weights else None

        weights = flow_signal_to_weights(
            signal, universe,
            n_long=self.n_long,
            n_short=self.n_short,
            long_only=self.long_only,
        )

        if not weights:
            return self._last_weights if self._last_weights else None

        self._last_weights = weights
        return weights
