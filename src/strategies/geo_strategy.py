"""
Geospatial / Macro Strategy -- wraps the geo signal for the backtesting engine.

Uses satellite luminosity, Google Trends stress index, and FRED macro
surprises for country/sector rotation. Monthly rebalancing.

This strategy operates at the ETF level (not stock-level) and is
designed for the EQUITY_ETFS + MULTI_ASSET universe.
"""

from __future__ import annotations

import pandas as pd

from src.signals.geo_signal import combine_geo_signals, geo_signal_to_weights


class GeoStrategy:
    """Strategy that trades on geospatial / macro signals.

    Parameters
    ----------
    luminosity_changes : pd.DataFrame or None
        Country-level luminosity changes (from satellite.py).
        Index=date, columns=country.
    stress_index : pd.Series or None
        Google Trends aggregate stress index (from google_trends.py).
    fred_surprises : pd.DataFrame or None
        FRED macro surprise z-scores (from fred.py).
    long_only : bool
        If True, only take long positions.
    top_n : int or None
        Number of ETFs to hold.
    """

    def __init__(
        self,
        luminosity_changes: pd.DataFrame | None = None,
        stress_index: pd.Series | None = None,
        fred_surprises: pd.DataFrame | None = None,
        long_only: bool = True,
        top_n: int | None = None,
    ):
        self.luminosity_changes = luminosity_changes
        self.stress_index = stress_index
        self.fred_surprises = fred_surprises
        self.long_only = long_only
        self.top_n = top_n

    def _get_luminosity_signal(self, date: pd.Timestamp) -> dict[str, float] | None:
        """Get luminosity signal for a date."""
        if self.luminosity_changes is None or self.luminosity_changes.empty:
            return None

        from src.data.satellite import luminosity_to_etf_signals, COUNTRY_ETF_MAP
        available = self.luminosity_changes.loc[:date]
        if available.empty:
            return None
        return luminosity_to_etf_signals(self.luminosity_changes, date)

    def _get_stress_signal(self, date: pd.Timestamp) -> float | None:
        """Get stress index for a date."""
        if self.stress_index is None or self.stress_index.empty:
            return None
        available = self.stress_index.loc[:date]
        if available.empty:
            return None
        return float(available.iloc[-1])

    def _get_fred_signal(self, date: pd.Timestamp) -> dict[str, float] | None:
        """Get FRED surprise signals for a date."""
        if self.fred_surprises is None or self.fred_surprises.empty:
            return None
        available = self.fred_surprises.loc[:date]
        if available.empty:
            return None
        latest = available.iloc[-1]
        return latest.dropna().to_dict()

    def generate_signals(
        self,
        date: pd.Timestamp,
        universe: list[str],
        lookback: dict[str, pd.DataFrame],
    ) -> dict[str, float] | None:
        """Generate portfolio weights from geo/macro signals.

        Returns None if no data sources have any data for this date.
        Returns {} if signals compute but produce no valid positions.
        """
        lum_signal = self._get_luminosity_signal(date)
        stress = self._get_stress_signal(date)
        fred = self._get_fred_signal(date)

        # If no data sources available at all, no opinion
        if lum_signal is None and stress is None and fred is None:
            return None

        combined = combine_geo_signals(
            luminosity_signals=lum_signal,
            stress_index=stress,
            fred_surprises=fred,
        )

        if not combined:
            return {}

        weights = geo_signal_to_weights(
            combined, universe,
            top_n=self.top_n,
            long_only=self.long_only,
        )

        # weights can be {} if no combined signals map to universe
        return weights
