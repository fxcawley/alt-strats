"""
VIIRS nighttime light data processor.

Processes NOAA VIIRS Day/Night Band composites to extract economic
activity proxies. Month-over-month luminosity changes at the country
level correlate with GDP growth and can be used as a leading indicator
for country/sector ETFs.

In practice, this module provides a framework for processing the data.
Actual VIIRS composites are large GeoTIFF files (~500MB per month).
For this project, we use pre-aggregated country-level data or
demonstrate the pipeline with synthetic data.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

SATELLITE_DIR = Path(__file__).resolve().parents[2] / "data" / "satellite"

# Country -> ETF mapping for satellite signals
COUNTRY_ETF_MAP = {
    "US": ["SPY", "QQQ", "IWM"],
    "China": ["FXI", "EEM"],
    "Japan": ["EWJ"],
    "Europe": ["VGK", "EFA"],
    "Germany": ["VGK"],
    "UK": ["EFA"],
    "India": ["EEM"],
    "Brazil": ["EEM"],
    "South Korea": ["EEM"],
}


def load_luminosity_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load country-level luminosity data.

    Expected format: CSV with columns [date, country, luminosity].
    If no file provided, returns a template DataFrame.

    In production, this would read from processed VIIRS GeoTIFF
    aggregates. For development, we support CSV input.
    """
    if filepath is not None:
        path = Path(filepath)
        if path.exists():
            df = pd.read_csv(path, parse_dates=["date"])
            return df

    # Return empty template
    return pd.DataFrame(columns=["date", "country", "luminosity"])


def compute_luminosity_change(
    luminosity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute month-over-month luminosity change per country.

    Returns a pivoted DataFrame: index=date, columns=country,
    values=pct_change in luminosity.
    """
    if luminosity_df.empty:
        return pd.DataFrame()

    pivot = luminosity_df.pivot_table(
        index="date",
        columns="country",
        values="luminosity",
        aggfunc="mean",
    )
    pivot = pivot.sort_index()
    return pivot.pct_change()


def luminosity_to_etf_signals(
    luminosity_change: pd.DataFrame,
    date: pd.Timestamp,
) -> dict[str, float]:
    """Map country luminosity changes to ETF signals.

    Positive luminosity change in a country -> bullish for associated ETFs.
    Returns a dict of {ticker: signal_value}.
    """
    if luminosity_change.empty:
        return {}

    # Get most recent data
    available = luminosity_change.loc[:date]
    if available.empty:
        return {}

    latest = available.iloc[-1]

    signals = {}
    for country, change in latest.items():
        if pd.isna(change):
            continue
        etfs = COUNTRY_ETF_MAP.get(country, [])
        for etf in etfs:
            if etf in signals:
                signals[etf] = (signals[etf] + change) / 2
            else:
                signals[etf] = change

    return signals


def create_synthetic_luminosity(
    countries: list[str] | None = None,
    start_date: str = "2012-01-01",
    end_date: str = "2025-12-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic luminosity data for testing.

    Generates monthly luminosity values with realistic statistical
    properties (trending upward with noise, correlated across countries).
    This is for pipeline testing only -- not for signal generation.
    """
    if countries is None:
        countries = list(COUNTRY_ETF_MAP.keys())

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="MS")
    n = len(dates)

    # Base trend + correlated noise
    base_trend = np.linspace(100, 150, n)
    correlation = 0.7

    rows = []
    shared_noise = rng.normal(0, 3, n)
    for country in countries:
        idiosyncratic = rng.normal(0, 2, n)
        luminosity = base_trend + correlation * shared_noise + (1 - correlation) * idiosyncratic
        # Add country-specific growth rate
        growth_factor = 1 + rng.uniform(-0.005, 0.01)
        luminosity *= np.cumprod(np.full(n, growth_factor))

        for date, lum in zip(dates, luminosity):
            rows.append({
                "date": date,
                "country": country,
                "luminosity": max(lum, 0),
            })

    return pd.DataFrame(rows)
