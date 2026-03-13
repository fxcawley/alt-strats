"""
Runner script for the satellite / geospatial macro signal pipeline.

Usage:
    python run_geo_signal.py [--backtest] [--fred-key YOUR_KEY]

This script:
  1. Loads macro data sources (FRED with publication lag, Google Trends
     via rolling windows, satellite luminosity from CSV)
  2. Computes the combined geo/macro signal
  3. Optionally runs a walk-forward backtest
  4. Runs the 5-gate validation framework on the results

Data requirements:
  - FRED: set FRED_API_KEY environment variable (free from fred.stlouisfed.org)
  - Google Trends: requires network access (rate-limited)
  - Satellite: requires pre-processed CSV in data/satellite/
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data.universe import EQUITY_ETF_TICKERS, MULTI_ASSET_TICKERS
from src.validation.gates import run_validation


def main():
    parser = argparse.ArgumentParser(description="Geospatial / Macro Signal Pipeline")
    parser.add_argument("--backtest", action="store_true",
                        help="Run walk-forward backtest")
    parser.add_argument("--fred-key", default=None,
                        help="FRED API key (or set FRED_API_KEY env var)")
    args = parser.parse_args()

    print("Geospatial / Macro Signal Pipeline")
    print()

    luminosity_changes = None
    stress_index = None
    fred_surprises = None

    # Satellite data (requires real CSV -- no synthetic fallback)
    from src.data.satellite import load_luminosity_data, compute_luminosity_change, SATELLITE_DIR
    csv_path = SATELLITE_DIR / "luminosity.csv"
    if csv_path.exists():
        print("Loading satellite luminosity data...")
        lum_df = load_luminosity_data(csv_path)
        luminosity_changes = compute_luminosity_change(lum_df, apply_pub_lag=True)
        print(f"  Luminosity data: {len(luminosity_changes)} months x "
              f"{len(luminosity_changes.columns)} countries")
    else:
        print(f"  Satellite data not found at {csv_path}")
        print("  Provide a CSV with columns: date, country, luminosity")

    # FRED data (with publication lag applied)
    if args.fred_key:
        import os
        os.environ["FRED_API_KEY"] = args.fred_key

    try:
        print("Fetching FRED macro data (with publication lag offsets)...")
        from src.data.fred import fetch_all_macro_series, compute_fred_surprise
        macro_df = fetch_all_macro_series(apply_pub_lag=True)
        fred_surprises = compute_fred_surprise(macro_df)
        print(f"  FRED data: {len(macro_df)} dates x {len(macro_df.columns)} series")
    except Exception as e:
        print(f"  FRED data unavailable: {e}")
        print("  Set FRED_API_KEY env var or pass --fred-key to enable")

    # Google Trends (rolling windows to avoid normalization bias)
    try:
        print("Fetching Google Trends data (rolling windows)...")
        from src.data.google_trends import (
            fetch_google_trends_rolling, compute_trends_zscore,
            compute_aggregate_stress_index, ALL_KEYWORDS,
        )
        trends_df = fetch_google_trends_rolling(ALL_KEYWORDS[:5])
        if not trends_df.empty:
            trends_zscore = compute_trends_zscore(trends_df)
            stress_index = compute_aggregate_stress_index(trends_zscore)
            print(f"  Google Trends data: {len(trends_df)} dates x "
                  f"{len(trends_df.columns)} keywords")
        else:
            print("  Google Trends returned empty data")
    except Exception as e:
        print(f"  Google Trends unavailable: {e}")

    # Check if we have any real data
    n_sources = sum([
        luminosity_changes is not None,
        stress_index is not None,
        fred_surprises is not None,
    ])
    if n_sources == 0:
        print("\nNo real data sources available. Cannot proceed.")
        print("At minimum, set FRED_API_KEY to enable FRED data.")
        sys.exit(1)
    else:
        print(f"\n{n_sources}/3 data sources loaded.")

    if args.backtest:
        from src.strategies.geo_strategy import GeoStrategy
        from src.backtest.engine import run_backtest

        universe = MULTI_ASSET_TICKERS
        print(f"\n{'=' * 60}")
        print(f"Running walk-forward backtest (monthly rebalance, "
              f"{len(universe)} ETFs)...")
        print(f"{'=' * 60}")

        strategy = GeoStrategy(
            luminosity_changes=luminosity_changes,
            stress_index=stress_index,
            fred_surprises=fred_surprises,
            long_only=True,
        )

        result = run_backtest(
            strategy=strategy,
            universe=universe,
            start="2013-01-01",
            rebalance_freq="ME",
            cost_pct=0.0003,
            rebalance_threshold=0.02,
        )

        summary = result.summary()
        print("\nBacktest Results:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        report = run_validation(
            signal_name="Geospatial / Macro Signal",
            strategy_returns=result.returns,
        )
        print(f"\n{report.summary()}")
    else:
        print("\nUse --backtest to run the walk-forward backtest")


if __name__ == "__main__":
    main()
