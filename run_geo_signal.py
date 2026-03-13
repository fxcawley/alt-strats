"""
Runner script for the satellite / geospatial macro signal pipeline.

Usage:
    python run_geo_signal.py [--backtest] [--use-synthetic]

This script:
  1. Loads macro data sources (FRED, Google Trends, satellite luminosity)
  2. Computes the combined geo/macro signal
  3. Optionally runs a walk-forward backtest
  4. Runs the 5-gate validation framework
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
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Use synthetic satellite data for testing")
    parser.add_argument("--fred-key", default=None,
                        help="FRED API key (or set FRED_API_KEY env var)")
    args = parser.parse_args()

    print("Geospatial / Macro Signal Pipeline")
    print()

    # Step 1: Load data sources
    luminosity_changes = None
    stress_index = None
    fred_surprises = None

    # Satellite data
    if args.use_synthetic:
        print("Generating synthetic satellite luminosity data...")
        from src.data.satellite import create_synthetic_luminosity, compute_luminosity_change
        synthetic = create_synthetic_luminosity()
        luminosity_changes = compute_luminosity_change(synthetic)
        print(f"  Luminosity data: {len(luminosity_changes)} months x {len(luminosity_changes.columns)} countries")
    else:
        print("Satellite data: skipped (use --use-synthetic for testing)")

    # FRED data
    if args.fred_key:
        import os
        os.environ["FRED_API_KEY"] = args.fred_key

    try:
        print("Fetching FRED macro data...")
        from src.data.fred import fetch_all_macro_series, compute_fred_surprise
        macro_df = fetch_all_macro_series()
        fred_surprises = compute_fred_surprise(macro_df)
        print(f"  FRED data: {len(macro_df)} dates x {len(macro_df.columns)} series")
    except Exception as e:
        print(f"  FRED data unavailable: {e}")
        print("  Set FRED_API_KEY env var or pass --fred-key to enable")

    # Google Trends
    try:
        print("Fetching Google Trends data...")
        from src.data.google_trends import (
            fetch_google_trends, compute_trends_zscore,
            compute_aggregate_stress_index, ALL_KEYWORDS,
        )
        trends_df = fetch_google_trends(ALL_KEYWORDS[:5])  # First 5 to avoid rate limit
        if not trends_df.empty:
            trends_zscore = compute_trends_zscore(trends_df)
            stress_index = compute_aggregate_stress_index(trends_zscore)
            print(f"  Google Trends data: {len(trends_df)} dates x {len(trends_df.columns)} keywords")
        else:
            print("  Google Trends returned empty data")
    except Exception as e:
        print(f"  Google Trends unavailable: {e}")

    # Step 2: Backtest
    if args.backtest:
        from src.strategies.geo_strategy import GeoStrategy
        from src.backtest.engine import run_backtest

        universe = MULTI_ASSET_TICKERS
        print(f"\n{'='*60}")
        print(f"Running walk-forward backtest (monthly rebalance, {len(universe)} ETFs)...")
        print(f"{'='*60}")

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

        # Run validation
        print(f"\n{'='*60}")
        print("Running 5-gate validation...")
        print(f"{'='*60}")

        report = run_validation(
            signal_name="Geospatial / Macro Signal",
            strategy_returns=result.returns,
        )
        print(report.summary())
    else:
        print("\nUse --backtest to run the walk-forward backtest")


if __name__ == "__main__":
    main()
