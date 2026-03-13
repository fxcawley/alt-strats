"""
Runner script for the order flow (COT positioning) signal pipeline.

Usage:
    python run_flow_signal.py [--start-year 2010] [--backtest]

This script:
  1. Downloads CFTC COT reports
  2. Computes net speculator positioning and z-scores
  3. Optionally runs a walk-forward backtest
  4. Runs the 5-gate validation framework
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.data.cot import get_cot_data, compute_net_positioning, compute_positioning_zscore
from src.data.universe import EQUITY_ETF_TICKERS, MULTI_ASSET_TICKERS
from src.signals.flow_signal import FlowSignalComputer
from src.validation.gates import run_validation


def main():
    parser = argparse.ArgumentParser(description="Order Flow Signal Pipeline")
    parser.add_argument("--start-year", type=int, default=2010,
                        help="Start year for COT data")
    parser.add_argument("--end-year", type=int, default=None,
                        help="End year for COT data")
    parser.add_argument("--backtest", action="store_true",
                        help="Run walk-forward backtest")
    parser.add_argument("--long-only", action="store_true",
                        help="Long-only (no shorting)")
    args = parser.parse_args()

    print("Order Flow Signal Pipeline")
    print(f"COT data: {args.start_year} to {args.end_year or 'current'}")
    print()

    # Step 1: Download COT data
    print("Downloading CFTC COT reports...")
    cot_df = get_cot_data(start_year=args.start_year, end_year=args.end_year)
    print(f"  Loaded {len(cot_df)} report rows")

    # Step 2: Compute net positioning
    print("Computing net speculator positioning...")
    positioning = compute_net_positioning(cot_df)
    print(f"  Positioning data: {len(positioning)} dates x {len(positioning.columns)} contracts")
    print(f"  Tickers covered: {list(positioning.columns)}")

    # Step 3: Compute z-scores
    print("Computing positioning z-scores (52-week lookback)...")
    zscore = compute_positioning_zscore(positioning, lookback_weeks=52)
    print(f"  Z-score data: {len(zscore)} dates")

    # Show latest z-scores
    latest = zscore.dropna(how="all").iloc[-1] if not zscore.dropna(how="all").empty else pd.Series()
    if not latest.empty:
        print("\nLatest positioning z-scores:")
        for ticker, z in latest.dropna().items():
            direction = "EXTREME LONG" if z > 2 else "EXTREME SHORT" if z < -2 else "normal"
            print(f"  {ticker}: {z:+.2f} ({direction})")

    # Step 4: Backtest
    if args.backtest:
        from src.strategies.flow_strategy import FlowStrategy
        from src.backtest.engine import run_backtest

        universe = MULTI_ASSET_TICKERS
        print(f"\n{'='*60}")
        print(f"Running walk-forward backtest (weekly rebalance, {len(universe)} ETFs)...")
        print(f"{'='*60}")

        strategy = FlowStrategy(
            positioning_zscore=zscore,
            long_only=args.long_only,
            contrarian=True,
        )

        # Use the earliest date with z-score data
        start_date = zscore.dropna(how="all").index[0].strftime("%Y-%m-%d")

        result = run_backtest(
            strategy=strategy,
            universe=universe,
            start=start_date,
            rebalance_freq="W",  # Weekly rebalance for flow signals
            cost_pct=0.0003,     # 3bps for ETFs
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
            signal_name="Order Flow (COT Contrarian)",
            strategy_returns=result.returns,
        )
        print(report.summary())


if __name__ == "__main__":
    main()
