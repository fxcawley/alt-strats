"""
Runner script for the NLP filing signal pipeline.

Usage:
    python run_nlp_signal.py [--ticker AAPL] [--start 2015-01-01] [--end 2025-01-01]
                             [--use-embeddings] [--backtest]

This script:
  1. Downloads filings from EDGAR for specified tickers
  2. Parses and extracts NLP features (sentiment, readability)
  3. Optionally runs a walk-forward backtest
  4. Runs the 5-gate validation framework on the results
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import pandas as pd

from src.data.edgar import download_all_filings, load_filing
from src.nlp.filing_parser import parse_filing
from src.nlp.sentiment import compute_sentiment
from src.signals.filing_signal import (
    extract_filing_features,
    build_filing_signal_df,
    compute_composite_score,
)
from src.validation.gates import run_validation


# Default tickers: S&P 500 sector ETF components (large caps with frequent filings)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",  # Tech
    "JPM", "BAC", "GS",                          # Financials
    "JNJ", "UNH", "PFE",                          # Healthcare
    "XOM", "CVX",                                  # Energy
    "PG", "KO", "WMT",                            # Consumer
]


def run_feature_extraction(
    tickers: list[str],
    start_date: str = "2015-01-01",
    end_date: str | None = None,
    use_embeddings: bool = False,
) -> list:
    """Download filings and extract NLP features for a list of tickers."""
    from src.signals.filing_signal import FilingFeatures

    all_features: list[FilingFeatures] = []

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Processing {ticker}...")
        print(f"{'='*60}")

        # Step 1: Download filings
        try:
            metadata = download_all_filings(
                ticker,
                filing_types=("10-K", "10-Q"),
                start_date=start_date,
                end_date=end_date,
            )
            print(f"  Found {len(metadata)} filings")
        except Exception as e:
            print(f"  ERROR downloading filings: {e}")
            continue

        # Step 2: Parse and extract features
        prior_parsed = None
        for _, row in metadata.iterrows():
            try:
                raw_content = load_filing(ticker, row["filing_date"])
                parsed = parse_filing(
                    raw_content,
                    filing_type=row["filing_type"],
                    filing_date=row["filing_date"],
                    ticker=ticker,
                )
                print(f"  {row['filing_date']} ({row['filing_type']}): "
                      f"MD&A={len(parsed.mda)} chars, "
                      f"Risk={len(parsed.risk_factors)} chars")

                features = extract_filing_features(
                    parsed,
                    prior_parsed=prior_parsed,
                    use_embeddings=use_embeddings,
                )
                features.composite_score = compute_composite_score(features)
                all_features.append(features)

                prior_parsed = parsed

            except Exception as e:
                print(f"  ERROR parsing {row['filing_date']}: {e}")
                continue

    return all_features


def run_backtest_nlp(
    features: list,
    universe: list[str],
    start: str = "2015-01-01",
    end: str | None = None,
):
    """Run a walk-forward backtest using NLP filing signals."""
    from src.strategies.nlp_strategy import NLPFilingStrategy
    from src.backtest.engine import run_backtest

    strategy = NLPFilingStrategy(
        filing_features=features,
        long_only=True,
        stale_days=120,
    )

    result = run_backtest(
        strategy=strategy,
        universe=universe,
        start=start,
        end=end,
        rebalance_freq="ME",
        cost_pct=0.001,
        rebalance_threshold=0.02,
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="NLP Filing Signal Pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="Tickers to process")
    parser.add_argument("--start", default="2015-01-01", help="Start date")
    parser.add_argument("--end", default=None, help="End date")
    parser.add_argument("--use-embeddings", action="store_true",
                        help="Compute embedding similarity (slower)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run walk-forward backtest after feature extraction")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloading, use cached filings only")
    args = parser.parse_args()

    print("NLP Filing Signal Pipeline")
    print(f"Tickers: {args.tickers}")
    print(f"Period: {args.start} to {args.end or 'now'}")
    print(f"Embeddings: {args.use_embeddings}")
    print()

    # Extract features
    features = run_feature_extraction(
        args.tickers,
        start_date=args.start,
        end_date=args.end,
        use_embeddings=args.use_embeddings,
    )

    print(f"\nExtracted features from {len(features)} filings")

    if not features:
        print("No features extracted. Check EDGAR access and filing availability.")
        sys.exit(1)

    # Build signal DataFrame
    signal_df = build_filing_signal_df(features)
    print(f"\nSignal DataFrame: {len(signal_df)} tickers")
    print(signal_df[["signal", "filing_type", "filing_date"]].to_string())

    # Run backtest if requested
    if args.backtest:
        print(f"\n{'='*60}")
        print("Running walk-forward backtest...")
        print(f"{'='*60}")

        result = run_backtest_nlp(features, args.tickers, args.start, args.end)
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
            signal_name="NLP Filing Signal",
            strategy_returns=result.returns,
        )
        print(report.summary())


if __name__ == "__main__":
    main()
