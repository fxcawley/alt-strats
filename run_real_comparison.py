"""
Honest comparison: real alternative data vs strat-testing baseline.

Only uses REAL data -- no synthetics, no proxies, no fallbacks.

Data sources actually tested:
  - CFTC COT positioning (free, weekly, downloaded from cftc.gov)
  - Price data (yfinance, for baseline strategies)

Data sources NOT available in this environment:
  - FRED (needs API key -- set FRED_API_KEY to enable)
  - Google Trends (needs direct network access)
  - EDGAR filings (needs direct network access to SEC)
  - VIIRS satellite (needs pre-processed CSV)

Usage:
    source .venv/Scripts/activate
    python run_real_comparison.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.backtest.engine import run_backtest, BacktestResult
from src.backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, alpha_beta
from src.analysis.alpha import t_test_alpha, bootstrap_alpha, information_coefficient
from src.data.universe import EQUITY_ETF_TICKERS, MULTI_ASSET_TICKERS
from src.validation.gates import run_validation


# ── Universes ────────────────────────────────────────────────────────────────
EQUITY_ETFS = [
    "SPY", "QQQ", "IWM", "IWD", "IWF", "EFA", "EEM", "EWJ",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU",
]
MULTI_ASSET = [
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "SHY", "LQD",
    "GLD",
]
FULL_UNIVERSE = sorted(set(EQUITY_ETFS + MULTI_ASSET))


# ── Baseline strategies (from strat-testing) ─────────────────────────────────

@dataclass
class CrossSectionalMomentum:
    lookback_days: int = 252
    skip_days: int = 21
    top_frac: float = 0.30
    long_only: bool = True

    def generate_signals(self, date, universe, lookback):
        scores = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback_days + self.skip_days + 5:
                continue
            close = df["Close"]
            n = len(close)
            end_pos = n - self.skip_days if self.skip_days > 0 else n
            start_pos = end_pos - self.lookback_days
            if start_pos < 0 or end_pos < 1:
                continue
            ret = close.iloc[end_pos - 1] / close.iloc[start_pos] - 1
            scores[ticker] = float(ret)
        if len(scores) < 3:
            return None
        sorted_tickers = sorted(scores, key=scores.get, reverse=True)
        n_long = max(1, int(len(sorted_tickers) * self.top_frac))
        longs = sorted_tickers[:n_long]
        return {t: 1.0 / n_long for t in longs}


@dataclass
class TrendFollowing:
    lookback_days: int = 252
    skip_days: int = 21
    vol_target: float | None = 0.10
    vol_lookback: int = 60
    long_only: bool = True

    def generate_signals(self, date, universe, lookback):
        raw_weights = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback_days + self.skip_days + 5:
                continue
            close = df["Close"]
            n = len(close)
            end_pos = n - self.skip_days if self.skip_days > 0 else n
            start_pos = end_pos - self.lookback_days
            if start_pos < 0 or end_pos < 1:
                continue
            ret = close.iloc[end_pos - 1] / close.iloc[start_pos] - 1
            if ret > 0:
                direction = 1.0
            elif not self.long_only:
                direction = -1.0
            else:
                continue
            if self.vol_target is not None:
                recent_vol = close.pct_change().iloc[-self.vol_lookback:].std() * np.sqrt(252)
                size = min(self.vol_target / recent_vol, 3.0) if recent_vol > 0.001 else 1.0
            else:
                size = 1.0
            raw_weights[ticker] = direction * size
        if not raw_weights:
            return None
        total = sum(abs(w) for w in raw_weights.values())
        if total > 1.0:
            raw_weights = {t: w / total for t, w in raw_weights.items()}
        return {t: w for t, w in raw_weights.items() if abs(w) > 1e-6}


class StaticBlend:
    def __init__(self, strategies):
        self.strategies = strategies

    def generate_signals(self, date, universe, lookback):
        blended = {}
        active_weight = 0.0
        for name, (strat, weight) in self.strategies.items():
            sig = strat.generate_signals(date, universe, lookback)
            if sig is None or not sig:
                continue
            active_weight += weight
            for tkr, tw in sig.items():
                blended[tkr] = blended.get(tkr, 0.0) + weight * tw
        if not blended:
            return None
        if 0 < active_weight < 0.99:
            blended = {t: w / active_weight for t, w in blended.items()}
        total = sum(abs(w) for w in blended.values())
        if total > 1.0:
            blended = {t: w / total for t, w in blended.items()}
        return {t: w for t, w in blended.items() if abs(w) > 1e-6}


# ── Run ──────────────────────────────────────────────────────────────────────

PERIODS = {
    "2016-2019": ("2016-06-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", None),
}


def run_one(name, strategy, universe, start, end, rebalance_freq="ME", cost_pct=0.0003):
    """Run a single backtest and return result + summary."""
    result = run_backtest(
        strategy=strategy,
        universe=universe,
        start=start,
        end=end,
        benchmark="SPY",
        rebalance_freq=rebalance_freq,
        initial_capital=100_000.0,
        cost_pct=cost_pct,
        lookback_buffer_days=500,
        rebalance_threshold=0.02,
    )
    return result


def main():
    print("=" * 70)
    print("  REAL DATA COMPARISON: COT flow signal vs strat-testing baseline")
    print("  (no synthetics, no proxies)")
    print("=" * 70)

    # ── Load real COT data ───────────────────────────────────────────────
    from src.data.cot import get_cot_data, compute_net_positioning, compute_positioning_zscore
    from src.strategies.flow_strategy import FlowStrategy
    from src.signals.flow_signal import compute_flow_signal

    print("\nLoading CFTC COT positioning data...")
    cot_df = get_cot_data(start_year=2015, end_year=2026, cache=True)
    positioning = compute_net_positioning(cot_df)
    zscore = compute_positioning_zscore(positioning, lookback_weeks=52)
    print(f"  {len(positioning)} weekly dates, {list(positioning.columns)}")

    # ── Run all strategies ───────────────────────────────────────────────
    rows = []

    for period_name, (start, end) in PERIODS.items():
        print(f"\n{'=' * 70}")
        print(f"  PERIOD: {period_name}")
        print(f"{'=' * 70}")

        # 1. XS Momentum + Trend Following (baseline from strat-testing)
        print("  Running XS Mom + Trend (baseline)...", end=" ", flush=True)
        baseline = run_one(
            "Baseline", StaticBlend(strategies={
                "xs_momentum": (CrossSectionalMomentum(), 0.5),
                "trend_following": (TrendFollowing(), 0.5),
            }),
            FULL_UNIVERSE, start, end,
        )
        s = baseline.summary()
        print(f"Sharpe={s['sharpe_ratio']:.2f}")

        # 2. XS Momentum standalone
        print("  Running XS Momentum...", end=" ", flush=True)
        xs_mom = run_one("XS Mom", CrossSectionalMomentum(), EQUITY_ETFS, start, end)
        s2 = xs_mom.summary()
        print(f"Sharpe={s2['sharpe_ratio']:.2f}")

        # 3. Real COT Flow Contrarian
        print("  Running Flow Contrarian (REAL COT)...", end=" ", flush=True)
        flow = run_one(
            "Flow", FlowStrategy(positioning_zscore=zscore, long_only=False, contrarian=True),
            MULTI_ASSET, start, end, rebalance_freq="W",
        )
        s3 = flow.summary()
        print(f"Sharpe={s3['sharpe_ratio']:.2f}")

        for name, result in [("XS Mom + Trend (baseline)", baseline),
                              ("XS Momentum", xs_mom),
                              ("Flow Contrarian (REAL COT)", flow)]:
            sm = result.summary()
            a, b = alpha_beta(result.returns, result.benchmark_returns)
            t = t_test_alpha(result.excess_returns)
            rows.append({
                "Strategy": name,
                "Period": period_name,
                "Return": f"{sm['total_return']:+.1%}",
                "CAGR": f"{sm['cagr']:.1%}",
                "Sharpe": f"{sm['sharpe_ratio']:.2f}",
                "Sortino": f"{sortino_ratio(result.returns):.2f}",
                "MaxDD": f"{sm['max_drawdown']:.1%}",
                "Alpha": f"{a:+.4f}",
                "Beta": f"{b:.2f}",
                "p-value": f"{t['p_value']:.3f}",
                "Costs": f"${sm['total_costs']:,.0f}",
                "Trades": sm["n_trades"],
            })

    # ── Comparison table ─────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  COMPARISON TABLE (real data only)")
    print(f"{'=' * 70}")
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    # ── Gate 1 & 2: IC and quintile analysis on real COT signal ──────────
    print(f"\n\n{'=' * 70}")
    print("  SIGNAL QUALITY: COT positioning vs next-week ETF returns")
    print(f"{'=' * 70}")

    # Build cross-sectional signal/return pairs from real COT data
    from src.data.prices import fetch_prices
    signals_list = []
    returns_list = []

    for ticker in positioning.columns:
        try:
            prices = fetch_prices(ticker, start="2016-01-01")
            weekly_returns = prices["Close"].resample("W").last().pct_change().shift(-1)

            for date in zscore.index:
                z = zscore.loc[date, ticker]
                if pd.isna(z):
                    continue
                # Find the next weekly return after this date
                future = weekly_returns.loc[date:]
                if len(future) < 2:
                    continue
                fwd_ret = future.iloc[1]
                if pd.isna(fwd_ret):
                    continue
                signals_list.append(-z)  # contrarian: negative zscore = bullish
                returns_list.append(fwd_ret)
        except Exception:
            continue

    if len(signals_list) > 50:
        sig_series = pd.Series(signals_list)
        ret_series = pd.Series(returns_list)

        from src.validation.gates import gate1_information_coefficient, gate2_quintile_monotonicity

        g1 = gate1_information_coefficient(sig_series, ret_series)
        g2 = gate2_quintile_monotonicity(sig_series, ret_series)
        print(f"\n  {g1}")
        print(f"  {g2}")
        if g2.details.get("quintile_returns"):
            print(f"\n  Quintile mean returns:")
            for q, r in g2.details["quintile_returns"].items():
                print(f"    Q{int(q)}: {r:+.4f}")
            print(f"    L-S spread: {g2.details.get('long_short_spread', 0):+.4f}")
    else:
        print(f"  Insufficient signal/return pairs ({len(signals_list)})")

    # ── Sharpe spread (Gate 4) ───────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  CROSS-PERIOD ROBUSTNESS")
    print(f"{'=' * 70}")

    strategies_by_name = {}
    for row in rows:
        name = row["Strategy"]
        if name not in strategies_by_name:
            strategies_by_name[name] = {}
        strategies_by_name[name][row["Period"]] = float(row["Sharpe"])

    for name, sharpes in strategies_by_name.items():
        vals = list(sharpes.values())
        spread = max(vals) - min(vals)
        avg = np.mean(vals)
        print(f"  {name:40s}  avg={avg:.2f}  spread={spread:.2f}  "
              f"{'PASS' if spread < 0.3 else 'FAIL'}")

    # ── 5-gate validation on Flow Contrarian ─────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  5-GATE VALIDATION: Flow Contrarian (REAL COT)")
    print(f"{'=' * 70}")

    flow_sharpes = strategies_by_name.get("Flow Contrarian (REAL COT)", {})

    report = run_validation(
        signal_name="Flow Contrarian (REAL COT)",
        predicted_scores=pd.Series(signals_list) if len(signals_list) > 50 else None,
        realized_returns=pd.Series(returns_list) if len(returns_list) > 50 else None,
        strategy_returns=flow.returns,
        sharpe_by_period=flow_sharpes if len(flow_sharpes) >= 2 else None,
    )
    print(f"\n{report.summary()}")

    print(f"\n\n{'=' * 70}")
    print("  NOTES")
    print(f"{'=' * 70}")
    print("  - COT data: real CFTC downloads, publication-date aligned (Tue+3bd)")
    print("  - VIX futures excluded from SPY mapping (separate signal)")
    print("  - Weekly rebalance, 3bps cost, 2% threshold")
    print("  - FRED/Google Trends/EDGAR/Satellite: need API keys or network access")
    print("  - Set FRED_API_KEY and re-run to include macro data")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
