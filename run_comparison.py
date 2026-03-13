"""
Run all alternative-data strategies and compare to strat-testing baseline.

This script:
  1. Runs the XS Momentum + Trend Following baseline (from strat-testing)
  2. Runs each alt-strats signal pipeline
  3. Produces a side-by-side comparison table
  4. Runs the 5-gate validation on each signal

Usage:
    source .venv/Scripts/activate
    python run_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.backtest.engine import run_backtest, BacktestResult, Strategy
from src.backtest.metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, alpha_beta
from src.analysis.alpha import t_test_alpha, bootstrap_alpha
from src.data.universe import EQUITY_ETF_TICKERS, MULTI_ASSET_TICKERS
from src.validation.gates import run_validation


# -- Universes ----------------------------------------------------------------
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


# -- Baseline strategies (replicated from strat-testing) ----------------------

@dataclass
class CrossSectionalMomentum:
    """12-1 momentum: rank by trailing return, go long top fraction."""
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

        weights = {}
        for t in longs:
            weights[t] = 1.0 / n_long
        return {t: w for t, w in weights.items() if abs(w) > 1e-6}


@dataclass
class TrendFollowing:
    """Time-series momentum: long assets with positive trailing return, vol-scaled."""
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
                if recent_vol > 0.001:
                    size = self.vol_target / recent_vol
                else:
                    size = 1.0
                size = min(size, 3.0)
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
    """Combine sub-strategies with fixed weights."""
    def __init__(self, strategies: dict[str, tuple[object, float]]):
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

        if active_weight > 0 and active_weight < 0.99:
            scale = 1.0 / active_weight
            blended = {t: w * scale for t, w in blended.items()}

        total = sum(abs(w) for w in blended.values())
        if total > 1.0:
            blended = {t: w / total for t, w in blended.items()}
        return {t: w for t, w in blended.items() if abs(w) > 1e-6}


# -- NLP Filing Strategy (using synthetic features for comparison) ------------

class SyntheticNLPStrategy:
    """NLP-like strategy using filing-date-aligned sentiment proxy.

    Since we can't download EDGAR filings in this environment (SSL proxy),
    we use a sentiment proxy: recent price momentum adjusted by volatility
    change (mimics the "deteriorating tone" signal). This is a placeholder
    that demonstrates the pipeline -- real performance would come from
    actual filing NLP features.

    In production, replace this with NLPFilingStrategy fed by real EDGAR data.
    """

    def __init__(self, sentiment_weight: float = 0.6, readability_weight: float = 0.4):
        self.sentiment_weight = sentiment_weight
        self.readability_weight = readability_weight

    def generate_signals(self, date, universe, lookback):
        scores = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < 260:
                continue
            close = df["Close"]
            n = len(close)

            # Proxy for "sentiment change": acceleration in returns
            # (3m return improvement vs 6m baseline)
            if n > 130:
                ret_3m = close.iloc[-1] / close.iloc[-63] - 1
                ret_6m = close.iloc[-1] / close.iloc[-126] - 1
                # "Improving sentiment" = recent returns accelerating
                sentiment_proxy = ret_3m - (ret_6m / 2)
            else:
                continue

            # Proxy for "readability/complexity": volatility regime change
            # (lower recent vol vs historical = "cleaner" signal)
            rets = close.pct_change().dropna()
            vol_1m = rets.iloc[-21:].std() * np.sqrt(252)
            vol_3m = rets.iloc[-63:].std() * np.sqrt(252)
            if vol_3m > 0:
                vol_change = -(vol_1m / vol_3m - 1)  # declining vol = positive
            else:
                vol_change = 0

            composite = (self.sentiment_weight * sentiment_proxy +
                         self.readability_weight * vol_change)
            scores[ticker] = composite

        if len(scores) < 3:
            return None

        # Go long top quintile
        ranked = sorted(scores, key=scores.get, reverse=True)
        n = len(ranked)
        n_long = max(1, n // 5)
        longs = ranked[:n_long]

        weights = {t: 1.0 / n_long for t in longs}
        return weights


# -- COT Flow Strategy (using synthetic positioning data) ---------------------

class SyntheticFlowStrategy:
    """Contrarian positioning strategy using volatility regime as proxy.

    Since we can't download CFTC COT data in this environment, we proxy
    "extreme positioning" with mean-reversion signals:
      - Extreme recent outperformance -> contrarian underweight
      - Extreme recent underperformance -> contrarian overweight

    This captures the same intuition as COT contrarian signals (crowded
    trades revert) using available price data. Real COT data would
    provide an information edge beyond this.
    """

    def __init__(self, lookback: int = 52 * 5, zscore_threshold: float = 1.5):
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold

    def generate_signals(self, date, universe, lookback):
        scores = {}
        for ticker in universe:
            df = lookback.get(ticker)
            if df is None or len(df) < self.lookback + 5:
                continue
            close = df["Close"]

            # Compute trailing 1-month return
            ret_1m = close.iloc[-1] / close.iloc[-21] - 1

            # Compute z-score vs trailing history
            monthly_rets = []
            for i in range(1, min(24, len(close) // 21)):
                r = close.iloc[-i * 21] / close.iloc[-(i + 1) * 21] - 1
                monthly_rets.append(r)

            if len(monthly_rets) < 6:
                continue

            mean_ret = np.mean(monthly_rets)
            std_ret = np.std(monthly_rets)
            if std_ret > 0:
                zscore = (ret_1m - mean_ret) / std_ret
            else:
                continue

            # Contrarian: flip the sign -- extreme gains suggest crowded long
            scores[ticker] = -zscore

        if len(scores) < 3:
            return None

        # Go long top third (most contrarian bullish), short bottom third
        ranked = sorted(scores, key=scores.get, reverse=True)
        n = len(ranked)
        n_long = max(1, n // 3)

        longs = ranked[:n_long]
        weights = {t: 1.0 / n_long for t in longs}
        return weights


# -- Geo/Macro Strategy (using macro proxies from price data) -----------------

class SyntheticGeoStrategy:
    """Macro rotation strategy using cross-asset signals.

    Proxies the satellite/Google Trends/FRED signal with observable
    cross-asset relationships:
      - Bond-equity correlation regime
      - Yield curve proxy (TLT vs SHY)
      - Cross-sectional dispersion (volatility regime)

    In production, real VIIRS luminosity + FRED surprises would
    provide additional information beyond price data.
    """

    def __init__(self):
        pass

    def generate_signals(self, date, universe, lookback):
        if "SPY" not in lookback or "TLT" not in lookback:
            return None
        if len(lookback.get("SPY", pd.DataFrame())) < 260:
            return None

        # Yield curve proxy: TLT vs SHY relative performance
        yield_signal = 0.0
        if "TLT" in lookback and "SHY" in lookback:
            tlt = lookback["TLT"]["Close"]
            shy = lookback["SHY"]["Close"]
            if len(tlt) > 63 and len(shy) > 63:
                tlt_ret = tlt.iloc[-1] / tlt.iloc[-63] - 1
                shy_ret = shy.iloc[-1] / shy.iloc[-63] - 1
                yield_signal = tlt_ret - shy_ret  # positive = flattening/risk-off

        # Market stress: SPY realized vol regime
        spy = lookback["SPY"]["Close"]
        spy_rets = spy.pct_change().dropna()
        vol_1m = spy_rets.iloc[-21:].std() * np.sqrt(252)
        vol_3m = spy_rets.iloc[-63:].std() * np.sqrt(252)
        stress_signal = vol_1m / vol_3m if vol_3m > 0 else 1.0

        # Dispersion: cross-sectional vol of 1m returns
        rets_1m = []
        for t in universe:
            if t in lookback and len(lookback[t]) > 22:
                c = lookback[t]["Close"]
                rets_1m.append(c.iloc[-1] / c.iloc[-22] - 1)

        dispersion = np.std(rets_1m) if len(rets_1m) > 3 else 0.0

        # Score each ETF: favor risk assets when stress is low, safe assets when high
        equity_tickers = set(EQUITY_ETFS)
        bond_tickers = {"TLT", "IEF", "SHY", "LQD"}

        weights = {}
        risk_appetite = 1.0 - stress_signal  # low vol ratio = risk-on
        risk_appetite += -yield_signal * 2    # flattening = risk-off

        # Clamp to [-1, 1]
        risk_appetite = max(-1.0, min(1.0, risk_appetite))

        eligible = [t for t in universe if t in lookback and len(lookback[t]) > 260]
        if not eligible:
            return None

        equity_eligible = [t for t in eligible if t in equity_tickers]
        bond_eligible = [t for t in eligible if t in bond_tickers]
        other = [t for t in eligible if t not in equity_tickers and t not in bond_tickers]

        # Allocate based on risk appetite
        equity_alloc = max(0.2, 0.5 + risk_appetite * 0.3)
        bond_alloc = max(0.1, 0.3 - risk_appetite * 0.2)
        other_alloc = 1.0 - equity_alloc - bond_alloc

        if equity_eligible:
            w = equity_alloc / len(equity_eligible)
            for t in equity_eligible:
                weights[t] = w
        if bond_eligible:
            w = bond_alloc / len(bond_eligible)
            for t in bond_eligible:
                weights[t] = w
        if other:
            w = other_alloc / len(other)
            for t in other:
                weights[t] = w

        total = sum(abs(v) for v in weights.values())
        if total > 1.0:
            weights = {t: w / total for t, w in weights.items()}

        return {t: w for t, w in weights.items() if abs(w) > 1e-6}


# -- Run all strategies -------------------------------------------------------

def run_strategy_periods(
    name: str,
    strategy_factory,
    universe: list[str],
    periods: dict[str, tuple[str, str | None]],
    rebalance_freq: str = "ME",
    cost_pct: float = 0.0003,
) -> dict[str, BacktestResult]:
    """Run a strategy across multiple periods."""
    results = {}
    for period_name, (start, end) in periods.items():
        print(f"  Running {name} -- {period_name}...", end=" ", flush=True)
        strat = strategy_factory()
        result = run_backtest(
            strategy=strat,
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
        s = result.summary()
        print(f"Sharpe={s['sharpe_ratio']:.2f}, Return={s['total_return']:+.1%}")
        results[period_name] = result
    return results


def print_comparison_table(all_results: dict[str, dict[str, BacktestResult]]):
    """Print a side-by-side comparison table."""
    rows = []
    for strategy_name, period_results in all_results.items():
        for period_name, result in period_results.items():
            s = result.summary()
            a, b = alpha_beta(result.returns, result.benchmark_returns)
            t_res = t_test_alpha(result.excess_returns)
            rows.append({
                "Strategy": strategy_name,
                "Period": period_name,
                "Return": f"{s['total_return']:+.1%}",
                "CAGR": f"{s['cagr']:.1%}",
                "Sharpe": f"{s['sharpe_ratio']:.2f}",
                "Sortino": f"{sortino_ratio(result.returns):.2f}",
                "MaxDD": f"{s['max_drawdown']:.1%}",
                "Alpha": f"{a:+.4f}",
                "Beta": f"{b:.2f}",
                "p-value": f"{t_res['p_value']:.3f}",
                "Costs": f"${s['total_costs']:,.0f}",
                "Trades": s["n_trades"],
            })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    return df


def plot_comparison(all_results: dict[str, dict[str, BacktestResult]]):
    """Plot equity curves for all strategies in each period."""
    output_dir = Path("data/results/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all periods
    all_periods = set()
    for period_results in all_results.values():
        all_periods.update(period_results.keys())

    for period_name in sorted(all_periods):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                        height_ratios=[3, 1], sharex=True)

        colors = ["steelblue", "darkorange", "forestgreen", "firebrick", "purple"]
        bench_plotted = False

        for idx, (strat_name, period_results) in enumerate(all_results.items()):
            if period_name not in period_results:
                continue
            result = period_results[period_name]
            color = colors[idx % len(colors)]

            # Normalize to 100 for comparison
            norm_eq = result.equity_curve / result.equity_curve.iloc[0] * 100
            ax1.plot(norm_eq.index, norm_eq.values,
                     label=strat_name, linewidth=2, color=color)

            if not bench_plotted:
                norm_bench = result.benchmark_curve / result.benchmark_curve.iloc[0] * 100
                ax1.plot(norm_bench.index, norm_bench.values,
                         label="SPY", linewidth=1.2, alpha=0.5,
                         linestyle="--", color="gray")
                bench_plotted = True

            # Drawdown
            eq = result.equity_curve
            dd = (eq - eq.cummax()) / eq.cummax()
            ax2.plot(dd.index, dd.values, color=color, alpha=0.6, linewidth=1)

        ax1.set_title(f"Strategy Comparison: {period_name}")
        ax1.set_ylabel("Normalized Value (100 = start)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("Drawdown")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        safe = period_name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(output_dir / f"comparison_{safe}.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / f'comparison_{safe}.png'}")


# -- Main ---------------------------------------------------------------------

PERIODS = {
    "2005-2013 (GFC)": ("2005-01-01", "2013-12-31"),
    "2014-2019": ("2014-01-01", "2019-12-31"),
    "2020-2026": ("2020-01-01", None),
}


def main():
    print("=" * 70)
    print("  STRATEGY COMPARISON: strat-testing baseline vs alt-strats signals")
    print("=" * 70)

    all_results: dict[str, dict[str, BacktestResult]] = {}

    # -- 1. Baseline: XS Momentum + Trend Following (50/50) --------------
    print(f"\n{'-' * 70}")
    print("BASELINE: XS Momentum + Trend Following (50/50)")
    print(f"{'-' * 70}")
    all_results["XS Mom + Trend (baseline)"] = run_strategy_periods(
        "XS Mom + Trend",
        lambda: StaticBlend(strategies={
            "xs_momentum": (CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30), 0.5),
            "trend_following": (TrendFollowing(lookback_days=252, skip_days=21, vol_target=0.10), 0.5),
        }),
        FULL_UNIVERSE,
        PERIODS,
    )

    # -- 2. XS Momentum standalone (the bar from HANDOFF.md) -------------
    print(f"\n{'-' * 70}")
    print("BASELINE: XS Momentum (standalone)")
    print(f"{'-' * 70}")
    all_results["XS Momentum"] = run_strategy_periods(
        "XS Momentum",
        lambda: CrossSectionalMomentum(lookback_days=252, skip_days=21, top_frac=0.30),
        EQUITY_ETFS,
        PERIODS,
    )

    # -- 3. NLP Filing Signal (synthetic proxy) ---------------------------
    print(f"\n{'-' * 70}")
    print("ALT-STRAT: NLP Filing Signal (sentiment proxy)")
    print(f"{'-' * 70}")
    all_results["NLP Filing Signal"] = run_strategy_periods(
        "NLP Filing",
        lambda: SyntheticNLPStrategy(sentiment_weight=0.6, readability_weight=0.4),
        EQUITY_ETFS,
        PERIODS,
    )

    # -- 4. Order Flow / Contrarian Positioning ---------------------------
    print(f"\n{'-' * 70}")
    print("ALT-STRAT: Flow Contrarian (positioning proxy)")
    print(f"{'-' * 70}")
    all_results["Flow Contrarian"] = run_strategy_periods(
        "Flow Contrarian",
        lambda: SyntheticFlowStrategy(zscore_threshold=1.5),
        MULTI_ASSET,
        PERIODS,
    )

    # -- 5. Geo/Macro Rotation --------------------------------------------
    print(f"\n{'-' * 70}")
    print("ALT-STRAT: Geo/Macro Rotation")
    print(f"{'-' * 70}")
    all_results["Geo/Macro Rotation"] = run_strategy_periods(
        "Geo/Macro",
        lambda: SyntheticGeoStrategy(),
        FULL_UNIVERSE,
        PERIODS,
    )

    # -- 6. Blend: NLP + Flow + Geo (equal weight) ------------------------
    print(f"\n{'-' * 70}")
    print("ALT-STRAT: Combined Alt-Data Blend (NLP + Flow + Geo)")
    print(f"{'-' * 70}")
    all_results["Alt-Data Blend"] = run_strategy_periods(
        "Alt-Data Blend",
        lambda: StaticBlend(strategies={
            "nlp": (SyntheticNLPStrategy(), 0.34),
            "flow": (SyntheticFlowStrategy(), 0.33),
            "geo": (SyntheticGeoStrategy(), 0.33),
        }),
        FULL_UNIVERSE,
        PERIODS,
    )

    # -- Comparison -------------------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("  FULL COMPARISON TABLE")
    print(f"{'=' * 70}")
    comp_df = print_comparison_table(all_results)

    # -- Sharpe spread analysis -------------------------------------------
    print(f"\n\n{'=' * 70}")
    print("  SHARPE SPREAD ACROSS PERIODS (< 0.3 = robust)")
    print(f"{'=' * 70}")
    for strat_name, period_results in all_results.items():
        sharpes = []
        for period_name, result in period_results.items():
            s = result.summary()
            sharpes.append(s["sharpe_ratio"])
        if len(sharpes) >= 2:
            spread = max(sharpes) - min(sharpes)
            avg = np.mean(sharpes)
            print(f"  {strat_name:35s}  avg={avg:.2f}  spread={spread:.2f}  "
                  f"{'PASS' if spread < 0.3 else 'FAIL'}")

    # -- Validation gates on most recent period ---------------------------
    print(f"\n\n{'=' * 70}")
    print("  5-GATE VALIDATION (2020-2026 period)")
    print(f"{'=' * 70}")
    target_period = "2020-2026"
    for strat_name, period_results in all_results.items():
        if target_period in period_results:
            result = period_results[target_period]
            sharpe_dict = {}
            for pn, pr in period_results.items():
                sharpe_dict[pn] = pr.summary()["sharpe_ratio"]

            report = run_validation(
                signal_name=strat_name,
                strategy_returns=result.returns,
                sharpe_by_period=sharpe_dict,
            )
            print(f"\n{report.summary()}")

    # -- Plot -------------------------------------------------------------
    plot_comparison(all_results)

    print(f"\n\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
