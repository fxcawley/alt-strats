"""
Core backtesting engine.

Design principles:
  - Strategies emit target weights per ticker on each rebalance day.
  - The engine converts weights to share counts at rebalance, then lets
    positions drift with prices until the next rebalance.  No implicit
    daily rebalancing.
  - Transaction costs (commissions + slippage) are applied on each trade.
  - Benchmark comparison (e.g. SPY buy-and-hold) is built in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.prices import fetch_prices, fetch_benchmark


# ---------------------------------------------------------------------------
# Signal / Strategy interface
# ---------------------------------------------------------------------------

class Strategy(Protocol):
    """Any object that can produce portfolio weights on a given date."""

    def generate_signals(
        self, date: pd.Timestamp, universe: list[str], lookback: dict[str, pd.DataFrame]
    ) -> dict[str, float] | None:
        """Return target portfolio weights keyed by ticker.

        Weights should sum to <= 1.0 (remainder is cash).
        Negative weights represent short positions.
        Return None to skip this rebalance (keep existing positions).
        """
        ...


# ---------------------------------------------------------------------------
# Portfolio bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class PortfolioSnapshot:
    date: pd.Timestamp
    weights: dict[str, float]
    equity: float
    cash_weight: float


@dataclass
class BacktestResult:
    """Container for everything a backtest produces."""
    equity_curve: pd.Series  # indexed by date
    benchmark_curve: pd.Series
    trades: list[dict]
    snapshots: list[PortfolioSnapshot]
    metadata: dict = field(default_factory=dict)

    # --- Derived metrics (computed lazily) ----------------------------------

    @property
    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.benchmark_curve.pct_change().dropna()

    @property
    def excess_returns(self) -> pd.Series:
        r = self.returns
        b = self.benchmark_returns.reindex(r.index)
        return (r - b).dropna()

    def summary(self) -> dict:
        """Compute standard performance metrics."""
        r = self.returns
        er = self.excess_returns
        n_years = len(r) / 252

        total_ret = self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
        bench_ret = self.benchmark_curve.iloc[-1] / self.benchmark_curve.iloc[0] - 1
        cagr = (1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1

        ann_vol = r.std() * np.sqrt(252)
        sharpe = (r.mean() * 252) / ann_vol if ann_vol > 0 else 0.0

        rolling_max = self.equity_curve.cummax()
        drawdowns = (self.equity_curve - rolling_max) / rolling_max
        max_dd = drawdowns.min()

        ann_alpha = er.mean() * 252
        ir = ann_alpha / (er.std() * np.sqrt(252)) if er.std() > 0 else 0.0

        return {
            "total_return": total_ret,
            "benchmark_return": bench_ret,
            "cagr": cagr,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "annualized_alpha": ann_alpha,
            "information_ratio": ir,
            "n_trades": len(self.trades),
            "n_days": len(r),
            "total_costs": self.metadata.get("total_costs", 0.0),
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def _get_close(price_data: dict[str, pd.DataFrame], ticker: str, date: pd.Timestamp) -> float | None:
    """Get the closing price for a ticker on a date, or None if not available."""
    df = price_data.get(ticker)
    if df is None or date not in df.index:
        return None
    return float(df.loc[date, "Close"])


def run_backtest(
    strategy: Strategy,
    universe: list[str],
    start: str = "2018-01-01",
    end: str | None = None,
    benchmark: str = "SPY",
    rebalance_freq: str = "ME",  # ME = month-end, W = weekly, D = daily
    initial_capital: float = 100_000.0,
    lookback_buffer_days: int = 252,
    cost_per_share: float = 0.0,
    cost_pct: float = 0.001,  # 10bps round-trip (slippage + spread)
    rebalance_threshold: float = 0.0,  # min weight change to trigger a trade
) -> BacktestResult:
    """Run a walk-forward backtest with share-count-based position tracking.

    Positions drift with market prices between rebalances.  Costs are
    applied as a percentage of the dollar value of each trade.

    Parameters
    ----------
    strategy : Strategy
        An object implementing ``generate_signals``.
    universe : list[str]
        Tickers the strategy can trade.
    start, end : str
        Date range for the equity curve.  Extra lookback is fetched
        automatically so strategies have history on day one.
    benchmark : str
        Ticker used as the passive benchmark.
    rebalance_freq : str
        Pandas offset alias for rebalance cadence.
    initial_capital : float
        Starting cash.
    lookback_buffer_days : int
        Extra calendar days of price history to fetch before *start*.
    cost_per_share : float
        Fixed cost per share traded (e.g. commission).
    cost_pct : float
        Proportional cost on the dollar value of each trade
        (models slippage + half spread).  Applied to the absolute
        dollar amount traded.
    rebalance_threshold : float
        Minimum absolute weight change to trigger a trade for a given
        ticker.  If the difference between current and target weight
        is below this threshold, the existing position is kept.
        Set to 0.02 (2%) to avoid unnecessary churn.
    """
    # 1. Fetch all price data up front -- every ticker must succeed
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    fetch_start = (start_dt - timedelta(days=lookback_buffer_days)).strftime("%Y-%m-%d")

    price_data: dict[str, pd.DataFrame] = {}
    for t in set(universe + [benchmark]):
        price_data[t] = fetch_prices(t, start=fetch_start, end=end)

    bench_close = price_data[benchmark]["Close"]

    # 2. Build rebalance schedule -- only trade from `start` onward
    all_dates = bench_close.loc[start:].index.sort_values()
    rebal_dates = all_dates.to_series().groupby(pd.Grouper(freq=rebalance_freq)).last().dropna().values
    rebal_set = set(pd.DatetimeIndex(rebal_dates))

    # 3. Walk forward with share-count tracking
    cash = initial_capital
    holdings: dict[str, float] = {}  # ticker -> number of shares (fractional)
    last_known_prices: dict[str, float] = {}  # for carry-forward on non-trading days
    total_costs = 0.0

    equity_series: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []
    snapshots: list[PortfolioSnapshot] = []

    for date in all_dates:
        # --- Mark to market: compute portfolio value from holdings + cash ---
        portfolio_value = cash
        for tkr, shares in holdings.items():
            price = _get_close(price_data, tkr, date)
            if price is not None:
                last_known_prices[tkr] = price
            else:
                price = last_known_prices.get(tkr)
            if price is not None:
                portfolio_value += shares * price

        equity_series[date] = portfolio_value

        # --- Rebalance ---
        if date in rebal_set:
            # Include benchmark in lookback so strategies can compute market-level signals
            lookback = {t: df.loc[:date] for t, df in price_data.items()
                        if t in universe or t == benchmark}
            target_weights = strategy.generate_signals(date, universe, lookback)

            # None means "skip this rebalance, keep existing positions"
            # But if we hold nothing, there's nothing to keep -- stay in cash.
            if target_weights is None and holdings:
                continue
            if target_weights is None:
                target_weights = {}  # no positions to keep, treat as empty

            # Convert target weights to target share counts
            # First, get current prices for everything we might trade
            current_prices: dict[str, float] = {}
            for tkr in set(list(target_weights.keys()) + list(holdings.keys())):
                p = _get_close(price_data, tkr, date)
                if p is not None and p > 0:
                    current_prices[tkr] = p

            # Compute target dollar allocations
            target_dollars: dict[str, float] = {}
            for tkr, w in target_weights.items():
                if tkr in current_prices:
                    target_dollars[tkr] = w * portfolio_value

            # Compute target shares
            target_shares: dict[str, float] = {}
            for tkr, dollars in target_dollars.items():
                target_shares[tkr] = dollars / current_prices[tkr]

            # Apply rebalance threshold: if the weight change for a ticker
            # is below the threshold, keep the existing shares instead.
            if rebalance_threshold > 0 and portfolio_value > 0:
                for tkr in set(list(target_shares.keys()) + list(holdings.keys())):
                    old_shares = holdings.get(tkr, 0.0)
                    new_shares = target_shares.get(tkr, 0.0)
                    price = current_prices.get(tkr)
                    if price is None:
                        continue
                    old_weight = old_shares * price / portfolio_value
                    new_weight = new_shares * price / portfolio_value
                    if abs(new_weight - old_weight) < rebalance_threshold:
                        # Keep existing position -- below threshold
                        if old_shares > 0:
                            target_shares[tkr] = old_shares
                        elif tkr in target_shares:
                            del target_shares[tkr]

            # Compute trades (delta in shares)
            rebalance_cost = 0.0
            for tkr in set(list(target_shares.keys()) + list(holdings.keys())):
                old_shares = holdings.get(tkr, 0.0)
                new_shares = target_shares.get(tkr, 0.0)
                delta_shares = new_shares - old_shares

                if abs(delta_shares) < 1e-9:
                    continue

                price = current_prices.get(tkr)
                if price is None:
                    raise RuntimeError(
                        f"Cannot trade {tkr} on {date}: no price available"
                    )

                dollar_traded = abs(delta_shares) * price
                trade_cost = abs(delta_shares) * cost_per_share + dollar_traded * cost_pct
                rebalance_cost += trade_cost

                old_weight = (old_shares * price / portfolio_value) if portfolio_value > 0 else 0.0
                new_weight = (new_shares * price / portfolio_value) if portfolio_value > 0 else 0.0

                trades.append({
                    "date": date,
                    "ticker": tkr,
                    "old_shares": old_shares,
                    "new_shares": new_shares,
                    "delta_shares": delta_shares,
                    "price": price,
                    "dollar_traded": dollar_traded,
                    "cost": trade_cost,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                })

            # Apply costs to cash
            cash -= rebalance_cost
            total_costs += rebalance_cost

            # Update holdings
            # Cash = portfolio_value - value of new positions - costs
            new_position_value = sum(
                target_shares.get(tkr, 0.0) * current_prices.get(tkr, 0.0)
                for tkr in target_shares
            )
            cash = portfolio_value - new_position_value - rebalance_cost
            holdings = {tkr: shares for tkr, shares in target_shares.items() if abs(shares) > 1e-9}

            # Record snapshot with actual weights at rebalance
            actual_weights = {}
            for tkr, shares in holdings.items():
                p = current_prices.get(tkr, 0.0)
                actual_weights[tkr] = (shares * p / portfolio_value) if portfolio_value > 0 else 0.0
            cash_w = cash / portfolio_value if portfolio_value > 0 else 1.0

            snapshots.append(PortfolioSnapshot(
                date=date,
                weights=actual_weights,
                equity=portfolio_value,
                cash_weight=cash_w,
            ))

    # 4. Build result
    eq_curve = pd.Series(equity_series, name="equity").sort_index()
    bench_curve = (
        bench_close.reindex(eq_curve.index).ffill()
        / bench_close.reindex(eq_curve.index).ffill().iloc[0]
        * initial_capital
    )
    bench_curve.name = "benchmark"

    return BacktestResult(
        equity_curve=eq_curve,
        benchmark_curve=bench_curve,
        trades=trades,
        snapshots=snapshots,
        metadata={
            "strategy": type(strategy).__name__,
            "universe_size": len(universe),
            "benchmark": benchmark,
            "rebalance_freq": rebalance_freq,
            "total_costs": total_costs,
            "cost_pct": cost_pct,
            "cost_per_share": cost_per_share,
        },
    )
