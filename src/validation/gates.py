"""
5-gate validation framework for alternative data signals.

Every signal must pass these gates before it's considered tradeable:

  Gate 1: Is the feature predictive in isolation? (IC > 0.02)
  Gate 2: Is the signal monotonic in quintiles?
  Gate 3: Does it survive walk-forward backtesting? (Sharpe > 0)
  Gate 4: Is it robust across time periods? (Sharpe spread < 0.3)
  Gate 5: Does it add value above existing signals? (incremental R² > 0.5%)

The baseline to beat is XS Momentum at 0.69-0.74 Sharpe across three periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.alpha import information_coefficient, t_test_alpha
from src.backtest.metrics import sharpe_ratio


@dataclass
class GateResult:
    """Result of a single validation gate."""
    gate_name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.gate_name}: "
            f"{self.metric_name}={self.metric_value:.4f} "
            f"(threshold: {self.threshold})"
        )


@dataclass
class ValidationReport:
    """Full validation report across all gates."""
    signal_name: str
    gates: list[GateResult]

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates)

    @property
    def n_passed(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    def summary(self) -> str:
        lines = [f"Validation Report: {self.signal_name}"]
        lines.append(f"Gates passed: {self.n_passed}/{len(self.gates)}")
        lines.append("-" * 60)
        for g in self.gates:
            lines.append(str(g))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate 1: Information Coefficient
# ---------------------------------------------------------------------------

def gate1_information_coefficient(
    predicted_scores: pd.Series,
    realized_returns: pd.Series,
    threshold: float = 0.02,
) -> GateResult:
    """Gate 1: Is the feature predictive in isolation?

    Computes rank IC (Spearman correlation) between predicted signal
    and realized returns. IC > 0.02 is meaningful for monthly returns.
    IC > 0.05 is strong.
    """
    try:
        ic = information_coefficient(predicted_scores, realized_returns)
    except ValueError as e:
        return GateResult(
            gate_name="Gate 1: Information Coefficient",
            passed=False,
            metric_name="IC",
            metric_value=0.0,
            threshold=threshold,
            details={"error": str(e)},
        )

    return GateResult(
        gate_name="Gate 1: Information Coefficient",
        passed=abs(ic) > threshold,
        metric_name="IC",
        metric_value=ic,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Gate 2: Quintile Monotonicity
# ---------------------------------------------------------------------------

def gate2_quintile_monotonicity(
    signal: pd.Series,
    returns: pd.Series,
    n_quintiles: int = 5,
) -> GateResult:
    """Gate 2: Is the signal monotonic in quintiles?

    Sorts the universe into quintiles by signal, computes average
    next-period return per quintile. The relationship should be
    monotonic (Q1 < Q2 < Q3 < Q4 < Q5 for a long signal).
    """
    aligned = pd.DataFrame({"signal": signal, "returns": returns}).dropna()
    if len(aligned) < n_quintiles * 5:
        return GateResult(
            gate_name="Gate 2: Quintile Monotonicity",
            passed=False,
            metric_name="monotonicity_score",
            metric_value=0.0,
            threshold=0.8,
            details={"error": f"Insufficient data: {len(aligned)} observations"},
        )

    # Assign quintiles
    aligned["quintile"] = pd.qcut(
        aligned["signal"], n_quintiles, labels=False, duplicates="drop"
    )

    # Average return per quintile
    quintile_returns = aligned.groupby("quintile")["returns"].mean()

    # Check monotonicity: compute rank correlation of quintile labels vs returns
    if len(quintile_returns) < 3:
        return GateResult(
            gate_name="Gate 2: Quintile Monotonicity",
            passed=False,
            metric_name="monotonicity_score",
            metric_value=0.0,
            threshold=0.8,
            details={"error": "Too few quintiles with data"},
        )

    # Monotonicity score: Spearman correlation of quintile rank vs quintile mean return
    mono_corr, _ = stats.spearmanr(
        quintile_returns.index.astype(float),
        quintile_returns.values,
    )

    # Long-short spread (top quintile - bottom quintile)
    ls_spread = quintile_returns.iloc[-1] - quintile_returns.iloc[0]

    return GateResult(
        gate_name="Gate 2: Quintile Monotonicity",
        passed=abs(mono_corr) >= 0.8,
        metric_name="monotonicity_score",
        metric_value=float(mono_corr),
        threshold=0.8,
        details={
            "quintile_returns": quintile_returns.to_dict(),
            "long_short_spread": float(ls_spread),
        },
    )


# ---------------------------------------------------------------------------
# Gate 3: Walk-Forward Backtest
# ---------------------------------------------------------------------------

def gate3_walk_forward(
    strategy_returns: pd.Series,
    min_sharpe: float = 0.0,
) -> GateResult:
    """Gate 3: Does it survive walk-forward backtesting?

    Measures Sharpe ratio from a walk-forward backtest.
    The bar is Sharpe > 0 (doesn't destroy value), with the
    aspiration being Sharpe > 0.5 to beat XS Momentum.
    """
    sr = sharpe_ratio(strategy_returns)

    return GateResult(
        gate_name="Gate 3: Walk-Forward Backtest",
        passed=sr > min_sharpe,
        metric_name="sharpe_ratio",
        metric_value=sr,
        threshold=min_sharpe,
        details={
            "annualized_return": float(strategy_returns.mean() * 252),
            "annualized_vol": float(strategy_returns.std() * np.sqrt(252)),
            "n_days": len(strategy_returns),
        },
    )


# ---------------------------------------------------------------------------
# Gate 4: Cross-Period Robustness
# ---------------------------------------------------------------------------

def gate4_cross_period_robustness(
    sharpe_by_period: dict[str, float],
    max_spread: float = 0.3,
) -> GateResult:
    """Gate 4: Is it robust across time periods?

    Checks that the Sharpe spread across non-overlapping periods
    is < 0.3. If the signal only works in one period, it's overfit.
    """
    if len(sharpe_by_period) < 2:
        return GateResult(
            gate_name="Gate 4: Cross-Period Robustness",
            passed=False,
            metric_name="sharpe_spread",
            metric_value=float("inf"),
            threshold=max_spread,
            details={"error": "Need at least 2 periods"},
        )

    sharpes = list(sharpe_by_period.values())
    spread = max(sharpes) - min(sharpes)

    return GateResult(
        gate_name="Gate 4: Cross-Period Robustness",
        passed=spread < max_spread,
        metric_name="sharpe_spread",
        metric_value=spread,
        threshold=max_spread,
        details={"sharpe_by_period": sharpe_by_period},
    )


# ---------------------------------------------------------------------------
# Gate 5: Incremental Value Above Existing Signals
# ---------------------------------------------------------------------------

def gate5_incremental_value(
    baseline_r_squared: float,
    combined_r_squared: float,
    min_increment: float = 0.005,
) -> GateResult:
    """Gate 5: Does it add value above existing signals?

    Checks that the incremental R-squared from adding the alternative
    signal to an existing model (momentum + vol) is > 0.5%.

    Parameters
    ----------
    baseline_r_squared : float
        Out-of-sample R² from the baseline model (momentum + vol only).
    combined_r_squared : float
        Out-of-sample R² from the model with alternative features added.
    min_increment : float
        Minimum R² improvement to pass (default 0.5% = 0.005).
    """
    increment = combined_r_squared - baseline_r_squared

    return GateResult(
        gate_name="Gate 5: Incremental R-squared",
        passed=increment > min_increment,
        metric_name="incremental_r_squared",
        metric_value=increment,
        threshold=min_increment,
        details={
            "baseline_r_squared": baseline_r_squared,
            "combined_r_squared": combined_r_squared,
        },
    )


# ---------------------------------------------------------------------------
# Run all gates
# ---------------------------------------------------------------------------

def run_validation(
    signal_name: str,
    predicted_scores: pd.Series | None = None,
    realized_returns: pd.Series | None = None,
    strategy_returns: pd.Series | None = None,
    sharpe_by_period: dict[str, float] | None = None,
    baseline_r_squared: float | None = None,
    combined_r_squared: float | None = None,
) -> ValidationReport:
    """Run all available validation gates for a signal.

    Not all gates need data -- gates without data are skipped.
    This allows incremental validation as the signal is developed.
    """
    gates = []

    # Gate 1
    if predicted_scores is not None and realized_returns is not None:
        gates.append(gate1_information_coefficient(predicted_scores, realized_returns))

    # Gate 2
    if predicted_scores is not None and realized_returns is not None:
        gates.append(gate2_quintile_monotonicity(predicted_scores, realized_returns))

    # Gate 3
    if strategy_returns is not None:
        gates.append(gate3_walk_forward(strategy_returns))

    # Gate 4
    if sharpe_by_period is not None:
        gates.append(gate4_cross_period_robustness(sharpe_by_period))

    # Gate 5
    if baseline_r_squared is not None and combined_r_squared is not None:
        gates.append(gate5_incremental_value(baseline_r_squared, combined_r_squared))

    return ValidationReport(signal_name=signal_name, gates=gates)
