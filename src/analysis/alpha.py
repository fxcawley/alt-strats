"""
Alpha assessment toolkit.

Tools for determining whether a strategy's excess return is statistically
meaningful or just noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def t_test_alpha(excess_returns: pd.Series) -> dict:
    """One-sample t-test: is mean excess return significantly different from 0?"""
    t_stat, p_value = stats.ttest_1samp(excess_returns.dropna(), 0)
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "mean_daily_alpha": float(excess_returns.mean()),
        "annualized_alpha": float(excess_returns.mean() * 252),
        "n_observations": len(excess_returns.dropna()),
        "significant_5pct": p_value < 0.05,
    }


def bootstrap_alpha(
    excess_returns: pd.Series,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap confidence interval for annualized alpha."""
    rng = np.random.default_rng(42)
    er = excess_returns.dropna().values
    n = len(er)

    boot_means = np.array([
        rng.choice(er, size=n, replace=True).mean() * 252
        for _ in range(n_bootstrap)
    ])

    lower = np.percentile(boot_means, (1 - confidence) / 2 * 100)
    upper = np.percentile(boot_means, (1 + confidence) / 2 * 100)

    return {
        "annualized_alpha_mean": float(boot_means.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": confidence,
        "pct_positive": float((boot_means > 0).mean()),
    }


def rolling_alpha(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 126,  # ~6 months
) -> pd.Series:
    """Rolling annualized alpha (strategy - beta * benchmark)."""
    aligned = pd.DataFrame({"s": strategy_returns, "b": benchmark_returns}).dropna()
    if len(aligned) < window:
        raise ValueError(
            f"Need at least {window} aligned observations, got {len(aligned)}"
        )

    alphas = []
    for i in range(len(aligned)):
        if i < window - 1:
            alphas.append(np.nan)
            continue
        chunk = aligned.iloc[i - window + 1 : i + 1]
        cov = np.cov(chunk["s"], chunk["b"])
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0
        alpha = (chunk["s"].mean() - beta * chunk["b"].mean()) * 252
        alphas.append(alpha)

    return pd.Series(alphas, index=aligned.index, name="rolling_alpha")


def information_coefficient(
    predicted_scores: pd.Series,
    realized_returns: pd.Series,
) -> float:
    """Rank IC: Spearman correlation between predicted signal and realized return."""
    aligned = pd.DataFrame({"pred": predicted_scores, "real": realized_returns}).dropna()
    if len(aligned) < 5:
        raise ValueError(
            f"Insufficient data for IC: need >= 5 observations, got {len(aligned)}"
        )
    corr, _ = stats.spearmanr(aligned["pred"], aligned["real"])
    return float(corr)
