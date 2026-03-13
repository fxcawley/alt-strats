"""
Feature engineering for ETF return prediction.

All features are computed from data available at the prediction date.
No look-ahead. Every feature uses only price/volume data up to and
including the current date.

Feature groups:
  1. Per-ETF price features (trailing returns, vol, RSI, drawdown)
  2. Market-level features (SPY regime, VIX proxy, breadth, dispersion)
  3. Cross-asset features (yield curve proxy, equity-bond correlation)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_etf_features(close: pd.Series, volume: pd.Series | None = None) -> dict[str, float]:
    """Compute per-ETF features from price history up to today.

    Parameters
    ----------
    close : pd.Series
        Dividend-adjusted close prices, ending at the prediction date.
    volume : pd.Series or None
        Volume data (optional).

    Returns dict of {feature_name: value}. Returns empty dict if
    insufficient data.
    """
    n = len(close)
    if n < 260:  # need ~1 year of history
        return {}

    features = {}
    c = close.values
    current = c[-1]

    # Trailing returns at multiple horizons
    for days, label in [(21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")]:
        if n > days:
            features[f"ret_{label}"] = current / c[-days - 1] - 1

    # 12-1 momentum (skip last month)
    if n > 273:
        features["ret_12_1"] = c[-22] / c[-273] - 1

    # Realized volatility at multiple horizons
    rets = np.diff(c) / c[:-1]
    for days, label in [(21, "1m"), (63, "3m")]:
        if len(rets) > days:
            features[f"vol_{label}"] = float(np.std(rets[-days:]) * np.sqrt(252))

    # Vol ratio (short-term vs long-term -- vol regime indicator)
    if "vol_1m" in features and "vol_3m" in features and features["vol_3m"] > 0:
        features["vol_ratio"] = features["vol_1m"] / features["vol_3m"]

    # RSI (14-day, simplified)
    if n > 20:
        deltas = np.diff(c[-15:])
        gains = np.mean(np.maximum(deltas, 0))
        losses = np.mean(np.maximum(-deltas, 0))
        if losses > 0:
            rs = gains / losses
            features["rsi_14"] = 100 - (100 / (1 + rs))
        else:
            features["rsi_14"] = 100.0

    # Drawdown from 52-week high
    if n >= 252:
        high_52w = np.max(c[-252:])
        features["drawdown_52w"] = current / high_52w - 1

    # Mean reversion z-score (20-day)
    if n > 25:
        roll_mean = np.mean(c[-20:])
        roll_std = np.std(c[-20:])
        if roll_std > 0:
            features["zscore_20d"] = (current - roll_mean) / roll_std

    # Volume trend (if available)
    if volume is not None and len(volume) > 42:
        v = volume.values
        avg_recent = np.mean(v[-21:])
        avg_prior = np.mean(v[-42:-21])
        if avg_prior > 0:
            features["volume_trend"] = avg_recent / avg_prior - 1

    return features


def compute_market_features(
    lookback: dict[str, pd.DataFrame],
    universe: list[str],
    spy_key: str = "SPY",
) -> dict[str, float]:
    """Compute market-level features from the full universe.

    These are the same for every ETF on a given date -- they describe
    the market regime, not the individual asset.
    """
    features = {}

    # SPY regime features
    if spy_key in lookback:
        spy_close = lookback[spy_key]["Close"]
        spy_feats = compute_etf_features(spy_close)
        for k in ["ret_1m", "ret_3m", "vol_1m", "vol_3m"]:
            if k in spy_feats:
                features[f"mkt_{k}"] = spy_feats[k]

    # VIX proxy: SPY 21-day realized vol (annualized)
    # Real VIX would be better but this avoids a separate data fetch
    if spy_key in lookback and len(lookback[spy_key]) > 25:
        spy_rets = lookback[spy_key]["Close"].pct_change().iloc[-21:]
        features["vix_proxy"] = float(spy_rets.std() * np.sqrt(252))

    # Yield curve proxy: TLT 3m return minus SHY 3m return
    # When this is positive, long bonds are outperforming short bonds (curve flattening/inversion)
    if "TLT" in lookback and "SHY" in lookback:
        for etf in ["TLT", "SHY"]:
            c = lookback[etf]["Close"]
            if len(c) > 63:
                features[f"{etf.lower()}_ret_3m"] = c.iloc[-1] / c.iloc[-64] - 1
        if "tlt_ret_3m" in features and "shy_ret_3m" in features:
            features["yield_slope_proxy"] = features["tlt_ret_3m"] - features["shy_ret_3m"]

    # Breadth: fraction of equity ETFs with positive 1-month return
    equity_tickers = [t for t in universe if t not in ("TLT", "IEF", "SHY", "LQD", "GLD")]
    n_positive = 0
    n_total = 0
    for t in equity_tickers:
        if t in lookback and len(lookback[t]) > 22:
            c = lookback[t]["Close"]
            if c.iloc[-1] / c.iloc[-22] - 1 > 0:
                n_positive += 1
            n_total += 1
    if n_total > 0:
        features["breadth"] = n_positive / n_total

    # Cross-sectional dispersion: std of 1-month returns across universe
    rets_1m = []
    for t in universe:
        if t in lookback and len(lookback[t]) > 22:
            c = lookback[t]["Close"]
            rets_1m.append(c.iloc[-1] / c.iloc[-22] - 1)
    if len(rets_1m) > 3:
        features["dispersion"] = float(np.std(rets_1m))

    # Equity-bond correlation (trailing 63d)
    if spy_key in lookback and "TLT" in lookback:
        spy_c = lookback[spy_key]["Close"]
        tlt_c = lookback["TLT"]["Close"]
        if len(spy_c) > 63 and len(tlt_c) > 63:
            spy_r = spy_c.pct_change().iloc[-63:]
            tlt_r = tlt_c.pct_change().iloc[-63:]
            aligned = pd.DataFrame({"spy": spy_r, "tlt": tlt_r}).dropna()
            if len(aligned) > 20:
                features["eq_bond_corr"] = float(aligned["spy"].corr(aligned["tlt"]))

    return features


def build_feature_matrix(
    lookback: dict[str, pd.DataFrame],
    universe: list[str],
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Build the full feature matrix for all ETFs on a given date.

    Returns a DataFrame with one row per ETF, columns are features.
    Market-level features are repeated across all rows.
    """
    market_feats = compute_market_features(lookback, universe)

    rows = []
    for ticker in universe:
        df = lookback.get(ticker)
        if df is None or len(df) < 260:
            continue

        close = df["Close"]
        volume = df.get("Volume")
        etf_feats = compute_etf_features(close, volume)

        if not etf_feats:
            continue

        row = {"ticker": ticker}
        row.update(etf_feats)
        row.update(market_feats)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ticker")
