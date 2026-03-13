"""
NLP filing signal: features -> prediction -> portfolio weights.

Computes a composite signal from SEC filing NLP features and converts
it to a cross-sectional score for portfolio construction.

Signal flow:
  1. For each new filing, compute sentiment (LM dictionary), readability,
     and optionally embedding similarity vs prior filing.
  2. Combine features into a composite score (z-scored, equal-weighted
     across feature groups by default).
  3. Rank stocks by composite score. Top quintile = long, bottom = short
     (or long-only if configured).

Point-in-time alignment: features are only available after the filing
date. Between filings, the signal for a stock is its most recent filing
score (stale signals decay toward neutral over time).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.nlp.sentiment import compute_sentiment, compute_sentiment_change, SentimentScores
from src.nlp.filing_parser import parse_filing, ParsedFiling


@dataclass
class FilingFeatures:
    """All NLP features extracted from a single filing."""
    ticker: str
    filing_date: str
    filing_type: str
    sentiment: dict[str, float]
    sentiment_change: dict[str, float] | None  # None if no prior filing
    readability: dict[str, float]
    embedding_change: dict[str, float] | None  # None if embeddings disabled
    composite_score: float | None = None


def extract_filing_features(
    parsed: ParsedFiling,
    prior_parsed: ParsedFiling | None = None,
    use_embeddings: bool = False,
) -> FilingFeatures:
    """Extract all NLP features from a parsed filing.

    Parameters
    ----------
    parsed : ParsedFiling
        The current filing (already parsed into sections).
    prior_parsed : ParsedFiling or None
        The previous filing for the same company (for change features).
    use_embeddings : bool
        Whether to compute embedding similarity (slower, needs model).
    """
    # Use MD&A as primary text; fall back to full text if MD&A extraction failed
    primary_text = parsed.mda if parsed.mda else parsed.full_text[:50_000]
    risk_text = parsed.risk_factors if parsed.risk_factors else ""

    # Sentiment on MD&A
    mda_sentiment = compute_sentiment(primary_text)
    # Sentiment on Risk Factors (separate signal)
    risk_sentiment = compute_sentiment(risk_text)

    # Combine into feature dict
    sentiment_feats = mda_sentiment.to_dict(prefix="mda_")
    sentiment_feats.update(risk_sentiment.to_dict(prefix="risk_"))

    # Readability features
    readability_feats = {
        "mda_gunning_fog": mda_sentiment.gunning_fog,
        "mda_word_count": float(mda_sentiment.word_count),
        "mda_complex_word_ratio": mda_sentiment.complex_word_ratio,
        "doc_length": float(parsed.raw_length),
    }

    # Sentiment change vs prior filing
    sentiment_change = None
    if prior_parsed is not None:
        prior_text = prior_parsed.mda if prior_parsed.mda else prior_parsed.full_text[:50_000]
        prior_sentiment = compute_sentiment(prior_text)
        sentiment_change = compute_sentiment_change(mda_sentiment, prior_sentiment)

    # Embedding change
    embedding_change = None
    if use_embeddings and prior_parsed is not None:
        from src.nlp.embeddings import compute_embedding_change
        prior_text = prior_parsed.mda if prior_parsed.mda else prior_parsed.full_text[:50_000]
        embedding_change = compute_embedding_change(primary_text, prior_text)

    return FilingFeatures(
        ticker=parsed.ticker,
        filing_date=parsed.filing_date,
        filing_type=parsed.filing_type,
        sentiment=sentiment_feats,
        sentiment_change=sentiment_change,
        readability=readability_feats,
        embedding_change=embedding_change,
    )


def compute_composite_score(features: FilingFeatures) -> float:
    """Compute a single composite score from all NLP features.

    Higher score = more positive signal (go long).
    The score is a simple equal-weighted average of z-scored components.
    In production, this would be replaced by a trained model.

    Components (direction):
      + net_sentiment (higher = more positive language)
      - negative_ratio (more negative words = worse)
      - uncertainty_ratio (more uncertainty = worse)
      - gunning_fog (more complex/obfuscated = worse)
      + sentiment_change (improving tone = positive)
      + embedding_distance (more change = potentially informative; signed by sentiment change)
    """
    components = []

    # Level features
    s = features.sentiment
    components.append(s.get("mda_net_sentiment", 0.0))
    components.append(-s.get("mda_negative_ratio", 0.0))
    components.append(-s.get("mda_uncertainty_ratio", 0.0))

    # Readability (negative: more complex = worse)
    fog = features.readability.get("mda_gunning_fog", 15.0)
    # Normalize fog to ~[-1, 1] range (typical range 12-22)
    components.append(-(fog - 17.0) / 5.0)

    # Change features (if available)
    if features.sentiment_change:
        # Improving net sentiment is positive
        components.append(features.sentiment_change.get("delta_net_sentiment", 0.0))
        # Decreasing negativity is positive
        components.append(-features.sentiment_change.get("delta_negative_ratio", 0.0))

    # Embedding change (if available, signed by sentiment direction)
    if features.embedding_change and features.sentiment_change:
        distance = features.embedding_change.get("embedding_distance", 0.0)
        sentiment_dir = features.sentiment_change.get("delta_net_sentiment", 0.0)
        # Large change + positive sentiment direction = positive signal
        components.append(distance * np.sign(sentiment_dir) if sentiment_dir != 0 else 0.0)

    if not components:
        return 0.0

    return float(np.mean(components))


def build_filing_signal_df(
    filing_features_list: list[FilingFeatures],
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a signal DataFrame from a list of filing features.

    For each ticker, uses the most recent filing as of `as_of_date`.
    Returns a DataFrame indexed by ticker with a 'signal' column
    (the composite score) and all individual features.
    """
    if not filing_features_list:
        return pd.DataFrame()

    # Flatten features
    rows = []
    for ff in filing_features_list:
        ff.composite_score = compute_composite_score(ff)
        row = {
            "ticker": ff.ticker,
            "filing_date": ff.filing_date,
            "filing_type": ff.filing_type,
            "signal": ff.composite_score,
        }
        row.update(ff.sentiment)
        row.update(ff.readability)
        if ff.sentiment_change:
            row.update(ff.sentiment_change)
        if ff.embedding_change:
            row.update(ff.embedding_change)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    # Filter to as_of_date if provided
    if as_of_date is not None:
        as_of = pd.Timestamp(as_of_date)
        df = df[df["filing_date"] <= as_of]

    if df.empty:
        return pd.DataFrame()

    # Keep most recent filing per ticker
    df = df.sort_values("filing_date").groupby("ticker").last().reset_index()
    return df.set_index("ticker")


def signal_to_weights(
    signal_df: pd.DataFrame,
    n_long: int | None = None,
    n_short: int | None = None,
    long_only: bool = True,
) -> dict[str, float]:
    """Convert signal scores to portfolio weights.

    Parameters
    ----------
    signal_df : pd.DataFrame
        Must have a 'signal' column, indexed by ticker.
    n_long : int or None
        Number of tickers to go long (top N). Defaults to top quintile.
    n_short : int or None
        Number of tickers to go short (bottom N). Ignored if long_only.
    long_only : bool
        If True, only go long (no short positions).

    Returns portfolio weights dict: {ticker: weight}.
    """
    if signal_df.empty or "signal" not in signal_df.columns:
        return {}

    ranked = signal_df["signal"].sort_values(ascending=False)
    n = len(ranked)

    if n_long is None:
        n_long = max(1, n // 5)  # top quintile
    if n_short is None:
        n_short = max(1, n // 5) if not long_only else 0

    long_tickers = ranked.head(n_long).index.tolist()
    short_tickers = ranked.tail(n_short).index.tolist() if n_short > 0 else []

    weights = {}
    if long_tickers:
        long_w = 1.0 / len(long_tickers) if long_only else 0.5 / len(long_tickers)
        for t in long_tickers:
            weights[t] = long_w

    if short_tickers and not long_only:
        short_w = -0.5 / len(short_tickers)
        for t in short_tickers:
            weights[t] = short_w

    return weights
