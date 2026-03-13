"""
Smoke test: verify all imports work and basic structure is correct.

Usage:
    python -m tests.smoke_test
"""

from __future__ import annotations

import sys


def test_imports():
    """Verify all modules can be imported."""
    print("Testing imports...")

    modules = [
        # Shared infrastructure (from strat-testing)
        "src.backtest.engine",
        "src.backtest.metrics",
        "src.analysis.alpha",
        "src.data.prices",
        "src.data.session",
        "src.data.universe",
        "src.ml.features",
        # NLP pipeline
        "src.data.edgar",
        "src.nlp.filing_parser",
        "src.nlp.sentiment",
        "src.nlp.embeddings",
        "src.signals.filing_signal",
        "src.strategies.nlp_strategy",
        # Order flow pipeline
        "src.data.cot",
        "src.signals.flow_signal",
        "src.strategies.flow_strategy",
        # Satellite / geo pipeline
        "src.data.fred",
        "src.data.google_trends",
        "src.data.satellite",
        "src.signals.geo_signal",
        "src.strategies.geo_strategy",
        # Validation
        "src.validation.gates",
    ]

    failed = []
    for mod in modules:
        try:
            __import__(mod)
            print(f"  OK  {mod}")
        except Exception as e:
            print(f"  FAIL {mod}: {e}")
            failed.append((mod, str(e)))

    return failed


def test_strategy_interface():
    """Verify strategies implement the correct interface."""
    print("\nTesting strategy interface...")

    from src.strategies.nlp_strategy import NLPFilingStrategy
    from src.strategies.flow_strategy import FlowStrategy
    from src.strategies.geo_strategy import GeoStrategy
    import pandas as pd

    strategies = [
        ("NLPFilingStrategy", NLPFilingStrategy(filing_features=[])),
        ("FlowStrategy", FlowStrategy(positioning_zscore=pd.DataFrame())),
        ("GeoStrategy", GeoStrategy()),
    ]

    for name, strategy in strategies:
        assert hasattr(strategy, "generate_signals"), f"{name} missing generate_signals"
        # Test with empty data -- should return None (no data available)
        result = strategy.generate_signals(
            pd.Timestamp("2024-01-01"),
            ["SPY", "QQQ"],
            {},
        )
        assert result is None or isinstance(result, dict), \
            f"{name}.generate_signals returned {type(result)}, expected dict or None"
        print(f"  OK  {name}.generate_signals -> {type(result).__name__}")


def test_validation_gates():
    """Verify the validation framework works."""
    print("\nTesting validation framework...")

    import numpy as np
    import pandas as pd
    from src.validation.gates import (
        gate1_information_coefficient,
        gate2_quintile_monotonicity,
        gate3_walk_forward,
        gate4_cross_period_robustness,
        gate5_incremental_value,
        run_validation,
    )

    rng = np.random.default_rng(42)
    n = 500

    # Create test data with a weak signal
    signal = pd.Series(rng.normal(0, 1, n))
    returns = pd.Series(signal * 0.02 + rng.normal(0, 1, n))
    strategy_returns = pd.Series(rng.normal(0.0003, 0.01, 252))

    # Gate 1
    g1 = gate1_information_coefficient(signal, returns)
    print(f"  Gate 1 (IC): {g1}")

    # Gate 2
    g2 = gate2_quintile_monotonicity(signal, returns)
    print(f"  Gate 2 (Quintile): {g2}")

    # Gate 3
    g3 = gate3_walk_forward(strategy_returns)
    print(f"  Gate 3 (Backtest): {g3}")

    # Gate 4
    g4 = gate4_cross_period_robustness({"2015-2019": 0.65, "2020-2025": 0.55})
    print(f"  Gate 4 (Robustness): {g4}")

    # Gate 5
    g5 = gate5_incremental_value(0.02, 0.03)
    print(f"  Gate 5 (Incremental): {g5}")

    # Full report
    report = run_validation(
        signal_name="Test Signal",
        predicted_scores=signal,
        realized_returns=returns,
        strategy_returns=strategy_returns,
        sharpe_by_period={"2015-2019": 0.65, "2020-2025": 0.55},
        baseline_r_squared=0.02,
        combined_r_squared=0.03,
    )
    print(f"\n{report.summary()}")


def test_sentiment():
    """Test Loughran-McDonald sentiment computation."""
    print("\nTesting sentiment computation...")

    from src.nlp.sentiment import compute_sentiment

    # Test with a sample financial text
    text = """
    The company reported strong revenue growth driven by increased demand.
    However, we face significant risks including market uncertainty,
    potential litigation, and regulatory constraints. Operating expenses
    increased due to restructuring charges. Despite these challenges,
    management remains optimistic about future profitability and
    expects continued improvement in margins.
    """

    scores = compute_sentiment(text)
    print(f"  Word count: {scores.word_count}")
    print(f"  Positive ratio: {scores.positive_ratio:.4f}")
    print(f"  Negative ratio: {scores.negative_ratio:.4f}")
    print(f"  Net sentiment: {scores.net_sentiment:.4f}")
    print(f"  Uncertainty ratio: {scores.uncertainty_ratio:.4f}")
    print(f"  Gunning Fog: {scores.gunning_fog:.1f}")
    assert scores.word_count > 0, "Should count words"
    print("  OK  Sentiment computation works")


def test_filing_parser():
    """Test filing section extraction."""
    print("\nTesting filing parser...")

    from src.nlp.filing_parser import parse_filing, strip_html

    # Test HTML stripping
    html = "<html><body><p>Test <b>content</b></p></body></html>"
    text = strip_html(html)
    assert "Test" in text and "content" in text, f"HTML stripping failed: {text}"
    print("  OK  HTML stripping")

    # Test section extraction with sample filing text
    # Sections need >500 chars to be considered valid (filters noise)
    risk_text = " ".join(["Market conditions may deteriorate significantly. "
                          "Competition is intense and could adversely affect results. "
                          "Regulatory changes pose material risks to our operations. "] * 20)
    mda_text = " ".join(["Revenue increased 15% year over year driven by strong demand. "
                         "Operating margins expanded due to cost discipline and efficiency. "
                         "We expect continued growth in all business segments. "] * 20)
    sample = f"""
    ITEM 1A. RISK FACTORS
    {risk_text}
    ITEM 1B. UNRESOLVED STAFF COMMENTS
    None.
    ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
    {mda_text}
    ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES
    See notes to financial statements.
    """

    parsed = parse_filing(sample, "10-K", "2024-01-15", "TEST")
    assert parsed.ticker == "TEST"
    assert len(parsed.risk_factors) > 0, "Should extract risk factors"
    assert len(parsed.mda) > 0, "Should extract MD&A"
    print(f"  OK  Section extraction (Risk: {len(parsed.risk_factors)} chars, MD&A: {len(parsed.mda)} chars)")


def main():
    print("=" * 60)
    print("alt-strats Smoke Test")
    print("=" * 60)

    # Test imports
    failed = test_imports()

    # Test components
    test_filing_parser()
    test_sentiment()
    test_strategy_interface()
    test_validation_gates()

    print("\n" + "=" * 60)
    if failed:
        print(f"SMOKE TEST FAILED: {len(failed)} import(s) failed")
        for mod, err in failed:
            print(f"  {mod}: {err}")
        sys.exit(1)
    else:
        print("SMOKE TEST PASSED: All modules imported and tested successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
