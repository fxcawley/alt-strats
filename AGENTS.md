# Project Notes (AGENTS.md)

## Build / Test Commands
- **Activate venv**: `source .venv/Scripts/activate` (Windows/Git Bash) or `source .venv/bin/activate` (Linux/Mac)
- **Install deps**: `pip install -r requirements.txt`
- **Smoke test**: `python -m tests.smoke_test`
- **NLP pipeline**: `python run_nlp_signal.py [--backtest] [--use-embeddings]`
- **Flow pipeline**: `python run_flow_signal.py [--backtest]`
- **Geo pipeline**: `python run_geo_signal.py [--backtest] [--use-synthetic]`
- **Clear cache**: `rm -f data/cache/*.parquet` (required after changing price fetch settings)

## Architecture
- All strategies implement `generate_signals(date, universe, lookback) -> dict[str, float] | None`
- Return `None` to keep existing positions; return `{}` to go to cash
- Price data uses `auto_adjust=True` (dividend-adjusted OHLCV) and is cached as parquet in `data/cache/`
- `src/backtest/engine.py` is the core loop: fetch data -> walk forward -> rebalance -> compute equity
- Engine tracks share counts (not weights) between rebalances -- positions drift with prices
- `rebalance_threshold` parameter skips trades below a weight-change minimum (default 0, set to 0.02 for production)
- Analysis modules in `src/analysis/` are standalone and can be used outside the backtest engine

## Signal Pipelines
- **NLP (Signal 1)**: EDGAR downloader -> filing parser -> LM sentiment + readability + embeddings -> composite score -> quintile ranking
- **Flow (Signal 3)**: CFTC COT reports -> net speculator positioning -> z-score -> contrarian signal
- **Geo (Signal 2)**: VIIRS luminosity + Google Trends + FRED surprises -> combined macro signal -> country/sector rotation

## Validation Framework (5 gates)
1. IC > 0.02 in isolation
2. Monotonic quintile returns
3. Walk-forward Sharpe > 0
4. Sharpe spread < 0.3 across periods
5. Incremental R-squared > 0.5% above baseline

## Anti-Patterns (from strat-testing)
- Use filing_date (point-in-time), NOT period-end date for NLP features
- Use publication_date (Friday), NOT report_date (Tuesday) for COT data
- Never silently swallow errors -- raise on parse/download failures
- `{}` means "go to cash"; `None` means "keep existing positions"
- Budget 3bps for ETFs, 5-15bps for individual stocks
- Prefer ridge over gradient boosting until >10K training observations
