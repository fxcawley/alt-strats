"""
CFTC Commitments of Traders (COT) report fetcher.

Downloads weekly COT reports from the CFTC website and parses them
into structured DataFrames. The COT report shows futures positioning
by trader category (commercial, large speculator, small speculator).

Key timing note: COT reports reflect positions as of Tuesday, but are
published on Friday. For backtesting, we use the publication date
(Friday) as the availability date to avoid look-ahead bias.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

COT_DIR = Path(__file__).resolve().parents[2] / "data" / "cot"
COT_URL_TEMPLATE = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
COT_CURRENT_URL = "https://www.cftc.gov/dea/newcot/deacom.txt"

HEADERS = {"User-Agent": "alt-strats-research"}

# Mapping from CFTC contract names to our ticker universe
# These map futures contracts to relevant ETF tickers
CONTRACT_TO_TICKER = {
    "E-MINI S&P 500": "SPY",
    "S&P 500 STOCK INDEX": "SPY",
    "NASDAQ-100": "QQQ",
    "E-MINI NASDAQ-100": "QQQ",
    "RUSSELL E-MINI": "IWM",
    "RUSSELL 2000 MINI": "IWM",
    "DJIA x $5": "SPY",
    "US TREASURY BONDS": "TLT",
    "10-YEAR U.S. TREASURY NOTES": "IEF",
    "2-YEAR U.S. TREASURY NOTES": "SHY",
    "5-YEAR U.S. TREASURY NOTES": "IEF",
    "GOLD": "GLD",
    "GOLD - COMMODITY EXCHANGE INC.": "GLD",
    "SILVER": "SLV",
    "SILVER - COMMODITY EXCHANGE INC.": "SLV",
    "CRUDE OIL, LIGHT SWEET": "XLE",
    "CRUDE OIL": "XLE",
    "NATURAL GAS": "XLE",
    "VIX FUTURES": "SPY",  # VIX as inverse signal for equities
}


def download_cot_historical(year: int) -> pd.DataFrame:
    """Download historical COT data for a given year.

    Returns a DataFrame with columns including:
        Market_and_Exchange_Names, As_of_Date_In_Form_YYMMDD,
        NonComm_Positions_Long_All, NonComm_Positions_Short_All,
        Comm_Positions_Long_All, Comm_Positions_Short_All, etc.
    """
    url = COT_URL_TEMPLATE.format(year=year)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # Find the CSV file in the zip
        csv_names = [n for n in zf.namelist() if n.endswith(".txt") or n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No data file found in COT zip for {year}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    return _clean_cot_df(df)


def download_cot_current() -> pd.DataFrame:
    """Download the most recent COT report."""
    resp = requests.get(COT_CURRENT_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
    return _clean_cot_df(df)


def _clean_cot_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize COT DataFrame column names and date parsing."""
    # Column names vary across years; standardize
    col_map = {}
    for c in df.columns:
        clean = c.strip()
        col_map[c] = clean
    df = df.rename(columns=col_map)

    # Parse date
    date_col = None
    for candidate in ["As_of_Date_In_Form_YYMMDD", "As of Date in Form YYMMDD",
                       "Report_Date_as_YYYY-MM-DD", "As_of_Date_In_Form_YYYY-MM-DD"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        # Try to find any date-like column
        for c in df.columns:
            if "date" in c.lower():
                date_col = c
                break

    if date_col:
        df["report_date"] = pd.to_datetime(df[date_col], format="mixed", dayfirst=False)
        # Publication date = report date (Tuesday) + 3 business days (Friday)
        df["publication_date"] = df["report_date"] + pd.offsets.BDay(3)

    return df


def get_cot_data(
    start_year: int = 2010,
    end_year: int | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Get COT data for a range of years.

    Downloads and concatenates yearly files. Caches to parquet.
    """
    if end_year is None:
        end_year = datetime.now().year

    cache_path = COT_DIR / f"cot_{start_year}_{end_year}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    COT_DIR.mkdir(parents=True, exist_ok=True)

    dfs = []
    for year in range(start_year, end_year + 1):
        try:
            df = download_cot_historical(year)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: could not download COT data for {year}: {e}")

    if not dfs:
        raise ValueError(f"No COT data downloaded for {start_year}-{end_year}")

    combined = pd.concat(dfs, ignore_index=True)

    if cache:
        combined.to_parquet(cache_path)

    return combined


def compute_net_positioning(cot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute net speculator positioning for each contract.

    Net positioning = NonComm Long - NonComm Short (large speculators).
    Returns a pivoted DataFrame: index=publication_date, columns=ticker,
    values=net positioning.
    """
    # Find the right column names
    market_col = None
    for c in ["Market_and_Exchange_Names", "Market and Exchange Names"]:
        if c in cot_df.columns:
            market_col = c
            break
    if market_col is None:
        raise ValueError("Cannot find market name column in COT data")

    long_col = None
    short_col = None
    for c in cot_df.columns:
        cl = c.lower().strip()
        if "noncomm" in cl and "long" in cl and "all" in cl and "spread" not in cl:
            long_col = c
        if "noncomm" in cl and "short" in cl and "all" in cl and "spread" not in cl:
            short_col = c

    if long_col is None or short_col is None:
        # Fallback: try simpler naming
        for c in cot_df.columns:
            cl = c.lower().strip()
            if "noncomm" in cl and "long" in cl and long_col is None:
                long_col = c
            if "noncomm" in cl and "short" in cl and short_col is None:
                short_col = c

    if long_col is None or short_col is None:
        raise ValueError(f"Cannot find positioning columns. Available: {list(cot_df.columns[:20])}")

    # Map contracts to tickers
    rows = []
    for _, row in cot_df.iterrows():
        market_name = str(row[market_col]).strip().upper()
        ticker = None
        for pattern, tkr in CONTRACT_TO_TICKER.items():
            if pattern.upper() in market_name:
                ticker = tkr
                break

        if ticker is None:
            continue

        net_pos = float(row[long_col]) - float(row[short_col])
        pub_date = row.get("publication_date")
        if pub_date is None or pd.isna(pub_date):
            continue

        rows.append({
            "publication_date": pub_date,
            "ticker": ticker,
            "net_positioning": net_pos,
        })

    if not rows:
        return pd.DataFrame()

    pos_df = pd.DataFrame(rows)
    # Pivot: one column per ticker
    pivot = pos_df.pivot_table(
        index="publication_date",
        columns="ticker",
        values="net_positioning",
        aggfunc="mean",
    )
    pivot = pivot.sort_index()
    return pivot


def compute_positioning_zscore(
    positioning: pd.DataFrame,
    lookback_weeks: int = 52,
) -> pd.DataFrame:
    """Compute z-score of positioning relative to trailing history.

    Z-score > 2 = extremely long (contrarian bearish signal).
    Z-score < -2 = extremely short (contrarian bullish signal).
    """
    window = lookback_weeks
    mean = positioning.rolling(window, min_periods=26).mean()
    std = positioning.rolling(window, min_periods=26).std()
    zscore = (positioning - mean) / std.replace(0, float("nan"))
    return zscore
