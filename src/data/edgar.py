"""
SEC EDGAR filing downloader.

Downloads 10-K, 10-Q, and 8-K filings from EDGAR for a given ticker.
Respects SEC rate limits (10 req/s) and stores raw filings as gzipped
text in data/filings/{ticker}/{filing_date}.txt.gz.

All access is point-in-time: we record the filing date (when the filing
was submitted to EDGAR), not the period-end date.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd

FILINGS_DIR = Path(__file__).resolve().parents[2] / "data" / "filings"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_TEXT_URL = "https://www.sec.gov/Archives/edgar/data"
EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_INDEX_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

# SEC requires identifying User-Agent
HEADERS = {
    "User-Agent": "alt-strats-research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# Rate limit: 10 requests/second
_last_request_time = 0.0
_MIN_INTERVAL = 0.11  # slightly over 100ms to stay under 10 req/s


def _rate_limit():
    """Enforce SEC rate limit."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET with rate limiting and error handling."""
    _rate_limit()
    resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
    resp.raise_for_status()
    return resp


def get_cik(ticker: str) -> str:
    """Look up the CIK (Central Index Key) for a ticker.

    Returns the CIK as a zero-padded 10-digit string.
    """
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = _get(url)
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"] == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker {ticker} not found in SEC company tickers")


def get_filing_metadata(
    cik: str,
    filing_types: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    start_date: str = "2010-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch filing metadata (dates, accession numbers) for a CIK.

    Returns a DataFrame with columns:
        filing_type, filing_date, accession_number, primary_document
    """
    url = EDGAR_COMPANY_URL.format(cik=cik)
    resp = _get(url)
    data = resp.json()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        raise ValueError(f"No filing data found for CIK {cik}")

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    rows = []
    for form, date_str, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form not in filing_types:
            continue
        filing_date = datetime.strptime(date_str, "%Y-%m-%d")
        if filing_date < datetime.strptime(start_date, "%Y-%m-%d"):
            continue
        if end_date and filing_date > datetime.strptime(end_date, "%Y-%m-%d"):
            continue
        rows.append({
            "filing_type": form,
            "filing_date": date_str,
            "accession_number": acc,
            "primary_document": doc,
        })

    if not rows:
        raise ValueError(
            f"No {filing_types} filings found for CIK {cik} "
            f"between {start_date} and {end_date or 'now'}"
        )

    return pd.DataFrame(rows)


def download_filing(
    cik: str,
    accession_number: str,
    primary_document: str,
) -> str:
    """Download the full text of a single filing from EDGAR.

    Returns the raw HTML/text content.
    """
    # Accession number with dashes removed for URL
    acc_no_dashes = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no_dashes}/{primary_document}"
    resp = _get(url)
    return resp.text


def save_filing(ticker: str, filing_date: str, content: str) -> Path:
    """Save a filing as gzipped text.

    Returns the path to the saved file.
    """
    ticker_dir = FILINGS_DIR / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    filepath = ticker_dir / f"{filing_date}.txt.gz"
    with gzip.open(filepath, "wt", encoding="utf-8") as f:
        f.write(content)
    return filepath


def load_filing(ticker: str, filing_date: str) -> str:
    """Load a previously saved filing from disk."""
    filepath = FILINGS_DIR / ticker.upper() / f"{filing_date}.txt.gz"
    if not filepath.exists():
        raise FileNotFoundError(f"No cached filing for {ticker} on {filing_date}")
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        return f.read()


def download_all_filings(
    ticker: str,
    filing_types: tuple[str, ...] = ("10-K", "10-Q"),
    start_date: str = "2010-01-01",
    end_date: str | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Download all filings for a ticker and save to disk.

    Returns metadata DataFrame with an additional 'filepath' column.
    """
    cik = get_cik(ticker)
    metadata = get_filing_metadata(cik, filing_types, start_date, end_date)

    filepaths = []
    for _, row in metadata.iterrows():
        filepath = FILINGS_DIR / ticker.upper() / f"{row['filing_date']}.txt.gz"
        if skip_existing and filepath.exists():
            filepaths.append(str(filepath))
            continue

        content = download_filing(cik, row["accession_number"], row["primary_document"])
        saved_path = save_filing(ticker, row["filing_date"], content)
        filepaths.append(str(saved_path))

    metadata["filepath"] = filepaths
    return metadata
