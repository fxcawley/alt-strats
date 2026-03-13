"""
SEC filing section parser.

Extracts specific sections from 10-K, 10-Q, and 8-K filings:
  - Item 1A: Risk Factors
  - Item 7: Management's Discussion and Analysis (MD&A)
  - Item 8: Financial Statements

Handles both HTML and plain-text filings. Uses BeautifulSoup for HTML
parsing and regex for section boundary detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


@dataclass
class ParsedFiling:
    """Container for extracted filing sections."""
    filing_type: str
    filing_date: str
    ticker: str
    full_text: str
    mda: str  # Item 7 / MD&A
    risk_factors: str  # Item 1A
    financial_statements: str  # Item 8
    raw_length: int


def strip_html(html_content: str) -> str:
    """Convert HTML filing to clean plain text."""
    soup = BeautifulSoup(html_content, "lxml")

    # Remove script and style elements
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse multiple whitespace/newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)

    return text.strip()


def _find_section(text: str, start_pattern: str, end_pattern: str) -> str:
    """Extract text between two section header patterns.

    Uses case-insensitive regex matching. Returns empty string if
    section not found.
    """
    start_match = re.search(start_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not start_match:
        return ""

    remaining = text[start_match.end():]
    end_match = re.search(end_pattern, remaining, re.IGNORECASE | re.MULTILINE)

    if end_match:
        return remaining[:end_match.start()].strip()
    # If no end pattern found, take a reasonable chunk (100K chars max)
    return remaining[:100_000].strip()


def extract_mda(text: str) -> str:
    """Extract Item 7 -- Management's Discussion and Analysis."""
    # Try multiple patterns (10-K vs 10-Q have different headers)
    patterns = [
        (r"item\s+7[\.\s]*[-—]?\s*management.?s\s+discussion", r"item\s+7a[\.\s]*[-—]?\s*quantitative"),
        (r"item\s+7[\.\s]*[-—]?\s*management.?s\s+discussion", r"item\s+8[\.\s]*[-—]?\s*financial\s+statements"),
        (r"item\s+2[\.\s]*[-—]?\s*management.?s\s+discussion", r"item\s+3[\.\s]*[-—]?\s*quantitative"),  # 10-Q
        (r"item\s+2[\.\s]*[-—]?\s*management.?s\s+discussion", r"item\s+4[\.\s]*[-—]?\s*controls"),  # 10-Q
    ]
    for start, end in patterns:
        result = _find_section(text, start, end)
        if len(result) > 500:  # meaningful section should be >500 chars
            return result
    return ""


def extract_risk_factors(text: str) -> str:
    """Extract Item 1A -- Risk Factors."""
    patterns = [
        (r"item\s+1a[\.\s]*[-—]?\s*risk\s+factors", r"item\s+1b[\.\s]*[-—]?\s*unresolved\s+staff"),
        (r"item\s+1a[\.\s]*[-—]?\s*risk\s+factors", r"item\s+2[\.\s]*[-—]?\s*properties"),
        (r"item\s+1a[\.\s]*[-—]?\s*risk\s+factors", r"item\s+2[\.\s]*[-—]?\s*management"),
    ]
    for start, end in patterns:
        result = _find_section(text, start, end)
        if len(result) > 500:
            return result
    return ""


def extract_financial_statements(text: str) -> str:
    """Extract Item 8 -- Financial Statements and Supplementary Data."""
    patterns = [
        (r"item\s+8[\.\s]*[-—]?\s*financial\s+statements", r"item\s+9[\.\s]*[-—]?\s*changes\s+in"),
        (r"item\s+8[\.\s]*[-—]?\s*financial\s+statements", r"item\s+9a[\.\s]*[-—]?\s*controls"),
    ]
    for start, end in patterns:
        result = _find_section(text, start, end)
        if len(result) > 200:
            return result
    return ""


def parse_filing(
    raw_content: str,
    filing_type: str,
    filing_date: str,
    ticker: str,
) -> ParsedFiling:
    """Parse a raw filing into structured sections.

    Handles both HTML and plain-text filings. Raises on empty content.
    """
    if not raw_content or len(raw_content.strip()) == 0:
        raise ValueError(f"Empty filing content for {ticker} on {filing_date}")

    # Strip HTML if present
    if "<html" in raw_content.lower() or "<body" in raw_content.lower():
        text = strip_html(raw_content)
    else:
        text = raw_content

    return ParsedFiling(
        filing_type=filing_type,
        filing_date=filing_date,
        ticker=ticker,
        full_text=text,
        mda=extract_mda(text),
        risk_factors=extract_risk_factors(text),
        financial_statements=extract_financial_statements(text),
        raw_length=len(raw_content),
    )
