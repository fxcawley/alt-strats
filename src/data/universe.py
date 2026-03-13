"""
Multi-asset ETF universes for factor research.

These are ETFs representing distinct asset classes and sub-classes.
All are highly liquid (tight spreads, deep books).  Realistic
transaction costs: 2-5bps.
"""

# Broad asset class ETFs
MULTI_ASSET = {
    # Equities
    "SPY": "US Large Cap",
    "QQQ": "US Nasdaq 100",
    "IWM": "US Small Cap",
    "EFA": "Intl Developed",
    "EEM": "Emerging Markets",
    # Fixed Income
    "TLT": "US Long Treasury",
    "IEF": "US 7-10Y Treasury",
    "SHY": "US Short Treasury",
    "LQD": "US IG Corporate",
    "HYG": "US High Yield",
    # Commodities
    "GLD": "Gold",
    "SLV": "Silver",
    "DBA": "Agriculture",
    # Real Estate
    "VNQ": "US REITs",
}

MULTI_ASSET_TICKERS = list(MULTI_ASSET.keys())

# Equity-only universe (for cross-sectional equity momentum)
EQUITY_ETFS = {
    "SPY": "US Large Cap",
    "QQQ": "US Nasdaq 100",
    "IWM": "US Small Cap",
    "IWD": "US Large Value",
    "IWF": "US Large Growth",
    "EFA": "Intl Developed",
    "EEM": "Emerging Markets",
    "VGK": "Europe",
    "EWJ": "Japan",
    "FXI": "China",
    "XLK": "US Tech",
    "XLF": "US Financials",
    "XLE": "US Energy",
    "XLV": "US Healthcare",
    "XLI": "US Industrials",
    "XLP": "US Staples",
    "XLY": "US Discretionary",
    "XLU": "US Utilities",
}

EQUITY_ETF_TICKERS = list(EQUITY_ETFS.keys())

# Sector ETFs
SECTOR_ETFS = {
    "tech": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "energy": "XLE",
    "consumer_disc": "XLY",
    "consumer_staples": "XLP",
    "industrials": "XLI",
    "materials": "XLB",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communications": "XLC",
}
