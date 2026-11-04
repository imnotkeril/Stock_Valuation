"""
Curated company universe and sector reference data.

This is the single source of truth for the list of companies offered for
peer comparison, sector analysis and screening, and for the sector ETFs used
as benchmarks. Sector names are the canonical names from
``config.SECTOR_MAPPING``.
"""

from typing import Dict, List, Optional

from StockAnalysisSystem.src.config import normalize_sector

#: Sector -> SPDR sector ETF used as performance benchmark
SECTOR_ETFS: Dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Utilities": "XLU",
}

#: Broad market benchmark
MARKET_BENCHMARK = "SPY"

# (symbol, name, subsector)
_UNIVERSE_DATA: Dict[str, List[tuple]] = {
    "Technology": [
        ("AAPL", "Apple Inc.", "Hardware"),
        ("MSFT", "Microsoft Corporation", "Software"),
        ("NVDA", "NVIDIA Corporation", "Semiconductors"),
        ("AVGO", "Broadcom Inc.", "Semiconductors"),
        ("ORCL", "Oracle Corporation", "Software"),
        ("CRM", "Salesforce, Inc.", "Software"),
        ("ADBE", "Adobe Inc.", "Software"),
        ("AMD", "Advanced Micro Devices, Inc.", "Semiconductors"),
        ("INTC", "Intel Corporation", "Semiconductors"),
        ("CSCO", "Cisco Systems, Inc.", "Hardware"),
        ("IBM", "International Business Machines", "IT Services"),
        ("ACN", "Accenture plc", "IT Services"),
    ],
    "Healthcare": [
        ("LLY", "Eli Lilly and Company", "Pharmaceuticals"),
        ("JNJ", "Johnson & Johnson", "Pharmaceuticals"),
        ("MRK", "Merck & Co., Inc.", "Pharmaceuticals"),
        ("PFE", "Pfizer Inc.", "Pharmaceuticals"),
        ("ABBV", "AbbVie Inc.", "Biotechnology"),
        ("AMGN", "Amgen Inc.", "Biotechnology"),
        ("GILD", "Gilead Sciences, Inc.", "Biotechnology"),
        ("ABT", "Abbott Laboratories", "Medical Devices"),
        ("MDT", "Medtronic plc", "Medical Devices"),
        ("TMO", "Thermo Fisher Scientific Inc.", "Medical Devices"),
        ("UNH", "UnitedHealth Group Incorporated", "Healthcare Services"),
        ("CVS", "CVS Health Corporation", "Healthcare Services"),
    ],
    "Financials": [
        ("JPM", "JPMorgan Chase & Co.", "Banks"),
        ("BAC", "Bank of America Corporation", "Banks"),
        ("WFC", "Wells Fargo & Company", "Banks"),
        ("C", "Citigroup Inc.", "Banks"),
        ("GS", "The Goldman Sachs Group, Inc.", "Diversified Financial Services"),
        ("MS", "Morgan Stanley", "Diversified Financial Services"),
        ("BLK", "BlackRock, Inc.", "Asset Management"),
        ("SCHW", "The Charles Schwab Corporation", "Asset Management"),
        ("V", "Visa Inc.", "Diversified Financial Services"),
        ("MA", "Mastercard Incorporated", "Diversified Financial Services"),
        ("PGR", "The Progressive Corporation", "Insurance"),
        ("CB", "Chubb Limited", "Insurance"),
    ],
    "Consumer Discretionary": [
        ("AMZN", "Amazon.com, Inc.", "Retail"),
        ("HD", "The Home Depot, Inc.", "Retail"),
        ("LOW", "Lowe's Companies, Inc.", "Retail"),
        ("TJX", "The TJX Companies, Inc.", "Retail"),
        ("TGT", "Target Corporation", "Retail"),
        ("TSLA", "Tesla, Inc.", "Automotive"),
        ("GM", "General Motors Company", "Automotive"),
        ("F", "Ford Motor Company", "Automotive"),
        ("MCD", "McDonald's Corporation", "Leisure"),
        ("SBUX", "Starbucks Corporation", "Leisure"),
        ("BKNG", "Booking Holdings Inc.", "Leisure"),
        ("NKE", "NIKE, Inc.", "Apparel"),
    ],
    "Consumer Staples": [
        ("WMT", "Walmart Inc.", "Food & Beverage"),
        ("COST", "Costco Wholesale Corporation", "Food & Beverage"),
        ("KO", "The Coca-Cola Company", "Food & Beverage"),
        ("PEP", "PepsiCo, Inc.", "Food & Beverage"),
        ("MDLZ", "Mondelez International, Inc.", "Food & Beverage"),
        ("GIS", "General Mills, Inc.", "Food & Beverage"),
        ("PG", "The Procter & Gamble Company", "Household Products"),
        ("CL", "Colgate-Palmolive Company", "Household Products"),
        ("KMB", "Kimberly-Clark Corporation", "Household Products"),
        ("EL", "The Estée Lauder Companies Inc.", "Personal Products"),
        ("PM", "Philip Morris International Inc.", "Food & Beverage"),
        ("MO", "Altria Group, Inc.", "Food & Beverage"),
    ],
    "Energy": [
        ("XOM", "Exxon Mobil Corporation", "Oil & Gas"),
        ("CVX", "Chevron Corporation", "Oil & Gas"),
        ("COP", "ConocoPhillips", "Oil & Gas"),
        ("EOG", "EOG Resources, Inc.", "Oil & Gas"),
        ("OXY", "Occidental Petroleum Corporation", "Oil & Gas"),
        ("SLB", "SLB", "Oil & Gas"),
        ("MPC", "Marathon Petroleum Corporation", "Oil & Gas"),
        ("PSX", "Phillips 66", "Oil & Gas"),
        ("VLO", "Valero Energy Corporation", "Oil & Gas"),
        ("KMI", "Kinder Morgan, Inc.", "Oil & Gas"),
        ("WMB", "The Williams Companies, Inc.", "Oil & Gas"),
        ("ENPH", "Enphase Energy, Inc.", "Renewable Energy"),
    ],
    "Industrials": [
        ("GE", "GE Aerospace", "Aerospace & Defense"),
        ("RTX", "RTX Corporation", "Aerospace & Defense"),
        ("LMT", "Lockheed Martin Corporation", "Aerospace & Defense"),
        ("BA", "The Boeing Company", "Aerospace & Defense"),
        ("HON", "Honeywell International Inc.", "Machinery"),
        ("CAT", "Caterpillar Inc.", "Machinery"),
        ("DE", "Deere & Company", "Machinery"),
        ("ETN", "Eaton Corporation plc", "Machinery"),
        ("UNP", "Union Pacific Corporation", "Transportation"),
        ("UPS", "United Parcel Service, Inc.", "Transportation"),
        ("FDX", "FedEx Corporation", "Transportation"),
        ("PWR", "Quanta Services, Inc.", "Construction"),
    ],
    "Materials": [
        ("LIN", "Linde plc", "Chemicals"),
        ("APD", "Air Products and Chemicals, Inc.", "Chemicals"),
        ("SHW", "The Sherwin-Williams Company", "Chemicals"),
        ("ECL", "Ecolab Inc.", "Chemicals"),
        ("DOW", "Dow Inc.", "Chemicals"),
        ("DD", "DuPont de Nemours, Inc.", "Chemicals"),
        ("NEM", "Newmont Corporation", "Metals & Mining"),
        ("FCX", "Freeport-McMoRan Inc.", "Metals & Mining"),
        ("NUE", "Nucor Corporation", "Metals & Mining"),
        ("BHP", "BHP Group Limited", "Metals & Mining"),
        ("IP", "International Paper Company", "Paper & Forest Products"),
        ("PKG", "Packaging Corporation of America", "Paper & Forest Products"),
    ],
    "Real Estate": [
        ("PLD", "Prologis, Inc.", "REIT"),
        ("AMT", "American Tower Corporation", "REIT"),
        ("EQIX", "Equinix, Inc.", "REIT"),
        ("CCI", "Crown Castle Inc.", "REIT"),
        ("PSA", "Public Storage", "REIT"),
        ("O", "Realty Income Corporation", "REIT"),
        ("SPG", "Simon Property Group, Inc.", "REIT"),
        ("DLR", "Digital Realty Trust, Inc.", "REIT"),
        ("AVB", "AvalonBay Communities, Inc.", "REIT"),
        ("EQR", "Equity Residential", "REIT"),
        ("CBRE", "CBRE Group, Inc.", "Real Estate Services"),
        ("CSGP", "CoStar Group, Inc.", "Real Estate Services"),
    ],
    "Communication Services": [
        ("GOOGL", "Alphabet Inc.", "Media"),
        ("META", "Meta Platforms, Inc.", "Media"),
        ("NFLX", "Netflix, Inc.", "Entertainment"),
        ("DIS", "The Walt Disney Company", "Entertainment"),
        ("EA", "Electronic Arts Inc.", "Entertainment"),
        ("TTWO", "Take-Two Interactive Software, Inc.", "Entertainment"),
        ("CMCSA", "Comcast Corporation", "Media"),
        ("CHTR", "Charter Communications, Inc.", "Media"),
        ("VZ", "Verizon Communications Inc.", "Telecom"),
        ("T", "AT&T Inc.", "Telecom"),
        ("TMUS", "T-Mobile US, Inc.", "Telecom"),
        ("WBD", "Warner Bros. Discovery, Inc.", "Entertainment"),
    ],
    "Utilities": [
        ("NEE", "NextEra Energy, Inc.", "Electric Utilities"),
        ("SO", "The Southern Company", "Electric Utilities"),
        ("DUK", "Duke Energy Corporation", "Electric Utilities"),
        ("AEP", "American Electric Power Company, Inc.", "Electric Utilities"),
        ("EXC", "Exelon Corporation", "Electric Utilities"),
        ("XEL", "Xcel Energy Inc.", "Electric Utilities"),
        ("D", "Dominion Energy, Inc.", "Multi-Utilities"),
        ("SRE", "Sempra", "Multi-Utilities"),
        ("WEC", "WEC Energy Group, Inc.", "Multi-Utilities"),
        ("ATO", "Atmos Energy Corporation", "Gas Utilities"),
        ("NI", "NiSource Inc.", "Gas Utilities"),
        ("AWK", "American Water Works Company, Inc.", "Water Utilities"),
    ],
}

#: Flat list of company records: {"symbol", "name", "sector", "subsector"}
UNIVERSE: List[Dict[str, str]] = [
    {"symbol": symbol, "name": name, "sector": sector, "subsector": subsector}
    for sector, companies in _UNIVERSE_DATA.items()
    for symbol, name, subsector in companies
]


def get_universe(sector: Optional[str] = None, subsector: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Return company records, optionally filtered by sector and subsector.

    ``sector`` accepts any provider spelling; ``None`` or ``"All"`` returns
    every sector. ``subsector`` ``None`` or ``"All"`` disables that filter.
    """
    records = UNIVERSE
    if sector and sector != "All":
        canonical = normalize_sector(sector)
        records = [r for r in records if r["sector"] == canonical]
    if subsector and subsector != "All":
        records = [r for r in records if r["subsector"].lower() == str(subsector).lower()]
    return [dict(r) for r in records]


def get_sector_companies(sector: str, subsector: Optional[str] = None) -> List[Dict[str, str]]:
    """Companies of a sector (and optional subsector) as records."""
    return get_universe(sector, subsector)


def get_sector_peers(ticker: str, sector: Optional[str], limit: int = 10,
                     max_peers: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Peer companies for ``ticker``: same-subsector companies first, then the
    rest of the sector, excluding ``ticker`` itself.

    Args:
        ticker: Company ticker.
        sector: Company sector (any spelling).
        limit: Maximum number of peers (``max_peers`` is an alias).

    Returns:
        List of records with ``symbol``, ``name``, ``sector`` and ``subsector``.
    """
    limit = max_peers if max_peers is not None else limit
    ticker = (ticker or "").upper()
    companies = get_universe(sector)
    own = next((c for c in UNIVERSE if c["symbol"] == ticker), None)
    others = [c for c in companies if c["symbol"] != ticker]
    if own:
        others.sort(key=lambda c: c["subsector"] != own["subsector"])
    return others[:max(limit, 0)]


def get_sector_etf(sector: Optional[str]) -> Optional[str]:
    """Sector benchmark ETF ticker, or None for unknown sectors."""
    return SECTOR_ETFS.get(normalize_sector(sector))


def get_subsectors(sector: Optional[str]) -> List[str]:
    """Distinct subsectors present in the universe for a sector."""
    seen: List[str] = []
    for record in get_universe(sector):
        if record["subsector"] not in seen:
            seen.append(record["subsector"])
    return seen


def lookup_company(symbol: str) -> Optional[Dict[str, str]]:
    """Return the universe record for ``symbol`` (case-insensitive), if any."""
    symbol = (symbol or "").upper()
    return next((dict(c) for c in UNIVERSE if c["symbol"] == symbol), None)


def search_universe(query: str, limit: int = 10) -> List[Dict[str, str]]:
    """Offline ticker/name search over the curated universe."""
    query = (query or "").strip().lower()
    if not query:
        return []
    exact = [c for c in UNIVERSE if c["symbol"].lower() == query]
    prefix = [c for c in UNIVERSE if c["symbol"].lower().startswith(query) and c not in exact]
    by_name = [c for c in UNIVERSE if query in c["name"].lower() and c not in exact and c not in prefix]
    return [dict(c) for c in (exact + prefix + by_name)[:limit]]
