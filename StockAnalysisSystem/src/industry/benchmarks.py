"""
Industry / sector ratio benchmarks.

:data:`SECTOR_BENCHMARKS` is the single source of truth for sector benchmark
ratios used by :class:`~StockAnalysisSystem.src.models.ratio_analysis.FinancialRatioAnalyzer`
and the UI. Every sector exposes the same categories and ratio keys (the keys
produced by ``FinancialRatioAnalyzer.calculate_ratios``); ratios that are not
meaningful for a sector (e.g. gross margin or current ratio for banks) are
``None``. Values are approximate large-cap sector medians; margins, returns,
yields and growth rates are fractions (0.25 == 25 %).

Optional overrides are read from ``SECTOR_DATA_DIR/sector_benchmark_overrides.json``
and are only written by :meth:`IndustryBenchmarks.update_benchmarks`. (The
legacy ``sector_benchmarks.json`` was an auto-generated copy of old defaults
and is intentionally ignored.)
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from StockAnalysisSystem.src.config import SECTOR_DATA_DIR, SECTOR_MAPPING, SECTORS, normalize_sector
from StockAnalysisSystem.src.industry.sector_mapping import map_industry_to_sector, normalize_sector_name
from StockAnalysisSystem.src.industry.universe import SECTOR_ETFS

logger = logging.getLogger(__name__)

BENCHMARKS_FILE = "sector_benchmark_overrides.json"
PERFORMANCE_FILE = "sector_performance.csv"

Benchmarks = Dict[str, Dict[str, Optional[float]]]

#: Ratio keys per category shared by every sector (order is display order).
BENCHMARK_KEYS: Dict[str, tuple] = {
    "valuation": ("pe_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "ev_revenue", "dividend_yield"),
    "profitability": ("gross_margin", "operating_margin", "net_margin", "roe", "roa", "roic"),
    "liquidity": ("current_ratio", "quick_ratio", "cash_ratio"),
    "leverage": ("debt_to_equity", "debt_to_assets", "debt_to_ebitda", "interest_coverage"),
    "efficiency": ("asset_turnover", "inventory_turnover", "receivables_turnover"),
    "growth": ("revenue_growth", "net_income_growth"),
}

#: Ratios where a lower value is better (used to orient benchmark comparisons).
LOWER_IS_BETTER = frozenset({
    "pe_ratio", "forward_pe", "peg_ratio", "ps_ratio", "pb_ratio", "ev_ebitda", "ev_revenue",
    "price_to_fcf", "price_to_cash_flow", "price_to_ffo",
    "debt_to_equity", "debt_to_assets", "debt_to_ebitda", "net_debt_to_ebitda", "liabilities_to_assets",
    "days_sales_outstanding", "days_inventory_outstanding",
    "efficiency_ratio", "inventory_to_sales", "sbc_to_revenue", "capex_to_revenue", "capex_to_ocf",
})


def _row(*values) -> list:
    return list(values)


# Compact table: one row per category in BENCHMARK_KEYS order (None = not meaningful).
_TABLE: Dict[str, Dict[str, list]] = {
    "Technology": {
        "valuation": _row(28.0, 6.0, 7.0, 20.0, 6.0, 0.008),
        "profitability": _row(0.55, 0.22, 0.18, 0.22, 0.10, 0.15),
        "liquidity": _row(2.0, 1.7, 0.8),
        "leverage": _row(0.5, 0.20, 1.5, 20.0),
        "efficiency": _row(0.7, 8.0, 6.0),
        "growth": _row(0.10, 0.12),
    },
    "Healthcare": {
        "valuation": _row(24.0, 4.0, 4.5, 16.0, 4.5, 0.015),
        "profitability": _row(0.60, 0.17, 0.12, 0.16, 0.07, 0.10),
        "liquidity": _row(1.8, 1.4, 0.6),
        "leverage": _row(0.7, 0.30, 2.5, 10.0),
        "efficiency": _row(0.6, 4.0, 6.0),
        "growth": _row(0.07, 0.08),
    },
    "Financials": {
        "valuation": _row(14.0, 3.0, 1.4, None, None, 0.025),
        "profitability": _row(None, 0.30, 0.22, 0.11, 0.011, None),
        "liquidity": _row(None, None, None),
        "leverage": _row(1.5, 0.12, None, None),
        "efficiency": _row(0.06, None, None),
        "growth": _row(0.05, 0.06),
    },
    "Consumer Discretionary": {
        "valuation": _row(20.0, 1.5, 4.0, 12.0, 1.8, 0.012),
        "profitability": _row(0.38, 0.10, 0.07, 0.18, 0.07, 0.12),
        "liquidity": _row(1.4, 0.8, 0.35),
        "leverage": _row(1.0, 0.35, 2.5, 8.0),
        "efficiency": _row(1.2, 6.0, 15.0),
        "growth": _row(0.06, 0.07),
    },
    "Consumer Staples": {
        "valuation": _row(21.0, 1.8, 5.0, 14.0, 2.2, 0.028),
        "profitability": _row(0.38, 0.15, 0.09, 0.20, 0.08, 0.12),
        "liquidity": _row(1.0, 0.6, 0.2),
        "leverage": _row(1.0, 0.35, 2.8, 10.0),
        "efficiency": _row(1.0, 7.0, 12.0),
        "growth": _row(0.04, 0.05),
    },
    "Energy": {
        "valuation": _row(12.0, 1.2, 1.8, 6.0, 1.5, 0.035),
        "profitability": _row(0.35, 0.15, 0.10, 0.14, 0.07, 0.10),
        "liquidity": _row(1.3, 1.0, 0.3),
        "leverage": _row(0.45, 0.20, 1.2, 12.0),
        "efficiency": _row(0.7, 12.0, 9.0),
        "growth": _row(0.03, 0.03),
    },
    "Industrials": {
        "valuation": _row(20.0, 1.8, 4.0, 13.0, 2.2, 0.016),
        "profitability": _row(0.32, 0.13, 0.09, 0.18, 0.07, 0.11),
        "liquidity": _row(1.4, 0.9, 0.3),
        "leverage": _row(0.9, 0.32, 2.3, 10.0),
        "efficiency": _row(0.8, 5.0, 6.0),
        "growth": _row(0.05, 0.06),
    },
    "Materials": {
        "valuation": _row(17.0, 1.6, 2.4, 9.0, 2.0, 0.022),
        "profitability": _row(0.28, 0.13, 0.09, 0.13, 0.06, 0.09),
        "liquidity": _row(1.8, 1.1, 0.4),
        "leverage": _row(0.6, 0.27, 2.0, 9.0),
        "efficiency": _row(0.7, 6.0, 7.0),
        "growth": _row(0.03, 0.04),
    },
    "Real Estate": {
        "valuation": _row(30.0, 6.0, 2.0, 18.0, 9.0, 0.038),
        "profitability": _row(0.65, 0.30, 0.20, 0.07, 0.03, 0.05),
        "liquidity": _row(None, None, None),
        "leverage": _row(1.1, 0.40, 6.0, 3.5),
        "efficiency": _row(0.10, None, 12.0),
        "growth": _row(0.05, 0.04),
    },
    "Communication Services": {
        "valuation": _row(18.0, 2.5, 3.0, 9.0, 3.0, 0.020),
        "profitability": _row(0.55, 0.18, 0.11, 0.14, 0.06, 0.09),
        "liquidity": _row(1.2, 1.0, 0.4),
        "leverage": _row(1.0, 0.35, 2.5, 7.0),
        "efficiency": _row(0.5, 20.0, 7.0),
        "growth": _row(0.05, 0.07),
    },
    "Utilities": {
        "valuation": _row(18.0, 2.4, 1.8, 11.0, 4.5, 0.033),
        "profitability": _row(0.45, 0.22, 0.13, 0.10, 0.03, 0.05),
        "liquidity": _row(0.8, 0.6, 0.05),
        "leverage": _row(1.5, 0.40, 5.5, 3.0),
        "efficiency": _row(0.25, 10.0, 8.0),
        "growth": _row(0.04, 0.05),
    },
}

_GENERIC: Dict[str, list] = {
    "valuation": _row(20.0, 2.5, 3.0, 12.0, 3.0, 0.018),
    "profitability": _row(0.40, 0.15, 0.10, 0.15, 0.07, 0.10),
    "liquidity": _row(1.5, 1.1, 0.5),
    "leverage": _row(0.8, 0.30, 2.5, 8.0),
    "efficiency": _row(0.8, 6.0, 8.0),
    "growth": _row(0.06, 0.07),
}


def _expand(rows: Dict[str, list]) -> Benchmarks:
    return {category: dict(zip(keys, rows[category])) for category, keys in BENCHMARK_KEYS.items()}


#: Canonical sector -> {category: {ratio: benchmark or None}}
SECTOR_BENCHMARKS: Dict[str, Benchmarks] = {sector: _expand(rows) for sector, rows in _TABLE.items()}

#: Benchmarks used for unknown sectors.
DEFAULT_BENCHMARKS: Benchmarks = _expand(_GENERIC)


def default_sector_benchmarks(sector: Optional[str]) -> Benchmarks:
    """Built-in benchmarks for a sector (any spelling); generic defaults for unknown sectors. Returns a copy."""
    return copy.deepcopy(SECTOR_BENCHMARKS.get(normalize_sector_name(sector), DEFAULT_BENCHMARKS))


def _merge(base: Benchmarks, override: Dict[str, Any]) -> Benchmarks:
    merged = copy.deepcopy(base)
    for category, ratios in (override or {}).items():
        if isinstance(ratios, dict):
            merged.setdefault(category, {}).update(ratios)
    return merged


class IndustryBenchmarks:
    """Sector / industry benchmark ratios and sector performance."""

    def __init__(self, data_dir: str = SECTOR_DATA_DIR, data_loader: Optional[Any] = None):
        """
        Args:
            data_dir: Directory with optional JSON/CSV overrides. Nothing is
                created or written on initialisation.
            data_loader: Optional ``DataLoader`` used by :meth:`get_sector_performance`
                to compute sector ETF returns.
        """
        self.data_dir = Path(data_dir)
        self.sector_mapping = SECTOR_MAPPING
        self.data_loader = data_loader
        self._overrides: Optional[Dict[str, Benchmarks]] = None

    @property
    def benchmarks(self) -> Dict[str, Benchmarks]:
        """Effective benchmarks for every canonical sector (defaults merged with overrides)."""
        return {sector: self.get_sector_benchmarks(sector) for sector in SECTORS}

    def get_sector_benchmarks(self, sector: Optional[str]) -> Benchmarks:
        """
        Benchmark ratios for a sector.

        Args:
            sector: Any sector spelling (``"Financial Services"``, ``"Information Technology"``...).

        Returns:
            ``{category: {ratio: value or None}}`` with the categories and keys of
            :data:`BENCHMARK_KEYS` (plus any keys added by overrides). Unknown
            sectors get :data:`DEFAULT_BENCHMARKS`. The dict is a copy.
        """
        canonical = normalize_sector_name(sector)
        base = SECTOR_BENCHMARKS.get(canonical, DEFAULT_BENCHMARKS)
        return _merge(base, self._load_overrides().get(canonical, {}))

    def get_industry_benchmarks(self, industry: str) -> Benchmarks:
        """Sector benchmarks of the industry's sector, refined by ``industry_<name>.json`` if present."""
        sector_benchmarks = self.get_sector_benchmarks(map_industry_to_sector(industry))
        return _merge(sector_benchmarks, self._get_industry_specific_benchmarks(industry))

    def get_sector_performance(self, data_loader: Optional[Any] = None) -> pd.DataFrame:
        """
        Recent sector performance (percent returns of the sector ETFs).

        Returns:
            DataFrame with columns ``Sector, 1D, 1W, 1M, 3M, 1Y, YTD, date, source``
            where ``source`` is ``"etf"`` (computed from prices via ``data_loader``),
            ``"file"`` (``sector_performance.csv``) or ``"sample"`` (illustrative
            static values used when nothing else is available).
        """
        loader = data_loader or self.data_loader
        if loader is not None:
            computed = self._compute_sector_performance(loader)
            if not computed.empty:
                return computed

        performance_file = self.data_dir / PERFORMANCE_FILE
        if performance_file.exists():
            try:
                df = pd.read_csv(performance_file)
                if "date" in df.columns and (pd.Timestamp.now() - pd.to_datetime(df["date"].max())).days <= 7:
                    df["source"] = "file"
                    return df
            except (OSError, ValueError) as e:
                logger.warning(f"Could not read {performance_file}: {e}")
        return self._get_sample_sector_performance()

    def update_benchmarks(self, sector: str, benchmarks: Dict[str, Dict[str, float]]) -> bool:
        """
        Override benchmarks for a sector and persist them (best effort).

        Returns:
            True when the override was stored in memory and written to disk.
        """
        canonical = normalize_sector_name(sector)
        if canonical == "Unknown" or not isinstance(benchmarks, dict):
            return False
        overrides = self._load_overrides()
        overrides[canonical] = _merge(overrides.get(canonical, {}), benchmarks)
        return self._save_benchmarks(overrides)

    def get_peer_companies(self, ticker: str, sector: str, max_peers: int = 5) -> list:
        """Peer tickers from the curated company universe."""
        from StockAnalysisSystem.src.industry.universe import get_sector_peers

        return [p["symbol"] for p in get_sector_peers(ticker, normalize_sector_name(sector), limit=max_peers)]

    # ------------------------------------------------------------------ helpers
    def _normalize_sector_name(self, sector: Optional[str]) -> str:
        return normalize_sector_name(sector)

    def _map_industry_to_sector(self, industry: Optional[str]) -> str:
        return map_industry_to_sector(industry)

    def _get_default_benchmarks(self) -> Benchmarks:
        return copy.deepcopy(DEFAULT_BENCHMARKS)

    def _get_default_sectors_benchmarks(self) -> Dict[str, Benchmarks]:
        return copy.deepcopy(SECTOR_BENCHMARKS)

    def _load_overrides(self) -> Dict[str, Benchmarks]:
        if self._overrides is None:
            self._overrides = {}
            benchmark_file = self.data_dir / BENCHMARKS_FILE
            if benchmark_file.exists():
                try:
                    data = json.loads(benchmark_file.read_text())
                    self._overrides = {normalize_sector_name(k): v for k, v in data.items() if isinstance(v, dict)}
                except (OSError, ValueError) as e:
                    logger.warning(f"Ignoring unreadable benchmark overrides {benchmark_file}: {e}")
        return self._overrides

    def _save_benchmarks(self, benchmarks: Optional[Dict[str, Benchmarks]] = None) -> bool:
        if benchmarks is not None:
            self._overrides = benchmarks
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / BENCHMARKS_FILE).write_text(json.dumps(self._overrides or {}, indent=2))
            return True
        except OSError as e:
            logger.warning(f"Could not save benchmark overrides: {e}")
            return False

    def _get_industry_specific_benchmarks(self, industry: str) -> Dict[str, Any]:
        industry_file = self.data_dir / f"industry_{str(industry).lower().replace(' ', '_')}.json"
        if not industry_file.exists():
            return {}
        try:
            return json.loads(industry_file.read_text())
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read industry benchmarks {industry_file}: {e}")
            return {}

    @staticmethod
    def _compute_sector_performance(data_loader: Any) -> pd.DataFrame:
        windows = {"1D": 1, "1W": 5, "1M": 21, "3M": 63, "1Y": 252}
        rows = []
        for sector, etf in SECTOR_ETFS.items():
            try:
                prices = data_loader.get_historical_prices(
                    etf, (pd.Timestamp.now() - pd.Timedelta(days=400)).strftime("%Y-%m-%d"))
                close = pd.to_numeric(prices["Close"], errors="coerce").dropna()
            except Exception as e:  # provider failures must not break the page
                logger.warning(f"No prices for {etf}: {e}")
                continue
            if len(close) < 2:
                continue
            last = float(close.iloc[-1])
            row: Dict[str, Any] = {"Sector": sector}
            for label, lag in windows.items():
                row[label] = (last / float(close.iloc[-1 - lag]) - 1) * 100 if len(close) > lag else None
            year_start = close[close.index < pd.Timestamp(year=close.index[-1].year, month=1, day=1)]
            row["YTD"] = (last / float(year_start.iloc[-1]) - 1) * 100 if not year_start.empty else None
            row["date"] = close.index[-1].strftime("%Y-%m-%d")
            row["source"] = "etf"
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _get_sample_sector_performance() -> pd.DataFrame:
        """Illustrative static values (``source == "sample"``), used when no data is available."""
        df = pd.DataFrame({
            "Sector": SECTORS,
            "1D": [0.5, -0.2, 0.3, -0.1, 0.2, 1.2, 0.4, 0.8, -0.5, 0.1, -0.3],
            "1W": [2.1, 1.5, -0.8, 1.2, 0.5, 3.2, 1.1, 2.3, -1.2, 0.7, -0.9],
            "1M": [5.3, 3.2, 2.1, 4.5, 1.8, 7.5, 3.5, 4.8, -3.5, 2.2, 1.5],
            "3M": [12.5, 7.8, 5.2, 10.3, 4.5, 15.2, 8.2, 9.5, -5.2, 6.5, 3.8],
            "1Y": [25.3, 15.2, 10.5, 18.7, 8.2, -12.5, 14.5, 17.2, -8.5, 12.8, 7.5],
            "YTD": [15.2, 8.5, 6.2, 11.2, 5.3, 9.8, 7.5, 10.2, -4.8, 7.2, 4.5],
        })
        df["date"] = pd.Timestamp.now().strftime("%Y-%m-%d")
        df["source"] = "sample"
        return df

# ---------------------------------------------------------------------------
# Module-level helpers (stable public API used by the UI pages)
# ---------------------------------------------------------------------------
def get_sector_benchmarks(sector: str) -> Dict[str, Any]:
    """
    Benchmark data for a sector.

    Returns:
        Dict with ``sector`` (canonical name), ``etf`` (benchmark ETF ticker or
        None) and the ratio benchmarks by category from
        :meth:`IndustryBenchmarks.get_sector_benchmarks` under ``ratios``.
        Empty dict for unknown sectors.
    """
    from StockAnalysisSystem.src.industry.universe import get_sector_etf

    canonical = normalize_sector(sector)
    if canonical == "Unknown":
        return {}
    try:
        ratios = IndustryBenchmarks().get_sector_benchmarks(canonical)
    except Exception as e:  # benchmarks are optional enrichment
        logger.warning(f"Could not load ratio benchmarks for {canonical}: {e}")
        ratios = {}
    return {"sector": canonical, "etf": get_sector_etf(canonical), "ratios": ratios}
