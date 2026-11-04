"""
Reports page.

Entry point: :func:`run_reports_page` (no arguments).

Builds company, valuation, comparison and sector reports with
:class:`~StockAnalysisSystem.src.utils.export.ReportGenerator` and offers
them for download as Excel, CSV, JSON, PDF and HTML. Generated reports
(including the rendered files) are kept in ``st.session_state`` per report
type, so they persist while other widgets change.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd
import streamlit as st

from StockAnalysisSystem.src.config import SECTORS, normalize_sector
from StockAnalysisSystem.src.industry.universe import (
    MARKET_BENCHMARK,
    UNIVERSE,
    get_sector_companies,
    get_sector_etf,
    lookup_company,
)
from StockAnalysisSystem.src.navigation import get_active_ticker
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.export import (
    EXPORT_FORMATS,
    Report,
    ReportGenerator,
    available_formats,
)
from StockAnalysisSystem.src.utils.financial_data import build_financial_data, is_empty

logger = logging.getLogger(__name__)

RESULTS_KEY = "reports_results"
DEFAULT_TICKER = "AAPL"
MAX_COMPARISON = 8

#: Report type -> description
REPORT_TYPES: Dict[str, str] = {
    "Company": "Overview, key metrics, price performance, financial statements, ratios, risk and valuation "
               "for one company.",
    "Valuation": "Sector-specific valuation with every method, assumptions, method details and the DCF "
                 "cash-flow forecast.",
    "Comparison": "Side-by-side fundamentals, price performance and valuation for up to "
                  f"{MAX_COMPARISON} companies.",
    "Sector": "Sector ETF performance against the market plus fundamentals of the sector's largest companies.",
}

FORMAT_ORDER = ["pdf", "excel", "csv", "json", "html"]
_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-^=]{0,14}$")


class _NoData(Exception):
    """Raised inside cached loaders so empty results are not cached."""


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_data_loader() -> DataLoader:
    return DataLoader()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    prices = _get_data_loader().get_historical_prices(ticker, start, end)
    if prices is None or prices.empty:
        raise _NoData(ticker)
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_statement(ticker: str, statement_type: str) -> pd.DataFrame:
    frame = _get_data_loader().get_financial_statements(ticker, statement_type, "annual")
    if frame is None or frame.empty:
        raise _NoData(ticker)
    return frame


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_info(ticker: str) -> Dict[str, Any]:
    info = _get_data_loader().get_company_info(ticker)
    if not info:
        raise _NoData(ticker)
    return dict(info)


def _guarded(loader: Callable[..., Any], empty: Any, *args: Any) -> Any:
    try:
        return loader(*args)
    except _NoData:
        return empty
    except Exception:
        logger.exception("Data loading failed: %s%s", getattr(loader, "__name__", loader), args)
        return empty


def load_prices(ticker: str, years: float = 1.0) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=int(365.25 * years))
    return _guarded(_cached_prices, pd.DataFrame(), ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def load_company(ticker: str, years: float = 1.0) -> Dict[str, Any]:
    """``{"ticker", "company_info", "price_data", "financial_data"}`` (pieces may be empty)."""
    info = _guarded(_cached_info, {}, ticker)
    prices = load_prices(ticker, years)
    statements = {key: _guarded(_cached_statement, pd.DataFrame(), ticker, stype)
                  for key, stype in (("income_statement", "income"), ("balance_sheet", "balance"),
                                     ("cash_flow", "cash"))}
    if not info:
        record = lookup_company(ticker)
        if record:
            info = {"symbol": record["symbol"], "name": record["name"], "sector": record["sector"],
                    "industry": record["subsector"]}
    return {"ticker": ticker, "company_info": info, "price_data": prices,
            "financial_data": build_financial_data(company_info=info, price_data=prices, **statements)}


def has_data(bundle: Mapping[str, Any]) -> bool:
    financial = bundle.get("financial_data") or {}
    return not (is_empty(bundle.get("price_data")) and all(
        is_empty(financial.get(k)) for k in ("income_statement", "balance_sheet", "cash_flow")))


# ---------------------------------------------------------------------------
# Analysis helpers (all defensive: failures are logged and skipped)
# ---------------------------------------------------------------------------
def run_valuation(bundle: Mapping[str, Any], **overrides: Any) -> Optional[Dict[str, Any]]:
    try:
        from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory

        result = ValuationFactory(_get_data_loader()).get_company_valuation(
            bundle["ticker"], financial_data=bundle["financial_data"], **overrides)
        return dict(result) if isinstance(result, Mapping) else None
    except Exception:
        logger.exception("Valuation failed for %s", bundle.get("ticker"))
        return None


def run_ratios(bundle: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

        ratios = FinancialRatioAnalyzer().calculate_ratios(bundle["financial_data"])
        return dict(ratios) if isinstance(ratios, Mapping) else None
    except Exception:
        logger.exception("Ratio analysis failed for %s", bundle.get("ticker"))
        return None


def run_risk(bundle: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer

        sector = (bundle.get("company_info") or {}).get("sector")
        risk = BankruptcyAnalyzer().get_comprehensive_risk_assessment(bundle["financial_data"], sector)
        return dict(risk) if isinstance(risk, Mapping) and not risk.get("error") else None
    except Exception:
        logger.exception("Risk analysis failed for %s", bundle.get("ticker"))
        return None


# ---------------------------------------------------------------------------
# Report builders (data loading + ReportGenerator)
# ---------------------------------------------------------------------------
def build_company_report(ticker: str, years: int = 3, include_valuation: bool = True, include_ratios: bool = True,
                         include_risk: bool = True, include_statements: bool = True,
                         include_charts: bool = True) -> Report:
    bundle = load_company(ticker, years)
    if not has_data(bundle):
        raise LookupError(f"No data is available for {ticker}.")
    return ReportGenerator().company_report(
        ticker, bundle["company_info"], bundle["financial_data"], bundle["price_data"],
        valuation=run_valuation(bundle) if include_valuation else None,
        ratios=run_ratios(bundle) if include_ratios else None,
        risk=run_risk(bundle) if include_risk else None,
        include_statements=include_statements, include_charts=include_charts)


def build_valuation_report(ticker: str, overrides: Optional[Mapping[str, Any]] = None,
                           include_charts: bool = True) -> Report:
    bundle = load_company(ticker, 1)
    if not has_data(bundle):
        raise LookupError(f"No data is available for {ticker}.")
    valuation = run_valuation(bundle, **dict(overrides or {}))
    if not valuation:
        raise RuntimeError(f"The valuation of {ticker} could not be completed.")
    return ReportGenerator().valuation_report(ticker, valuation, bundle["company_info"], bundle["price_data"],
                                              financial_data=bundle["financial_data"],
                                              include_charts=include_charts)


def build_comparison_report(tickers: List[str], include_valuation: bool = True,
                            include_charts: bool = True) -> Report:
    companies: Dict[str, Dict[str, Any]] = {}
    missing = []
    for ticker in tickers:
        bundle = load_company(ticker, 1)
        if not has_data(bundle):
            missing.append(ticker)
            continue
        if include_valuation:
            bundle["valuation"] = run_valuation(bundle)
        companies[ticker] = bundle
    if len(companies) < 2:
        raise LookupError("At least two companies with available data are needed"
                          + (f" (no data for {', '.join(missing)})." if missing else "."))
    report = ReportGenerator().comparison_report(companies, include_charts=include_charts)
    if missing:
        report.metadata["missing"] = missing
    return report


def build_sector_report(sector: str, count: int = 6, include_valuation: bool = False,
                        include_charts: bool = True) -> Report:
    sector = normalize_sector(sector)
    records = get_sector_companies(sector)[:count]
    if not records:
        raise LookupError(f"No companies are listed for the {sector} sector.")
    companies = []
    for record in records:
        bundle = load_company(record["symbol"], 1)
        entry = {**record, **bundle}
        if include_valuation and has_data(bundle):
            entry["valuation"] = run_valuation(bundle)
        companies.append(entry)
    etf = get_sector_etf(sector)
    return ReportGenerator().sector_report(
        sector, companies, etf_ticker=etf, etf_prices=load_prices(etf, 1) if etf else None,
        benchmark_ticker=MARKET_BENCHMARK, benchmark_prices=load_prices(MARKET_BENCHMARK, 1),
        include_charts=include_charts)


def render_report_files(report: Report) -> Dict[str, Any]:
    """Render every available format; returns ``{"files": {fmt: bytes}, "errors": {fmt: message}}``."""
    files, errors = report.render_all(available_formats())
    for fmt in FORMAT_ORDER:
        if fmt not in files and fmt not in errors:
            errors[fmt] = "not available in this installation"
    return {"files": files, "errors": errors}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def _results() -> Dict[str, Dict[str, Any]]:
    if RESULTS_KEY not in st.session_state:
        st.session_state[RESULTS_KEY] = {}
    return st.session_state[RESULTS_KEY]


def _ticker_input(key: str, label: str = "Ticker symbol") -> str:
    if key not in st.session_state:
        st.session_state[key] = get_active_ticker(DEFAULT_TICKER)
    raw = st.text_input(label, key=key, max_chars=15)
    ticker = str(raw or "").strip().upper()
    if ticker and not _TICKER_PATTERN.match(ticker):
        st.warning(f"'{raw}' does not look like a valid ticker symbol.")
        return ""
    return ticker


def _generate(report_type: str, params: Mapping[str, Any], builder: Callable[[], Report]) -> None:
    """Build and render a report, storing the outcome (or a friendly error) in session state."""
    with st.spinner("Generating report ..."):
        try:
            report = builder()
            outcome = {"report": report, "params": dict(params), **render_report_files(report)}
        except (LookupError, RuntimeError) as exc:
            outcome = {"error": str(exc), "params": dict(params)}
        except Exception:
            logger.exception("%s report generation failed", report_type)
            outcome = {"error": "The report could not be generated because of an unexpected error.",
                       "params": dict(params)}
    _results()[report_type] = outcome


def _company_form() -> None:
    ticker = _ticker_input("reports_company_ticker")
    years = st.select_slider("Price history (years)", options=[1, 2, 3, 5], value=3, key="reports_company_years")
    cols = st.columns(5)
    options = {
        "valuation": cols[0].checkbox("Valuation", value=True, key="reports_company_inc_valuation"),
        "ratios": cols[1].checkbox("Ratios", value=True, key="reports_company_inc_ratios"),
        "risk": cols[2].checkbox("Risk", value=True, key="reports_company_inc_risk"),
        "statements": cols[3].checkbox("Statements", value=True, key="reports_company_inc_statements"),
        "charts": cols[4].checkbox("Charts", value=True, key="reports_company_inc_charts"),
    }
    params = {"ticker": ticker, "years": years, **options}
    if st.button("Generate company report", type="primary", key="reports_company_generate", disabled=not ticker):
        _generate("Company", params, lambda: build_company_report(
            ticker, years, options["valuation"], options["ratios"], options["risk"], options["statements"],
            options["charts"]))
    _render_result("Company", params)


def _valuation_form() -> None:
    ticker = _ticker_input("reports_valuation_ticker")
    override = st.toggle("Override DCF assumptions", value=False, key="reports_valuation_override")
    overrides: Dict[str, Any] = {}
    if override:
        cols = st.columns(3)
        overrides["discount_rate"] = cols[0].slider("Discount rate (%)", 5.0, 20.0, 10.0, 0.5,
                                                    key="reports_valuation_discount") / 100
        overrides["terminal_growth"] = cols[1].slider("Terminal growth (%)", 0.0, 5.0, 2.5, 0.1,
                                                      key="reports_valuation_terminal") / 100
        overrides["forecast_years"] = cols[2].slider("Forecast years", 3, 10, 5,
                                                     key="reports_valuation_years")
    charts = st.checkbox("Include charts", value=True, key="reports_valuation_charts")
    params = {"ticker": ticker, "overrides": overrides, "charts": charts}
    if st.button("Generate valuation report", type="primary", key="reports_valuation_generate", disabled=not ticker):
        _generate("Valuation", params, lambda: build_valuation_report(ticker, overrides, charts))
    _render_result("Valuation", params)


def _comparison_form() -> None:
    symbols = [c["symbol"] for c in UNIVERSE]
    names = {c["symbol"]: c["name"] for c in UNIVERSE}
    if "reports_comparison_tickers" not in st.session_state:
        active = get_active_ticker(DEFAULT_TICKER)
        default = [active] if active in symbols else []
        default += [s for s in ("AAPL", "MSFT", "GOOGL") if s not in default]
        st.session_state["reports_comparison_tickers"] = default[:3]
    selected = st.multiselect("Companies", options=symbols, key="reports_comparison_tickers",
                              format_func=lambda s: f"{s} - {names.get(s, s)}",
                              max_selections=MAX_COMPARISON)
    extra_raw = st.text_input("Other tickers (comma separated)", key="reports_comparison_extra",
                              placeholder="e.g. SHOP, ASML")
    extra = [t.strip().upper() for t in str(extra_raw or "").split(",") if t.strip()]
    invalid = [t for t in extra if not _TICKER_PATTERN.match(t)]
    if invalid:
        st.warning("Ignoring invalid tickers: " + ", ".join(invalid))
    tickers = list(dict.fromkeys(list(selected) + [t for t in extra if t not in invalid]))[:MAX_COMPARISON]
    cols = st.columns(2)
    valuation = cols[0].checkbox("Include valuation", value=True, key="reports_comparison_valuation")
    charts = cols[1].checkbox("Include charts", value=True, key="reports_comparison_charts")
    params = {"tickers": tickers, "valuation": valuation, "charts": charts}
    if len(tickers) < 2:
        st.caption("Select at least two companies.")
    if st.button("Generate comparison report", type="primary", key="reports_comparison_generate",
                 disabled=len(tickers) < 2):
        _generate("Comparison", params, lambda: build_comparison_report(tickers, valuation, charts))
    _render_result("Comparison", params)


def _sector_form() -> None:
    sector = st.selectbox("Sector", SECTORS, key="reports_sector_name")
    available = len(get_sector_companies(sector))
    count = st.slider("Companies", min_value=2, max_value=max(available, 2), value=min(6, max(available, 2)),
                      key="reports_sector_count")
    cols = st.columns(2)
    valuation = cols[0].checkbox("Include valuations (slower)", value=False, key="reports_sector_valuation")
    charts = cols[1].checkbox("Include charts", value=True, key="reports_sector_charts")
    params = {"sector": sector, "count": count, "valuation": valuation, "charts": charts}
    if st.button("Generate sector report", type="primary", key="reports_sector_generate"):
        _generate("Sector", params, lambda: build_sector_report(sector, count, valuation, charts))
    _render_result("Sector", params)


def _render_result(report_type: str, current_params: Mapping[str, Any]) -> None:
    outcome = _results().get(report_type)
    if not outcome:
        return
    prefix = f"reports_{report_type.lower()}"
    if dict(outcome.get("params") or {}) != dict(current_params):
        st.info("Showing the last generated report. Click the generate button to apply the current settings.")
    if outcome.get("error"):
        st.warning(outcome["error"])
        return
    report: Report = outcome["report"]
    files: Dict[str, bytes] = outcome.get("files") or {}
    st.success(f"Report ready: {report.title}")
    if report.metadata.get("missing"):
        st.caption("No data for: " + ", ".join(report.metadata["missing"]))
    formats = [f for f in FORMAT_ORDER if f in files]
    columns = st.columns(max(len(formats), 1) + 1)
    generator = ReportGenerator()
    for column, fmt in zip(columns, formats):
        spec = EXPORT_FORMATS[fmt]
        column.download_button(spec.label, data=files[fmt], file_name=generator.file_name(report, fmt),
                               mime=spec.mime, key=f"{prefix}_download_{fmt}", on_click="ignore",
                               icon=":material/download:")
    if columns[-1].button("Clear", key=f"{prefix}_clear"):
        _results().pop(report_type, None)
        st.rerun()
    unavailable = {f: e for f, e in (outcome.get("errors") or {}).items()}
    if unavailable:
        st.caption("Unavailable formats: " + "; ".join(f"{EXPORT_FORMATS[f].label} ({e})"
                                                       for f, e in unavailable.items()))
    with st.expander("Preview", expanded=True):
        _preview(report, prefix)


def _preview(report: Report, prefix: str) -> None:
    st.markdown(f"### {report.title}")
    if report.subtitle:
        st.caption(report.subtitle)
    for i, section in enumerate(report.sections):
        st.markdown(f"#### {section.title}")
        for text in section.text:
            st.markdown(text)
        if section.metrics:
            metrics = section.formatted_metrics()
            st.dataframe(pd.DataFrame({"Metric": list(metrics), "Value": list(metrics.values())}),
                         hide_index=True, key=f"{prefix}_preview_metrics_{i}")
        table = section.display_table()
        if table is not None:
            st.dataframe(table, hide_index=not section.table_index, key=f"{prefix}_preview_table_{i}")
        if section.image:
            st.image(section.image, caption=section.image_caption or None)
    if report.appendix:
        st.caption("Also included in the Excel, CSV and JSON downloads: " + ", ".join(report.appendix))
    st.caption(report.disclaimer)


def run_reports_page() -> None:
    """Render the reports page."""
    st.title("Reports")
    st.markdown("Generate analysis reports and download them as PDF, Excel, CSV, JSON or HTML.")
    report_type = st.radio("Report type", list(REPORT_TYPES), horizontal=True, key="reports_type")
    st.caption(REPORT_TYPES[report_type])
    forms = {"Company": _company_form, "Valuation": _valuation_form, "Comparison": _comparison_form,
             "Sector": _sector_form}
    try:
        forms[report_type]()
    except Exception:  # never show a traceback to users
        logger.exception("Reports page failed")
        st.error("The reports page encountered an unexpected error. Please try different settings.")


if __name__ == "__main__":  # pragma: no cover
    run_reports_page()
