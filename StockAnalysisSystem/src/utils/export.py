"""
Export and report generation.

Two layers are provided:

* **In-memory serialisers** returning ``bytes`` (ready for
  ``st.download_button``): :func:`to_excel_bytes`, :func:`to_csv_bytes`,
  :func:`to_json_bytes`, :func:`to_pdf_bytes` and :func:`to_html_bytes`.
  They accept DataFrames, dictionaries (including nested dicts, numpy and
  pandas scalars, NaN/inf and timestamps) or a :class:`Report`.
* **Reports**: :class:`ReportGenerator` assembles :class:`Report` documents
  (company, valuation, comparison and sector reports) from the canonical
  ``financial_data`` bundle, price history and the standardised valuation
  result of ``ValuationFactory.get_company_valuation``. A report renders to
  every format with :meth:`Report.render`::

      generator = ReportGenerator()
      report = generator.company_report("AAPL", company_info, financial_data, price_data,
                                        valuation=valuation)
      pdf_bytes = report.render("pdf")          # or "excel", "csv", "json", "html"
      file_name = generator.file_name(report, "pdf")

:class:`ExportManager` keeps the older file-writing API (``export_to_pdf``,
``export_to_excel``, ``export_to_csv``, ``export_to_json``). Files are written
to :data:`DEFAULT_EXPORT_DIR` (``src/data/exports``, git-ignored, overridable
with the ``STOCK_ANALYSIS_EXPORT_DIR`` environment variable) unless an
explicit directory is given.

PDF output uses fpdf2 with the built-in core fonts only; text is converted to
Latin-1 (typographic punctuation is mapped to ASCII, other characters become
``?``) so PDF generation never fails on non-Latin text.
"""

from __future__ import annotations

import base64
import dataclasses
import html
import io
import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from StockAnalysisSystem.src.config import DATA_DIR
from StockAnalysisSystem.src.utils.financial_data import (
    cagr,
    format_large_number,
    get_latest,
    get_series,
    get_value,
    is_empty,
    safe_divide,
    to_float,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the environment
    import xlsxwriter  # noqa: F401

    _EXCEL_ENGINE: Optional[str] = "xlsxwriter"
except ImportError:  # pragma: no cover
    try:
        import openpyxl  # noqa: F401

        _EXCEL_ENGINE = "openpyxl"
    except ImportError:
        _EXCEL_ENGINE = None

try:  # pragma: no cover - depends on the environment
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    FPDF = None  # type: ignore[assignment,misc]
    PDF_AVAILABLE = False

try:  # pragma: no cover - depends on the environment
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    CHARTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHARTS_AVAILABLE = False

EXCEL_AVAILABLE = _EXCEL_ENGINE is not None
#: Kept for backward compatibility; PDF export uses fpdf2, not ReportLab.
REPORTLAB_AVAILABLE = False

#: Default directory for files written by :class:`ExportManager` (git-ignored).
DEFAULT_EXPORT_DIR = Path(os.environ.get("STOCK_ANALYSIS_EXPORT_DIR") or (DATA_DIR / "exports"))


class ExportError(RuntimeError):
    """Raised when a document cannot be produced in the requested format."""


@dataclass(frozen=True)
class ExportFormat:
    key: str
    label: str
    extension: str
    mime: str


#: Supported output formats keyed by format key.
EXPORT_FORMATS: Dict[str, ExportFormat] = {
    "excel": ExportFormat("excel", "Excel", "xlsx",
                          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "csv": ExportFormat("csv", "CSV", "csv", "text/csv"),
    "json": ExportFormat("json", "JSON", "json", "application/json"),
    "pdf": ExportFormat("pdf", "PDF", "pdf", "application/pdf"),
    "html": ExportFormat("html", "HTML", "html", "text/html"),
}

_FORMAT_ALIASES = {"xlsx": "excel", "xls": "excel", "htm": "html"}


def normalize_format(fmt: str) -> str:
    """Return the canonical format key for ``fmt`` (e.g. ``"xlsx"`` -> ``"excel"``)."""
    key = str(fmt or "").strip().lower().lstrip(".")
    key = _FORMAT_ALIASES.get(key, key)
    if key not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format: {fmt!r}")
    return key


def available_formats() -> List[str]:
    """Format keys usable in this environment."""
    formats = ["csv", "json", "html"]
    if EXCEL_AVAILABLE:
        formats.insert(0, "excel")
    if PDF_AVAILABLE:
        formats.append("pdf")
    return formats


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------
#: Format kinds understood by :func:`format_value`.
FORMAT_KINDS = ("auto", "money", "count", "price", "pct", "pct_points", "ratio", "multiple", "int", "number", "text")


def format_value(value: Any, kind: str = "auto") -> str:
    """
    Human readable representation of ``value``.

    Kinds: ``money`` ($2.76T), ``count`` (15.12B), ``price`` ($184.45), ``pct`` (fraction ->
    ``12.3%``), ``pct_points`` (already in percent), ``ratio`` (1.23),
    ``multiple`` (12.3x), ``int`` (1,234), ``number``/``auto`` and ``text``.
    Missing, NaN and infinite values become ``"N/A"``.
    """
    if value is None:
        return "N/A"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return _date_label(value)
    if isinstance(value, str):
        return value
    if kind == "text":
        return str(value)
    number = to_float(value) if isinstance(value, (int, float, np.integer, np.floating, Decimal)) else None
    if number is None:
        if isinstance(value, (int, float, np.integer, np.floating, Decimal)):
            return "N/A"  # NaN / inf
        try:
            if pd.isna(value):
                return "N/A"
        except (TypeError, ValueError):
            pass
        return str(value)
    if kind == "money":
        return format_large_number(number)
    if kind == "count":
        return format_large_number(number, prefix="")
    if kind == "price":
        return f"${number:,.2f}"
    if kind == "pct":
        return f"{number * 100:.1f}%"
    if kind == "pct_points":
        return f"{number:.1f}%"
    if kind == "ratio":
        return f"{number:.2f}"
    if kind == "multiple":
        return f"{number:.1f}x"
    if kind == "int":
        return f"{number:,.0f}"
    if abs(number) >= 1e6:
        return format_large_number(number, prefix="")
    if number == int(number) and abs(number) < 1e6 and isinstance(value, (int, np.integer)):
        return f"{int(number):,}"
    return f"{number:,.2f}"


def _date_label(value: Any) -> str:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if ts is pd.NaT:
        return "N/A"
    if ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        return ts.strftime("%Y-%m-%d")
    return ts.strftime("%Y-%m-%d %H:%M")


def _label(value: Any) -> str:
    """String label for dict keys, column and index labels."""
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return _date_label(value)
    if isinstance(value, tuple):
        return " | ".join(_label(v) for v in value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def humanize(key: Any) -> str:
    """``"value_per_share"`` -> ``"Value Per Share"`` (acronyms kept)."""
    text = _label(key)
    if not re.search(r"[_]", text) and not text.islower():
        return text
    words = text.replace("_", " ").split()
    keep = {"dcf", "ddm", "eps", "fcf", "ev", "pe", "roe", "roa", "roic", "ebitda", "ebit", "nav", "ffo", "affo",
            "wacc", "cagr", "p/e", "p/b", "p/s"}
    return " ".join(w.upper() if w.lower() in keep else w.capitalize() for w in words)


def guess_kind(key: Any, value: Any = None) -> str:
    """Heuristic format kind for a metric name."""
    name = _label(key).lower()
    number = to_float(value) if isinstance(value, (int, float, np.integer, np.floating)) else None
    if "upside" in name or "pct" in name or name.endswith("%") or "(%)" in name:
        return "pct_points"
    if any(t in name for t in ("rate", "growth", "margin", "yield", "return", "ratio_pct", "weight", "volatility",
                               "drawdown", "roe", "roa", "roic", "safety")):
        if number is None or abs(number) <= 5:
            return "pct"
    if any(t in name for t in ("price", "per_share", "per share", "value_per", "eps", "fair value", "target")):
        return "price"
    if any(t in name for t in ("p/e", "p/b", "p/s", "ev/", "multiple", "pe_ratio", "pb_ratio")):
        return "multiple"
    if number is not None and abs(number) >= 1e6:
        return "money"
    return "auto"


# ---------------------------------------------------------------------------
# JSON-safe conversion
# ---------------------------------------------------------------------------
def to_serializable(obj: Any, _depth: int = 0) -> Any:
    """
    Convert ``obj`` into plain JSON-compatible Python objects.

    numpy/pandas scalars become Python numbers, NaN/inf/NaT/NA become ``None``,
    timestamps become ISO strings, DataFrames become ``{"columns", "index",
    "data"}`` (``pd.DataFrame(**result)`` rebuilds them), Series and dicts
    become dicts with string keys, binary data becomes a short placeholder.
    Unknown objects fall back to ``str(obj)``.
    """
    if _depth > 50:
        return str(obj)
    nxt = _depth + 1
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating, Decimal)):
        number = float(obj)
        return number if math.isfinite(number) else None
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return None if obj is pd.NaT else obj.isoformat()
    if isinstance(obj, (pd.Timedelta, timedelta, pd.Period)):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        frame = obj.replace([np.inf, -np.inf], np.nan)
        return {
            "columns": [_label(c) for c in frame.columns],
            "index": [to_serializable(i, nxt) if not isinstance(i, tuple) else _label(i) for i in frame.index],
            "data": [[to_serializable(v, nxt) for v in row] for row in frame.itertuples(index=False, name=None)],
        }
    if isinstance(obj, pd.Series):
        return {_label(k): to_serializable(v, nxt) for k, v in obj.items()}
    if isinstance(obj, pd.Index):
        return [to_serializable(v, nxt) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_serializable(v, nxt) for v in obj.tolist()]
    if isinstance(obj, Mapping):
        return {_label(k): to_serializable(v, nxt) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj, key=str) if isinstance(obj, (set, frozenset)) else obj
        return [to_serializable(v, nxt) for v in items]
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<binary data: {len(obj)} bytes>"
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, complex):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_serializable(getattr(obj, f.name), nxt) for f in dataclasses.fields(obj)}
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return to_serializable(obj.to_dict(), nxt)
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_serializable(obj.item(), nxt)
        except Exception:  # pragma: no cover - defensive
            pass
    return str(obj)


def to_json_bytes(data: Any, indent: Optional[int] = 2) -> bytes:
    """Serialise ``data`` (anything :func:`to_serializable` handles, or a Report) to UTF-8 JSON."""
    if isinstance(data, Report):
        data = data.to_dict()
    payload = to_serializable(data)
    return json.dumps(payload, indent=indent, allow_nan=False, ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Tabular conversion (CSV / Excel)
# ---------------------------------------------------------------------------
def flatten_dict(data: Any, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested mappings/sequences/DataFrames into ``{"a.b.c": value}``.

    DataFrames contribute ``"<key>.<row>.<column>"`` entries, Series
    ``"<key>.<index>"`` entries.
    """
    items: Dict[str, Any] = {}

    def _walk(value: Any, key: str) -> None:
        if isinstance(value, pd.DataFrame):
            for row_label, row in value.iterrows():
                for col_label, cell in row.items():
                    _walk(cell, f"{key}{sep}{_label(row_label)}{sep}{_label(col_label)}" if key else
                          f"{_label(row_label)}{sep}{_label(col_label)}")
        elif isinstance(value, pd.Series):
            for idx, cell in value.items():
                _walk(cell, f"{key}{sep}{_label(idx)}" if key else _label(idx))
        elif isinstance(value, Mapping):
            if not value and key:
                items[key] = None
            for k, v in value.items():
                _walk(v, f"{key}{sep}{_label(k)}" if key else _label(k))
        elif isinstance(value, (list, tuple)) and any(isinstance(v, (Mapping, list, tuple, pd.Series,
                                                                     pd.DataFrame)) for v in value):
            for i, v in enumerate(value):
                _walk(v, f"{key}{sep}{i}" if key else str(i))
        else:
            items[key or "value"] = to_serializable(value)

    _walk(data, parent_key)
    return items


def _cell_value(value: Any) -> Any:
    """Scalar suitable for a spreadsheet cell / CSV field."""
    if isinstance(value, (Mapping, list, tuple, set, np.ndarray, pd.Series, pd.DataFrame)):
        return json.dumps(to_serializable(value), ensure_ascii=False, allow_nan=False)
    if isinstance(value, (bytes, bytearray)):
        return f"<binary data: {len(value)} bytes>"
    if isinstance(value, pd.Timestamp):
        return value.tz_localize(None) if value.tzinfo is not None else value
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.replace(tzinfo=None)
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, str) and len(value) > 32000:
        return value[:32000] + "..."
    return value


def tabular_frame(data: Any) -> pd.DataFrame:
    """
    Convert ``data`` to a DataFrame suitable for CSV/Excel output.

    DataFrames are cleaned (labels to strings, inf to NaN, nested objects to
    JSON strings, time zones dropped). Lists of dicts become rows. Any other
    mapping becomes a two-column ``field``/``value`` table of the flattened
    keys.
    """
    if isinstance(data, Report):
        return data.to_frame()
    if isinstance(data, pd.Series):
        data = data.to_frame(name=_label(data.name) if data.name is not None else "value")
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        if isinstance(frame.columns, pd.MultiIndex) or not all(isinstance(c, str) for c in frame.columns):
            frame.columns = [_label(c) for c in frame.columns]
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame.index = frame.index.tz_localize(None)
        elif isinstance(frame.index, pd.MultiIndex):
            frame.index = [_label(i) for i in frame.index]
        for col in frame.columns:
            series = frame[col]
            if isinstance(series.dtype, pd.DatetimeTZDtype):
                frame[col] = series.dt.tz_localize(None)
            elif pd.api.types.is_float_dtype(series.dtype):
                frame[col] = series.replace([np.inf, -np.inf], np.nan)
            elif series.dtype == object:
                frame[col] = series.map(_cell_value)
        return frame
    if isinstance(data, (list, tuple)) and data and all(isinstance(r, Mapping) for r in data):
        return tabular_frame(pd.DataFrame([{k: _cell_value(v) for k, v in r.items()} for r in data]))
    flat = flatten_dict(data)
    return pd.DataFrame({"field": list(flat.keys()), "value": [_cell_value(v) for v in flat.values()]})


def _keep_index(frame: pd.DataFrame) -> bool:
    """Write the index only when it carries labels (not a default RangeIndex)."""
    return not (isinstance(frame.index, pd.RangeIndex) and frame.index.name is None)


def to_csv_bytes(data: Any) -> bytes:
    """Serialise a DataFrame, mapping or Report to UTF-8 CSV."""
    frame = tabular_frame(data)
    return frame.to_csv(index=_keep_index(frame)).encode("utf-8")


_INVALID_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


def _sheet_name(name: Any, used: Iterable[str]) -> str:
    base = _INVALID_SHEET_CHARS.sub(" ", _label(name)).strip().strip("'") or "Sheet"
    base = base[:31]
    used_lower = {u.lower() for u in used}
    candidate, counter = base, 2
    while candidate.lower() in used_lower:
        suffix = f" ({counter})"
        candidate = base[:31 - len(suffix)] + suffix
        counter += 1
    return candidate


def _as_sheets(data: Any) -> Dict[str, pd.DataFrame]:
    if isinstance(data, Report):
        return data.tables()
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return {"Data": tabular_frame(data)}
    if isinstance(data, Mapping):
        sheets: Dict[str, pd.DataFrame] = {}
        scalars: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                if not value.empty:
                    sheets[_label(key)] = tabular_frame(value)
            elif isinstance(value, (list, tuple)) and value and all(isinstance(r, Mapping) for r in value):
                sheets[_label(key)] = tabular_frame(list(value))
            else:
                scalars[_label(key)] = value
        if scalars:
            sheets = {"Summary": tabular_frame(scalars), **sheets}
        return sheets or {"Data": pd.DataFrame({"field": [], "value": []})}
    return {"Data": tabular_frame(data)}


def to_excel_bytes(data: Any) -> bytes:
    """
    Serialise data to an ``.xlsx`` workbook.

    ``data`` may be a DataFrame (one sheet), a mapping of sheet name ->
    DataFrame (scalar entries are collected in a ``Summary`` sheet) or a
    :class:`Report`.
    """
    if not EXCEL_AVAILABLE:
        raise ExportError("Excel export requires xlsxwriter or openpyxl.")
    sheets = _as_sheets(data)
    buffer = io.BytesIO()
    engine_kwargs = {"options": {"strings_to_formulas": False, "strings_to_urls": False}} \
        if _EXCEL_ENGINE == "xlsxwriter" else {}
    with pd.ExcelWriter(buffer, engine=_EXCEL_ENGINE, engine_kwargs=engine_kwargs) as writer:
        used: List[str] = []
        for name, frame in sheets.items():
            sheet = _sheet_name(name, used)
            used.append(sheet)
            frame = tabular_frame(frame)
            if _EXCEL_ENGINE == "openpyxl":
                frame = frame.map(lambda v: "'" + v if isinstance(v, str) and v.startswith("=") else v)
            index = _keep_index(frame)
            frame.to_excel(writer, sheet_name=sheet, index=index)
            _autofit(writer, sheet, frame, index)
    return buffer.getvalue()


def _autofit(writer: pd.ExcelWriter, sheet: str, frame: pd.DataFrame, index: bool) -> None:
    try:
        labels = ([_label(frame.index.name or "")] if index else []) + [_label(c) for c in frame.columns]
        widths = []
        if index:
            widths.append(max([len(_label(i)) for i in frame.index] + [len(labels[0]), 8]))
        for col in frame.columns:
            sample = frame[col].head(200).map(lambda v: len(_label(v)) if v is not None else 0)
            widths.append(max([len(_label(col))] + sample.tolist() + [8]))
        worksheet = writer.sheets[sheet]
        for i, width in enumerate(widths):
            width = min(width + 2, 60)
            if _EXCEL_ENGINE == "xlsxwriter":
                worksheet.set_column(i, i, width)
            else:  # openpyxl
                from openpyxl.utils import get_column_letter

                worksheet.column_dimensions[get_column_letter(i + 1)].width = width
    except Exception as exc:  # cosmetic only
        logger.debug("Column autofit failed for sheet %s: %s", sheet, exc)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------
_PDF_REPLACEMENTS = {
    "‘": "'", "’": "'", "‚": ",", "‛": "'", "“": '"', "”": '"', "„": '"',
    "′": "'", "″": '"', "–": "-", "—": "-", "―": "-", "‐": "-", "‑": "-",
    "−": "-", "…": "...", "•": "*", "·": "*", "‣": ">", "→": "->", "←": "<-",
    "↑": "^", "↓": "v", "≤": "<=", "≥": ">=", "≠": "!=", "≈": "~", "×": "x",
    "€": "EUR", "™": "(TM)", "‰": "per mille", " ": " ", " ": " ", "​": "",
    " ": " ", "﻿": "", "✓": "v", "✔": "v", "✗": "x", "✘": "x",
}


def sanitize_pdf_text(text: Any) -> str:
    """Return ``text`` restricted to Latin-1 characters printable with PDF core fonts."""
    text = "" if text is None else str(text)
    out: List[str] = []
    for char in text:
        if char in ("\n", "\t"):
            out.append("    " if char == "\t" else char)
            continue
        if unicodedata.category(char) in ("Cc", "Cf", "Cs"):
            out.append(_PDF_REPLACEMENTS.get(char, ""))
            continue
        if char in _PDF_REPLACEMENTS:
            out.append(_PDF_REPLACEMENTS[char])
            continue
        try:
            char.encode("latin-1")
            out.append(char)
            continue
        except UnicodeEncodeError:
            pass
        decomposed = unicodedata.normalize("NFKD", char)
        ascii_part = decomposed.encode("ascii", "ignore").decode("ascii")
        out.append(ascii_part if ascii_part.strip() else "?")
    return "".join(out)


if PDF_AVAILABLE:

    class _ReportPDF(FPDF):  # type: ignore[misc,valid-type]
        def __init__(self, title: str, footer_note: str = ""):
            super().__init__(orientation="P", unit="mm", format="A4")
            self._title = sanitize_pdf_text(title)
            self._footer_note = sanitize_pdf_text(footer_note)
            self.set_auto_page_break(auto=True, margin=15)
            self.set_margins(15, 15, 15)
            self.set_title(self._title)
            self.set_creator("Stock Analysis System")

        def header(self):  # noqa: D401 - fpdf hook
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, self._title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="R")
            self.set_text_color(0, 0, 0)
            self.ln(2)

        def footer(self):  # noqa: D401 - fpdf hook
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"{self._footer_note}    Page {self.page_no()}/{{nb}}".strip(), align="C")
            self.set_text_color(0, 0, 0)


def _pdf_table(pdf: Any, header: Sequence[str], rows: Sequence[Sequence[str]],
               first_col_wider: bool = True) -> None:
    n_cols = len(header)
    if n_cols == 0:
        return
    if first_col_wider and n_cols > 1:
        col_widths: Tuple[float, ...] = tuple([2.2] + [1.0] * (n_cols - 1))
    else:
        col_widths = tuple([1.0] * n_cols)
    font_size = 9 if n_cols <= 5 else (8 if n_cols <= 7 else 7)
    pdf.set_font("Helvetica", size=font_size)
    data = [[sanitize_pdf_text(h) for h in header]] + [[sanitize_pdf_text(c) for c in row] for row in rows]
    with pdf.table(data, col_widths=col_widths, first_row_as_headings=True, line_height=font_size * 0.5,
                   text_align="LEFT", cell_fill_color=(245, 245, 245), cell_fill_mode="ROWS",
                   repeat_headings=1):
        pass
    pdf.ln(3)


def _pdf_heading(pdf: Any, text: str, level: int = 1) -> None:
    size = {0: 18, 1: 13, 2: 11}.get(level, 11)
    if pdf.get_y() > pdf.h - 40:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(25, 55, 95)
    pdf.multi_cell(0, size * 0.55, sanitize_pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


def _pdf_paragraph(pdf: Any, text: str, size: int = 10, style: str = "") -> None:
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, size * 0.5, sanitize_pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _pdf_image(pdf: Any, png: bytes, caption: str = "") -> None:
    try:
        width = pdf.epw
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        pdf.image(io.BytesIO(png), w=width)
        if caption:
            _pdf_paragraph(pdf, caption, size=8, style="I")
        pdf.ln(2)
    except Exception as exc:  # an image must never break the document
        logger.warning("Could not embed image in PDF: %s", exc)


# ---------------------------------------------------------------------------
# Charts (static PNG for PDF/HTML reports)
# ---------------------------------------------------------------------------
_CHART_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
                 "#7f7f7f", "#bcbd22"]


def line_chart_png(series: Mapping[str, pd.Series], title: str = "", ylabel: str = "",
                   normalize: bool = False, width: float = 8.0, height: float = 3.6) -> Optional[bytes]:
    """Render one or more time series as a PNG line chart (``None`` when nothing to plot)."""
    if not CHARTS_AVAILABLE:
        return None
    try:
        fig = Figure(figsize=(width, height), dpi=110)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        plotted = 0
        for i, (name, values) in enumerate(series.items()):
            values = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty:
                continue
            if normalize and values.iloc[0] not in (0, None):
                values = values / values.iloc[0] * 100
            ax.plot(values.index, values.values, label=sanitize_pdf_text(name), linewidth=1.4,
                    color=_CHART_COLORS[i % len(_CHART_COLORS)])
            plotted += 1
        if not plotted:
            return None
        ax.set_title(sanitize_pdf_text(title), fontsize=11)
        ax.set_ylabel(sanitize_pdf_text(ylabel or ("Indexed (start = 100)" if normalize else "")), fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        if plotted > 1:
            ax.legend(fontsize=8, loc="best")
        fig.autofmt_xdate()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()
    except Exception as exc:
        logger.warning("Line chart rendering failed: %s", exc)
        return None


def bar_chart_png(values: Mapping[str, Any], title: str = "", ylabel: str = "", percent: bool = False,
                  width: float = 8.0, height: float = 3.4) -> Optional[bytes]:
    """Render ``{label: value}`` as a PNG bar chart (``None`` when nothing to plot)."""
    if not CHARTS_AVAILABLE:
        return None
    try:
        clean = {sanitize_pdf_text(k): to_float(v) for k, v in values.items()}
        clean = {k: v for k, v in clean.items() if v is not None}
        if not clean:
            return None
        fig = Figure(figsize=(width, height), dpi=110)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        numbers = [v * 100 if percent else v for v in clean.values()]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in numbers]
        ax.bar(list(clean.keys()), numbers, color=colors)
        ax.set_title(sanitize_pdf_text(title), fontsize=11)
        ax.set_ylabel(sanitize_pdf_text(ylabel or ("%" if percent else "")), fontsize=9)
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8, rotation=45 if len(clean) > 6 else 0)
        ax.tick_params(axis="y", labelsize=8)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        return buffer.getvalue()
    except Exception as exc:
        logger.warning("Bar chart rendering failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------
@dataclass
class ReportSection:
    """
    One section of a report.

    Attributes:
        title: Section heading.
        text: Paragraphs.
        metrics: Ordered ``{label: raw value}`` pairs (raw numbers are kept for
            Excel/JSON; :attr:`metric_formats` controls display formatting).
        metric_formats: ``{label: kind}`` (see :func:`format_value`).
        table: Optional DataFrame with raw values.
        table_formats: ``{column: kind}``; ``"*"`` applies to every column
            without an explicit kind.
        table_index: Whether the table index carries row labels.
        image: Optional PNG chart.
        image_caption: Caption below the chart.
        sheet_name: Excel sheet name for the table (defaults to the title).
    """

    title: str
    text: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metric_formats: Dict[str, str] = field(default_factory=dict)
    table: Optional[pd.DataFrame] = None
    table_formats: Dict[str, str] = field(default_factory=dict)
    table_index: bool = False
    image: Optional[bytes] = None
    image_caption: str = ""
    sheet_name: Optional[str] = None

    def formatted_metrics(self) -> Dict[str, str]:
        return {label: format_value(value, self.metric_formats.get(label) or guess_kind(label, value))
                for label, value in self.metrics.items()}

    def display_table(self) -> Optional[pd.DataFrame]:
        """Table with every cell formatted as a display string."""
        if self.table is None or self.table.empty:
            return None
        frame = self.table.copy()
        out = pd.DataFrame(index=frame.index)
        for col in frame.columns:
            kind = self.table_formats.get(_label(col)) or self.table_formats.get("*") or "auto"
            out[_label(col)] = [format_value(v, kind) for v in frame[col].tolist()]
        out.index = [_label(i) for i in frame.index]
        out.index.name = _label(frame.index.name) if frame.index.name is not None else None
        return out

    def is_empty(self) -> bool:
        return not (self.text or self.metrics or (self.table is not None and not self.table.empty) or self.image)


@dataclass
class Report:
    """A renderable report (see :meth:`render`)."""

    title: str
    report_type: str = "custom"
    subtitle: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    appendix: Dict[str, pd.DataFrame] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = ("For informational purposes only. Not investment advice. Figures are derived from "
                       "third-party data and model assumptions that may be incomplete or inaccurate.")

    def __post_init__(self):
        self.metadata.setdefault("generated_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.metadata.setdefault("report_type", self.report_type)

    # ----- helpers ------------------------------------------------------------
    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "_", self.title).strip("_").lower() or "report"

    def add_section(self, section: ReportSection) -> ReportSection:
        if not section.is_empty():
            self.sections.append(section)
        return section

    def section(self, title: str) -> Optional[ReportSection]:
        return next((s for s in self.sections if s.title == title), None)

    # ----- tabular views ------------------------------------------------------
    def summary_frame(self) -> pd.DataFrame:
        rows = [{"section": "Report", "item": "Title", "value": self.title}]
        if self.subtitle:
            rows.append({"section": "Report", "item": "Subtitle", "value": self.subtitle})
        rows += [{"section": "Report", "item": humanize(k), "value": _cell_value(v)} for k, v in self.metadata.items()]
        for section in self.sections:
            for label, value in section.metrics.items():
                rows.append({"section": section.title, "item": label, "value": _cell_value(value)})
        return pd.DataFrame(rows, columns=["section", "item", "value"])

    def tables(self) -> Dict[str, pd.DataFrame]:
        """Sheets for Excel: ``Summary`` (metadata + metrics) then every table and appendix."""
        sheets: Dict[str, pd.DataFrame] = {"Summary": self.summary_frame()}
        notes = [{"section": s.title, "text": t} for s in self.sections for t in s.text]
        for section in self.sections:
            if section.table is not None and not section.table.empty:
                name = section.sheet_name or section.title
                sheets[_unique_key(name, sheets)] = section.table
        for name, frame in self.appendix.items():
            if isinstance(frame, (pd.DataFrame, pd.Series)) and not frame.empty:
                sheets[_unique_key(name, sheets)] = frame
        if notes:
            sheets[_unique_key("Notes", sheets)] = pd.DataFrame(notes)
        return sheets

    def to_frame(self) -> pd.DataFrame:
        """Long-format table (``section``, ``row``, ``column``, ``value``) holding every value of the report."""
        rows: List[Dict[str, Any]] = []
        for _, item in self.summary_frame().iterrows():
            rows.append({"section": item["section"], "row": item["item"], "column": "value", "value": item["value"]})
        for section in self.sections:
            for i, text in enumerate(section.text):
                rows.append({"section": section.title, "row": f"text_{i + 1}", "column": "text", "value": text})
            if section.table is not None and not section.table.empty:
                rows += _long_rows(section.title, section.table, section.table_index)
        for name, frame in self.appendix.items():
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                rows += _long_rows(name, frame, True)
        return pd.DataFrame(rows, columns=["section", "row", "column", "value"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "report_type": self.report_type,
            "metadata": to_serializable(self.metadata),
            "sections": [
                {
                    "title": s.title,
                    "text": list(s.text),
                    "metrics": to_serializable(s.metrics),
                    "table": to_serializable(s.table) if s.table is not None else None,
                    "has_chart": s.image is not None,
                }
                for s in self.sections
            ],
            "appendix": {name: to_serializable(frame) for name, frame in self.appendix.items()},
            "data": to_serializable(self.data),
            "disclaimer": self.disclaimer,
        }

    # ----- rendering ----------------------------------------------------------
    def to_excel(self) -> bytes:
        return to_excel_bytes(self)

    def to_csv(self) -> bytes:
        return to_csv_bytes(self)

    def to_json(self) -> bytes:
        return to_json_bytes(self)

    def to_pdf(self) -> bytes:
        return to_pdf_bytes(self)

    def to_html(self) -> bytes:
        return to_html_bytes(self)

    def render(self, fmt: str) -> bytes:
        """Render to ``excel``, ``csv``, ``json``, ``pdf`` or ``html`` bytes."""
        key = normalize_format(fmt)
        return {"excel": self.to_excel, "csv": self.to_csv, "json": self.to_json,
                "pdf": self.to_pdf, "html": self.to_html}[key]()

    def render_all(self, formats: Optional[Iterable[str]] = None) -> Tuple[Dict[str, bytes], Dict[str, str]]:
        """Render several formats; returns ``(files, errors)`` keyed by format key."""
        files: Dict[str, bytes] = {}
        errors: Dict[str, str] = {}
        for fmt in formats or available_formats():
            key = normalize_format(fmt)
            try:
                files[key] = self.render(key)
            except Exception as exc:
                logger.exception("Rendering %s report as %s failed", self.report_type, key)
                errors[key] = str(exc) or exc.__class__.__name__
        return files, errors


def _unique_key(name: str, existing: Mapping[str, Any]) -> str:
    candidate, counter = name, 2
    while candidate in existing:
        candidate = f"{name} ({counter})"
        counter += 1
    return candidate


def _long_rows(section: str, frame: pd.DataFrame, with_index: bool) -> List[Dict[str, Any]]:
    rows = []
    for position, (row_label, row) in enumerate(frame.iterrows()):
        label = _label(row_label) if with_index else str(position + 1)
        for col, value in row.items():
            rows.append({"section": section, "row": label, "column": _label(col), "value": _cell_value(value)})
    return rows


def _metadata_line(report: Report) -> str:
    parts = []
    for key in ("ticker", "tickers", "sector", "generated_at"):
        value = report.metadata.get(key)
        if value:
            value = ", ".join(value) if isinstance(value, (list, tuple)) else value
            parts.append(f"{humanize(key)}: {value}")
    return "  |  ".join(parts)


_PDF_MAX_ROWS = 60
_PDF_MAX_COLS = 8


def to_pdf_bytes(data: Any, title: Optional[str] = None) -> bytes:
    """
    Render a :class:`Report` (or any data accepted by
    :meth:`ReportGenerator.from_data`) to PDF bytes using fpdf2 core fonts.
    """
    if not PDF_AVAILABLE:
        raise ExportError("PDF export requires the fpdf2 package.")
    report = data if isinstance(data, Report) else ReportGenerator().from_data(data, title=title)
    pdf = _ReportPDF(report.title, footer_note="Stock Analysis System")
    pdf.alias_nb_pages()
    pdf.add_page()
    _pdf_heading(pdf, report.title, level=0)
    if report.subtitle:
        _pdf_paragraph(pdf, report.subtitle, size=11, style="I")
    meta = _metadata_line(report)
    if meta:
        _pdf_paragraph(pdf, meta, size=8)
    pdf.ln(2)
    for section in report.sections:
        _pdf_heading(pdf, section.title, level=1)
        for text in section.text:
            _pdf_paragraph(pdf, text)
        if section.metrics:
            rows = [[label, value] for label, value in section.formatted_metrics().items()]
            _pdf_table(pdf, ["Metric", "Value"], rows, first_col_wider=True)
        table = section.display_table()
        if table is not None:
            truncated = len(table) > _PDF_MAX_ROWS or len(table.columns) > _PDF_MAX_COLS - int(section.table_index)
            max_cols = _PDF_MAX_COLS - int(section.table_index)
            table = table.iloc[:_PDF_MAX_ROWS, :max_cols]
            header = ([table.index.name or ""] if section.table_index else []) + [str(c) for c in table.columns]
            rows = [([str(idx)] if section.table_index else []) + list(map(str, row))
                    for idx, row in zip(table.index, table.itertuples(index=False, name=None))]
            _pdf_table(pdf, header, rows, first_col_wider=section.table_index)
            if truncated:
                _pdf_paragraph(pdf, "Table truncated - see the Excel or CSV export for all data.", size=8,
                               style="I")
        if section.image:
            _pdf_image(pdf, section.image, section.image_caption)
    if report.disclaimer:
        pdf.ln(4)
        _pdf_paragraph(pdf, report.disclaimer, size=8, style="I")
    return bytes(pdf.output())


def to_html_bytes(data: Any, title: Optional[str] = None) -> bytes:
    """Render a :class:`Report` (or generic data) as a standalone HTML document."""
    report = data if isinstance(data, Report) else ReportGenerator().from_data(data, title=title)
    esc = html.escape
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{esc(report.title)}</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;"
        "color:#222}h1{color:#19375f}h2{color:#19375f;border-bottom:1px solid #ddd;padding-bottom:4px;"
        "margin-top:28px}table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}"
        "th,td{border:1px solid #ddd;padding:4px 8px;text-align:left}th{background:#f0f3f7}"
        "tr:nth-child(even) td{background:#fafafa}.meta{color:#666;font-size:12px}"
        ".disclaimer{color:#777;font-size:11px;font-style:italic;margin-top:32px}img{max-width:100%}</style>",
        "</head><body>",
        f"<h1>{esc(report.title)}</h1>",
    ]
    if report.subtitle:
        parts.append(f"<p><em>{esc(report.subtitle)}</em></p>")
    meta = _metadata_line(report)
    if meta:
        parts.append(f"<p class='meta'>{esc(meta)}</p>")
    for section in report.sections:
        parts.append(f"<h2>{esc(section.title)}</h2>")
        parts += [f"<p>{esc(t)}</p>" for t in section.text]
        if section.metrics:
            parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            parts += [f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in section.formatted_metrics().items()]
            parts.append("</table>")
        table = section.display_table()
        if table is not None:
            parts.append(table.to_html(index=section.table_index, escape=True, border=0))
        if section.image:
            encoded = base64.b64encode(section.image).decode("ascii")
            parts.append(f"<figure><img alt='{esc(section.image_caption or section.title)}' "
                         f"src='data:image/png;base64,{encoded}'>")
            if section.image_caption:
                parts.append(f"<figcaption class='meta'>{esc(section.image_caption)}</figcaption>")
            parts.append("</figure>")
    if report.disclaimer:
        parts.append(f"<p class='disclaimer'>{esc(report.disclaimer)}</p>")
    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")


# ---------------------------------------------------------------------------
# Analytics used by the report builders
# ---------------------------------------------------------------------------
def _close_series(prices: Optional[pd.DataFrame]) -> pd.Series:
    if is_empty(prices):
        return pd.Series(dtype=float)
    column = "Close" if "Close" in prices.columns else ("Adj Close" if "Adj Close" in prices.columns else None)
    if column is None:
        return pd.Series(dtype=float)
    close = pd.to_numeric(prices[column], errors="coerce").dropna()
    close = close[close > 0]
    try:
        close.index = pd.to_datetime(close.index)
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
    except (TypeError, ValueError):
        return pd.Series(dtype=float)
    return close.sort_index()


def price_performance(prices: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    """
    Trailing returns (fractions), annualised volatility, maximum drawdown and
    52-week range from a price history with a ``Close`` column.
    """
    close = _close_series(prices)
    result: Dict[str, Optional[float]] = {
        "last_price": None, "return_1m": None, "return_3m": None, "return_6m": None, "return_1y": None,
        "return_ytd": None, "return_total": None, "volatility": None, "max_drawdown": None,
        "high_52w": None, "low_52w": None,
    }
    if close.empty:
        return result
    last_date, last = close.index[-1], float(close.iloc[-1])
    result["last_price"] = last
    for key, offset in (("return_1m", pd.DateOffset(months=1)), ("return_3m", pd.DateOffset(months=3)),
                        ("return_6m", pd.DateOffset(months=6)), ("return_1y", pd.DateOffset(years=1))):
        start = last_date - offset
        if close.index[0] <= start:
            base = close.asof(start)
            result[key] = safe_divide(last, base) - 1 if safe_divide(last, base) is not None else None
    year_start = close[close.index >= pd.Timestamp(year=last_date.year, month=1, day=1)]
    prior = close[close.index < pd.Timestamp(year=last_date.year, month=1, day=1)]
    if not prior.empty:
        result["return_ytd"] = last / float(prior.iloc[-1]) - 1
    elif not year_start.empty:
        result["return_ytd"] = last / float(year_start.iloc[0]) - 1
    result["return_total"] = last / float(close.iloc[0]) - 1
    log_returns = np.log(close).diff().dropna().tail(252)
    if len(log_returns) > 5:
        result["volatility"] = float(log_returns.std() * math.sqrt(252))
    result["max_drawdown"] = float((close / close.cummax() - 1).min())
    trailing = close[close.index >= last_date - pd.DateOffset(years=1)]
    result["high_52w"] = float(trailing.max())
    result["low_52w"] = float(trailing.min())
    return result


PERFORMANCE_LABELS = {
    "last_price": ("Last Price", "price"), "return_1m": ("1M Return", "pct"), "return_3m": ("3M Return", "pct"),
    "return_6m": ("6M Return", "pct"), "return_1y": ("1Y Return", "pct"), "return_ytd": ("YTD Return", "pct"),
    "return_total": ("Period Return", "pct"), "volatility": ("Volatility (ann.)", "pct"),
    "max_drawdown": ("Max Drawdown", "pct"), "high_52w": ("52W High", "price"), "low_52w": ("52W Low", "price"),
}


def key_metrics(financial_data: Optional[Mapping[str, Any]],
                company_info: Optional[Mapping[str, Any]] = None) -> Dict[str, Optional[float]]:
    """
    Headline fundamentals from the canonical ``financial_data`` bundle.

    Returns raw values: amounts in currency units, ratios as fractions,
    multiples as plain numbers. Missing inputs yield ``None``.
    """
    financial_data = financial_data or {}
    info = dict(company_info or financial_data.get("company_info") or {})
    income = financial_data.get("income_statement")
    balance = financial_data.get("balance_sheet")
    cash_flow = financial_data.get("cash_flow")
    market = financial_data.get("market_data") or {}

    revenue = get_latest(income, "revenue")
    previous_revenue = get_value(income, "revenue", 1)
    gross_profit = get_latest(income, "gross_profit")
    operating_income = get_latest(income, "operating_income")
    net_income = get_latest(income, "net_income")
    ebitda = get_latest(income, "ebitda")
    eps = get_latest(income, "diluted_eps")
    operating_cf = get_latest(cash_flow, "operating_cash_flow")
    capex = get_latest(cash_flow, "capital_expenditure")
    fcf = get_latest(cash_flow, "free_cash_flow")
    if fcf is None and operating_cf is not None and capex is not None:
        fcf = operating_cf - abs(capex)
    total_assets = get_latest(balance, "total_assets")
    equity = get_latest(balance, "total_equity")
    debt = get_latest(balance, "total_debt")
    cash = get_latest(balance, "cash_and_short_term_investments")
    current_assets = get_latest(balance, "current_assets")
    current_liabilities = get_latest(balance, "current_liabilities")

    price = to_float(market.get("share_price")) or to_float(info.get("current_price"))
    market_cap = to_float(market.get("market_cap")) or to_float(info.get("market_cap"))
    shares = to_float(market.get("shares_outstanding"))
    enterprise_value = None
    if market_cap is not None:
        enterprise_value = market_cap + (debt or 0.0) - (cash or 0.0)

    revenue_history = get_series(income, "revenue", ascending=True)
    return {
        "price": price,
        "market_cap": market_cap,
        "shares_outstanding": shares,
        "enterprise_value": enterprise_value,
        "revenue": revenue,
        "revenue_growth": (safe_divide(revenue, previous_revenue) - 1)
        if safe_divide(revenue, previous_revenue) is not None and previous_revenue and previous_revenue > 0 else None,
        "revenue_cagr": cagr(revenue_history) if len(revenue_history) >= 2 else None,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "ebitda": ebitda,
        "eps": eps,
        "free_cash_flow": fcf,
        "gross_margin": safe_divide(gross_profit, revenue),
        "operating_margin": safe_divide(operating_income, revenue),
        "net_margin": safe_divide(net_income, revenue),
        "fcf_margin": safe_divide(fcf, revenue),
        "roe": safe_divide(net_income, equity) if equity and equity > 0 else None,
        "roa": safe_divide(net_income, total_assets),
        "debt_to_equity": safe_divide(debt, equity) if equity and equity > 0 else None,
        "current_ratio": safe_divide(current_assets, current_liabilities),
        "pe_ratio": safe_divide(market_cap, net_income) if net_income and net_income > 0 else None,
        "ps_ratio": safe_divide(market_cap, revenue),
        "pb_ratio": safe_divide(market_cap, equity) if equity and equity > 0 else None,
        "ev_to_ebitda": safe_divide(enterprise_value, ebitda) if ebitda and ebitda > 0 else None,
        "fcf_yield": safe_divide(fcf, market_cap),
        "dividend_yield": to_float(market.get("dividend_yield")),
        "beta": to_float(market.get("beta")) or to_float(info.get("beta")),
        "total_assets": total_assets,
        "total_equity": equity,
        "total_debt": debt,
        "cash": cash,
    }


#: Display label and format kind for every :func:`key_metrics` key.
KEY_METRIC_LABELS: Dict[str, Tuple[str, str]] = {
    "price": ("Share Price", "price"), "market_cap": ("Market Cap", "money"),
    "shares_outstanding": ("Shares Outstanding", "count"), "enterprise_value": ("Enterprise Value", "money"),
    "revenue": ("Revenue", "money"), "revenue_growth": ("Revenue Growth (YoY)", "pct"),
    "revenue_cagr": ("Revenue CAGR", "pct"), "gross_profit": ("Gross Profit", "money"),
    "operating_income": ("Operating Income", "money"), "net_income": ("Net Income", "money"),
    "ebitda": ("EBITDA", "money"), "eps": ("Diluted EPS", "price"), "free_cash_flow": ("Free Cash Flow", "money"),
    "gross_margin": ("Gross Margin", "pct"), "operating_margin": ("Operating Margin", "pct"),
    "net_margin": ("Net Margin", "pct"), "fcf_margin": ("FCF Margin", "pct"), "roe": ("Return on Equity", "pct"),
    "roa": ("Return on Assets", "pct"), "debt_to_equity": ("Debt / Equity", "ratio"),
    "current_ratio": ("Current Ratio", "ratio"), "pe_ratio": ("P/E", "multiple"), "ps_ratio": ("P/S", "multiple"),
    "pb_ratio": ("P/B", "multiple"), "ev_to_ebitda": ("EV/EBITDA", "multiple"), "fcf_yield": ("FCF Yield", "pct"),
    "dividend_yield": ("Dividend Yield", "pct"), "beta": ("Beta", "ratio"), "total_assets": ("Total Assets", "money"),
    "total_equity": ("Total Equity", "money"), "total_debt": ("Total Debt", "money"),
    "cash": ("Cash & Short-Term Investments", "money"),
}

#: Line items shown in the condensed statement summary (canonical key, label, kind).
STATEMENT_SUMMARY_ITEMS: List[Tuple[str, str, str, str]] = [
    ("income_statement", "revenue", "Revenue", "money"),
    ("income_statement", "gross_profit", "Gross Profit", "money"),
    ("income_statement", "operating_income", "Operating Income", "money"),
    ("income_statement", "net_income", "Net Income", "money"),
    ("income_statement", "diluted_eps", "Diluted EPS", "price"),
    ("cash_flow", "operating_cash_flow", "Operating Cash Flow", "money"),
    ("cash_flow", "capital_expenditure", "Capital Expenditure", "money"),
    ("cash_flow", "free_cash_flow", "Free Cash Flow", "money"),
    ("balance_sheet", "total_assets", "Total Assets", "money"),
    ("balance_sheet", "total_liabilities", "Total Liabilities", "money"),
    ("balance_sheet", "total_equity", "Shareholders' Equity", "money"),
    ("balance_sheet", "total_debt", "Total Debt", "money"),
    ("balance_sheet", "cash_and_short_term_investments", "Cash & ST Investments", "money"),
]


def statement_summary(financial_data: Optional[Mapping[str, Any]], max_periods: int = 5) -> pd.DataFrame:
    """Condensed multi-period table of key line items (rows) by period (columns, most recent first)."""
    rows: Dict[str, pd.Series] = {}
    for statement, item, label, _ in STATEMENT_SUMMARY_ITEMS:
        series = get_series((financial_data or {}).get(statement), item, ascending=False)
        if not series.empty:
            rows[label] = series
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).T
    try:
        ordered = sorted(frame.columns, key=lambda c: pd.Timestamp(c), reverse=True)
        frame = frame[ordered]
    except (TypeError, ValueError):
        pass
    frame = frame.iloc[:, :max_periods]
    frame.columns = [_label(c) for c in frame.columns]
    frame.index.name = "Line Item"
    return frame


def _statement_formats(frame: pd.DataFrame) -> Dict[str, str]:
    return {"*": "money"}


def normalized_prices(price_map: Mapping[str, Optional[pd.DataFrame]]) -> pd.DataFrame:
    """Close prices rebased to 100 at the first common date (columns = tickers)."""
    closes = {ticker: _close_series(prices) for ticker, prices in price_map.items()}
    closes = {t: s for t, s in closes.items() if not s.empty}
    if not closes:
        return pd.DataFrame()
    frame = pd.DataFrame(closes).sort_index().ffill().dropna()
    if frame.empty:
        return frame
    frame = frame / frame.iloc[0] * 100
    frame.index.name = "Date"
    return frame


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------
_OVERVIEW_FIELDS = [("name", "Company"), ("symbol", "Ticker"), ("sector", "Sector"), ("industry", "Industry"),
                    ("country", "Country"), ("exchange", "Exchange"), ("currency", "Currency"),
                    ("employees", "Employees"), ("website", "Website")]


class ReportGenerator:
    """
    Build :class:`Report` documents from analysis data.

    Every builder accepts plain data (no loading happens here) and tolerates
    missing pieces: sections without data are omitted.
    """

    def __init__(self, include_charts: bool = True):
        self.include_charts = include_charts and CHARTS_AVAILABLE

    # ----- public builders ------------------------------------------------------
    def company_report(self, ticker: str, company_info: Optional[Mapping[str, Any]] = None,
                       financial_data: Optional[Mapping[str, Any]] = None,
                       price_data: Optional[pd.DataFrame] = None,
                       valuation: Optional[Mapping[str, Any]] = None,
                       ratios: Optional[Mapping[str, Any]] = None,
                       risk: Optional[Mapping[str, Any]] = None,
                       include_statements: bool = True,
                       include_charts: Optional[bool] = None) -> Report:
        """Comprehensive single-company report."""
        ticker = (ticker or "").upper()
        info = dict(company_info or (financial_data or {}).get("company_info") or {})
        charts = self.include_charts if include_charts is None else (include_charts and CHARTS_AVAILABLE)
        name = info.get("name") or ticker
        report = Report(title=f"{name} ({ticker}) - Company Report" if name != ticker else f"{ticker} - Company Report",
                        report_type="company", subtitle=_subtitle(info),
                        metadata={"ticker": ticker, "company": name})

        report.add_section(self._overview_section(info))
        metrics = key_metrics(financial_data, info)
        report.add_section(self._metrics_section("Key Metrics", metrics))
        report.add_section(self._performance_section(ticker, price_data, charts))
        if include_statements:
            summary = statement_summary(financial_data)
            if not summary.empty:
                report.add_section(ReportSection(
                    "Financial Summary", table=summary, table_formats=_statement_formats(summary), table_index=True,
                    text=["Key line items by reporting period (most recent first)."]))
            for key, label in (("income_statement", "Income Statement"), ("balance_sheet", "Balance Sheet"),
                               ("cash_flow", "Cash Flow")):
                frame = (financial_data or {}).get(key)
                if not is_empty(frame):
                    report.appendix[label] = frame
        if ratios:
            report.add_section(self._ratios_section(ratios))
        if risk:
            report.add_section(self._risk_section(risk))
        if valuation:
            for section in self._valuation_sections(valuation, brief=True):
                report.add_section(section)
        close = _close_series(price_data)
        if not close.empty:
            report.appendix["Price History"] = close.to_frame(name="Close").rename_axis("Date")
        report.data = {"ticker": ticker, "company_info": info, "key_metrics": metrics,
                       "valuation": _valuation_payload(valuation), "ratios": ratios or {}, "risk": risk or {}}
        return report

    def valuation_report(self, ticker: str, valuation: Mapping[str, Any],
                         company_info: Optional[Mapping[str, Any]] = None,
                         price_data: Optional[pd.DataFrame] = None,
                         financial_data: Optional[Mapping[str, Any]] = None,
                         include_charts: Optional[bool] = None) -> Report:
        """Detailed valuation report from a ``ValuationFactory`` result."""
        ticker = (ticker or valuation.get("company") or "").upper()
        info = dict(company_info or (financial_data or {}).get("company_info") or {})
        charts = self.include_charts if include_charts is None else (include_charts and CHARTS_AVAILABLE)
        name = info.get("name") or ticker
        report = Report(title=f"{name} ({ticker}) - Valuation Report" if name != ticker
                        else f"{ticker} - Valuation Report",
                        report_type="valuation", subtitle=_subtitle(info) or str(valuation.get("sector") or ""),
                        metadata={"ticker": ticker, "company": name, "model": valuation.get("model")})
        for section in self._valuation_sections(valuation, brief=False, charts=charts):
            report.add_section(section)
        if financial_data:
            report.add_section(self._metrics_section("Fundamentals", key_metrics(financial_data, info),
                                                     keys=("revenue", "net_income", "free_cash_flow", "eps",
                                                           "net_margin", "roe", "pe_ratio", "ev_to_ebitda",
                                                           "debt_to_equity", "dividend_yield")))
        if price_data is not None:
            report.add_section(self._performance_section(ticker, price_data, charts=False))
        report.data = {"ticker": ticker, "valuation": _valuation_payload(valuation)}
        return report

    def comparison_report(self, companies: Mapping[str, Mapping[str, Any]],
                          include_charts: Optional[bool] = None) -> Report:
        """
        Side-by-side comparison.

        Args:
            companies: ``{ticker: {"company_info", "financial_data", "price_data", "valuation"}}``
                (every entry optional).
        """
        charts = self.include_charts if include_charts is None else (include_charts and CHARTS_AVAILABLE)
        tickers = [t.upper() for t in companies]
        report = Report(title=f"Company Comparison - {', '.join(tickers)}", report_type="comparison",
                        subtitle=f"{len(tickers)} companies", metadata={"tickers": tickers})
        metric_rows, perf_rows, val_rows = {}, {}, {}
        for ticker, bundle in companies.items():
            bundle = bundle or {}
            info = dict(bundle.get("company_info") or (bundle.get("financial_data") or {}).get("company_info") or {})
            metrics = key_metrics(bundle.get("financial_data"), info)
            metric_rows[ticker.upper()] = {"Name": info.get("name") or ticker.upper(),
                                           "Sector": info.get("sector") or "N/A",
                                           **{KEY_METRIC_LABELS[k][0]: metrics.get(k) for k in _COMPARISON_KEYS}}
            perf = price_performance(bundle.get("price_data"))
            perf_rows[ticker.upper()] = {PERFORMANCE_LABELS[k][0]: perf.get(k) for k in _PERF_KEYS}
            valuation = bundle.get("valuation")
            if valuation:
                val_rows[ticker.upper()] = _valuation_row(valuation)
        metrics_table = pd.DataFrame.from_dict(metric_rows, orient="index").rename_axis("Ticker")
        report.add_section(ReportSection(
            "Fundamentals", table=metrics_table, table_index=True, sheet_name="Fundamentals",
            table_formats={"Name": "text", "Sector": "text",
                           **{KEY_METRIC_LABELS[k][0]: KEY_METRIC_LABELS[k][1] for k in _COMPARISON_KEYS}},
            text=_comparison_highlights(metrics_table)))
        perf_table = pd.DataFrame.from_dict(perf_rows, orient="index").rename_axis("Ticker")
        if perf_table.notna().any().any():
            prices = normalized_prices({t: (b or {}).get("price_data") for t, b in companies.items()})
            image = line_chart_png({c: prices[c] for c in prices.columns}, "Relative price performance",
                                   ylabel="Indexed (start = 100)") if charts and not prices.empty else None
            report.add_section(ReportSection(
                "Price Performance", table=perf_table, table_index=True, sheet_name="Performance",
                table_formats={PERFORMANCE_LABELS[k][0]: PERFORMANCE_LABELS[k][1] for k in _PERF_KEYS},
                image=image, image_caption="Close prices rebased to 100 at the first common date."))
            if not prices.empty:
                report.appendix["Normalized Prices"] = prices
        if val_rows:
            val_table = pd.DataFrame.from_dict(val_rows, orient="index").rename_axis("Ticker")
            report.add_section(ReportSection("Valuation", table=val_table, table_index=True, sheet_name="Valuation",
                                             table_formats=_VALUATION_ROW_FORMATS))
        report.data = {"tickers": tickers,
                       "valuations": {t.upper(): _valuation_payload((b or {}).get("valuation"))
                                      for t, b in companies.items() if (b or {}).get("valuation")}}
        return report

    def sector_report(self, sector: str, companies: Sequence[Mapping[str, Any]],
                      etf_ticker: Optional[str] = None, etf_prices: Optional[pd.DataFrame] = None,
                      benchmark_ticker: str = "SPY", benchmark_prices: Optional[pd.DataFrame] = None,
                      include_charts: Optional[bool] = None) -> Report:
        """
        Sector overview.

        Args:
            sector: Canonical sector name.
            companies: Records with ``symbol``, ``name``, ``subsector`` and
                optionally ``company_info``, ``financial_data``,
                ``price_data`` and ``valuation``.
            etf_ticker / etf_prices: Sector ETF benchmark.
            benchmark_ticker / benchmark_prices: Broad market benchmark.
        """
        charts = self.include_charts if include_charts is None else (include_charts and CHARTS_AVAILABLE)
        report = Report(title=f"{sector} Sector Report", report_type="sector",
                        subtitle=f"{len(companies)} companies" + (f" | benchmark ETF {etf_ticker}" if etf_ticker else ""),
                        metadata={"sector": sector, "etf": etf_ticker})
        etf_perf = price_performance(etf_prices)
        bench_perf = price_performance(benchmark_prices)
        overview = ReportSection("Sector Performance")
        if etf_perf["last_price"] is not None:
            for key in ("last_price", "return_1m", "return_3m", "return_6m", "return_1y", "volatility",
                        "max_drawdown"):
                label, kind = PERFORMANCE_LABELS[key]
                overview.metrics[f"{etf_ticker} {label}"] = etf_perf[key]
                overview.metric_formats[f"{etf_ticker} {label}"] = kind
            if bench_perf["return_1y"] is not None and etf_perf["return_1y"] is not None:
                overview.metrics[f"{benchmark_ticker} 1Y Return"] = bench_perf["return_1y"]
                overview.metrics["Relative 1Y Performance"] = etf_perf["return_1y"] - bench_perf["return_1y"]
                overview.metric_formats.update({f"{benchmark_ticker} 1Y Return": "pct",
                                                "Relative 1Y Performance": "pct"})
                verb = "outperformed" if etf_perf["return_1y"] >= bench_perf["return_1y"] else "underperformed"
                overview.text.append(
                    f"Over the last year {etf_ticker} returned {format_value(etf_perf['return_1y'], 'pct')} and "
                    f"{verb} {benchmark_ticker} ({format_value(bench_perf['return_1y'], 'pct')}).")
            if charts:
                series = {etf_ticker or sector: _close_series(etf_prices)}
                if benchmark_prices is not None:
                    series[benchmark_ticker] = _close_series(benchmark_prices)
                overview.image = line_chart_png(series, f"{sector} vs market", normalize=True)
                overview.image_caption = "Close prices rebased to 100."
        else:
            overview.text.append("Sector ETF price data is not available.")
        report.add_section(overview)

        rows = {}
        for record in companies:
            symbol = str(record.get("symbol") or record.get("ticker") or "").upper()
            if not symbol:
                continue
            info = dict(record.get("company_info") or {})
            metrics = key_metrics(record.get("financial_data"), info)
            perf = price_performance(record.get("price_data"))
            row = {"Name": record.get("name") or info.get("name") or symbol,
                   "Subsector": record.get("subsector") or info.get("industry") or "N/A",
                   "Market Cap": metrics.get("market_cap"), "Price": perf.get("last_price") or metrics.get("price"),
                   "1Y Return": perf.get("return_1y"), "Revenue": metrics.get("revenue"),
                   "Net Margin": metrics.get("net_margin"), "ROE": metrics.get("roe"), "P/E": metrics.get("pe_ratio"),
                   "EV/EBITDA": metrics.get("ev_to_ebitda")}
            valuation = record.get("valuation")
            if valuation:
                row["Fair Value"] = to_float(valuation.get("value_per_share"))
                row["Upside (%)"] = to_float(valuation.get("potential_upside"))
                row["Recommendation"] = (valuation.get("recommendation") or {}).get("label", "N/A")
            rows[symbol] = row
        if rows:
            table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Ticker")
            if "Market Cap" in table and table["Market Cap"].notna().any():
                table = table.sort_values("Market Cap", ascending=False, na_position="last")
            formats = {"Name": "text", "Subsector": "text", "Market Cap": "money", "Price": "price",
                       "1Y Return": "pct", "Revenue": "money", "Net Margin": "pct", "ROE": "pct",
                       "P/E": "multiple", "EV/EBITDA": "multiple", "Fair Value": "price",
                       "Upside (%)": "pct_points", "Recommendation": "text"}
            image = None
            if charts and table["1Y Return"].notna().any():
                image = bar_chart_png(table["1Y Return"].dropna().to_dict(), "1-year price return", percent=True)
            report.add_section(ReportSection("Companies", table=table, table_index=True, table_formats=formats,
                                             image=image, sheet_name="Companies"))
            aggregates = ReportSection("Sector Aggregates")
            numeric = {"Total Market Cap": ("Market Cap", "sum", "money"),
                       "Median P/E": ("P/E", "median", "multiple"),
                       "Median Net Margin": ("Net Margin", "median", "pct"),
                       "Median ROE": ("ROE", "median", "pct"),
                       "Median 1Y Return": ("1Y Return", "median", "pct"),
                       "Median Upside (%)": ("Upside (%)", "median", "pct_points")}
            for label, (column, how, kind) in numeric.items():
                if column in table and pd.to_numeric(table[column], errors="coerce").notna().any():
                    values = pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
                    aggregates.metrics[label] = float(getattr(values, how)())
                    aggregates.metric_formats[label] = kind
            best = pd.to_numeric(table["1Y Return"], errors="coerce").dropna()
            if not best.empty:
                aggregates.text.append(f"Best 1Y performer: {best.idxmax()} ({format_value(best.max(), 'pct')}); "
                                       f"weakest: {best.idxmin()} ({format_value(best.min(), 'pct')}).")
            report.add_section(aggregates)
        closes = {r.get("symbol"): r.get("price_data") for r in companies if r.get("price_data") is not None}
        prices = normalized_prices(closes)
        if not prices.empty:
            report.appendix["Normalized Prices"] = prices
        report.data = {"sector": sector, "etf": etf_ticker, "etf_performance": etf_perf,
                       "benchmark_performance": bench_perf}
        return report

    def from_data(self, data: Any, title: Optional[str] = None, report_type: str = "custom") -> Report:
        """
        Generic report from arbitrary data: scalars become metrics,
        DataFrames/lists of records become tables, nested mappings become
        sections. Standardised valuation results get the valuation layout.
        """
        if isinstance(data, Report):
            return data
        if isinstance(data, Mapping) and "valuation_methods" in data:
            ticker = str(data.get("company") or data.get("ticker") or "")
            return self.valuation_report(ticker, data)
        report = Report(title=title or (f"{data.get('ticker')} - {humanize(report_type)} Report"
                                        if isinstance(data, Mapping) and data.get("ticker")
                                        else f"{humanize(report_type)} Report"),
                        report_type=report_type)
        if isinstance(data, (pd.DataFrame, pd.Series)):
            frame = data.to_frame() if isinstance(data, pd.Series) else data
            report.add_section(ReportSection("Data", table=frame, table_index=_keep_index(frame)))
            return report
        if not isinstance(data, Mapping):
            report.add_section(ReportSection("Data", text=[str(to_serializable(data))]))
            return report
        main = ReportSection("Summary")
        for key, value in data.items():
            label = humanize(key)
            if isinstance(value, (pd.DataFrame, pd.Series)):
                frame = value.to_frame() if isinstance(value, pd.Series) else value
                if not frame.empty:
                    report.add_section(ReportSection(label, table=frame, table_index=_keep_index(frame)))
            elif isinstance(value, (list, tuple)) and value and all(isinstance(r, Mapping) for r in value):
                report.add_section(ReportSection(label, table=tabular_frame(list(value))))
            elif isinstance(value, Mapping):
                flat = flatten_dict(value)
                section = ReportSection(label)
                for k, v in flat.items():
                    section.metrics[humanize(k)] = v
                    section.metric_formats[humanize(k)] = guess_kind(k, v)
                report.add_section(section)
            else:
                main.metrics[label] = to_serializable(value) if not isinstance(value, str) else value
                main.metric_formats[label] = guess_kind(key, value)
        if main.metrics:
            report.sections.insert(0, main)
        report.data = dict(data)
        return report

    # ----- rendering ----------------------------------------------------------------
    @staticmethod
    def render(report: Report, fmt: str) -> bytes:
        return report.render(fmt)

    @staticmethod
    def file_name(report: Report, fmt: str) -> str:
        ext = EXPORT_FORMATS[normalize_format(fmt)].extension
        stamp = datetime.now().strftime("%Y%m%d")
        return f"{report.slug}_{stamp}.{ext}"

    @staticmethod
    def mime_type(fmt: str) -> str:
        return EXPORT_FORMATS[normalize_format(fmt)].mime

    # Backward-compatible helpers (older pages called these with a report dict)
    def generate_pdf_report(self, report: Any) -> bytes:
        return self.from_data(report).to_pdf()

    def generate_excel_report(self, report: Any) -> bytes:
        return self.from_data(report).to_excel()

    def generate_html_report(self, report: Any) -> bytes:
        return self.from_data(report).to_html()

    def generate_csv_report(self, report: Any) -> bytes:
        return self.from_data(report).to_csv()

    def generate_json_report(self, report: Any) -> bytes:
        return self.from_data(report).to_json()

    # ----- section builders ----------------------------------------------------
    @staticmethod
    def _overview_section(info: Mapping[str, Any]) -> ReportSection:
        section = ReportSection("Company Overview")
        for key, label in _OVERVIEW_FIELDS:
            value = info.get(key)
            if value not in (None, "", "N/A"):
                section.metrics[label] = value
                section.metric_formats[label] = "int" if key == "employees" else "text"
        if info.get("market_cap"):
            section.metrics["Market Cap"] = to_float(info.get("market_cap"))
            section.metric_formats["Market Cap"] = "money"
        description = info.get("description") or info.get("long_business_summary")
        if description:
            section.text.append(str(description))
        return section

    @staticmethod
    def _metrics_section(title: str, metrics: Mapping[str, Any], keys: Optional[Sequence[str]] = None) -> ReportSection:
        section = ReportSection(title)
        for key in keys or KEY_METRIC_LABELS.keys():
            value = metrics.get(key)
            if value is None:
                continue
            label, kind = KEY_METRIC_LABELS[key]
            section.metrics[label] = value
            section.metric_formats[label] = kind
        return section

    @staticmethod
    def _performance_section(ticker: str, price_data: Optional[pd.DataFrame], charts: bool) -> ReportSection:
        perf = price_performance(price_data)
        section = ReportSection("Price Performance")
        if perf["last_price"] is None:
            return section
        for key, (label, kind) in PERFORMANCE_LABELS.items():
            if perf.get(key) is not None:
                section.metrics[label] = perf[key]
                section.metric_formats[label] = kind
        close = _close_series(price_data)
        section.text.append(f"Price history from {_date_label(close.index[0])} to {_date_label(close.index[-1])} "
                            f"({len(close)} observations).")
        if charts:
            series = {ticker: close}
            if len(close) >= 50:
                series["50-day average"] = close.rolling(50).mean()
            if len(close) >= 200:
                series["200-day average"] = close.rolling(200).mean()
            section.image = line_chart_png(series, f"{ticker} share price", ylabel="Price")
            section.image_caption = "Daily close with moving averages."
        return section

    @staticmethod
    def _ratios_section(ratios: Mapping[str, Any]) -> ReportSection:
        rows = []
        for category, values in ratios.items():
            if not isinstance(values, Mapping):
                continue
            for name, value in values.items():
                if isinstance(value, Mapping):
                    value = value.get("value")
                number = to_float(value) if not isinstance(value, str) else None
                if number is None:
                    continue
                rows.append({"Category": humanize(category), "Ratio": humanize(name), "Value": number})
        if not rows:
            return ReportSection("Financial Ratios")
        return ReportSection("Financial Ratios", table=pd.DataFrame(rows), table_formats={"Value": "ratio"},
                             sheet_name="Ratios")

    @staticmethod
    def _risk_section(risk: Mapping[str, Any]) -> ReportSection:
        section = ReportSection("Financial Risk")
        overall = risk.get("overall_assessment") or risk.get("assessment")
        if overall:
            section.metrics["Overall Assessment"] = humanize(overall) if isinstance(overall, str) else overall
            section.metric_formats["Overall Assessment"] = "text"
        description = risk.get("overall_description") or risk.get("description")
        if description:
            section.text.append(str(description))
        rows = []
        for key, value in risk.items():
            if isinstance(value, Mapping) and ("score" in value or "assessment" in value):
                rows.append({"Model": value.get("model") or humanize(key), "Score": to_float(value.get("score")),
                             "Assessment": humanize(value.get("assessment") or "N/A"),
                             "Description": value.get("description") or ""})
        if rows:
            section.table = pd.DataFrame(rows)
            section.table_formats = {"Score": "ratio", "Model": "text", "Assessment": "text", "Description": "text"}
            section.sheet_name = "Risk"
        return section

    def _valuation_sections(self, valuation: Mapping[str, Any], brief: bool,
                            charts: bool = False) -> List[ReportSection]:
        sections: List[ReportSection] = []
        summary = ReportSection("Valuation Summary")
        rec = valuation.get("recommendation") or {}
        values = valuation.get("valuation_summary") or {}
        for label, value, kind in (
                ("Model", valuation.get("model"), "text"), ("Sector", valuation.get("sector"), "text"),
                ("Subsector", valuation.get("subsector"), "text"),
                ("Current Price", to_float(valuation.get("current_price")), "price"),
                ("Fair Value (weighted)", to_float(valuation.get("value_per_share")), "price"),
                ("Potential Upside", to_float(valuation.get("potential_upside")), "pct_points"),
                ("Recommendation", rec.get("label"), "text"),
                ("Fair Value Low", to_float(values.get("min")), "price"),
                ("Fair Value High", to_float(values.get("max")), "price"),
                ("Methods Used", ", ".join(humanize(m) for m in values.get("methods_used") or []) or None, "text")):
            if value not in (None, ""):
                summary.metrics[label] = value
                summary.metric_formats[label] = kind
        if rec.get("rationale"):
            summary.text.append(str(rec["rationale"]))
        if valuation.get("error"):
            summary.text.append(f"Valuation could not be completed: {valuation['error']}")
        summary.text += [str(n) for n in valuation.get("notes") or []]
        sections.append(summary)

        methods = valuation.get("valuation_methods") or {}
        weights = values.get("weights") or {}
        excluded = values.get("excluded") or {}
        method_rows = []
        for key, result in methods.items():
            if not isinstance(result, Mapping):
                continue
            status = "Used" if key in weights else ("Excluded: " + str(excluded.get(key) or result.get("error") or
                                                                       "no value"))
            method_rows.append({"Method": humanize(result.get("method") or key),
                                "Value per Share": to_float(result.get("value_per_share")),
                                "Weight": weights.get(key), "Status": status})
        if method_rows:
            image = None
            if charts:
                bars = {r["Method"]: r["Value per Share"] for r in method_rows if r["Value per Share"]}
                if bars:
                    image = bar_chart_png(bars, "Fair value per share by method", ylabel="Value per share")
            sections.append(ReportSection("Valuation Methods", table=pd.DataFrame(method_rows),
                                          table_formats={"Method": "text", "Value per Share": "price",
                                                         "Weight": "pct", "Status": "text"},
                                          image=image, sheet_name="Methods"))
        if brief:
            return sections

        assumptions = valuation.get("assumptions") or {}
        if assumptions:
            section = ReportSection("Assumptions")
            for key, value in flatten_dict(assumptions).items():
                section.metrics[humanize(key)] = value
                section.metric_formats[humanize(key)] = guess_kind(key, value)
            sections.append(section)

        detail_rows = []
        for key, result in methods.items():
            if not isinstance(result, Mapping):
                continue
            for field_name, value in result.items():
                if field_name in ("company", "method", "error") or isinstance(value, (Mapping, list, tuple,
                                                                                         pd.Series, pd.DataFrame)):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                    detail_rows.append({"Method": humanize(result.get("method") or key), "Field": humanize(field_name),
                                        "Value": to_float(value), "Format": guess_kind(field_name, value)})
        if detail_rows:
            frame = pd.DataFrame(detail_rows)
            frame["Display"] = [format_value(v, k) for v, k in zip(frame["Value"], frame["Format"])]
            sections.append(ReportSection("Method Details", table=frame[["Method", "Field", "Value", "Display"]],
                                          table_formats={"Value": "auto", "Method": "text", "Field": "text",
                                                         "Display": "text"},
                                          sheet_name="Method Details"))

        dcf = methods.get("dcf") or next((m for k, m in methods.items() if "dcf" in str(k).lower()
                                          and isinstance(m, Mapping)), None)
        if isinstance(dcf, Mapping) and isinstance(dcf.get("forecast_fcf"), (list, tuple, np.ndarray)):
            forecast = [to_float(v) for v in dcf["forecast_fcf"]]
            frame = pd.DataFrame({"Year": range(1, len(forecast) + 1), "Forecast FCF": forecast})
            discount = to_float(dcf.get("discount_rate"))
            if discount is not None and discount > -1:
                frame["Present Value"] = [v / (1 + discount) ** y if v is not None else None
                                          for y, v in zip(frame["Year"], forecast)]
            sections.append(ReportSection("DCF Cash Flow Forecast", table=frame,
                                          table_formats={"Year": "int", "Forecast FCF": "money",
                                                         "Present Value": "money"},
                                          sheet_name="DCF Forecast"))
        return sections


_COMPARISON_KEYS = ("price", "market_cap", "revenue", "revenue_growth", "gross_margin", "operating_margin",
                    "net_margin", "roe", "debt_to_equity", "pe_ratio", "ps_ratio", "ev_to_ebitda", "fcf_yield",
                    "dividend_yield")
_PERF_KEYS = ("return_1m", "return_3m", "return_6m", "return_1y", "volatility", "max_drawdown")
_VALUATION_ROW_FORMATS = {"Model": "text", "Current Price": "price", "Fair Value": "price",
                          "Upside (%)": "pct_points", "Recommendation": "text"}


def _valuation_row(valuation: Mapping[str, Any]) -> Dict[str, Any]:
    return {"Model": valuation.get("model") or "N/A",
            "Current Price": to_float(valuation.get("current_price")),
            "Fair Value": to_float(valuation.get("value_per_share")),
            "Upside (%)": to_float(valuation.get("potential_upside")),
            "Recommendation": (valuation.get("recommendation") or {}).get("label", "N/A")}


def _valuation_payload(valuation: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(valuation) if valuation else {}


def _subtitle(info: Mapping[str, Any]) -> str:
    parts = [str(info[k]) for k in ("sector", "industry", "exchange") if info.get(k) not in (None, "", "N/A")]
    return " | ".join(parts)


def _comparison_highlights(table: pd.DataFrame) -> List[str]:
    notes = []
    for label, kind, best in (("Net Margin", "pct", "max"), ("Revenue Growth (YoY)", "pct", "max"),
                              ("P/E", "multiple", "min"), ("Return on Equity", "pct", "max")):
        if label not in table:
            continue
        column = pd.to_numeric(table[label], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(column) < 2:
            continue
        ticker = column.idxmax() if best == "max" else column.idxmin()
        word = "Highest" if best == "max" else "Lowest"
        notes.append(f"{word} {label}: {ticker} ({format_value(column[ticker], kind)}).")
    return notes


# ---------------------------------------------------------------------------
# File-based export (backward compatible API)
# ---------------------------------------------------------------------------
def _safe_file_name(name: str, extension: str) -> str:
    stem = Path(str(name)).name  # drop any directory component
    if stem.lower().endswith("." + extension):
        stem = stem[: -(len(extension) + 1)]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "export"
    return f"{stem}.{extension}"


class ExportManager:
    """
    Export analysis results to files or bytes.

    File methods return the path of the written file and raise
    :class:`ExportError` when the format is unavailable. The ``*_bytes``
    method family never touches the file system.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_EXPORT_DIR
        self.generator = ReportGenerator()

    # ----- in-memory -------------------------------------------------------------
    def to_bytes(self, data: Any, fmt: str, report_type: str = "custom", title: Optional[str] = None) -> bytes:
        """Serialise ``data`` (DataFrame, mapping, valuation result or Report) to ``fmt`` bytes."""
        key = normalize_format(fmt)
        if key == "json":
            return to_json_bytes(data)
        if key == "csv":
            return to_csv_bytes(data)
        if key == "excel":
            return to_excel_bytes(data if isinstance(data, (Report, pd.DataFrame, pd.Series))
                                  else self.generator.from_data(data, title=title, report_type=report_type)
                                  if isinstance(data, Mapping) and "valuation_methods" in data else data)
        report = self.generator.from_data(data, title=title, report_type=report_type)
        return report.to_pdf() if key == "pdf" else report.to_html()

    # ----- files -------------------------------------------------------------------
    def _write(self, payload: bytes, filename: Optional[str], default_stem: str, extension: str) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = _safe_file_name(filename or default_stem, extension)
        path = self.output_dir / name
        path.write_bytes(payload)
        logger.info("Exported %s", path)
        return str(path)

    @staticmethod
    def _default_stem(data: Any, kind: str) -> str:
        ticker = data.get("ticker") or data.get("company") if isinstance(data, Mapping) else None
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{ticker or 'export'}_{kind}_{stamp}"

    def export_to_pdf(self, data: Any, report_type: str = "custom", filename: Optional[str] = None) -> str:
        payload = self.to_bytes(data, "pdf", report_type=report_type)
        return self._write(payload, filename, self._default_stem(data, f"{report_type}_report"), "pdf")

    def export_to_excel(self, data: Any, report_type: str = "custom", filename: Optional[str] = None) -> str:
        payload = self.to_bytes(data, "excel", report_type=report_type)
        return self._write(payload, filename, self._default_stem(data, f"{report_type}_data"), "xlsx")

    def export_to_csv(self, data: Any, filename: Optional[str] = None) -> str:
        return self._write(to_csv_bytes(data), filename, self._default_stem(data, "data"), "csv")

    def export_to_json(self, data: Any, filename: Optional[str] = None) -> str:
        return self._write(to_json_bytes(data), filename, self._default_stem(data, "data"), "json")

    def export_to_html(self, data: Any, report_type: str = "custom", filename: Optional[str] = None) -> str:
        payload = self.to_bytes(data, "html", report_type=report_type)
        return self._write(payload, filename, self._default_stem(data, f"{report_type}_report"), "html")

    def generate_visualization(self, data: Any, viz_type: str = "bar") -> str:
        """Base64-encoded PNG chart (``bar``: {label: value}; ``line``: DataFrame/{name: Series})."""
        png: Optional[bytes] = None
        if viz_type == "line":
            if isinstance(data, pd.DataFrame):
                series = {str(c): data[c] for c in data.columns}
            elif isinstance(data, pd.Series):
                series = {str(data.name or "value"): data}
            else:
                series = {str(k): pd.Series(v) for k, v in dict(data or {}).items()}
            png = line_chart_png(series)
        elif viz_type == "bar":
            values = data.to_dict() if isinstance(data, pd.Series) else dict(data or {})
            png = bar_chart_png(values)
        else:
            logger.warning("Unsupported visualization type: %s", viz_type)
        return base64.b64encode(png).decode("ascii") if png else ""
