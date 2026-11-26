"""
Documentation gate: README, USER_GUIDE and METHODOLOGY stay linked, complete and honest.

Run from anywhere:  python StockAnalysisSystem/tools/docs_check.py
Exit 0 = every check passed, 1 = at least one failed (each failure is printed).

Checks:
  1. README links every docs/screenshots/*.jpg that exists and links docs/METHODOLOGY.md.
  2. docs/METHODOLOGY.md exists and mentions WACC, liquidation, Sortino and Zmijewski.
  3. The phrase "not investment advice" appears in README, USER_GUIDE and METHODOLOGY.
  4. Every relative markdown link in README, USER_GUIDE and METHODOLOGY resolves to an existing file.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
USER_GUIDE = ROOT / "docs" / "USER_GUIDE.md"
METHODOLOGY = ROOT / "docs" / "METHODOLOGY.md"
SCREENSHOTS = ROOT / "docs" / "screenshots"

DISCLAIMER = "not investment advice"
METHODOLOGY_TERMS = ("WACC", "liquidation", "Sortino", "Zmijewski")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "#")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_readme_links() -> List[str]:
    failures = []
    text = _read(README)
    for shot in sorted(SCREENSHOTS.glob("*.jpg")):
        target = f"docs/screenshots/{shot.name}"
        if target not in text:
            failures.append(f"README does not link {target}")
    if "docs/METHODOLOGY.md" not in text:
        failures.append("README does not link docs/METHODOLOGY.md")
    return failures


def check_methodology() -> List[str]:
    if not METHODOLOGY.is_file():
        return ["docs/METHODOLOGY.md does not exist"]
    text = _read(METHODOLOGY).lower()
    return [f"METHODOLOGY.md does not mention {term}" for term in METHODOLOGY_TERMS
            if term.lower() not in text]


def check_disclaimer() -> List[str]:
    failures = []
    for path in (README, USER_GUIDE, METHODOLOGY):
        if not path.is_file():
            failures.append(f"{path.relative_to(ROOT)} does not exist")
        elif DISCLAIMER not in _read(path).lower():
            failures.append(f"{path.relative_to(ROOT)} lacks the phrase '{DISCLAIMER}'")
    return failures


def check_relative_links() -> List[str]:
    failures = []
    for path in (README, USER_GUIDE, METHODOLOGY):
        if not path.is_file():
            continue
        for match in LINK.finditer(_read(path)):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith(EXTERNAL_PREFIXES):
                continue
            if not (path.parent / target).resolve().exists():
                failures.append(f"{path.relative_to(ROOT)}: broken link '{match.group(1)}'")
    return failures


def main() -> int:
    if not README.is_file():
        print("FAIL README.md does not exist")
        return 1
    checks = {
        "readme links (screenshots, METHODOLOGY)": check_readme_links(),
        "METHODOLOGY content": check_methodology(),
        "disclaimer phrase": check_disclaimer(),
        "relative links resolve": check_relative_links(),
    }
    failed = 0
    for name, failures in checks.items():
        if failures:
            failed += len(failures)
            for failure in failures:
                print(f"FAIL {name}: {failure}")
        else:
            print(f"OK   {name}")
    print("docs_check: ok" if not failed else f"docs_check: {failed} failure(s)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
