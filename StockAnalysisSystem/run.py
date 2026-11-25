"""
Launch the Stock Analysis System.

    python StockAnalysisSystem/run.py [extra streamlit options]

Equivalent to ``streamlit run StockAnalysisSystem/src/main.py`` from the
repository root (where ``.streamlit/config.toml`` defines the dark theme).
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_SCRIPT = REPO_ROOT / "StockAnalysisSystem" / "src" / "main.py"


def main() -> None:
    from streamlit.web import cli as stcli

    os.chdir(REPO_ROOT)  # so Streamlit picks up .streamlit/config.toml
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    sys.argv = ["streamlit", "run", str(MAIN_SCRIPT), *sys.argv[1:]]
    print(f"Starting Stock Analysis System ({MAIN_SCRIPT.relative_to(REPO_ROOT)})...")
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
