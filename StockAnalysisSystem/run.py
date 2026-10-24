import os
import sys
import logging
from dotenv import load_dotenv
import streamlit.web.cli as stcli

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'stock_analysis.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_environment_variables():
    """
    Load environment variables from .env file.
    Ensures critical configuration is available before application startup.
    """
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info("Environment variables loaded successfully")
    else:
        logger.warning("No .env file found. Using system environment variables.")


def validate_api_keys():
    """
    Warn about missing optional API keys. yfinance works without any key;
    Alpha Vantage / Finnhub / FMP keys only enable extra providers.
    """
    optional_keys = ['ALPHA_VANTAGE_API_KEY', 'FINNHUB_API_KEY', 'FINANCIAL_MODELING_PREP_API_KEY']
    missing = [k for k in optional_keys if not os.getenv(k)]
    if missing:
        logger.warning(f"Optional API keys not set (extra providers disabled): {', '.join(missing)}")


def configure_system_path():
    """
    Add project root to Python path to enable module imports.
    """
    root_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, root_dir)
    # package imports use the `StockAnalysisSystem.src...` prefix
    sys.path.insert(0, os.path.dirname(root_dir))
    logger.info(f"Added {root_dir} to system path")


def main():
    """
    Main entry point for the Stock Analysis System application.
    Handles environment setup, validation, and application launch.
    """
    try:
        # Load environment variables
        load_environment_variables()

        # Validate critical configuration
        validate_api_keys()

        # Configure system path
        configure_system_path()

        # Configure Streamlit command line arguments
        sys.argv = [
            "streamlit",
            "run",
            os.path.join(os.path.dirname(__file__), "src/main.py"),
            "--theme.base=dark",
            "--theme.primaryColor=#74f174",
            "--theme.backgroundColor=#121212",
            "--theme.secondaryBackgroundColor=#1f1f1f",
            "--theme.textColor=#e0e0e0"
        ]

        logger.info("Starting Stock Analysis System...")
        logger.info(f"Application path: {os.path.join(os.path.dirname(__file__), 'src/main.py')}")

        # Run Streamlit
        sys.exit(stcli.main())

    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()