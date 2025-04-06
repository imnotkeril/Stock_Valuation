import os
import sys
import streamlit.web.cli as stcli


def main():
    """
    Main entry point for the Stock Analysis System application.
    Sets up the Python path and launches the Streamlit application.
    """
    # Add project root to Python path
    root_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, root_dir)

    # Configure Streamlit command line arguments
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(root_dir, "src/main.py"),
        "--server.enableCORS=false",
        "--theme.base=dark",
        "--theme.primaryColor=#74f174",
        "--theme.backgroundColor=#121212",
        "--theme.secondaryBackgroundColor=#1f1f1f",
        "--theme.textColor=#e0e0e0"
    ]

    print("Starting Stock Analysis System...")
    print(f"Application path: {os.path.join(root_dir, 'src/main.py')}")

    # Run Streamlit
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()