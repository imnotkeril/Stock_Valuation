import os
import pathlib


def create_directories_and_files():
    """Create all directories and files for the Stock Analysis System project."""

    # Root directory
    root_dir = "StockAnalysisSystem"

    # Create root directory if it doesn't exist
    pathlib.Path(root_dir).mkdir(exist_ok=True)

    # Directory structure with empty files
    structure = {
        "src": {
            "files": ["main.py", "config.py"],
            "utils": {
                "files": ["data_loader.py", "visualization.py", "export.py"]
            },
            "models": {
                "files": ["financial_statements.py", "ratio_analysis.py", "bankruptcy_models.py"],
                "forecasting": {
                    "files": ["time_series.py", "ml_models.py"]
                }
            },
            "valuation": {
                "files": ["base_valuation.py", "dcf_models.py"],
                "sector_specific": {
                    "files": [
                        "financial_sector.py",
                        "tech_sector.py",
                        "energy_sector.py",
                        "retail_sector.py",
                        "manufacturing.py",
                        "real_estate.py",
                        "healthcare.py"
                    ]
                }
            },
            "industry": {
                "files": ["sector_mapping.py", "benchmarks.py", "sector_ratios.py"]
            },
            "pages": {
                "files": ["home.py", "forecasting.py", "reports.py"],
                "company_analysis": {
                    "files": ["overview.py", "financials.py", "valuation.py", "risk.py"]
                },
                "comparison": {
                    "files": ["peer_comparison.py", "sector_analysis.py"]
                },
                "screening": {
                    "files": ["stock_screener.py", "valuation_filter.py"]
                }
            },
            "data": {
                "cache": {
                    "files": [".gitkeep"]  # Just to keep the directory in git
                },
                "sector_data": {
                    "files": [".gitkeep"]  # Just to keep the directory in git
                }
            }
        }
    }

    # Root level files
    root_files = ["README.md", "requirements.txt", "run.py", "Dockerfile"]

    # Create all directories and files
    create_structure(root_dir, structure)

    # Create root files
    for file in root_files:
        file_path = os.path.join(root_dir, file)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("# " + file + "\n")

    print(f"Project structure created successfully in '{root_dir}' directory!")


def create_structure(parent_path, structure):
    """Recursively create directories and files based on the structure dictionary."""
    for dir_name, contents in structure.items():
        # Create directory
        dir_path = os.path.join(parent_path, dir_name)
        pathlib.Path(dir_path).mkdir(exist_ok=True)

        # Create files in this directory
        if "files" in contents:
            for file in contents["files"]:
                file_path = os.path.join(dir_path, file)
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write("# " + file + "\n")

        # Create subdirectories
        for key, value in contents.items():
            if key != "files" and isinstance(value, dict):
                create_structure(dir_path, {key: value})


if __name__ == "__main__":
    create_directories_and_files()