# create_project.py
import os
import sys


def create_directory(path):
    """Создает директорию, если она не существует"""
    os.makedirs(path, exist_ok=True)


def create_file(path):
    """Создает пустой файл, если он не существует"""
    if not os.path.exists(path):
        open(path, 'w').close()


def create_project_structure():
    # Базовая директория проекта
    base_dir = 'StockAnalysisSystem'

    # Основные директории
    main_dirs = [
        'src',
        'src/utils',
        'src/models',
        'src/models/forecasting',
        'src/valuation',
        'src/valuation/sector_specific',
        'src/industry',
        'src/comparison',
        'src/screening',
        'src/pages',
        'src/pages/company_analysis',
        'src/pages/comparison',
        'src/pages/screening',
        'data/cache',
        'data/sector_data',
        'tests/unit',
        'tests/integration'
    ]

    # Создание директорий
    for dir_path in main_dirs:
        create_directory(os.path.join(base_dir, dir_path))

    # Создание __init__.py файлов
    init_files = [
        'src/__init__.py',
        'src/utils/__init__.py',
        'src/models/__init__.py',
        'src/models/forecasting/__init__.py',
        'src/valuation/__init__.py',
        'src/valuation/sector_specific/__init__.py',
        'src/industry/__init__.py',
        'src/comparison/__init__.py',
        'src/screening/__init__.py',
        'src/pages/__init__.py',
        'src/pages/company_analysis/__init__.py',
        'src/pages/comparison/__init__.py',
        'src/pages/screening/__init__.py'
    ]

    for init_file in init_files:
        create_file(os.path.join(base_dir, init_file))

    # Создание основных файлов проекта
    main_files = [
        'src/main.py',
        'src/config.py',
        'run.py',
        'requirements.txt',
        'README.md',

        # Утилиты
        'src/utils/data_loader.py',
        'src/utils/visualization.py',
        'src/utils/export.py',
        'src/utils/helpers.py',

        # Модели
        'src/models/financial_statements.py',
        'src/models/ratio_analysis.py',
        'src/models/bankruptcy_models.py',
        'src/models/forecasting/time_series.py',
        'src/models/forecasting/ml_models.py',

        # Valuation
        'src/valuation/base_valuation.py',
        'src/valuation/dcf_models.py',
        'src/valuation/sector_factor.py',

        # Секторные модели оценки
        'src/valuation/sector_specific/financial_sector.py',
        'src/valuation/sector_specific/tech_sector.py',
        'src/valuation/sector_specific/energy_sector.py',
        'src/valuation/sector_specific/retail_sector.py',
        'src/valuation/sector_specific/manufacturing.py',
        'src/valuation/sector_specific/real_estate.py',
        'src/valuation/sector_specific/healthcare.py',
        'src/valuation/sector_specific/communication_sector.py',

        # Industry
        'src/industry/sector_mapping.py',
        'src/industry/benchmarks.py',
        'src/industry/sector_ratios.py',

        # Comparison
        'src/comparison/peer_finder.py',
        'src/comparison/peer_comparison.py',
        'src/comparison/industry_benchmarking.py',

        # Screening
        'src/screening/screener_engine.py',
        'src/screening/filters.py',
        'src/screening/strategies.py',

        # Pages
        'src/pages/home.py',
        'src/pages/forecasting.py',
        'src/pages/reports.py',
        'src/pages/company_analysis/overview.py',
        'src/pages/company_analysis/financials.py',
        'src/pages/company_analysis/valuation.py',
        'src/pages/company_analysis/risk.py',
        'src/pages/comparison/peer_comparison.py',
        'src/pages/comparison/sector_analysis.py',
        'src/pages/screening/stock_screener.py',
        'src/pages/screening/valuation_filter.py'
    ]

    for main_file in main_files:
        create_file(os.path.join(base_dir, main_file))

    print("Структура проекта StockAnalysisSystem создана успешно!")


if __name__ == "__main__":
    create_project_structure()