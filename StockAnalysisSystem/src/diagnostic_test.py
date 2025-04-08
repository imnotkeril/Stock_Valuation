# Сохраните этот код в src/diagnostic_test.py

import streamlit as st
import sys
import os
import traceback
import pandas as pd
import plotly.graph_objects as go

# Добавление пути к проекту
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os

# Получаем абсолютный путь к корню проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Теперь можно импортировать
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer



st.write("Current working directory:", os.getcwd())
st.write("Python path:", sys.path)

def test_stock_analysis():
    st.title("Stock Analysis System - Diagnostic Test")

    # Боковая панель для ввода
    with st.sidebar:
        st.header("Company Selection")
        ticker = st.text_input("Enter Stock Ticker", "AAPL")
        analyze_button = st.button("Analyze Company")

    # Проверка загрузки данных
    if analyze_button:
        try:
            # Импорт модулей
            from StockAnalysisSystem.src.utils.data_loader import DataLoader
            from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer

            st.write("### 1. Data Loader Test")

            # Инициализация загрузчика данных
            data_loader = DataLoader()

            # Тест загрузки исторических цен
            st.write(f"#### Loading Price Data for {ticker}")
            price_data = data_loader.get_historical_prices(ticker)

            if not price_data.empty:
                st.success("✅ Price Data Loaded Successfully")
                st.dataframe(price_data.head())

                # Создание простого графика цен
                fig = go.Figure(data=[go.Candlestick(
                    x=price_data.index,
                    open=price_data['Open'],
                    high=price_data['High'],
                    low=price_data['Low'],
                    close=price_data['Close']
                )])
                st.plotly_chart(fig)
            else:
                st.error("❌ Failed to Load Price Data")

            # Тест загрузки финансовой отчетности
            st.write("### 2. Financial Statements Test")

            statements = {
                'Income Statement': data_loader.get_financial_statements(ticker, 'income'),
                'Balance Sheet': data_loader.get_financial_statements(ticker, 'balance'),
                'Cash Flow': data_loader.get_financial_statements(ticker, 'cash')
            }

            for name, statement in statements.items():
                st.write(f"#### {name}")
                if not statement.empty:
                    st.success(f"✅ {name} Loaded Successfully")
                    st.dataframe(statement.head())
                else:
                    st.error(f"❌ Failed to Load {name}")

            # Тест расчета финансовых коэффициентов
            st.write("### 3. Financial Ratios Test")

            ratio_analyzer = FinancialRatioAnalyzer()

            # Подготовка финансовых данных
            financial_data = {
                'income_statement': statements['Income Statement'],
                'balance_sheet': statements['Balance Sheet'],
                'cash_flow': statements['Cash Flow'],
                'market_data': data_loader.get_company_info(ticker)
            }

            try:
                ratios = ratio_analyzer.calculate_ratios(financial_data)
                st.success("✅ Ratios Calculated Successfully")

                # Отображение коэффициентов
                for category, category_ratios in ratios.items():
                    st.write(f"#### {category.capitalize()} Ratios")
                    st.json(category_ratios)

            except Exception as ratio_error:
                st.error(f"❌ Failed to Calculate Ratios: {ratio_error}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error(traceback.format_exc())


def main():
    test_stock_analysis()


if __name__ == "__main__":
    main()