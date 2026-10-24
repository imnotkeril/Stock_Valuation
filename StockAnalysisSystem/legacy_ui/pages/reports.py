import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import base64
import io

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Import project modules
from StockAnalysisSystem.src.config import UI_SETTINGS, COLORS, VIZ_SETTINGS
from StockAnalysisSystem.src.utils.data_loader import DataLoader
from StockAnalysisSystem.src.utils.visualization import FinancialVisualizer
from StockAnalysisSystem.src.models.ratio_analysis import FinancialRatioAnalyzer
from StockAnalysisSystem.src.models.financial_statements import FinancialStatementAnalyzer
from StockAnalysisSystem.src.models.bankruptcy_models import BankruptcyAnalyzer
from StockAnalysisSystem.src.valuation.sector_factor import ValuationFactory
from StockAnalysisSystem.src.utils.export import ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reports')


def run_reports_page():
    """Main function to run the reports page"""
    st.title("Financial Reports Generator")
    st.markdown(
        "Generate comprehensive financial reports and analysis documents for companies, sectors, or custom portfolio analysis.")

    # Initialize data loader and analyzers
    data_loader = DataLoader()
    visualizer = FinancialVisualizer(theme="dark")
    ratio_analyzer = FinancialRatioAnalyzer()
    statement_analyzer = FinancialStatementAnalyzer()
    bankruptcy_analyzer = BankruptcyAnalyzer()
    valuation_factory = ValuationFactory(data_loader)
    report_generator = ReportGenerator()

    # Sidebar for report options
    with st.sidebar:
        st.header("Report Options")

        # Report type selection
        report_types = {
            "Company Analysis": "Comprehensive analysis of a single company",
            "Sector Analysis": "Overview and analysis of a market sector",
            "Comparative Analysis": "Compare multiple companies",
            "Valuation Report": "Detailed valuation of a company",
            "Portfolio Analysis": "Analysis of a custom portfolio"
        }

        report_type = st.selectbox(
            "Select report type:",
            options=list(report_types.keys())
        )

        st.info(report_types[report_type])

        # Report options based on selection
        if report_type == "Company Analysis":
            # Company selection
            ticker = st.text_input("Enter ticker symbol:", "AAPL").upper()

            # Sections to include
            st.subheader("Sections to Include")

            include_company_overview = st.checkbox("Company Overview", value=True)
            include_financial_analysis = st.checkbox("Financial Analysis", value=True)
            include_ratio_analysis = st.checkbox("Ratio Analysis", value=True)
            include_valuation = st.checkbox("Valuation", value=True)
            include_risk_analysis = st.checkbox("Risk Analysis", value=True)
            include_technical_analysis = st.checkbox("Technical Analysis", value=False)
            include_charts = st.checkbox("Include Charts", value=True)

            # Time period
            st.subheader("Financial Data Period")

            period_options = {
                "1 Year": 1,
                "3 Years": 3,
                "5 Years": 5,
                "10 Years": 10
            }

            financial_period = st.selectbox(
                "Financial data period:",
                options=list(period_options.keys()),
                index=2  # Default to 5 years
            )

            years = period_options[financial_period]

            # Statement frequency
            statement_frequency = st.selectbox(
                "Financial statement frequency:",
                options=["Annual", "Quarterly"],
                index=0  # Default to annual
            )

        elif report_type == "Sector Analysis":
            # Sector selection
            sectors = [
                "Technology",
                "Healthcare",
                "Financials",
                "Consumer Discretionary",
                "Consumer Staples",
                "Energy",
                "Industrials",
                "Materials",
                "Real Estate",
                "Communication Services",
                "Utilities"
            ]

            selected_sector = st.selectbox(
                "Select sector:",
                options=sectors
            )

            # Number of companies to include
            top_n = st.slider(
                "Number of companies to include:",
                min_value=5,
                max_value=30,
                value=10,
                step=5
            )

            # Sections to include
            st.subheader("Sections to Include")

            include_sector_overview = st.checkbox("Sector Overview", value=True)
            include_performance_analysis = st.checkbox("Performance Analysis", value=True)
            include_top_companies = st.checkbox("Top Companies Analysis", value=True)
            include_valuation_metrics = st.checkbox("Valuation Metrics", value=True)
            include_financial_metrics = st.checkbox("Financial Metrics", value=True)
            include_charts = st.checkbox("Include Charts", value=True)

            # Time period
            st.subheader("Analysis Period")

            period_options = {
                "1 Year": 365,
                "3 Years": 3 * 365,
                "5 Years": 5 * 365,
                "10 Years": 10 * 365
            }

            analysis_period = st.selectbox(
                "Analysis period:",
                options=list(period_options.keys()),
                index=1  # Default to 3 years
            )

            days = period_options[analysis_period]

        elif report_type == "Comparative Analysis":
            # Companies selection
            st.subheader("Select Companies")

            company1 = st.text_input("Company 1:", "AAPL").upper()
            company2 = st.text_input("Company 2:", "MSFT").upper()
            company3 = st.text_input("Company 3 (optional):", "").upper()
            company4 = st.text_input("Company 4 (optional):", "").upper()
            company5 = st.text_input("Company 5 (optional):", "").upper()

            # Create list of companies
            companies = [company1, company2]
            if company3:
                companies.append(company3)
            if company4:
                companies.append(company4)
            if company5:
                companies.append(company5)

            # Sections to include
            st.subheader("Sections to Include")

            include_companies_overview = st.checkbox("Companies Overview", value=True)
            include_performance_comparison = st.checkbox("Performance Comparison", value=True)
            include_financial_comparison = st.checkbox("Financial Comparison", value=True)
            include_valuation_comparison = st.checkbox("Valuation Comparison", value=True)
            include_ratio_comparison = st.checkbox("Ratio Comparison", value=True)
            include_charts = st.checkbox("Include Charts", value=True)

            # Time period
            st.subheader("Analysis Period")

            period_options = {
                "1 Year": 365,
                "3 Years": 3 * 365,
                "5 Years": 5 * 365,
                "10 Years": 10 * 365
            }

            analysis_period = st.selectbox(
                "Analysis period:",
                options=list(period_options.keys()),
                index=1  # Default to 3 years
            )

            days = period_options[analysis_period]

            # Statement frequency
            statement_frequency = st.selectbox(
                "Financial statement frequency:",
                options=["Annual", "Quarterly"],
                index=0  # Default to annual
            )

        elif report_type == "Valuation Report":
            # Company selection
            ticker = st.text_input("Enter ticker symbol:", "AAPL").upper()

            # Valuation methods
            st.subheader("Valuation Methods")

            include_dcf = st.checkbox("Discounted Cash Flow (DCF)", value=True)
            include_relative = st.checkbox("Relative Valuation", value=True)
            include_dividend = st.checkbox("Dividend Discount Model", value=True)
            include_asset = st.checkbox("Asset-Based Valuation", value=False)

            # DCF parameters
            if include_dcf:
                with st.expander("DCF Parameters", expanded=True):
                    forecast_years = st.slider(
                        "Forecast Period (Years):",
                        min_value=3,
                        max_value=10,
                        value=5
                    )

                    discount_rate = st.slider(
                        "Discount Rate (%):",
                        min_value=5.0,
                        max_value=20.0,
                        value=10.0,
                        step=0.5
                    )

                    terminal_growth = st.slider(
                        "Terminal Growth Rate (%):",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.1
                    )

            # Sections to include
            st.subheader("Sections to Include")

            include_company_overview = st.checkbox("Company Overview", value=True)
            include_historical_financials = st.checkbox("Historical Financials", value=True)
            include_valuation_summary = st.checkbox("Valuation Summary", value=True)
            include_valuation_details = st.checkbox("Valuation Details", value=True)
            include_sensitivity_analysis = st.checkbox("Sensitivity Analysis", value=True)
            include_fair_value_range = st.checkbox("Fair Value Range", value=True)
            include_charts = st.checkbox("Include Charts", value=True)

        elif report_type == "Portfolio Analysis":
            # Portfolio composition
            st.subheader("Portfolio Composition")

            # Option to upload CSV with portfolio
            upload_option = st.radio(
                "Portfolio data:",
                options=["Enter manually", "Upload CSV"]
            )

            portfolio_data = []

            if upload_option == "Enter manually":
                # Manual entry
                for i in range(1, 11):
                    col1, col2 = st.columns(2)

                    with col1:
                        ticker = st.text_input(f"Stock {i}:", "").upper()

                    with col2:
                        weight = st.number_input(f"Weight {i} (%):", min_value=0.0, max_value=100.0, value=0.0,
                                                 step=5.0)

                    if ticker and weight > 0:
                        portfolio_data.append({"ticker": ticker, "weight": weight / 100})
            else:
                # CSV upload
                uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"])

                if uploaded_file is not None:
                    try:
                        # Read CSV
                        portfolio_df = pd.read_csv(uploaded_file)

                        # Check required columns
                        if "ticker" in portfolio_df.columns and "weight" in portfolio_df.columns:
                            # Convert to list of dictionaries
                            portfolio_data = portfolio_df[["ticker", "weight"]].to_dict('records')

                            # Normalize weights if needed
                            total_weight = sum(item["weight"] for item in portfolio_data)
                            if total_weight != 1.0:
                                for item in portfolio_data:
                                    item["weight"] = item["weight"] / total_weight

                            # Show preview
                            st.success(f"Successfully loaded {len(portfolio_data)} positions.")
                            st.dataframe(portfolio_df)
                        else:
                            st.error("CSV must contain 'ticker' and 'weight' columns.")
                    except Exception as e:
                        st.error(f"Error loading CSV: {str(e)}")

            # Sections to include
            st.subheader("Sections to Include")

            include_portfolio_overview = st.checkbox("Portfolio Overview", value=True)
            include_performance_analysis = st.checkbox("Performance Analysis", value=True)
            include_risk_analysis = st.checkbox("Risk Analysis", value=True)
            include_diversification = st.checkbox("Diversification Analysis", value=True)
            include_valuation = st.checkbox("Valuation Assessment", value=True)
            include_individual_analysis = st.checkbox("Individual Holdings Analysis", value=True)
            include_charts = st.checkbox("Include Charts", value=True)

            # Time period
            st.subheader("Analysis Period")

            period_options = {
                "1 Year": 365,
                "3 Years": 3 * 365,
                "5 Years": 5 * 365,
                "10 Years": 10 * 365
            }

            analysis_period = st.selectbox(
                "Analysis period:",
                options=list(period_options.keys()),
                index=1  # Default to 3 years
            )

            days = period_options[analysis_period]

        # Report format
        st.subheader("Report Format")

        report_format = st.selectbox(
            "Select report format:",
            options=["PDF", "Excel", "HTML"],
            index=0  # Default to PDF
        )

        # Include cover page
        include_cover = st.checkbox("Include Cover Page", value=True)
        # Generate report button
        generate_report_btn = st.sidebar.button("Generate Report", type="primary")

        # Main content area
        if generate_report_btn:
            try:
                with st.spinner("Generating report... This may take a moment."):
                    if report_type == "Company Analysis":
                        # Validate input
                        if not ticker:
                            st.error("Please enter a valid ticker symbol.")
                            st.stop()

                        # Check if ticker exists
                        try:
                            company_info = data_loader.get_company_info(ticker)
                            if not company_info or "name" not in company_info:
                                st.error(f"Could not find information for ticker '{ticker}'. Please verify the symbol.")
                                st.stop()
                        except Exception as e:
                            st.error(f"Error retrieving company information: {str(e)}")
                            st.stop()

                        st.subheader(f"Generating report for {company_info.get('name', ticker)} ({ticker})")

                        # Load data
                        start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
                        end_date = datetime.now().strftime('%Y-%m-%d')

                        # Progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 1. Load price data
                        status_text.text("Loading price data...")
                        price_data = data_loader.get_historical_prices(ticker, start_date, end_date)
                        progress_bar.progress(10)

                        # 2. Load financial statements
                        status_text.text("Loading financial statements...")
                        freq = "annual" if statement_frequency == "Annual" else "quarterly"
                        income_stmt = data_loader.get_financial_statements(ticker, 'income', freq)
                        balance_sheet = data_loader.get_financial_statements(ticker, 'balance', freq)
                        cash_flow = data_loader.get_financial_statements(ticker, 'cash', freq)
                        progress_bar.progress(30)

                        # 3. Perform analysis
                        status_text.text("Performing financial analysis...")
                        financial_data = {
                            'income_statement': income_stmt,
                            'balance_sheet': balance_sheet,
                            'cash_flow': cash_flow,
                            'market_data': {
                                'share_price': price_data['Close'].iloc[-1] if not price_data.empty else None,
                                'market_cap': company_info.get('market_cap'),
                                'beta': company_info.get('beta')
                            }
                        }

                        # Calculate ratios
                        ratios = ratio_analyzer.calculate_ratios(financial_data)
                        progress_bar.progress(50)

                        # 4. Perform valuation
                        status_text.text("Calculating valuation...")
                        sector = company_info.get('sector', 'Unknown')
                        valuation_models = {}

                        if include_valuation:
                            # Get appropriate valuation factory
                            valuation_model = valuation_factory.get_valuation_model(sector)

                            # DCF Valuation
                            if include_dcf:
                                dcf_params = {
                                    'forecast_years': forecast_years,
                                    'discount_rate': discount_rate / 100,
                                    'terminal_growth': terminal_growth / 100
                                }
                                dcf_result = valuation_model.discounted_cash_flow_valuation(ticker, financial_data,
                                                                                            sector, **dcf_params)
                                valuation_models['DCF'] = dcf_result

                            # Relative Valuation
                            if include_relative:
                                relative_result = valuation_model.relative_valuation(ticker, financial_data, sector)
                                valuation_models['Relative'] = relative_result

                            # Dividend Discount Model
                            if include_dividend:
                                dividend_result = valuation_model.dividend_discount_valuation(ticker, financial_data,
                                                                                              sector)
                                valuation_models['DDM'] = dividend_result

                            # Asset-Based Valuation
                            if include_asset:
                                asset_result = valuation_model.asset_based_valuation(ticker, financial_data, sector)
                                valuation_models['Asset-Based'] = asset_result

                        progress_bar.progress(70)

                        # 5. Risk analysis
                        status_text.text("Analyzing risks...")
                        risk_assessment = None
                        if include_risk_analysis:
                            risk_assessment = bankruptcy_analyzer.get_comprehensive_risk_assessment(financial_data,
                                                                                                    sector)

                        progress_bar.progress(80)

                        # 6. Generate visualizations if needed
                        charts = {}
                        if include_charts:
                            status_text.text("Generating charts...")
                            # Price chart
                            price_chart = visualizer.plot_stock_price(price_data, ticker,
                                                                      company_name=company_info.get('name'))
                            charts['price'] = price_chart

                            # Financial ratio charts
                            if include_ratio_analysis:
                                ratio_chart = visualizer.plot_financial_ratios(
                                    {category: ratio_analyzer.analyze_ratios(ratios, sector).get(category, {})
                                     for category in ratios.keys()},
                                    benchmark_data=ratio_analyzer.get_sector_benchmarks(sector)
                                )
                                charts['ratios'] = ratio_chart

                        progress_bar.progress(90)

                        # 7. Generate report
                        status_text.text("Generating final report...")

                        report_data = {
                            "report_type": "Company Analysis",
                            "company_info": company_info,
                            "price_data": price_data,
                            "financial_data": financial_data,
                            "ratios": ratios,
                            "valuation_models": valuation_models,
                            "risk_assessment": risk_assessment,
                            "charts": charts,
                            "period": financial_period,
                            "sections": {
                                "company_overview": include_company_overview,
                                "financial_analysis": include_financial_analysis,
                                "ratio_analysis": include_ratio_analysis,
                                "valuation": include_valuation,
                                "risk_analysis": include_risk_analysis,
                                "technical_analysis": include_technical_analysis
                            },
                            "include_charts": include_charts,
                            "include_cover": include_cover
                        }

                        # Generate the actual report based on format
                        if report_format == "PDF":
                            report_file = report_generator.generate_pdf_report(report_data)
                            mime_type = "application/pdf"
                            file_extension = "pdf"
                        elif report_format == "Excel":
                            report_file = report_generator.generate_excel_report(report_data)
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            file_extension = "xlsx"
                        else:  # HTML
                            report_file = report_generator.generate_html_report(report_data)
                            mime_type = "text/html"
                            file_extension = "html"

                        progress_bar.progress(100)
                        status_text.text("Report generated successfully!")

                        # Provide download link
                        st.success(f"Report for {company_info.get('name', ticker)} generated successfully!")

                        # Create download button
                        report_filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d')}.{file_extension}"

                        # Convert to base64 for download
                        b64 = base64.b64encode(report_file).decode()
                        href = f'<a href="data:{mime_type};base64,{b64}" download="{report_filename}">Download {report_format} Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Preview section
                        with st.expander("Report Preview", expanded=True):
                            if report_format == "PDF":
                                st.warning("PDF preview not available. Please download the report to view.")
                            elif report_format == "Excel":
                                st.warning("Excel preview not available. Please download the report to view.")
                            else:  # HTML
                                st.components.v1.html(report_file.decode(), height=600)

                    elif report_type == "Sector Analysis":
                        # Implementation for sector analysis report
                        st.subheader(f"Generating Sector Analysis for {selected_sector}")

                        # Load sector data
                        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                        end_date = datetime.now().strftime('%Y-%m-%d')

                        # Progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # 1. Load sector ETF data
                        status_text.text("Loading sector data...")

                        # Define sector ETFs mapping
                        sector_etfs = {
                            "Technology": "XLK",
                            "Healthcare": "XLV",
                            "Financials": "XLF",
                            "Consumer Discretionary": "XLY",
                            "Consumer Staples": "XLP",
                            "Energy": "XLE",
                            "Industrials": "XLI",
                            "Materials": "XLB",
                            "Real Estate": "XLRE",
                            "Communication Services": "XLC",
                            "Utilities": "XLU"
                        }

                        etf_ticker = sector_etfs.get(selected_sector)
                        etf_data = data_loader.get_historical_prices(etf_ticker, start_date, end_date)
                        progress_bar.progress(20)

                        # 2. Get top companies in sector
                        status_text.text(f"Identifying top {top_n} companies in {selected_sector}...")
                        # In a real implementation, we would query a database or API to get top companies
                        # For now, we'll use hardcoded values as a placeholder
                        top_companies = get_top_companies_in_sector(selected_sector, top_n)
                        progress_bar.progress(40)

                        # 3. Load data for top companies
                        status_text.text("Loading company data...")
                        companies_data = {}
                        for company in top_companies:
                            ticker = company["ticker"]
                            companies_data[ticker] = {
                                "info": data_loader.get_company_info(ticker),
                                "prices": data_loader.get_historical_prices(ticker, start_date, end_date)
                            }
                        progress_bar.progress(70)

                        # 4. Generate visualizations
                        charts = {}
                        if include_charts:
                            status_text.text("Generating charts...")
                            # Sector performance chart
                            performance_chart = visualizer.plot_sector_performance(etf_data, etf_ticker,
                                                                                   selected_sector)
                            charts['sector_performance'] = performance_chart

                            # Companies comparison chart
                            comparison_data = {ticker: data["prices"] for ticker, data in companies_data.items()}
                            companies_chart = visualizer.plot_companies_comparison(comparison_data)
                            charts['companies_comparison'] = companies_chart

                        progress_bar.progress(85)

                        # 5. Generate report
                        status_text.text("Generating final report...")

                        report_data = {
                            "report_type": "Sector Analysis",
                            "sector": selected_sector,
                            "etf_data": etf_data,
                            "top_companies": top_companies,
                            "companies_data": companies_data,
                            "charts": charts,
                            "period": analysis_period,
                            "sections": {
                                "sector_overview": include_sector_overview,
                                "performance_analysis": include_performance_analysis,
                                "top_companies": include_top_companies,
                                "valuation_metrics": include_valuation_metrics,
                                "financial_metrics": include_financial_metrics
                            },
                            "include_charts": include_charts,
                            "include_cover": include_cover
                        }

                        # Generate the actual report based on format
                        if report_format == "PDF":
                            report_file = report_generator.generate_pdf_report(report_data)
                            mime_type = "application/pdf"
                            file_extension = "pdf"
                        elif report_format == "Excel":
                            report_file = report_generator.generate_excel_report(report_data)
                            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            file_extension = "xlsx"
                        else:  # HTML
                            report_file = report_generator.generate_html_report(report_data)
                            mime_type = "text/html"
                            file_extension = "html"

                        progress_bar.progress(100)
                        status_text.text("Report generated successfully!")

                        # Provide download link
                        st.success(f"Sector Analysis for {selected_sector} generated successfully!")

                        # Create download button
                        report_filename = f"{selected_sector}_analysis_{datetime.now().strftime('%Y%m%d')}.{file_extension}"

                        # Convert to base64 for download
                        b64 = base64.b64encode(report_file).decode()
                        href = f'<a href="data:{mime_type};base64,{b64}" download="{report_filename}">Download {report_format} Report</a>'
                        st.markdown(href, unsafe_allow_html=True)

                        # Preview section
                        with st.expander("Report Preview", expanded=True):
                            if report_format == "PDF":
                                st.warning("PDF preview not available. Please download the report to view.")
                            elif report_format == "Excel":
                                st.warning("Excel preview not available. Please download the report to view.")
                            else:  # HTML
                                st.components.v1.html(report_file.decode(), height=600)

                    # Implementations for other report types would go here
                    # For now we'll just show a placeholder message
                    else:
                        st.info(f"Implementation for {report_type} is coming soon.")

            except Exception as e:
                st.error(f"An error occurred while generating the report: {str(e)}")
                st.exception(e)
        else:
            # Show instructions when no report is being generated
            if report_type == "Company Analysis":
                st.subheader("Company Analysis Report")
                st.write("""
                This report provides a comprehensive analysis of a single company, including:

                - Company overview and business description
                - Historical financial analysis and key metrics
                - Ratio analysis with industry comparison
                - Valuation using appropriate methods
                - Risk assessment and financial health evaluation

                Configure your report options in the sidebar and click "Generate Report" to create your custom analysis.
                """)

                # Example screenshot or placeholder
                st.image("https://via.placeholder.com/800x400?text=Company+Analysis+Report+Example",
                         use_column_width=True)

            elif report_type == "Sector Analysis":
                st.subheader("Sector Analysis Report")
                st.write("""
                This report provides insights and analysis of an entire market sector, including:

                - Sector overview and current trends
                - Performance analysis against broader market
                - Top performing companies in the sector
                - Valuation metrics across the sector
                - Key financial metrics and growth drivers

                Configure your report options in the sidebar and click "Generate Report" to create your sector analysis.
                """)

                # Example screenshot or placeholder
                st.image("https://via.placeholder.com/800x400?text=Sector+Analysis+Report+Example",
                         use_column_width=True)

            elif report_type == "Comparative Analysis":
                st.subheader("Comparative Analysis Report")
                st.write("""
                This report provides side-by-side comparison of multiple companies, including:

                - Overview of each company
                - Comparative performance analysis
                - Financial metrics comparison
                - Valuation multiples comparison
                - Strengths and weaknesses analysis

                Enter ticker symbols for the companies you want to compare, configure options in the sidebar, and click "Generate Report".
                """)

                # Example screenshot or placeholder
                st.image("https://via.placeholder.com/800x400?text=Comparative+Analysis+Report+Example",
                         use_column_width=True)

            elif report_type == "Valuation Report":
                st.subheader("Valuation Report")
                st.write("""
                This report provides an in-depth valuation analysis of a company, including:

                - Multiple valuation methodologies (DCF, Relative, etc.)
                - Detailed assumptions and inputs
                - Sensitivity analysis for key variables
                - Fair value range estimation
                - Comparison with current market price

                Configure valuation parameters in the sidebar and click "Generate Report" to create your valuation analysis.
                """)

                # Example screenshot or placeholder
                st.image("https://via.placeholder.com/800x400?text=Valuation+Report+Example", use_column_width=True)

            elif report_type == "Portfolio Analysis":
                st.subheader("Portfolio Analysis Report")
                st.write("""
                This report provides analysis of a custom investment portfolio, including:

                - Portfolio composition and allocation
                - Performance analysis and attribution
                - Risk metrics (volatility, drawdowns, etc.)
                - Diversification assessment
                - Individual holdings analysis

                Enter your portfolio holdings or upload a CSV file, configure options in the sidebar, and click "Generate Report".
                """)

                # Example CSV template
                st.subheader("CSV Template")
                st.write("If uploading a CSV file, please use the following format:")

                df_template = pd.DataFrame({
                    "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                    "weight": [0.25, 0.25, 0.25, 0.25],
                    "shares": [10, 5, 2, 1]  # Optional
                })

                st.dataframe(df_template)

                # Download template button
                csv = df_template.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="portfolio_template.csv">Download CSV Template</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Example screenshot or placeholder
                st.image("https://via.placeholder.com/800x400?text=Portfolio+Analysis+Report+Example",
                         use_column_width=True)

        def get_top_companies_in_sector(sector, n=10):
            """
            Get top companies in a specific sector.
            This is a placeholder implementation - in a real application,
            this would query a database or API.

            Args:
                sector: Market sector name
                n: Number of companies to return

            Returns:
                List of dictionaries with company information
            """
            # Sample data for demonstration
            sector_companies = {
                "Technology": [
                    {"ticker": "AAPL", "name": "Apple Inc.", "market_cap": 2500000000000},
                    {"ticker": "MSFT", "name": "Microsoft Corporation", "market_cap": 2300000000000},
                    {"ticker": "GOOGL", "name": "Alphabet Inc.", "market_cap": 1800000000000},
                    {"ticker": "AMZN", "name": "Amazon.com Inc.", "market_cap": 1600000000000},
                    {"ticker": "META", "name": "Meta Platforms Inc.", "market_cap": 1000000000000},
                    {"ticker": "NVDA", "name": "NVIDIA Corporation", "market_cap": 900000000000},
                    {"ticker": "TSLA", "name": "Tesla, Inc.", "market_cap": 800000000000},
                    {"ticker": "AVGO", "name": "Broadcom Inc.", "market_cap": 400000000000},
                    {"ticker": "CSCO", "name": "Cisco Systems, Inc.", "market_cap": 200000000000},
                    {"ticker": "INTC", "name": "Intel Corporation", "market_cap": 150000000000}
                ],
                "Healthcare": [
                    {"ticker": "JNJ", "name": "Johnson & Johnson", "market_cap": 400000000000},
                    {"ticker": "UNH", "name": "UnitedHealth Group Inc.", "market_cap": 380000000000},
                    {"ticker": "PFE", "name": "Pfizer Inc.", "market_cap": 200000000000},
                    {"ticker": "ABBV", "name": "AbbVie Inc.", "market_cap": 180000000000},
                    {"ticker": "MRK", "name": "Merck & Co., Inc.", "market_cap": 170000000000},
                    {"ticker": "LLY", "name": "Eli Lilly and Company", "market_cap": 160000000000},
                    {"ticker": "TMO", "name": "Thermo Fisher Scientific Inc.", "market_cap": 150000000000},
                    {"ticker": "ABT", "name": "Abbott Laboratories", "market_cap": 140000000000},
                    {"ticker": "DHR", "name": "Danaher Corporation", "market_cap": 130000000000},
                    {"ticker": "BMY", "name": "Bristol-Myers Squibb Company", "market_cap": 120000000000}
                ],
                # Add data for other sectors as needed
            }

            # Return companies for the requested sector
            return sector_companies.get(sector, [])[:n]

        if __name__ == "__main__":
            run_reports_page()