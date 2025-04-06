import os
import logging
import pandas as pd
import json
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import xlsxwriter

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import COLORS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('export')


class ExportManager:
    """
    Class for exporting analysis results to various formats like PDF, Excel, CSV, etc.
    Provides functions for generating reports, visualizations, and data files.
    """

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize export manager

        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "exports"

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.colors = COLORS
        logger.info(f"Initialized ExportManager with output directory: {self.output_dir}")

    def export_to_pdf(self, data: Dict[str, Any], report_type: str, filename: Optional[str] = None) -> str:
        """
        Export analysis results to PDF report

        Args:
            data: Dictionary with analysis data
            report_type: Type of report ('valuation', 'financial', 'comparison', 'forecast')
            filename: Custom filename for the report

        Returns:
            Path to the saved PDF file
        """
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available, PDF export disabled")
            return "PDF export not available (ReportLab library required)"

        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ticker = data.get('ticker', 'unknown')
                filename = f"{ticker}_{report_type}_report_{timestamp}.pdf"

            # Ensure .pdf extension
            if not filename.lower().endswith('.pdf'):
                filename += '.pdf'

            # Full file path
            filepath = self.output_dir / filename

            # Create PDF document
            doc = SimpleDocTemplate(str(filepath), pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []

            # Title style
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=18,
                alignment=1  # Center alignment
            )

            # Subtitle style
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=styles['Heading2'],
                fontSize=14,
                alignment=1
            )

            # Normal text style
            normal_style = styles['Normal']

            # Add report content based on type
            if report_type == 'valuation':
                self._add_valuation_report_content(data, elements, title_style, subtitle_style, normal_style)
            elif report_type == 'financial':
                self._add_financial_report_content(data, elements, title_style, subtitle_style, normal_style)
            elif report_type == 'comparison':
                self._add_comparison_report_content(data, elements, title_style, subtitle_style, normal_style)
            elif report_type == 'forecast':
                self._add_forecast_report_content(data, elements, title_style, subtitle_style, normal_style)
            else:
                elements.append(Paragraph(f"Unknown report type: {report_type}", normal_style))

            # Build PDF document
            doc.build(elements)

            logger.info(f"Successfully exported PDF report to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting to PDF: {e}")
            return f"Error exporting to PDF: {str(e)}"

    def export_to_excel(self, data: Dict[str, Any], report_type: str, filename: Optional[str] = None) -> str:
        """
        Export analysis results to Excel workbook

        Args:
            data: Dictionary with analysis data
            report_type: Type of report
            filename: Custom filename for the Excel file

        Returns:
            Path to the saved Excel file
        """
        if not EXCEL_AVAILABLE:
            logger.warning("XlsxWriter not available, Excel export disabled")
            return "Excel export not available (XlsxWriter library required)"

        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ticker = data.get('ticker', 'unknown')
                filename = f"{ticker}_{report_type}_data_{timestamp}.xlsx"

            # Ensure .xlsx extension
            if not filename.lower().endswith('.xlsx'):
                filename += '.xlsx'

            # Full file path
            filepath = self.output_dir / filename

            # Create Excel workbook
            with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
                workbook = writer.book

                # Create formats
                header_format = workbook.add_format({
                    'bold': True,
                    'text_wrap': True,
                    'valign': 'top',
                    'bg_color': '#D3D3D3',
                    'border': 1
                })

                # Add worksheets based on report type
                if report_type == 'valuation':
                    self._add_valuation_excel_sheets(data, writer, workbook, header_format)
                elif report_type == 'financial':
                    self._add_financial_excel_sheets(data, writer, workbook, header_format)
                elif report_type == 'comparison':
                    self._add_comparison_excel_sheets(data, writer, workbook, header_format)
                elif report_type == 'forecast':
                    self._add_forecast_excel_sheets(data, writer, workbook, header_format)
                else:
                    # Add a simple summary sheet
                    summary_df = pd.DataFrame({
                        'Key': ['Report Type', 'Export Date', 'Status'],
                        'Value': [report_type, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Unknown report type']
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)

            logger.info(f"Successfully exported Excel file to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            return f"Error exporting to Excel: {str(e)}"

    def export_to_csv(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export data to CSV file

        Args:
            data: Dictionary with data to export
            filename: Custom filename for the CSV file

        Returns:
            Path to the saved CSV file
        """
        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ticker = data.get('ticker', 'unknown')
                filename = f"{ticker}_data_{timestamp}.csv"

            # Ensure .csv extension
            if not filename.lower().endswith('.csv'):
                filename += '.csv'

            # Full file path
            filepath = self.output_dir / filename

            # Convert data to DataFrame
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, dict):
                # Try to convert dict to DataFrame
                try:
                    # First attempt: assume it's a dict of series or lists
                    df = pd.DataFrame(data)
                except ValueError:
                    # Second attempt: flatten the dict
                    flat_data = self._flatten_dict(data)
                    df = pd.DataFrame([flat_data])
            else:
                logger.error("Data must be a DataFrame or a compatible dictionary")
                return "Error: Data must be a DataFrame or a compatible dictionary"

            # Export to CSV
            df.to_csv(filepath, index=True)

            logger.info(f"Successfully exported CSV file to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return f"Error exporting to CSV: {str(e)}"

    def export_to_json(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        Export data to JSON file

        Args:
            data: Dictionary with data to export
            filename: Custom filename for the JSON file

        Returns:
            Path to the saved JSON file
        """
        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ticker = data.get('ticker', 'unknown')
                filename = f"{ticker}_data_{timestamp}.json"

            # Ensure .json extension
            if not filename.lower().endswith('.json'):
                filename += '.json'

            # Full file path
            filepath = self.output_dir / filename

            # Convert non-serializable objects to strings
            serializable_data = self._make_json_serializable(data)

            # Export to JSON
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)

            logger.info(f"Successfully exported JSON file to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return f"Error exporting to JSON: {str(e)}"

    def generate_visualization(self, data: Dict[str, Any], viz_type: str) -> str:
        """
        Generate visualization and return as base64 encoded string

        Args:
            data: Data to visualize
            viz_type: Type of visualization ('bar', 'line', 'radar', etc.)

        Returns:
            Base64 encoded string of the visualization image
        """
        try:
            plt.figure(figsize=(10, 6))

            # Set seaborn style
            sns.set(style="darkgrid")

            # Generate visualization based on type
            if viz_type == 'bar':
                self._create_bar_chart(data)
            elif viz_type == 'line':
                self._create_line_chart(data)
            elif viz_type == 'radar':
                self._create_radar_chart(data)
            elif viz_type == 'heatmap':
                self._create_heatmap(data)
            else:
                plt.text(0.5, 0.5, f"Unknown visualization type: {viz_type}",
                         horizontalalignment='center', verticalalignment='center')

            # Save to in-memory file
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)

            # Encode to base64
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Close the figure to free up memory
            plt.close()

            return img_str

        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return ""

    # Private helper methods

    def _add_valuation_report_content(self, data: Dict[str, Any], elements: List[Any],
                                      title_style: Any, subtitle_style: Any, normal_style: Any) -> None:
        """Add valuation report content to PDF elements list"""
        # Title
        ticker = data.get('ticker', 'Unknown')
        elements.append(Paragraph(f"Valuation Report: {ticker}", title_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Summary
        elements.append(Paragraph("Valuation Summary", subtitle_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Fair value and current price
        fair_value = data.get('fair_value')
        current_price = data.get('current_price')

        if fair_value is not None and current_price is not None:
            # Calculate upside
            upside_pct = (fair_value / current_price - 1) * 100 if current_price > 0 else None

            # Summary table
            summary_data = [
                ["Metric", "Value"],
                ["Fair Value", f"${fair_value:.2f}"],
                ["Current Price", f"${current_price:.2f}"],
            ]

            if upside_pct is not None:
                direction = "Upside" if upside_pct >= 0 else "Downside"
                summary_data.append([f"{direction} Potential", f"{abs(upside_pct):.1f}%"])

            # Assessment
            assessment = data.get('upside', {}).get('assessment', 'N/A')
            if assessment != 'N/A':
                assessment_text = assessment.replace('_', ' ').title()
                summary_data.append(["Assessment", assessment_text])

            # Create table
            summary_table = Table(summary_data, colWidths=[2 * inch, 3 * inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ]))

            elements.append(summary_table)
            elements.append(Spacer(1, 0.25 * inch))

        # Valuation method details
        elements.append(Paragraph("Valuation Method Details", subtitle_style))
        elements.append(Spacer(1, 0.1 * inch))

        method = data.get('valuation_method', 'Unknown')
        elements.append(Paragraph(f"Primary Valuation Method: {method}", normal_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Assumptions
        elements.append(Paragraph("Key Assumptions", subtitle_style))
        elements.append(Spacer(1, 0.1 * inch))

        assumptions = data.get('assumptions', {})
        if assumptions:
            # Create assumptions table
            assumptions_data = [["Assumption", "Value"]]

            for key, value in assumptions.items():
                # Format key for better readability
                formatted_key = key.replace('_', ' ').title()

                # Format value based on type
                if isinstance(value, (int, float)):
                    if 'rate' in key.lower() or 'percentage' in key.lower():
                        formatted_value = f"{value * 100:.2f}%" if -1 < value < 1 else f"{value:.2f}%"
                    else:
                        formatted_value = f"{value:,.2f}"
                elif isinstance(value, dict):
                    formatted_value = ", ".join([f"{k}: {v}" for k, v in value.items()])
                else:
                    formatted_value = str(value)

                assumptions_data.append([formatted_key, formatted_value])

            # Create table
            assumptions_table = Table(assumptions_data, colWidths=[2.5 * inch, 2.5 * inch])
            assumptions_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                ('BACKGROUND', (0, 1), (1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))

            elements.append(assumptions_table)
            elements.append(Spacer(1, 0.25 * inch))

        # Sensitivity analysis
        sensitivity = data.get('sensitivity', {})
        if sensitivity:
            elements.append(Paragraph("Sensitivity Analysis", subtitle_style))
            elements.append(Spacer(1, 0.1 * inch))

            elements.append(
                Paragraph("The table below shows how the fair value changes with different assumptions:", normal_style))
            elements.append(Spacer(1, 0.1 * inch))

            # This would typically include a visualization, but for simplicity
            # we'll just describe it textually
            elements.append(
                Paragraph("A sensitivity analysis chart would be included here in a full report.", normal_style))
            elements.append(Spacer(1, 0.25 * inch))

        # Footer
        elements.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))

    def _add_financial_report_content(self, data: Dict[str, Any], elements: List[Any],
                                      title_style: Any, subtitle_style: Any, normal_style: Any) -> None:
        """Add financial report content to PDF elements list"""
        # Title
        ticker = data.get('ticker', 'Unknown')
        elements.append(Paragraph(f"Financial Analysis Report: {ticker}", title_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Summary section
        elements.append(Paragraph("Financial Health Summary", subtitle_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Financial health overview
        health_score = data.get('health_score', {}).get('overall_score')
        if health_score is not None:
            elements.append(Paragraph(f"Overall Financial Health Score: {health_score:.1f}/100", normal_style))

            # Health components
            components = data.get('health_score', {}).get('components', {})
            if components:
                # Create health components table
                health_data = [["Component", "Score", "Assessment"]]

                for component, score in components.items():
                    # Format component name
                    component_name = component.title()

                    # Determine assessment
                    if score >= 80:
                        assessment = "Excellent"
                    elif score >= 60:
                        assessment = "Good"
                    elif score >= 40:
                        assessment = "Average"
                    elif score >= 20:
                        assessment = "Weak"
                    else:
                        assessment = "Poor"

                    health_data.append([component_name, f"{score:.1f}/100", assessment])

                # Create table
                health_table = Table(health_data, colWidths=[1.5 * inch, 1 * inch, 1.5 * inch])
                health_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (2, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (2, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (2, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (2, 0), 12),
                    ('BACKGROUND', (0, 1), (2, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (1, 1), (1, -1), 'CENTER'),
                ]))

                elements.append(health_table)
                elements.append(Spacer(1, 0.25 * inch))

        # Key ratios section
        elements.append(Paragraph("Key Financial Ratios", subtitle_style))
        elements.append(Spacer(1, 0.1 * inch))

        ratios = data.get('ratios', {})
        if ratios:
            # Create table for key ratios by category
            for category, category_ratios in ratios.items():
                if not category_ratios or not isinstance(category_ratios, dict):
                    continue

                # Skip non-ratio categories
                if category in ['sector', 'error']:
                    continue

                # Format category name
                category_name = category.replace('_', ' ').title()
                elements.append(Paragraph(f"{category_name} Ratios", normal_style))

                # Create ratio table
                ratio_data = [["Ratio", "Value", "Industry Avg", "Assessment"]]

                for ratio_name, ratio_info in category_ratios.items():
                    if not isinstance(ratio_info, dict) or 'value' not in ratio_info:
                        continue

                    # Format ratio name
                    formatted_name = ratio_name.replace('_', ' ').title()

                    # Format value
                    value = ratio_info.get('value')
                    if isinstance(value, float):
                        if value < 0.1:
                            formatted_value = f"{value:.4f}"
                        elif value < 1:
                            formatted_value = f"{value:.3f}"
                        elif value < 10:
                            formatted_value = f"{value:.2f}"
                        else:
                            formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = str(value)

                    # Format benchmark
                    benchmark = ratio_info.get('benchmark')
                    if benchmark is None:
                        formatted_benchmark = "N/A"
                    elif isinstance(benchmark, float):
                        if benchmark < 0.1:
                            formatted_benchmark = f"{benchmark:.4f}"
                        elif benchmark < 1:
                            formatted_benchmark = f"{benchmark:.3f}"
                        elif benchmark < 10:
                            formatted_benchmark = f"{benchmark:.2f}"
                        else:
                            formatted_benchmark = f"{benchmark:.1f}"
                    else:
                        formatted_benchmark = str(benchmark)

                    # Get assessment
                    assessment = ratio_info.get('assessment', 'N/A')
                    if assessment != 'N/A':
                        assessment = assessment.title()

                    ratio_data.append([formatted_name, formatted_value, formatted_benchmark, assessment])

                # Create table if we have ratio data
                if len(ratio_data) > 1:
                    ratio_table = Table(ratio_data, colWidths=[1.5 * inch, 1 * inch, 1 * inch, 1 * inch])
                    ratio_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (3, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (3, 0), 12),
                        ('BACKGROUND', (0, 1), (3, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ALIGN', (1, 1), (2, -1), 'RIGHT'),
                    ]))

                    elements.append(ratio_table)
                    elements.append(Spacer(1, 0.2 * inch))

        # Footer
        elements.append(Paragraph(f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))

    def _add_comparison_report_content(self, data: Dict[str, Any], elements: List[Any],
                                       title_style: Any, subtitle_style: Any, normal_style: Any) -> None:
        """Add comparison report content to PDF elements list"""
        # Title
        ticker = data.get('ticker', 'Unknown')
        elements.append(Paragraph(f"Peer Comparison Report: {ticker}", title_style))
        elements.append(Spacer(1, 0.25 * inch))

        # Peer companies section
        elements.append(Paragraph("Peer Companies Analysis", subtitle_style))
        elements.append(Spacer(1, 0.1 * inch))

        # Get peer data
        peers = data.get('peers', [])
        if peers:
            elements.append(Paragraph(f"Companies compared: {ticker}, {', '.join(peers)}", normal_style))
            elements.append(Spacer(1, 0.1 * inch))

        # Comparison summary
        comparison = data.get('comparison', {})
        if comparison:
            elements.append(Paragraph("Comparison Summary", normal_style))
            elements.append(Spacer(1, 0.1 * inch))

            # Create summary table
            summary_data = [["Metric", ticker, "Peer Average", "% Difference"]]

            metrics = comparison.get('metrics', {})
            for metric, values in metrics.items():
                if 'company_value' not in values or 'peer_average' not in values:
                    continue

                # Format metric name
                formatted_metric = metric.replace('_', ' ').title()

                # Get values
                company_value = values.get('company_value')
                peer_value = values.get('peer_average')

                # Format values
                if isinstance(company_value, float):
                    if abs(company_value) < 0.1:
                        formatted_company = f"{company_value:.4f}"
                    elif abs(company_value) < 1:
                        formatted_company = f"{company_value:.3f}"
                    elif abs(company_value) < 10:
                        formatted_company = f"{company_value:.2f}"
                    else:
                        formatted_company = f"{company_value:.1f}"
                else:
                    formatted_company = str(company_value) if company_value is not None else "N/A"

                if isinstance(peer_value, float):
                    if abs(peer_value) < 0.1:
                        formatted_peer = f"{peer_value:.4f}"
                    elif abs(peer_value) < 1:
                        formatted_peer = f"{peer_value:.3f}"
                    elif abs(peer_value) < 10:
                        formatted_peer = f"{peer_value:.2f}"
                    else:
                        formatted_peer = f"{peer_value:.1f}"
                else:
                    formatted_peer = str(peer_value) if peer_value is not None else "N/A"

                # Calculate percentage difference
                if company_value is not None and peer_value is not None and peer_value != 0:
                    pct_diff = (company_value / peer_value - 1) * 100
                    formatted_diff = f"{pct_diff:+.1f}%"
                else:
                    formatted_diff = "N/A"

                summary_data.append([formatted_metric, formatted_company, formatted_peer, formatted_diff])

            # Create table if we have data
            if len(summary_data) > 1:
                summary_table = Table(summary_data, colWidths=[1.5 * inch, 1 * inch, 1.2 * inch, 1 * inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (3, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (3, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (3, 0), 'CENTER'),
                    ('FONTNAME', (0, 0), (3, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (3, 0), 12),
                    ('BACKGROUND', (0, 1), (3, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (1, 1), (3, -1), 'RIGHT'),
                ]))

                elements.append(summary_table)
                elements.append(Spacer(1, 0.25 * inch))

        # Sector position
        sector_