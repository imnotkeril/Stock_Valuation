import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any

# Add the parent directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sector_ratios')


class SectorRatios:
    """
    Provides sector-specific financial ratios, benchmarks, and analysis methods.
    Based on industry standards and best practices for different sectors of the economy.
    """

    def __init__(self):
        """Initialize SectorRatios with predefined sector information"""
        # Define key ratios and benchmarks for each sector
        self.sector_ratios = self._initialize_sector_ratios()

        # Define sector-specific valuation methods
        self.valuation_methods = self._initialize_valuation_methods()

        # Define sector mapping for subsectors
        self.subsector_mapping = self._initialize_subsector_mapping()

        logger.info("Initialized SectorRatios")

    def _initialize_sector_ratios(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize sector-specific ratios and benchmarks

        Returns:
            Dictionary with sector-specific ratios and benchmarks
        """
        # Information derived from "Сектора_фондового_рынка.pdf" and industry standards
        sector_ratios = {
            "Information Technology": {
                "key_ratios": [
                    "P/E", "Forward P/E", "PEG", "P/S", "P/B", "EV/EBITDA", "EV/Revenue",
                    "R&D/Revenue", "Gross Margin", "Operating Margin", "ROIC"
                ],
                "benchmarks": {
                    "P/E": 25.0,
                    "Forward P/E": 20.0,
                    "P/S": 5.0,
                    "P/B": 5.0,
                    "EV/EBITDA": 15.0,
                    "EV/Revenue": 5.0,
                    "R&D/Revenue": 0.15,
                    "Gross Margin": 0.50,
                    "Operating Margin": 0.20,
                    "ROIC": 0.15,
                    "Revenue Growth": 0.15
                },
                "description": "Technology companies often trade at higher multiples due to growth prospects. Focus on R&D investments, margins, and scalability. Cash flow generation and return on invested capital are key indicators of sustainable competitive advantage."
            },

            "Healthcare": {
                "key_ratios": [
                    "P/E", "Forward P/E", "P/S", "P/B", "EV/EBITDA",
                    "R&D/Revenue", "Operating Margin", "ROE", "Net Margin"
                ],
                "benchmarks": {
                    "P/E": 20.0,
                    "Forward P/E": 16.0,
                    "P/S": 4.0,
                    "P/B": 4.0,
                    "EV/EBITDA": 12.0,
                    "R&D/Revenue": 0.12,
                    "Operating Margin": 0.15,
                    "Net Margin": 0.15,
                    "ROE": 0.18
                },
                "description": "Healthcare sector includes pharmaceuticals, biotech, equipment, and services. Pharmaceutical and biotech firms focus on R&D and drug pipelines. Equipment and services companies emphasize operational efficiency and margin stability. Regulatory factors have significant impact."
            },

            "Financials": {
                "key_ratios": [
                    "P/B", "P/E", "ROE", "Net Interest Margin", "Efficiency Ratio",
                    "NPL Ratio", "Loan-to-Deposit Ratio", "CET1 Ratio"
                ],
                "benchmarks": {
                    "P/B": 1.2,
                    "P/E": 12.0,
                    "ROE": 0.12,
                    "Net Interest Margin": 0.03,
                    "Efficiency Ratio": 0.60,  # Lower is better
                    "NPL Ratio": 0.01,  # Lower is better
                    "Loan-to-Deposit Ratio": 0.90,
                    "CET1 Ratio": 0.12
                },
                "description": "Financial sector valuations focus on book value and return on equity. Banks are valued on P/B, net interest margin, and loan quality. Insurance firms emphasize combined ratio and investment returns. Asset managers focus on AUM growth and fee generation."
            },

            "Energy": {
                "key_ratios": [
                    "EV/EBITDA", "P/CF", "Reserve Replacement Ratio", "Production Cost per Barrel",
                    "Debt/EBITDA", "FCF Yield", "Reserve Life Index"
                ],
                "benchmarks": {
                    "EV/EBITDA": 6.0,
                    "P/CF": 5.0,
                    "Reserve Replacement Ratio": 1.0,  # >1 means growing reserves
                    "Debt/EBITDA": 2.5,  # Lower is better
                    "FCF Yield": 0.06,
                    "PEG": 1.5
                },
                "description": "Energy companies are valued based on reserves, production costs, and free cash flow. Oil and gas producers focus on reserve replacement and production costs. Midstream companies emphasize stable cash flows and distribution growth. Renewable energy firms focus on capacity growth and project returns."
            },

            "Consumer Discretionary": {
                "key_ratios": [
                    "P/E", "P/S", "EV/EBITDA", "Same-Store Sales Growth", "Inventory Turnover",
                    "Gross Margin", "Operating Margin", "ROA"
                ],
                "benchmarks": {
                    "P/E": 18.0,
                    "P/S": 1.5,
                    "EV/EBITDA": 10.0,
                    "Same-Store Sales Growth": 0.03,
                    "Inventory Turnover": 6.0,
                    "Gross Margin": 0.35,
                    "Operating Margin": 0.10,
                    "ROA": 0.08
                },
                "description": "Consumer discretionary companies are sensitive to economic cycles. Retail firms focus on same-store sales growth, inventory turnover, and operating margins. Luxury goods emphasize brand value and pricing power. Restaurants and hospitality businesses focus on location efficiency and customer loyalty metrics."
            },

            "Consumer Staples": {
                "key_ratios": [
                    "P/E", "EV/EBITDA", "Dividend Yield", "Operating Margin",
                    "FCF Yield", "Debt/EBITDA", "ROE"
                ],
                "benchmarks": {
                    "P/E": 20.0,
                    "EV/EBITDA": 12.0,
                    "Dividend Yield": 0.02,
                    "Operating Margin": 0.15,
                    "FCF Yield": 0.04,
                    "Debt/EBITDA": 2.0,
                    "ROE": 0.20
                },
                "description": "Consumer staples offer stability across economic cycles. Food and beverage companies focus on brand strength and market share. Household products emphasize pricing power and international growth. Defensive characteristics make dividend yield and stability important metrics."
            },

            "Industrials": {
                "key_ratios": [
                    "EV/EBITDA", "P/E", "ROIC", "Operating Margin", "FCF Conversion",
                    "ROA", "Asset Turnover", "Order Backlog"
                ],
                "benchmarks": {
                    "EV/EBITDA": 9.0,
                    "P/E": 16.0,
                    "ROIC": 0.10,
                    "Operating Margin": 0.12,
                    "FCF Conversion": 0.90,  # FCF / Net Income
                    "ROA": 0.06,
                    "Asset Turnover": 0.75
                },
                "description": "Industrial companies focus on operational efficiency and capital allocation. Aerospace and defense track order backlog and program execution. Machinery and equipment manufacturers emphasize product innovation and service revenue. Transportation companies focus on asset utilization and fuel efficiency."
            },

            "Materials": {
                "key_ratios": [
                    "EV/EBITDA", "P/B", "ROIC", "ROE", "Operating Margin",
                    "Debt/EBITDA", "FCF Yield", "CAPEX/Sales"
                ],
                "benchmarks": {
                    "EV/EBITDA": 8.0,
                    "P/B": 2.0,
                    "ROIC": 0.09,
                    "ROE": 0.12,
                    "Operating Margin": 0.12,
                    "Debt/EBITDA": 2.5,
                    "FCF Yield": 0.05,
                    "CAPEX/Sales": 0.08
                },
                "description": "Materials companies are cyclical and commodity-price sensitive. Mining companies focus on production costs and reserve quality. Chemical companies emphasize product differentiation and R&D pipeline. Building materials track housing starts and infrastructure spending."
            },

            "Real Estate": {
                "key_ratios": [
                    "P/FFO", "P/NAV", "Cap Rate", "Occupancy Rate", "Debt/Assets",
                    "FFO Yield", "NOI Margin", "Same-Property NOI Growth"
                ],
                "benchmarks": {
                    "P/FFO": 18.0,
                    "P/NAV": 0.95,
                    "Cap Rate": 0.05,
                    "Occupancy Rate": 0.90,
                    "Debt/Assets": 0.50,
                    "FFO Yield": 0.08,
                    "NOI Margin": 0.65,
                    "Same-Property NOI Growth": 0.03
                },
                "description": "Real estate companies are valued based on funds from operations (FFO) and net asset value (NAV). REITs emphasize occupancy rates, rental growth, and dividend yield. Different property types have unique valuation considerations, with data centers and logistics commanding premium multiples."
            },

            "Communication Services": {
                "key_ratios": [
                    "P/E", "EV/EBITDA", "EV/Revenue", "ARPU", "Churn Rate",
                    "EBITDA Margin", "FCF Yield", "Debt/EBITDA"
                ],
                "benchmarks": {
                    "P/E": 18.0,
                    "EV/EBITDA": 8.0,
                    "EV/Revenue": 3.0,
                    "ARPU": None,  # Varies widely by service
                    "Churn Rate": 0.02,  # Lower is better
                    "EBITDA Margin": 0.35,
                    "FCF Yield": 0.06,
                    "Debt/EBITDA": 2.5
                },
                "description": "Communication services spans telecom, media, and entertainment. Telecom companies focus on subscriber growth, ARPU, and infrastructure investment. Media and entertainment firms emphasize content value, subscriber growth, and digital transition. Social media platforms focus on user engagement, ad revenue, and platform expansion."
            },

            "Utilities": {
                "key_ratios": [
                    "P/E", "EV/EBITDA", "Dividend Yield", "Debt/EBITDA",
                    "Interest Coverage", "ROA", "Regulated Asset Base", "Payout Ratio"
                ],
                "benchmarks": {
                    "P/E": 16.0,
                    "EV/EBITDA": 9.0,
                    "Dividend Yield": 0.04,
                    "Debt/EBITDA": 4.0,
                    "Interest Coverage": 3.0,
                    "ROA": 0.03,
                    "Payout Ratio": 0.70
                },
                "description": "Utilities offer stable cash flows but limited growth. Electric utilities focus on regulated asset base and allowed return on equity. Gas utilities emphasize customer growth and infrastructure investment. Renewable utilities focus on capacity growth and power purchase agreement terms."
            }
        }

        return sector_ratios

    def _initialize_valuation_methods(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Initialize sector-specific preferred valuation methods

        Returns:
            Dictionary with preferred valuation methods by sector
        """
        valuation_methods = {
            "Information Technology": {
                "primary": ["DCF", "EV/Revenue", "P/S", "EV/EBITDA"],
                "secondary": ["PEG", "ROIC Analysis", "P/E"],
                "description": "Technology companies are best valued using growth-adjusted metrics and DCF models that capture long-term potential. SaaS companies often use EV/Revenue and Rule of 40 (growth rate + profit margin)."
            },

            "Healthcare": {
                "primary": ["DCF", "P/E", "EV/EBITDA", "Sum-of-Parts"],
                "secondary": ["P/S", "PEG", "R&D Efficiency Analysis"],
                "description": "Pharmaceutical and biotech companies often require pipeline-based valuation and sum-of-parts analysis, while healthcare services companies focus more on cash flow stability and EBITDA multiples."
            },

            "Financials": {
                "primary": ["P/B", "DDM", "Residual Income Model", "P/E"],
                "secondary": ["Excess Return Model", "Net Interest Margin Analysis", "Deposit Premium"],
                "description": "Banks are primarily valued based on book value and return on equity, while insurance companies focus on embedded value and combined ratio analysis."
            },

            "Energy": {
                "primary": ["EV/EBITDA", "P/CF", "NAV", "Reserve Valuation"],
                "secondary": ["DCF", "EV/Reserve", "EV/Production"],
                "description": "Energy companies require specialized valuation considering reserve life, production costs, and commodity price scenarios. Midstream assets focus on stable cash flows and distribution growth."
            },

            "Consumer Discretionary": {
                "primary": ["DCF", "EV/EBITDA", "P/E", "P/S"],
                "secondary": ["EV/EBITDAR for Retail", "Brand Value Analysis", "Same-Store Sales Growth"],
                "description": "Retail companies focus on comparable store sales growth and inventory efficiency, while luxury brands emphasize brand value and pricing power in valuation models."
            },

            "Consumer Staples": {
                "primary": ["DCF", "EV/EBITDA", "P/E", "Dividend Yield"],
                "secondary": ["P/FCF", "Brand Value Analysis", "P/S"],
                "description": "Consumer staples are valued for stability and strong cash flows, with emphasis on brand strength, market share, and dividend sustainability."
            },

            "Industrials": {
                "primary": ["DCF", "EV/EBITDA", "ROIC Analysis", "P/E"],
                "secondary": ["EV/Sales", "Replacement Value", "Sum-of-Parts"],
                "description": "Industrial companies require valuation based on capital efficiency and through-cycle performance, with focus on free cash flow generation and return on invested capital."
            },

            "Materials": {
                "primary": ["EV/EBITDA", "NAV", "P/CF", "Replacement Value"],
                "secondary": ["DCF with Commodity Price Scenarios", "EV/Reserves", "P/B"],
                "description": "Materials companies are often valued using asset-based approaches and scenario analysis around commodity prices, with focus on production costs and reserve quality."
            },

            "Real Estate": {
                "primary": ["NAV", "Cap Rate", "P/FFO", "DDM"],
                "secondary": ["P/AFFO", "Replacement Cost", "DCF", "Dividend Yield"],
                "description": "Real estate valuation emphasizes property-level metrics like cap rates and net operating income, as well as portfolio-level metrics like funds from operations."
            },

            "Communication Services": {
                "primary": ["DCF", "EV/EBITDA", "P/E", "EV/Subscriber"],
                "secondary": ["EV/Revenue", "Customer Lifetime Value", "Network Value"],
                "description": "Telecom companies focus on subscriber economics and infrastructure value, while media companies emphasize content libraries and audience reach in valuation models."
            },

            "Utilities": {
                "primary": ["DDM", "Regulated Asset Base", "P/E", "EV/EBITDA"],
                "secondary": ["DCF", "Dividend Yield", "Replacement Value"],
                "description": "Utilities are valued primarily based on regulated returns, dividend yield, and predictable cash flows, with emphasis on regulatory environment and allowed return on capital."
            }
        }

        return valuation_methods

    def _initialize_subsector_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize mapping of subsectors to main sectors

        Returns:
            Dictionary mapping subsectors to main sectors
        """
        subsector_mapping = {
            # Technology subsectors
            "Software": "Information Technology",
            "Hardware": "Information Technology",
            "Semiconductors": "Information Technology",
            "IT Services": "Information Technology",
            "Internet Software & Services": "Information Technology",
            "Electronic Equipment": "Information Technology",
            "Technology Hardware": "Information Technology",

            # Healthcare subsectors
            "Pharmaceuticals": "Healthcare",
            "Biotechnology": "Healthcare",
            "Medical Devices": "Healthcare",
            "Healthcare Services": "Healthcare",
            "Healthcare Equipment": "Healthcare",
            "Life Sciences Tools": "Healthcare",
            "Healthcare Technology": "Healthcare",

            # Financials subsectors
            "Banks": "Financials",
            "Insurance": "Financials",
            "Diversified Financial Services": "Financials",
            "Capital Markets": "Financials",
            "Consumer Finance": "Financials",
            "Mortgage Finance": "Financials",
            "Asset Management": "Financials",

            # Energy subsectors
            "Oil & Gas Exploration & Production": "Energy",
            "Oil & Gas Equipment & Services": "Energy",
            "Oil & Gas Refining & Marketing": "Energy",
            "Oil & Gas Storage & Transportation": "Energy",
            "Coal & Consumable Fuels": "Energy",
            "Renewable Energy": "Energy",

            # Consumer Discretionary subsectors
            "Automobiles & Components": "Consumer Discretionary",
            "Consumer Durables & Apparel": "Consumer Discretionary",
            "Consumer Services": "Consumer Discretionary",
            "Retailing": "Consumer Discretionary",
            "Leisure Products": "Consumer Discretionary",
            "Hotels, Restaurants & Leisure": "Consumer Discretionary",
            "Specialty Retail": "Consumer Discretionary",
            "Apparel Retail": "Consumer Discretionary",

            # Consumer Staples subsectors
            "Food & Staples Retailing": "Consumer Staples",
            "Food Products": "Consumer Staples",
            "Beverages": "Consumer Staples",
            "Tobacco": "Consumer Staples",
            "Household Products": "Consumer Staples",
            "Personal Products": "Consumer Staples",

            # Industrials subsectors
            "Aerospace & Defense": "Industrials",
            "Industrial Conglomerates": "Industrials",
            "Machinery": "Industrials",
            "Building Products": "Industrials",
            "Electrical Equipment": "Industrials",
            "Trading Companies & Distributors": "Industrials",
            "Commercial Services & Supplies": "Industrials",
            "Professional Services": "Industrials",
            "Air Freight & Logistics": "Industrials",
            "Airlines": "Industrials",
            "Marine": "Industrials",
            "Road & Rail": "Industrials",
            "Transportation Infrastructure": "Industrials",

            # Materials subsectors
            "Chemicals": "Materials",
            "Construction Materials": "Materials",
            "Containers & Packaging": "Materials",
            "Metals & Mining": "Materials",
            "Paper & Forest Products": "Materials",

            # Real Estate subsectors
            "Equity Real Estate Investment Trusts (REITs)": "Real Estate",
            "Real Estate Management & Development": "Real Estate",
            "Diversified REITs": "Real Estate",
            "Industrial REITs": "Real Estate",
            "Office REITs": "Real Estate",
            "Residential REITs": "Real Estate",
            "Retail REITs": "Real Estate",
            "Specialized REITs": "Real Estate",
            "Hotel & Resort REITs": "Real Estate",
            "Healthcare REITs": "Real Estate",

            # Communication Services subsectors
            "Diversified Telecommunication Services": "Communication Services",
            "Wireless Telecommunication Services": "Communication Services",
            "Media": "Communication Services",
            "Entertainment": "Communication Services",
            "Interactive Media & Services": "Communication Services",

            # Utilities subsectors
            "Electric Utilities": "Utilities",
            "Gas Utilities": "Utilities",
            "Multi-Utilities": "Utilities",
            "Water Utilities": "Utilities",
            "Independent Power and Renewable Electricity Producers": "Utilities"
        }

        return subsector_mapping

    def get_sector_key_ratios(self, sector: str) -> List[str]:
        """
        Get key financial ratios for a specific sector

        Args:
            sector: Sector name (main sector or subsector)

        Returns:
            List of key ratios for the sector
        """
        # If a subsector is provided, map it to main sector
        main_sector = self._get_main_sector(sector)

        # Get sector data
        sector_data = self.sector_ratios.get(main_sector, None)

        if sector_data:
            return sector_data.get("key_ratios", [])
        else:
            logger.warning(f"No key ratios found for sector: {sector}")
            return []

    def get_sector_benchmarks(self, sector: str) -> Dict[str, float]:
        """
        Get benchmark values for a specific sector

        Args:
            sector: Sector name (main sector or subsector)

        Returns:
            Dictionary with benchmark values for the sector
        """
        # If a subsector is provided, map it to main sector
        main_sector = self._get_main_sector(sector)

        # Get sector data
        sector_data = self.sector_ratios.get(main_sector, None)

        if sector_data:
            return sector_data.get("benchmarks", {})
        else:
            logger.warning(f"No benchmarks found for sector: {sector}")
            return {}

    def get_sector_description(self, sector: str) -> str:
        """
        Get descriptive information about a sector

        Args:
            sector: Sector name (main sector or subsector)

        Returns:
            Description string for the sector
        """
        # If a subsector is provided, map it to main sector
        main_sector = self._get_main_sector(sector)

        # Get sector data
        sector_data = self.sector_ratios.get(main_sector, None)

        if sector_data:
            return sector_data.get("description", "No description available")
        else:
            return f"No description available for sector: {sector}"

    def get_valuation_methods(self, sector: str) -> Dict[str, List[str]]:
        """
        Get preferred valuation methods for a specific sector

        Args:
            sector: Sector name (main sector or subsector)

        Returns:
            Dictionary with primary and secondary valuation methods for the sector
        """
        # If a subsector is provided, map it to main sector
        main_sector = self._get_main_sector(sector)

        # Get valuation methods data
        valuation_data = self.valuation_methods.get(main_sector, None)

        if valuation_data:
            return {
                "primary": valuation_data.get("primary", []),
                "secondary": valuation_data.get("secondary", []),
                "description": valuation_data.get("description", "")
            }
        else:
            logger.warning(f"No valuation methods found for sector: {sector}")
            return {"primary": [], "secondary": [], "description": ""}

    def _get_main_sector(self, sector: str) -> str:
        """
        Map a subsector to its main sector if needed

        Args:
            sector: Sector name (could be main sector or subsector)

        Returns:
            Main sector name
        """
        # Check if this is already a main sector
        if sector in self.sector_ratios:
            return sector

        # Check if this is a subsector
        if sector in self.subsector_mapping:
            return self.subsector_mapping[sector]

        # Try case-insensitive matching
        for subsector, main_sector in self.subsector_mapping.items():
            if sector.lower() == subsector.lower():
                return main_sector

        # If no match found, try partial matching
        for subsector, main_sector in self.subsector_mapping.items():
            if sector.lower() in subsector.lower() or subsector.lower() in sector.lower():
                return main_sector

        # If no match found for subsector, return the original input
        # It might already be a main sector with slight spelling difference
        for main_sector in self.sector_ratios.keys():
            if sector.lower() in main_sector.lower() or main_sector.lower() in sector.lower():
                return main_sector

        logger.warning(f"Could not map {sector} to a main sector, returning as is")
        return sector

    def get_all_main_sectors(self) -> List[str]:
        """
        Get a list of all main sectors

        Returns:
            List of main sector names
        """
        return list(self.sector_ratios.keys())

    def get_all_subsectors(self) -> List[str]:
        """
        Get a list of all subsectors

        Returns:
            List of subsector names
        """
        return list(self.subsector_mapping.keys())

    def get_subsectors_for_sector(self, main_sector: str) -> List[str]:
        """
        Get all subsectors for a given main sector

        Args:
            main_sector: Main sector name

        Returns:
            List of subsector names
        """
        subsectors = []
        for subsector, sector in self.subsector_mapping.items():
            if sector == main_sector:
                subsectors.append(subsector)

        return subsectors

    def compare_sectors(self, sector1: str, sector2: str) -> Dict[str, Any]:
        """
        Compare two sectors based on their key financial metrics

        Args:
            sector1: First sector name
            sector2: Second sector name

        Returns:
            Dictionary with comparison results
        """
        # Get main sectors
        main_sector1 = self._get_main_sector(sector1)
        main_sector2 = self._get_main_sector(sector2)

        # Get benchmarks
        benchmarks1 = self.get_sector_benchmarks(main_sector1)
        benchmarks2 = self.get_sector_benchmarks(main_sector2)

        # Initialize comparison results
        comparison = {
            "sector1": main_sector1,
            "sector2": main_sector2,
            "metrics_comparison": {},
            "valuation_methods_comparison": {
                "sector1_unique": [],
                "sector2_unique": [],
                "common": []
            },
            "key_differences": [],
            "common_metrics": []
        }

        # Compare benchmarks
        for metric in set(benchmarks1.keys()) | set(benchmarks2.keys()):
            value1 = benchmarks1.get(metric)
            value2 = benchmarks2.get(metric)

            if value1 is not None and value2 is not None:
                # Both sectors have this metric
                comparison["metrics_comparison"][metric] = {
                    "sector1_value": value1,
                    "sector2_value": value2,
                    "difference": value1 - value2,
                    "percentage_difference": (value1 / value2 - 1) * 100 if value2 != 0 else None
                }
                comparison["common_metrics"].append(metric)
            elif value1 is not None:
                # Only sector1 has this metric
                comparison["metrics_comparison"][metric] = {
                    "sector1_value": value1,
                    "sector2_value": None,
                    "difference": None,
                    "percentage_difference": None
                }
            elif value2 is not None:
                # Only sector2 has this metric
                comparison["metrics_comparison"][metric] = {
                    "sector1_value": None,
                    "sector2_value": value2,
                    "difference": None,
                    "percentage_difference": None
                }

        # Compare valuation methods
        valuation1 = self.get_valuation_methods(main_sector1)
        valuation2 = self.get_valuation_methods(main_sector2)

        primary1 = set(valuation1.get("primary", []))
        primary2 = set(valuation2.get("primary", []))

        comparison["valuation_methods_comparison"]["sector1_unique"] = list(primary1 - primary2)
        comparison["valuation_methods_comparison"]["sector2_unique"] = list(primary2 - primary1)
        comparison["valuation_methods_comparison"]["common"] = list(primary1 & primary2)

        # Identify key differences
        # Add significant differences in metrics (>20%)
        for metric, data in comparison["metrics_comparison"].items():
            if data["percentage_difference"] is not None and abs(data["percentage_difference"]) > 20:
                direction = "higher" if data["percentage_difference"] > 0 else "lower"
                comparison["key_differences"].append(
                    f"{main_sector1} typically has {abs(data['percentage_difference']):.1f}% {direction} {metric} than {main_sector2}"
                )

        # Add differences in valuation approach
        if comparison["valuation_methods_comparison"]["sector1_unique"]:
            comparison["key_differences"].append(
                f"{main_sector1} uniquely emphasizes {', '.join(comparison['valuation_methods_comparison']['sector1_unique'])} for valuation"
            )

        if comparison["valuation_methods_comparison"]["sector2_unique"]:
            comparison["key_differences"].append(
                f"{main_sector2} uniquely emphasizes {', '.join(comparison['valuation_methods_comparison']['sector2_unique'])} for valuation"
            )

        return comparison

    def get_sector_specific_ratio_explanations(self, sector: str) -> Dict[str, str]:
        """
        Get explanations for sector-specific ratios

        Args:
            sector: Sector name

        Returns:
            Dictionary with ratio explanations
        """
        main_sector = self._get_main_sector(sector)

        # Define explanations for sector-specific ratios
        explanations = {
            "Information Technology": {
                "R&D/Revenue": "Research and Development expenses as a percentage of revenue, measuring innovation investment. Higher values indicate greater focus on developing new technologies and products.",
                "Gross Margin": "For tech companies, high gross margins often indicate software focus or strong intellectual property, while lower margins suggest hardware or service orientation.",
                "EV/Revenue": "Often used for high-growth tech companies not yet profitable. Lower values may indicate undervaluation, though must be considered alongside growth rate.",
                "PEG": "Price/Earnings to Growth ratio. Particularly important for tech companies where growth expectations significantly impact valuation."
            },

            "Healthcare": {
                "R&D/Revenue": "Critical metric for pharmaceutical and biotech companies, indicating investment in drug discovery and development pipeline.",
                "Pipeline Value": "Estimated value of drugs/treatments in development, considering probability of approval, market size, and competition.",
                "Patent Cliff Exposure": "Percentage of revenue vulnerable to patent expirations in the next 3-5 years."
            },

            "Financials": {
                "Net Interest Margin": "Difference between interest income generated and interest paid, relative to interest-earning assets. Core profitability metric for banks.",
                "Efficiency Ratio": "Non-interest expenses as a percentage of revenue. Lower values indicate greater operational efficiency in financial institutions.",
                "NPL Ratio": "Non-Performing Loans as a percentage of total loans. Key indicator of loan portfolio quality and potential future write-offs.",
                "Loan-to-Deposit Ratio": "Total loans divided by total deposits. Measures liquidity and lending capacity.",
                "CET1 Ratio": "Common Equity Tier 1 capital as a percentage of risk-weighted assets. Key regulatory capital measure."
            },

            "Energy": {
                "Reserve Replacement Ratio": "New reserves added relative to production. Values above 100% indicate growing reserves, sustainable production.",
                "Production Cost per Barrel": "Total cost to produce one barrel of oil equivalent. Critical for profitability in varying price environments.",
                "Reserve Life Index": "Reserves divided by annual production, indicating how many years current reserves will last."
            },

            "Consumer Discretionary": {
                "Same-Store Sales Growth": "Revenue growth from stores open for at least one year. Key indicator of organic growth separate from new store openings.",
                "Inventory Turnover": "Cost of goods sold divided by average inventory. Particularly important for retailers to measure inventory management efficiency."
            },

            "Consumer Staples": {
                "Volume Growth": "Growth in units sold, separate from price increases. Important to distinguish between price-driven and volume-driven growth.",
                "Market Share": "Company's sales as a percentage of total market. Particularly important in consumer products with brand loyalty."
            },

            "Industrials": {
                "Order Backlog": "Value of unfilled orders. Important leading indicator of future revenue for industrial equipment manufacturers.",
                "Book-to-Bill Ratio": "Orders received divided by revenue billed in the same period. Values above 1.0 indicate growing demand."
            },

            "Real Estate": {
                "FFO": "Funds From Operations (net income plus depreciation, amortization, minus gains on sales). Primary earnings metric for REITs.",
                "AFFO": "Adjusted Funds From Operations (FFO minus recurring capital expenditures). More conservative measure of cash flow.",
                "Cap Rate": "Net Operating Income divided by property value. Higher cap rates suggest higher yield but potentially higher risk.",
                "NOI Margin": "Net Operating Income as a percentage of property revenue. Measures property management efficiency."
            },

            "Communication Services": {
                "ARPU": "Average Revenue Per User. Key metric for subscription-based telecom and media companies.",
                "Churn Rate": "Percentage of subscribers who discontinue service in a period. Lower values indicate stronger customer retention.",
                "Cost Per Acquisition": "Marketing expense to acquire a new customer. Important for evaluating subscriber economics."
            },

            "Utilities": {
                "Regulated Asset Base": "Value of assets on which regulators allow utilities to earn a return. Fundamental driver of earnings potential.",
                "Allowed ROE": "Return on Equity permitted by regulators. Directly impacts profitability of regulated activities."
            },

            "Materials": {
                "Capacity Utilization": "Production output as a percentage of total capacity. Higher values indicate operational efficiency and strong demand.",
                "EBITDA/Ton": "EBITDA divided by production volume. Measures profitability per unit of production, allowing comparison across companies of different sizes."
            }
        }

        return explanations.get(main_sector, {})

    def assess_company_sector_fit(self, company_ratios: Dict[str, float], sector: str) -> Dict[str, Any]:
        """
        Assess how well a company's financial ratios align with sector norms

        Args:
            company_ratios: Dictionary with company's financial ratios
            sector: Sector name

        Returns:
            Dictionary with sector fit assessment
        """
        main_sector = self._get_main_sector(sector)
        benchmarks = self.get_sector_benchmarks(main_sector)

        assessment = {
            "sector": main_sector,
            "overall_fit_score": 0.0,
            "metric_assessments": {},
            "strengths": [],
            "weaknesses": [],
            "overall_assessment": ""
        }

        # Check for metrics that are relevant to both company and sector
        common_metrics = set(company_ratios.keys()) & set(benchmarks.keys())

        if not common_metrics:
            assessment["overall_assessment"] = f"Insufficient data to assess fit with {main_sector} sector"
            return assessment

        fit_scores = []

        # Assess each common metric
        for metric in common_metrics:
            company_value = company_ratios[metric]
            benchmark_value = benchmarks[metric]

            # Skip if benchmark is None
            if benchmark_value is None:
                continue

            # Determine if higher values are better for this metric
            higher_is_better = True
            if metric in ["Debt/EBITDA", "Debt/Equity", "Efficiency Ratio", "NPL Ratio", "Churn Rate"]:
                higher_is_better = False

            # Calculate relative performance
            if benchmark_value != 0:
                relative_value = company_value / benchmark_value
            else:
                relative_value = 1.0 if company_value == 0 else 2.0

            # For metrics where lower is better, invert the relationship
            if not higher_is_better:
                relative_value = 1 / relative_value if relative_value != 0 else 2.0

            # Calculate score (1.0 means exactly matching benchmark)
            # Score decreases as values diverge from benchmark
            # Score of 0 means value is twice as high or half as low as benchmark
            if relative_value > 1:
                score = max(0, 2 - relative_value)
            else:
                score = relative_value

            # Convert score to 0-1 scale
            normalized_score = min(1.0, score)

            fit_scores.append(normalized_score)

            # Store metric assessment
            assessment["metric_assessments"][metric] = {
                "company_value": company_value,
                "sector_benchmark": benchmark_value,
                "fit_score": normalized_score,
                "deviation_percentage": (relative_value - 1) * 100
            }

            # Add to strengths or weaknesses
            if normalized_score >= 0.8:
                assessment["strengths"].append(
                    f"{metric} ({company_value:.2f}) aligns well with {main_sector} sector norms"
                )
            elif normalized_score <= 0.4:
                if higher_is_better:
                    direction = "lower" if company_value < benchmark_value else "higher"
                else:
                    direction = "higher" if company_value > benchmark_value else "lower"

                assessment["weaknesses"].append(
                    f"{metric} ({company_value:.2f}) is significantly {direction} than typical for {main_sector} sector"
                )

        # Calculate overall fit score
        if fit_scores:
            assessment["overall_fit_score"] = sum(fit_scores) / len(fit_scores)

        # Generate overall assessment
        if assessment["overall_fit_score"] >= 0.8:
            assessment["overall_assessment"] = f"Strong alignment with {main_sector} sector characteristics"
        elif assessment["overall_fit_score"] >= 0.6:
            assessment["overall_assessment"] = f"Good alignment with {main_sector} sector, with some deviations"
        elif assessment["overall_fit_score"] >= 0.4:
            assessment["overall_assessment"] = f"Moderate alignment with {main_sector} sector, with notable deviations"
        else:
            assessment[
                "overall_assessment"] = f"Poor alignment with {main_sector} sector, may be misclassified or a sector outlier"

        return assessment

    def get_sector_lifecycle_characteristics(self, sector: str) -> Dict[str, Any]:
        """
        Get lifecycle characteristics of the specified sector

        Args:
            sector: Sector name

        Returns:
            Dictionary with sector lifecycle characteristics
        """
        main_sector = self._get_main_sector(sector)

        # Lifecycle characteristics by sector
        lifecycle_data = {
            "Information Technology": {
                "lifecycle_stage": "Growth to Mature (varies by subsector)",
                "growth_expectation": "High (10-15% annually)",
                "volatility": "High",
                "disruption_risk": "High",
                "consolidation_trend": "Active in mature subsectors",
                "key_subsector_stages": {
                    "Cloud Computing": "Growth",
                    "AI/Machine Learning": "Early Growth",
                    "Enterprise Software": "Mature",
                    "Semiconductors": "Cyclical Mature",
                    "Hardware": "Mature/Declining"
                },
                "description": "Technology sector spans multiple lifecycle stages, with emerging technologies in rapid growth phases while legacy hardware and software segments mature. The sector faces continuous innovation cycles and disruption."
            },

            "Healthcare": {
                "lifecycle_stage": "Growth to Mature (varies by subsector)",
                "growth_expectation": "Moderate to High (6-12% annually)",
                "volatility": "Moderate",
                "disruption_risk": "Moderate to High",
                "consolidation_trend": "Active",
                "key_subsector_stages": {
                    "Biotechnology": "Growth",
                    "Pharmaceuticals": "Mature",
                    "Healthcare Services": "Mature",
                    "Digital Health": "Growth",
                    "Medical Devices": "Mature"
                },
                "description": "Healthcare combines stable service businesses with high-growth innovation segments. Demographic trends provide long-term growth drivers, while innovation and regulatory changes create both opportunities and risks."
            },

            "Financials": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Low to Moderate (3-6% annually)",
                "volatility": "Moderate to High",
                "disruption_risk": "Moderate (FinTech disruption)",
                "consolidation_trend": "Ongoing",
                "key_subsector_stages": {
                    "Traditional Banking": "Mature",
                    "Insurance": "Mature",
                    "Asset Management": "Mature",
                    "FinTech": "Growth",
                    "Payments": "Growth to Mature"
                },
                "description": "Financial sector is largely mature with established players, though experiencing disruption from technology. Growth typically tracks overall economic expansion, with cyclical sensitivity to interest rates and credit quality."
            },

            "Energy": {
                "lifecycle_stage": "Mature to Declining (Traditional) / Growth (Renewables)",
                "growth_expectation": "Low (1-3% annually) for traditional, High (15%+) for renewables",
                "volatility": "High",
                "disruption_risk": "High (energy transition)",
                "consolidation_trend": "Active in traditional segments",
                "key_subsector_stages": {
                    "Oil & Gas Exploration": "Mature/Declining",
                    "Oil & Gas Midstream": "Mature",
                    "Refining": "Mature",
                    "Renewable Energy": "Growth",
                    "Energy Storage": "Early Growth"
                },
                "description": "Energy sector is bifurcated between traditional fossil fuels in mature/declining phases and renewable energy in growth phase. The energy transition creates significant disruption risk for traditional players while offering growth opportunities in clean energy."
            },

            "Consumer Discretionary": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Moderate (4-7% annually)",
                "volatility": "High (cyclical)",
                "disruption_risk": "Moderate to High (e-commerce)",
                "consolidation_trend": "Moderate",
                "key_subsector_stages": {
                    "Retail": "Mature (disrupted)",
                    "Automotive": "Mature (disrupted)",
                    "Luxury Goods": "Mature",
                    "Homebuilding": "Cyclical Mature",
                    "E-commerce": "Growth to Mature"
                },
                "description": "Consumer discretionary encompasses mature businesses experiencing varying levels of disruption. Traditional retail faces significant challenges from e-commerce, while automotive sector confronts electric and autonomous vehicle transitions."
            },

            "Consumer Staples": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Low (3-5% annually)",
                "volatility": "Low",
                "disruption_risk": "Low to Moderate",
                "consolidation_trend": "Ongoing",
                "key_subsector_stages": {
                    "Food & Beverage": "Mature",
                    "Household Products": "Mature",
                    "Tobacco": "Mature/Declining",
                    "Food Retail": "Mature"
                },
                "description": "Consumer staples sector consists of mature businesses with stable, defensive characteristics. Growth typically comes from emerging markets, premiumization, and modest innovation, with limited disruption risk compared to other sectors."
            },

            "Industrials": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Moderate (4-6% annually)",
                "volatility": "Moderate to High (cyclical)",
                "disruption_risk": "Moderate",
                "consolidation_trend": "Active",
                "key_subsector_stages": {
                    "Aerospace & Defense": "Mature",
                    "Machinery": "Mature",
                    "Electrical Equipment": "Mature",
                    "Transportation": "Mature",
                    "Industrial Services": "Growth to Mature"
                },
                "description": "Industrial sector consists primarily of mature businesses with cyclical exposure to economic activity. Growth opportunities exist in automation, electrification, and sustainability transitions, though overall sector growth tracks industrial production."
            },

            "Materials": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Low to Moderate (3-5% annually)",
                "volatility": "High (cyclical)",
                "disruption_risk": "Moderate",
                "consolidation_trend": "Active",
                "key_subsector_stages": {
                    "Chemicals": "Mature",
                    "Metals & Mining": "Mature (cyclical)",
                    "Construction Materials": "Mature (cyclical)",
                    "Paper & Packaging": "Mature"
                },
                "description": "Materials sector comprises mature businesses with high cyclicality and commodity price exposure. Growth typically tracks overall economic activity, with periodic supply-demand imbalances creating significant volatility."
            },

            "Real Estate": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Low to Moderate (3-6% annually)",
                "volatility": "Moderate",
                "disruption_risk": "Moderate",
                "consolidation_trend": "Moderate",
                "key_subsector_stages": {
                    "Residential": "Mature (cyclical)",
                    "Office": "Mature (disrupted)",
                    "Retail": "Mature (disrupted)",
                    "Industrial/Logistics": "Growth",
                    "Data Centers": "Growth"
                },
                "description": "Real estate sector shows varying dynamics across property types, with traditional office and retail facing disruption while industrial, logistics, and data centers grow. Overall sector growth typically exceeds inflation by a modest margin."
            },

            "Communication Services": {
                "lifecycle_stage": "Mature (Telecom) / Growth (Digital Media)",
                "growth_expectation": "Low (1-3%) for Telecom, High (8-12%) for Digital Media",
                "volatility": "Moderate",
                "disruption_risk": "High",
                "consolidation_trend": "Active",
                "key_subsector_stages": {
                    "Telecommunications": "Mature",
                    "Traditional Media": "Mature/Declining",
                    "Digital Media": "Growth",
                    "Social Media": "Growth to Mature",
                    "Gaming": "Growth"
                },
                "description": "Communications sector spans mature telecommunications businesses and growing digital media platforms. Traditional telecom and media face disruption, while digital platforms continue to grow through user engagement and monetization innovations."
            },

            "Utilities": {
                "lifecycle_stage": "Mature",
                "growth_expectation": "Low (2-4% annually)",
                "volatility": "Low",
                "disruption_risk": "Moderate (renewables transition)",
                "consolidation_trend": "Moderate",
                "key_subsector_stages": {
                    "Electric Utilities": "Mature",
                    "Gas Utilities": "Mature",
                    "Water Utilities": "Mature",
                    "Renewable Utilities": "Growth"
                },
                "description": "Utilities sector consists of mature, regulated businesses with stable cash flows and modest growth. The energy transition creates both challenges and opportunities, with traditional fossil generation declining while renewable capacity expands."
            }
        }

        return lifecycle_data.get(main_sector, {
            "lifecycle_stage": "Unknown",
            "growth_expectation": "Unknown",
            "volatility": "Unknown",
            "disruption_risk": "Unknown",
            "consolidation_trend": "Unknown",
            "description": f"No lifecycle data available for sector: {sector}"
        })

    def get_sector_risk_profile(self, sector: str) -> Dict[str, Any]:
        """
        Get risk profile characteristics of the specified sector

        Args:
            sector: Sector name

        Returns:
            Dictionary with sector risk profile
        """
        main_sector = self._get_main_sector(sector)

        # Risk profiles by sector
        risk_profiles = {
            "Information Technology": {
                "overall_risk": "High",
                "specific_risks": {
                    "Business Cycle Sensitivity": "High (enterprise) / Moderate (consumer)",
                    "Regulatory Risk": "Increasing (data privacy, antitrust)",
                    "Disruption Risk": "Very High (rapid innovation cycles)",
                    "Pricing Power Risk": "Varies (high for software, low for hardware)",
                    "Input Cost Risk": "Low to Moderate",
                    "Geopolitical Risk": "High (supply chain, market access)"
                },
                "beta_range": "1.0-1.5",
                "description": "Technology sector faces high disruption risk from rapid innovation cycles, increasing regulatory scrutiny, and geopolitical tensions affecting supply chains and market access. Software companies typically enjoy stronger pricing power and margins than hardware."
            },

            "Healthcare": {
                "overall_risk": "Moderate",
                "specific_risks": {
                    "Business Cycle Sensitivity": "Low",
                    "Regulatory Risk": "Very High (pricing, approval, reimbursement)",
                    "Disruption Risk": "Moderate to High (pharmaceuticals, digital health)",
                    "Pricing Power Risk": "Increasing (political pressure)",
                    "Input Cost Risk": "Low",
                    "Geopolitical Risk": "Moderate (supply chain, market access)"
                },
                "beta_range": "0.8-1.1",
                "description": "Healthcare sector combines stable demand characteristics with significant regulatory uncertainty. Pharmaceutical and biotech companies face pipeline risk and patent cliffs, while all healthcare businesses navigate complex reimbursement environments and pricing pressures."
            },

            "Financials": {
                "overall_risk": "Moderate to High",
                "specific_risks": {
                    "Business Cycle Sensitivity": "High",
                    "Regulatory Risk": "High (capital requirements, consumer protection)",
                    "Disruption Risk": "Moderate to High (fintech)",
                    "Pricing Power Risk": "Moderate",
                    "Interest Rate Risk": "High",
                    "Credit Risk": "High",
                    "Systemic Risk": "High"
                },
                "beta_range": "1.1-1.4",
                "description": "Financial sector faces significant cyclical exposure through credit quality and trading activity. Interest rate changes impact net interest margins, while financial institutions must navigate complex regulatory requirements and technological disruption."
            },

            "Energy": {
                "overall_risk": "Very High",
                "specific_risks": {
                    "Business Cycle Sensitivity": "High",
                    "Regulatory Risk": "High (environmental, carbon pricing)",
                    "Commodity Price Risk": "Very High",
                    "Disruption Risk": "High (energy transition)",
                    "Geopolitical Risk": "Very High",
                    "Stranded Asset Risk": "High (fossil fuels)"
                },
                "beta_range": "1.3-1.7",
                "description": "Energy sector faces extreme commodity price volatility, significant geopolitical risk affecting supply and demand, and long-term disruption from the clean energy transition. Traditional oil and gas companies must navigate declining long-term demand and stranded asset risk."
            },

            "Consumer Discretionary": {
                "overall_risk": "High",
                "specific_risks": {
                    "Business Cycle Sensitivity": "Very High",
                    "Consumer Preference Risk": "High",
                    "Disruption Risk": "High (e-commerce, direct-to-consumer)",
                    "Pricing Power Risk": "Moderate to High",
                    "Input Cost Risk": "Moderate",
                    "Competition Risk": "High"
                },
                "beta_range": "1.2-1.5",
                "description": "Materials sector experiences extreme cyclicality tied to industrial production and construction activity. Companies face significant commodity price volatility, high capital intensity, and increasing environmental regulation affecting operations and capital requirements."
            },

            "Real Estate": {
                "overall_risk": "Moderate",
                "specific_risks": {
                    "Business Cycle Sensitivity": "Moderate to High",
                    "Interest Rate Risk": "High",
                    "Occupancy Risk": "Moderate (varies by property type)",
                    "Disruption Risk": "Moderate (varies by property type)",
                    "Regulatory Risk": "Moderate",
                    "Geographic Concentration Risk": "Varies"
                },
                "beta_range": "0.8-1.1",
                "description": "Real estate combines stable income streams with cyclical exposure to economic conditions affecting occupancy and rental rates. Interest rate sensitivity creates valuation pressure during tightening cycles, while different property types face varying levels of disruption risk."
            },

            "Communication Services": {
                "overall_risk": "Moderate",
                "specific_risks": {
                    "Business Cycle Sensitivity": "Low (telecom) / Moderate (media)",
                    "Regulatory Risk": "High (antitrust, content, privacy)",
                    "Disruption Risk": "Moderate to High",
                    "Competition Risk": "High",
                    "Technology Obsolescence Risk": "Moderate to High",
                    "Content Cost Inflation Risk": "High (media)"
                },
                "beta_range": "0.8-1.2",
                "description": "Communications sector combines utility-like telecom businesses with more volatile media and entertainment. Content creation businesses face high competition for audience attention, while infrastructure providers navigate heavy capital requirements and technological disruption."
            },

            "Utilities": {
                "overall_risk": "Low",
                "specific_risks": {
                    "Business Cycle Sensitivity": "Very Low",
                    "Regulatory Risk": "High (rate cases, environmental)",
                    "Interest Rate Risk": "High",
                    "Disruption Risk": "Moderate (renewables, distributed generation)",
                    "Weather/Disaster Risk": "Moderate to High",
                    "Input Cost Risk": "Moderate"
                },
                "beta_range": "0.5-0.7",
                "description": "Utilities offer the lowest business cycle sensitivity with regulated returns and essential service characteristics. Primary risks include regulatory outcomes affecting allowed returns, interest rate sensitivity, and long-term disruption from distributed energy resources and renewable transition."
            }
        }

        return risk_profiles.get(main_sector, {
            "overall_risk": "Unknown",
            "specific_risks": {},
            "beta_range": "Unknown",
            "description": f"No risk profile available for sector: {sector}"
        })

    def get_sector_growth_drivers(self, sector: str) -> Dict[str, Any]:
        """
        Get key growth drivers for the specified sector

        Args:
            sector: Sector name

        Returns:
            Dictionary with sector growth drivers
        """
        main_sector = self._get_main_sector(sector)

        # Growth drivers by sector
        growth_drivers = {
            "Information Technology": {
                "structural_drivers": [
                    "Digital transformation across industries",
                    "Cloud computing and SaaS transition",
                    "Artificial intelligence and machine learning adoption",
                    "Data analytics and big data",
                    "Cybersecurity requirements",
                    "5G and connectivity expansion"
                ],
                "growth_metrics_to_track": [
                    "Annual recurring revenue (ARR) growth",
                    "Customer acquisition cost (CAC)",
                    "Customer lifetime value (LTV)",
                    "Net dollar retention rate",
                    "R&D as percentage of revenue"
                ],
                "historical_cagr": "12-15% (last decade)",
                "long_term_outlook": "Strong sustained growth expected, though varying significantly across subsectors with software growing faster than hardware"
            },

            "Healthcare": {
                "structural_drivers": [
                    "Aging global population",
                    "Increasing prevalence of chronic diseases",
                    "Access expansion in emerging markets",
                    "Precision medicine and diagnostics",
                    "Digital health and telemedicine",
                    "Biotechnology innovation"
                ],
                "growth_metrics_to_track": [
                    "Pipeline progression and approvals",
                    "R&D productivity metrics",
                    "Patent expiration timeline",
                    "Procedure/utilization growth",
                    "Pricing trends and reimbursement changes"
                ],
                "historical_cagr": "7-9% (last decade)",
                "long_term_outlook": "Strong sustainable growth driven by demographics and innovation, though cost containment pressures may moderate pricing components"
            },

            "Financials": {
                "structural_drivers": [
                    "Interest rate environment",
                    "Credit growth and quality",
                    "GDP and economic expansion",
                    "Wealth accumulation and asset growth",
                    "Digital payments adoption",
                    "Financial inclusion in emerging markets"
                ],
                "growth_metrics_to_track": [
                    "Net interest margin trends",
                    "Loan growth",
                    "Fee income growth",
                    "Assets under management (AUM) growth",
                    "Digital adoption metrics"
                ],
                "historical_cagr": "4-6% (last decade)",
                "long_term_outlook": "Moderate growth correlated with economic expansion, with digital finance offering pockets of higher growth potential"
            },

            "Energy": {
                "structural_drivers": [
                    "Global energy demand growth",
                    "Energy transition to renewables",
                    "Battery storage deployment",
                    "Hydrogen and alternative fuels",
                    "Electrification trends",
                    "Energy efficiency requirements"
                ],
                "growth_metrics_to_track": [
                    "Production growth",
                    "Reserve replacement ratio",
                    "Renewable capacity additions",
                    "Refining/processing margins",
                    "Return on capital employed (ROCE)"
                ],
                "historical_cagr": "1-3% (traditional) / 15-20% (renewables) (last decade)",
                "long_term_outlook": "Divergent outlook with traditional fossil fuels facing long-term structural decline while renewables experience rapid growth"
            },

            "Consumer Discretionary": {
                "structural_drivers": [
                    "Middle class expansion globally",
                    "E-commerce penetration",
                    "Omnichannel retail integration",
                    "Experience economy growth",
                    "Personalization and customization",
                    "Sustainability and ethical consumption"
                ],
                "growth_metrics_to_track": [
                    "Same-store sales growth",
                    "E-commerce sales growth",
                    "Average transaction value",
                    "Customer acquisition cost",
                    "Brand value metrics"
                ],
                "historical_cagr": "5-7% (last decade)",
                "long_term_outlook": "Moderate growth with significant divergence between traditional retail (low/negative) and digital-first businesses (high)"
            },

            "Consumer Staples": {
                "structural_drivers": [
                    "Global population growth",
                    "Emerging market consumption growth",
                    "Premiumization trends",
                    "Health and wellness focus",
                    "Private label penetration",
                    "Sustainability initiatives"
                ],
                "growth_metrics_to_track": [
                    "Organic sales growth",
                    "Volume vs. price growth components",
                    "Market share trends",
                    "New product contribution to growth",
                    "Emerging market growth"
                ],
                "historical_cagr": "3-5% (last decade)",
                "long_term_outlook": "Stable modest growth slightly above population growth, with emerging markets and premium categories offering higher growth potential"
            },

            "Industrials": {
                "structural_drivers": [
                    "Infrastructure investment",
                    "Automation and robotics adoption",
                    "Reshoring and supply chain reconfiguration",
                    "Sustainability and energy efficiency",
                    "Urbanization in emerging markets",
                    "Defense modernization"
                ],
                "growth_metrics_to_track": [
                    "Order intake and book-to-bill ratio",
                    "Backlog growth",
                    "Service revenue growth",
                    "Aftermarket vs. OEM sales mix",
                    "Operating margin trends"
                ],
                "historical_cagr": "4-6% (last decade)",
                "long_term_outlook": "Moderate growth driven by automation, infrastructure renewal, and sustainability investments, with cyclical fluctuations around the trend"
            },

            "Materials": {
                "structural_drivers": [
                    "Global infrastructure development",
                    "Urbanization in emerging markets",
                    "Construction and housing activity",
                    "Electric vehicle and battery materials",
                    "Lightweighting and sustainable materials",
                    "Packaging innovation"
                ],
                "growth_metrics_to_track": [
                    "Volume growth vs. price",
                    "Capacity utilization",
                    "Capacity addition plans",
                    "Supply-demand balance by material",
                    "Input cost trends"
                ],
                "historical_cagr": "3-5% (last decade)",
                "long_term_outlook": "Moderate cyclical growth with high volatility around trend, with specialty materials growing faster than commodities"
            },

            "Real Estate": {
                "structural_drivers": [
                    "Population growth and household formation",
                    "Urbanization trends",
                    "E-commerce driving logistics demand",
                    "Data center and digital infrastructure growth",
                    "Flexible work affecting office demand",
                    "Aging demographics driving healthcare real estate"
                ],
                "growth_metrics_to_track": [
                    "Net operating income (NOI) growth",
                    "Occupancy rates",
                    "Leasing spreads",
                    "Development pipeline",
                    "Same-property NOI growth"
                ],
                "historical_cagr": "4-6% (last decade)",
                "long_term_outlook": "Moderate growth slightly above inflation, with significant divergence between property types (industrial/data centers growing, retail/office challenged)"
            },

            "Communication Services": {
                "structural_drivers": [
                    "Data consumption growth",
                    "5G infrastructure deployment",
                    "Streaming media adoption",
                    "Digital advertising shift",
                    "Gaming and interactive entertainment",
                    "Social media engagement"
                ],
                "growth_metrics_to_track": [
                    "ARPU trends",
                    "Subscriber growth/churn",
                    "Content engagement metrics",
                    "Ad revenue per user",
                    "Average revenue per user (ARPU)"
                ],
                "historical_cagr": "4-8% (last decade)",
                "long_term_outlook": "Moderate growth with digital media and entertainment growing faster than traditional telecommunications"
            },

            "Utilities": {
                "structural_drivers": [
                    "Electrification trends",
                    "Renewable energy transition",
                    "Grid modernization and resilience",
                    "Population and household growth",
                    "Energy efficiency offsetting demand",
                    "Electric vehicle charging infrastructure"
                ],
                "growth_metrics_to_track": [
                    "Rate base growth",
                    "Regulatory allowed ROE",
                    "Capital expenditure plans",
                    "Customer growth",
                    "Renewable capacity additions"
                ],
                "historical_cagr": "2-4% (last decade)",
                "long_term_outlook": "Stable modest growth slightly above inflation, with potential acceleration from electrification and grid investments"
            }
        }

        return growth_drivers.get(main_sector, {
            "structural_drivers": [],
            "growth_metrics_to_track": [],
            "historical_cagr": "Unknown",
            "long_term_outlook": f"No growth driver information available for sector: {sector}"
        })



    # Example usage
    if __name__ == "__main__":

        sector_ratios = SectorRatios()

        # Example: Get key ratios for Technology sector
        tech_ratios = sector_ratios.get_sector_key_ratios("Information Technology")
        print(f"Key ratios for Technology sector: {tech_ratios}")

        # Example: Get sector benchmarks for a subsector
        software_benchmarks = sector_ratios.get_sector_benchmarks("Software")
        print(f"Benchmarks for Software subsector: {software_benchmarks}")

        # Example: Compare two sectors
        sector_comparison = sector_ratios.compare_sectors("Technology", "Healthcare")
        print(f"Key differences: {sector_comparison['key_differences']}")

        # Example: Get sector risk profile
        risk_profile = sector_ratios.get_sector_risk_profile("Financials")
        print(f"Financial sector risk profile: {risk_profile['overall_risk']} - {risk_profile['description']}")