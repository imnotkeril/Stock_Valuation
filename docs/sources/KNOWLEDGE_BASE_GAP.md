# Owner's knowledge base vs the software (2026-11-26)

Source: `E:\Main\Main\Папки\WildMarketCapital\База знаний, Шаблоны и тд` (owner's notes, Russian PDFs;
the Notion page "Finance clerk guide" was not readable here - no Notion connector in the session).
The software was built from textbook methods, not from these notes; this file maps one to the other.

## Covered
| Note | In the software |
|---|---|
| Income approach: DCF, DDM, capitalisation, WACC, CAPM | DCF tab + sector DCFs, DDM (Gordon), WACC/CAPM (Blume beta), APV |
| Comparative approach: multiples, precedent transactions | Relative tab, Precedents tab |
| Cost approach: NAV | Asset-Based tab (haircut slider), REIT NAV |
| Fundamental models: DCF, DDM, RIM, multiples, SOTP | all present (RIM = financial-sector excess return) |
| Ratios: liquidity, profitability, leverage, efficiency, valuation (P/E, P/S, P/B, P/CF, P/FCF, PEG, ROE, ROA, ROIC) | Financials > Ratios, Overview key ratios |
| Risk: stress testing, VaR, drawdown, volatility, Z-score | Risk tab (Altman/Springate/Zmijewski, scenarios, VaR/CVaR) |
| Sectors (11 S&P sectors, key metrics) | 8 sector models; NIM, loan/deposit, FFO/P-FFO, NAV, cap rate, occupancy, reserves, ARPU, churn, same-store, inventory turnover, rate base, backlog, R&D/revenue, dividend yield |

## Gaps (candidates, owner decides)
| Gap | Source note | Effort |
|---|---|---|
| Liquidation value as a separate method | Оценка бизнеса 3.2 | small |
| Value adjustments: control premium, excess/non-core assets, contingent liabilities | Оценка бизнеса 5.x | medium (inputs are user-entered) |
| Sortino, Treynor, Calmar, Information ratio, Jensen's alpha on the Risk tab | Список моделей §2 | small |
| Capital adequacy (CET1) for banks | Секторы, Financials | blocked: not in yfinance statements |
| Materials / Utilities / Staples as own models (today: manufacturing / energy / retail) | Секторы 6, 8, 10 | medium |
| Insider transactions | Фундаментальная оценка 11.4 | small with Finnhub key |
| Russian A1-A4 / P1-P4 balance liquidity grouping | Финанализ из универа | small |
| Qualitative: Porter five forces, moat, management | Фундаментальная оценка 3-4 | out of scope (not computable) |

Out of scope by SPEC §6: portfolio optimisation (Markowitz, Black-Litterman, Kelly), GARCH family,
microstructure / HFT, stat-arb, macro (Notion Macro folder; owner dropped MacroLens ideas).
