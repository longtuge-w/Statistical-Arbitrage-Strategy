# Statistical-Arbitrage-Strategy

## Dependencies

os, matplotlib, pandas, numpy, tqdm, sklearn, sys, statsmodels.api

## Main Script

The `main.py` file orchestrates the workflow for analyzing financial data, generating signals, and backtesting a trading strategy. It utilizes classes and methods defined in other files to carry out the analysis and backtesting over a specified date range.

### Workflow:

1. **Data Preparation**: Reads price and universe data.
2. **Signal Generation**: Utilizes `SignalGenerator` to create trading signals based on predefined thresholds.
3. **Analysis**: Uses `FinancialAnalyzer` to perform various analyses, including computing s-scores and plotting results.
4. **Backtesting**: Employs `PortfolioBacktester` to simulate trading based on generated signals and evaluates performance.

### Parameters:

- Date ranges for analysis and backtesting.
- Thresholds for signal generation.
- Tokens to include in the analysis.

### Execution:

Run the script directly to perform the entire workflow from data preparation to backtesting.

## Portfolio Backtester

Defined in `Backtest.py`, the `PortfolioBacktester` class simulates trading based on input signals and evaluates the performance of a trading strategy.

### Key Methods:

- **`backtest_portfolio`**: Simulates trading over a given period based on trading signals and price data. It tracks portfolio value and records transactions.
- **`process_signal`**: Processes individual trading signals and updates portfolio holdings and cash.
- **`plot_cumulative_return_and_calculate_metrics`**: Plots cumulative returns of the portfolio and calculates performance metrics such as Sharpe ratio and maximum drawdown.

### Portfolio Management:

The portfolio is managed by executing trades based on signals and adjusting positions while considering transaction fees.

## Financial Analyzer

`FinancialAnalyzer` and `utils.py` contain functionalities for analyzing financial data, such as selecting common tokens, computing returns, and applying PCA for risk factor construction.

### Key Features:

- **Selection of Common Tokens**: Identifies tokens with significant market presence over a specified time frame.
- **Hourly Returns Calculation**: Computes hourly returns for selected tokens.
- **Risk Factor Construction**: Uses PCA to identify principal components and constructs risk factors.
- **s-Score Calculation**: Estimates Ornstein-Uhlenbeck parameters and calculates s-scores for trading signal generation.

### Analysis and Visualization:

These scripts also provide methods for visualizing cumulative returns, eigenportfolio weights, and s-scores, aiding in the interpretation of analysis results.

## Modifiable Parameters

Parameters that can be modified to generate different results include:

- Starting capital in the PortfolioBacktester class
- Transaction fees in the PortfolioBacktester class
- Maximum shares per stock in the PortfolioBacktester class
- Thresholds for generating trading signals in the FinancialAnalyzer class

## How to Run the Codes

1. Ensure all dependencies are installed.
2. Run the `main.py` script to execute the backtesting process.
3. Modify the parameters as needed in the script for different scenarios.

```bash
python main.py
```
