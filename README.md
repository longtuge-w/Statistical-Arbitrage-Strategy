# Statistical-Arbitrage-Strategy

## Dependencies

os, matplotlib, pandas, numpy, tqdm, sklearn, sys, statsmodels.api

## Analyzer Module

### Classes

- **FinancialAnalyzer**
  - **__init__(self)**
  - **select_common_tokens(self, market_cap_df, t, M)**
  - **compute_hourly_returns(self, price_df, tokens_list, t, M)**
  - **compute_matrices(self, returns_df)**
  - **compute_pca(self, covariance_matrix, n_components)**
  - **construct_risk_factors(self, eigenvectors, std_devs)**
  - **calculate_factor_return(self, risk_factors, returns_at_k)**
  - **estimate_residuals_and_coefficients(self, returns_df, factor_returns)**
  - **estimate_ou_parameters_all_tokens(self, residuals, delta_t)**
  - **calculate_s_scores_all_tokens(self, ou_parameters, residuals)**
  - **generate_trading_signals(self, s_scores, thresholds)**
  - **generate_signals_for_dates(self, start_date, end_date, prices, universe)**
  - **plot_cumulative_returns(self, stock_prices, portfolio_df, start_date, end_date)**
  - **plot_eigenportfolio_weights(self, prices, universe, timestamp1, timestamp2)**
  - **ensure_btc_eth_in_list(self, str_list)**
  - **plot_s_scores(self, prices, universe, tokens, start_date, end_date)**

### Functions

- **__init__(self, prices, universe, tokens, start_date, end_date)**
- **select_common_tokens(self, prices, universe, tokens, start_date, end_date)**
- **compute_hourly_returns(self, prices, universe, tokens, start_date, end_date)**
- **compute_matrices(self, prices, universe, tokens, start_date, end_date)**
- **compute_pca(self, prices, universe, tokens, start_date, end_date)**
- **construct_risk_factors(self, prices, universe, tokens, start_date, end_date)**
- **calculate_factor_return(self, prices, universe, tokens, start_date, end_date)**
- **estimate_residuals_and_coefficients(self, prices, universe, tokens, start_date, end_date)**
- **estimate_ou_parameters_all_tokens(self, prices, universe, tokens, start_date, end_date)**
- **calculate_s_scores_all_tokens(self, prices, universe, tokens, start_date, end_date)**
- **generate_trading_signals(self, prices, universe, tokens, start_date, end_date)**
- **generate_signals_for_dates(self, prices, universe, tokens, start_date, end_date)**
- **plot_cumulative_returns(self, prices, universe, tokens, start_date, end_date)**
- **plot_eigenportfolio_weights(self, prices, universe, tokens, start_date, end_date)**
- **ensure_btc_eth_in_list(self, prices, universe, tokens, start_date, end_date)**
- **plot_s_scores(self, prices, universe, tokens, start_date, end_date)**

## Backtest Module

### Classes

- **PortfolioBacktester**
  - **__init__(self, starting_capital, transaction_fee, max_shares_per_stock)**
  - **backtest_portfolio(self, signals_df, prices_df)**
  - **process_signal(self, signal, token, price, fee, timestamp)**
  - **buy_to_open(self, token, price, fee, timestamp)**
  - **sell_to_open(self, token, price, fee, timestamp)**
  - **close_long_position(self, token, price, fee, timestamp)**
  - **close_short_position(self, token, price, fee, timestamp)**
  - **update_portfolio_value(self, timestamp, prices_df)**
  - **write_transactions_to_file(self)**
  - **plot_cumulative_return_and_calculate_metrics(self, df, risk_free_rate)**

### Functions

- **__init__(self, df, risk_free_rate)**
- **backtest_portfolio(self, df, risk_free_rate)**
- **process_signal(self, df, risk_free_rate)**
- **buy_to_open(self, df, risk_free_rate)**
- **sell_to_open(self, df, risk_free_rate)**
- **close_long_position(self, df, risk_free_rate)**
- **close_short_position(self, df, risk_free_rate)**
- **update_portfolio_value(self, df, risk_free_rate)**
- **write_transactions_to_file(self, df, risk_free_rate)**
- **plot_cumulative_return_and_calculate_metrics(self, df, risk_free_rate)**

## main Module

## Signal_Generator Module

### Classes

- **SignalGenerator**
  - **__init__(self)**
  - **generate_trading_signals(self, df, thresholds)**

### Functions

- **__init__(self, df, thresholds)**
- **generate_trading_signals(self, df, thresholds)**

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
