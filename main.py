# Import necessary modules from Analyzer, Signal_Generator, and Backtest files
from Analyzer import *
from Signal_Generator import *
from Backtest import *

# The main section of the script
if __name__ == "__main__":

    # Define time period for analysis and backtesting
    start_date = '2021-09-26T00:00:00+00:00'
    end_date = '2022-09-25T23:00:00+00:00'

    # Additional timestamps for specific analyses
    timestamp1 = '2021-09-26T12:00:00+00:00'
    timestamp2 = '2022-04-15T20:00:00+00:00'

    # Define time period for calculating S-scores
    s_score_start_date = '2021-09-26T00:00:00+00:00'
    s_score_end_date = '2021-10-25T23:00:00+00:00'

    # Set thresholds for different trading signals
    thresholds = {
        's_bo': 1.25,  # Threshold for buying to open
        's_so': 1.25,  # Threshold for selling to open
        's_bc': 0.75,  # Threshold for closing short positions
        's_sc': 0.5    # Threshold for closing long positions
    }

    # Define the tokens to analyze
    tokens = ['BTC', 'ETH']

    # Read in price and universe data from CSV files
    prices = pd.read_csv('coin_all_prices_full.csv')
    universe = pd.read_csv('coin_universe_150K_40.csv')

    # Initialize Analyzer and Signal Generator
    analyzer = FinancialAnalyzer()
    sg = SignalGenerator()

    # Generate signals and S-scores for the specified dates
    eigen1, eigen2, eigenport, s_scores = analyzer.generate_signals_for_dates(start_date, end_date, prices, universe)

    # Generate trading signals based on the S-scores and predefined thresholds
    signals_df = sg.generate_trading_signals(s_scores, thresholds)

    # # Plot various analyses including cumulative returns, eigenportfolio weights, and S-scores
    analyzer.plot_cumulative_returns(prices, eigenport, start_date, end_date)
    analyzer.plot_eigenportfolio_weights(prices, universe, timestamp1, timestamp2)
    analyzer.plot_s_scores(prices, universe, tokens, s_score_start_date, s_score_end_date)

    # Initialize the backtester with a starting capital and transaction fee
    backtester = PortfolioBacktester(starting_capital=1000000, transaction_fee=0, max_shares_per_stock=1e4)

    # Perform the backtesting with the generated signals and price data
    returns = backtester.backtest_portfolio(signals_df, prices)

    # Calculate and plot performance metrics such as Sharpe ratio and maximum drawdown
    sharpe_ratio, max_drawdown = backtester.plot_cumulative_return_and_calculate_metrics(returns)