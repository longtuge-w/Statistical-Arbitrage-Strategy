# Import required libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the PortfolioBacktester class
class PortfolioBacktester:
    def __init__(self, starting_capital, transaction_fee=0.00, max_shares_per_stock=1e5):
        # Initialize the portfolio with starting capital, transaction fees, and maximum shares per stock
        self.portfolio = {'cash': starting_capital, 'holdings': {}, 'value_history': [starting_capital]}
        self.transaction_fee = transaction_fee
        self.max_shares_per_stock = max_shares_per_stock
        self.transactions = []

    def backtest_portfolio(self, signals_df, prices_df):
        # Prepare data for backtesting and initialize progress bar
        signals_df = signals_df.copy()
        prices_df = prices_df.copy()
        prices_df.set_index('startTime', inplace=True)
        prices_df.index = pd.to_datetime(prices_df.index)
        pbar = tqdm(total=signals_df.shape[0], desc='Doing Backtest')

        # Process each signal in the data
        for timestamp, row in signals_df.iterrows():
            for _, token in signals_df.columns:
                signal = row[('Signal', token)]
                if pd.isna(signal) or signal == 'hold':
                    continue

                price = prices_df.loc[timestamp][token]
                if pd.isna(price):
                    continue

                # Calculate fee and process the signal
                fee = price * self.transaction_fee
                self.process_signal(signal, token, price, fee, timestamp)

            # Update progress bar and portfolio value
            pbar.update(1)
            self.update_portfolio_value(timestamp, prices_df)

        # Finalize progress bar and write transactions to file
        pbar.close()
        self.write_transactions_to_file()

        # Prepare and return the dataframe of portfolio value history
        return_df = pd.DataFrame(data=self.portfolio['value_history'][1:], index=signals_df.index)
        return_df.rename(columns={0: 'Value'}, inplace=True)
        return return_df

    def process_signal(self, signal, token, price, fee, timestamp):
        if signal == 'buy to open' and self.portfolio['cash'] >= (price + fee):
            self.buy_to_open(token, price, fee, timestamp)
        elif signal == 'sell to open':
            self.sell_to_open(token, price, fee, timestamp)
        elif signal == 'close long position' and self.portfolio['holdings'].get(token, 0) > 0:
            self.close_long_position(token, price, fee, timestamp)
        elif signal == 'close short position' and self.portfolio['holdings'].get(token, 0) < 0:
            self.close_short_position(token, price, fee, timestamp)

    def buy_to_open(self, token, price, fee, timestamp):
        current_shares = self.portfolio['holdings'].get(token, 0)
        if current_shares < self.max_shares_per_stock:
            shares_to_buy = min(1, self.max_shares_per_stock - current_shares)
            self.portfolio['holdings'][token] = current_shares + shares_to_buy
            self.portfolio['cash'] -= (shares_to_buy * price + fee)
            self.transactions.append((timestamp, token, 'buy', shares_to_buy, price, self.portfolio['cash']))

    def sell_to_open(self, token, price, fee, timestamp):
        current_shares = self.portfolio['holdings'].get(token, 0)
        if current_shares > -self.max_shares_per_stock:
            shares_to_sell = min(1, self.max_shares_per_stock + current_shares)
            self.portfolio['holdings'][token] = current_shares - shares_to_sell
            self.portfolio['cash'] += (shares_to_sell * price - fee)
            self.transactions.append((timestamp, token, 'sell', shares_to_sell, price, self.portfolio['cash']))

    def close_long_position(self, token, price, fee, timestamp):
        self.portfolio['cash'] += (self.portfolio['holdings'][token] * price) - (self.portfolio['holdings'][token] * fee)
        self.transactions.append((timestamp, token, 'sell', self.portfolio['holdings'][token], price, self.portfolio['cash']))
        del self.portfolio['holdings'][token]

    def close_short_position(self, token, price, fee, timestamp):
        self.portfolio['cash'] -= (abs(self.portfolio['holdings'][token]) * price) + (abs(self.portfolio['holdings'][token]) * fee)
        self.transactions.append((timestamp, token, 'buy', abs(self.portfolio['holdings'][token]), price, self.portfolio['cash']))
        del self.portfolio['holdings'][token]

    def update_portfolio_value(self, timestamp, prices_df):
        portfolio_value = self.portfolio['cash'] + sum(self.portfolio['holdings'].get(token, 0) * prices_df.loc[timestamp][token] for token in self.portfolio['holdings'])
        self.portfolio['value_history'].append(portfolio_value)

    def write_transactions_to_file(self):
        with open('transactions.txt', 'w') as file:
            for transaction in self.transactions:
                file.write(f"{transaction[0]}: {transaction[2]} {transaction[3]} shares of {transaction[1]} at {transaction[4]} each. Remaining cash: {transaction[5]:.2f}\n")

    def plot_cumulative_return_and_calculate_metrics(self, df, risk_free_rate=0.0):
        # Create a copy of the dataframe and convert the index to datetime
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        # Calculate the percentage change to get returns and fill NaN values with 0
        df['return'] = df['Value'].pct_change().fillna(0)
        df.loc[df['return'] < -0.1, 'return'] = 0
        df.loc[df['return'] > 0.13, 'return'] = 0

        # Calculate cumulative return from the 'return' column
        df['Cumulative Return'] = (df['return'] + 1).cumprod() - 1

        # Plot the cumulative return over time
        plt.figure(figsize=(12, 8))
        plt.plot(df.index, df['Cumulative Return'])
        plt.title("Portfolio Cumulative Return")
        plt.xlabel("Timestamp")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_return.png')  # Save the figure as 'cumulative_return.png'
        plt.show()  # Display the plot
        plt.close()  # Close the plotting context

        # Plot the distribution of hourly returns
        plt.figure(figsize=(12, 8))
        plt.hist(df['return'], bins=50, alpha=0.75)
        plt.title("Distribution of Hourly Returns")
        plt.xlabel("Hourly Return")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('hist_return.png')  # Save the figure as 'hist_return.png'
        plt.show()  # Display the plot
        plt.close()  # Close the plotting context

        # Calculate Sharpe ratio using daily returns
        daily_returns = df['return'].values
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std()

        # Calculate maximum drawdown
        rolling_max = (df['Cumulative Return']+1).cummax()  # Track the rolling maximum value
        daily_drawdown = (df['Cumulative Return']+1) / rolling_max - 1.0  # Compute drawdown
        max_drawdown = daily_drawdown.min()  # Find the maximum drawdown

        # Print the calculated Sharpe ratio and maximum drawdown
        print(f"Hourly Sharpe Ratio: {sharpe_ratio}")
        print(f"Daily Sharpe Ratio: {sharpe_ratio * np.sqrt(24)}")  # Adjust Sharpe ratio for daily returns
        print(f"Annually Sharpe Ratio: {sharpe_ratio * np.sqrt(24 * 252)}")  # Adjust Sharpe ratio for annual returns
        print(f"Max Drawdown: {max_drawdown}")

        # Return the calculated Sharpe ratio and maximum drawdown
        return sharpe_ratio, max_drawdown