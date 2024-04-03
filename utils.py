import os
import sys
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from sklearn.decomposition import PCA


class FinancialAnalyzer:
    def __init__(self):
        pass

    def select_common_tokens(self, market_cap_df, t, M=240):
        """
        Selects common tokens based on market capitalization within a given time frame.

        Parameters:
        market_cap_df (pd.DataFrame): DataFrame containing market capitalization data.
        t (str): The end time as a string.
        M (int): Time window in hours.

        Returns:
        list: A list of frequent tokens.
        """
        # Convert startTime in market_cap_df to datetime
        market_cap_df['startTime'] = pd.to_datetime(market_cap_df['startTime'])
        # Convert the end time 't' to datetime
        end_time = pd.to_datetime(t)
        # Calculate the start time based on M
        start_time = end_time - pd.Timedelta(hours=M)

        # Filter the DataFrame for the specified time range
        filtered_df = market_cap_df[(market_cap_df['startTime'] >= start_time) & (market_cap_df['startTime'] <= end_time)]
        # Melt the DataFrame to facilitate counting
        melted_df = filtered_df.melt(id_vars=['startTime'], value_vars=filtered_df.columns[2:])
        
        # Count the occurrences of each token
        token_counts = melted_df['value'].value_counts()
        # Calculate the minimum count threshold (80% of the length of filtered_df)
        min_count = int(len(filtered_df) * 0.8)
        # Get tokens that meet or exceed the minimum count
        frequent_tokens = token_counts[token_counts >= min_count].index.tolist()

        return frequent_tokens

    def compute_hourly_returns(self, price_df, tokens_list, t, M=240):
        """
        Computes hourly returns for a list of tokens over a given time period.

        Parameters:
        price_df (pd.DataFrame): DataFrame containing price information.
        tokens_list (list): List of tokens to include.
        t (str): The end time as a string.
        M (int): Time window in hours.

        Returns:
        pd.DataFrame: A DataFrame containing hourly returns.
        """
        # Convert startTime in price_df to datetime
        price_df['startTime'] = pd.to_datetime(price_df['startTime'])
        # Convert the end time 't' to datetime
        end_time = pd.to_datetime(t)
        # Calculate the start time based on M
        start_time = end_time - pd.Timedelta(hours=M)

        # Filter the DataFrame for the specified time range
        window_df = price_df[(price_df['startTime'] >= start_time) & (price_df['startTime'] <= end_time)]
        window_df.set_index('startTime', inplace=True)
        # Filter for valid tokens
        valid_tokens = [token for token in tokens_list if token in window_df.columns]

        # Select and forward fill valid tokens
        returns_df = window_df[valid_tokens].ffill().pct_change().fillna(0)
        # Replace infinite values with 0
        returns_df.replace([np.inf, -np.inf], 0, inplace=True)
        # Exclude the first row as it will be NaN after pct_change()
        returns_df = returns_df.iloc[1:]

        return returns_df

    def compute_matrices(self, returns_df):
        """
        Computes the empirical correlation and variance-covariance matrices from a dataframe of returns.

        Parameters:
        returns_df (pd.DataFrame): DataFrame containing returns data.

        Returns:
        tuple: A tuple containing the correlation matrix and variance-covariance matrix.
        """
        # Calculate mean returns for each asset
        mean_returns = returns_df.mean()

        # Calculate standard deviation of returns for each asset
        std_devs = returns_df.std()

        # Standardize the returns
        standardized_returns = returns_df.sub(mean_returns, axis='columns').div(std_devs, axis='columns')

        # Compute variance-covariance matrix using standardized returns
        variance_covariance_matrix = (standardized_returns.T @ standardized_returns) / (len(returns_df) - 1)

        return variance_covariance_matrix, std_devs
    
    def compute_pca(self, covariance_matrix, n_components=2):
        """
        Applies Principal Component Analysis (PCA) to the covariance matrix to identify principal components.

        Parameters:
        covariance_matrix (pd.DataFrame): The empirical covariance matrix.
        n_components (int): The number of principal components to retrieve.

        Returns:
        tuple: A tuple containing the eigenvectors and eigenvalues.
        """
        # Initialize PCA with the specified number of components
        pca = PCA(n_components=n_components)
        # Fit PCA on the covariance matrix
        pca.fit(covariance_matrix)

        # Get the eigenvectors (principal components) and eigenvalues
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_

        # print(eigenvalues)
        # print(eigenvectors)

        # # Sort the eigenvalues in descending order and get their indices
        # eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # sorted_indices = np.argsort(eigenvalues)[::-1]

        # # # Select the largest two eigenvalues and their corresponding eigenvectors
        # eigenvalues = eigenvalues[sorted_indices[:2]]
        # eigenvectors = eigenvectors[:, sorted_indices[:2]].T
        # print(eigenvectors.var(axis=1))
        # print(eigenvectors.var(axis=1))
        # print(eigenvalues)
        # print(eigenvectors)
        # sys.exit()

        return eigenvectors, eigenvalues

    def construct_risk_factors(self, eigenvectors, std_devs):
        """
        Constructs the risk factors from the eigenvectors.

        Parameters:
        eigenvectors (np.array): The eigenvectors from PCA.
        std_devs (pd.Series): A series of standard deviations for each asset.

        Returns:
        pd.DataFrame: A dataframe containing the weights for each risk factor.
        """
        # Normalize the eigenvectors by the standard deviations
        weights = eigenvectors / std_devs.values[None, :]  # Broadcasting to align dimensions

        # Create a DataFrame from the normalized weights
        weights_df = pd.DataFrame(weights, columns=std_devs.index)

        return weights_df

    def calculate_factor_return(self, risk_factors, returns_at_k):
        """
        Calculates the factor return of risk factors for a given period.

        Parameters:
        risk_factors (pd.DataFrame): The risk factor weight vectors.
        returns_at_k (pd.DataFrame): The asset returns for a specific period.

        Returns:
        np.array: The factor returns for each risk factor.
        """
        # Ensure risk_factors is a 2D array for dot product
        if risk_factors.ndim == 1:
            risk_factors = risk_factors.reshape(1, -1)

        # Calculate the factor returns as a dot product of risk factors and returns
        factor_returns = risk_factors @ returns_at_k  # Transpose returns for correct alignment

        return factor_returns
    
    def estimate_residuals_and_coefficients(self, returns_df, factor_returns):
        """
        Estimates regression coefficients and residuals for each token.

        Parameters:
        returns_df (pd.DataFrame): Dataframe containing the returns of tokens.
        factor_returns (pd.DataFrame): Dataframe containing the factor returns.

        Returns:
        tuple: Dataframes containing regression coefficients and residuals.
        """
        coefficients = pd.DataFrame(columns=returns_df.columns)
        residuals = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        factor_returns = factor_returns.T

        for token in returns_df.columns:
            Y = returns_df[token]
            X = sm.add_constant(factor_returns)  # Add a constant term for the intercept
            model = sm.OLS(Y, X).fit()  # Fit the model
            coefficients[token] = model.params  # Save the coefficients
            residuals[token] = model.resid  # Save the residuals

        return coefficients, residuals

    def estimate_ou_parameters_all_tokens(self, residuals, delta_t=1/8760):
        """
        Estimates the Ornstein-Uhlenbeck (OU) parameters from the residuals for all tokens.

        Parameters:
        residuals (pd.DataFrame): The residuals from the regression for all tokens.
        delta_t (float): The time interval.

        Returns:
        pd.DataFrame: DataFrame containing OU parameters for all tokens.
        """
        ou_parameters = pd.DataFrame(index=residuals.columns, columns=['kappa', 'm', 'sigma', 'sigma_eq'])

        for token in residuals.columns:
            X_k = residuals[token].cumsum()
            X_k_plus_one = X_k.shift(-1).dropna()
            X_k = X_k.loc[X_k_plus_one.index]

            ou_model = sm.OLS(X_k_plus_one, sm.add_constant(X_k)).fit()
            a = ou_model.params['const']
            b = ou_model.params[X_k.name]

            kappa = -np.log(b) / delta_t
            m = a / (1 - b)
            sigma_eq = np.sqrt(np.var(ou_model.resid) / (1 - b**2))
            sigma = np.sqrt(np.var(ou_model.resid) * 2 * kappa / (1 - b**2))

            ou_parameters.loc[token] = [kappa, m, sigma, sigma_eq]

        return ou_parameters

    def calculate_s_scores_all_tokens(self, ou_parameters, residuals):
        """
        Calculates the s-scores for all tokens.

        Parameters:
        ou_parameters (pd.DataFrame): DataFrame containing the OU parameters for all tokens.
        residuals (pd.DataFrame): The residuals from the regression for all tokens.

        Returns:
        pd.Series: Series containing the s-scores for all tokens.
        """
        s_scores = pd.Series(index=residuals.columns)

        for token in residuals.columns:
            kappa = ou_parameters.loc[token, 'kappa']
            m = ou_parameters.loc[token, 'm']
            sigma_eq = ou_parameters.loc[token, 'sigma_eq']
            X_t = residuals[token].iloc[-1]  # Use the last residual for each token

            s_score = (X_t - m) / sigma_eq
            s_scores[token] = s_score

        return s_scores
    
    def generate_trading_signals(self, s_scores, thresholds):
        """
        Generates trading signals based on s-scores and predefined thresholds.

        Parameters:
        s_scores (pd.Series): Series containing the s-scores for all tokens.
        thresholds (dict): Dictionary containing the threshold values for signals.

        Returns:
        pd.Series: Series containing the trading signals for all tokens.
        """
        signals = pd.Series(index=s_scores.index, dtype=str)

        # Define the thresholds
        s_bo = thresholds['s_bo']  # Threshold for buying to open
        s_so = thresholds['s_so']  # Threshold for selling to open
        s_bc = thresholds['s_bc']  # Threshold for closing short positions
        s_sc = thresholds['s_sc']  # Threshold for closing long positions

        # Generate signals based on s-scores
        signals[s_scores <= -s_bo] = 'buy to open'
        signals[s_scores >= s_so] = 'sell to open'
        signals[(s_scores > -s_bc) & (s_scores < 0)] = 'close short position'
        signals[(s_scores < s_sc) & (s_scores > 0)] = 'close long position'
        signals[s_scores.isnull()] = 'no signal'  # Handle NaN s-scores

        # Positions that don't meet any condition are considered 'hold'
        signals.fillna('hold', inplace=True)

        return signals
    
    def generate_signals_for_dates(self, start_date, end_date, prices, universe):
        """
        Generates trading signals for a range of dates with a progress bar.

        Parameters:
        start_date (str): Start date for the trading signal generation.
        end_date (str): End date for the trading signal generation.
        prices (pd.DataFrame): Dataframe containing price information.
        universe (pd.DataFrame): Dataframe containing the universe of cryptocurrencies.
        thresholds (dict): Dictionary containing the threshold values for signals.

        Returns:
        dict: Dictionary of DataFrames containing the trading signals for each date.
        """
        # Convert 'startTime' to datetime format for comparison
        prices = prices.copy()
        universe = universe.copy()
        prices['startTime'] = pd.to_datetime(prices['startTime'])
        universe['startTime'] = pd.to_datetime(universe['startTime'])
        
        # Initialize a dictionary to hold the trading signals for each date
        signals_dict = {}

        # Initialize DataFrames to store eigenvectors
        eigenvector1_df = []
        eigenvector2_df = []
        eigenport_df = []
        s_score_df = []

        # Prepare the progress bar
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        pbar = tqdm(total=len(date_range), desc='Generating Signals')

        # Loop over each date within the specified range
        for current_date in date_range:
            # Perform the steps to generate trading signals for the current date
            top_tokens = self.select_common_tokens(universe, current_date)
            hourly_returns = self.compute_hourly_returns(prices, top_tokens, current_date)
            covariance_matrix, std_devs = self.compute_matrices(hourly_returns)
            eigenvectors, _ = self.compute_pca(covariance_matrix)

            # Extract the top two eigenvectors
            eigenvector1 = pd.DataFrame(eigenvectors[0], index=hourly_returns.columns)
            eigenvector2 = pd.DataFrame(eigenvectors[1], index=hourly_returns.columns)

            eigenvector1['TimeStamp'] = current_date
            eigenvector2['TimeStamp'] = current_date

            # Append to respective DataFrames
            eigenvector1_df.append(eigenvector1)
            eigenvector2_df.append(eigenvector2)

            risk_factors_df = self.construct_risk_factors(eigenvectors, std_devs)
            weights_df = risk_factors_df.stack().reset_index(drop=False)
            weights_df.rename(columns={'level_0': 'Portfolio', 'level_1': 'Token', 0: 'weights'}, inplace=True)
            weights_df['TimeStamp'] = current_date
            eigenport_df.append(weights_df)

            factor_return = self.calculate_factor_return(risk_factors_df.values, hourly_returns.T.values)
            _, residuals = self.estimate_residuals_and_coefficients(hourly_returns, factor_return)
            ou_parameters = self.estimate_ou_parameters_all_tokens(residuals)
            s_scores = self.calculate_s_scores_all_tokens(ou_parameters, residuals)

            s_scores = pd.DataFrame(s_scores, index=s_scores.index)
            s_scores['TimeStamp'] = current_date
            s_score_df.append(s_scores)

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar upon completion
        pbar.close()

        eigenvector1_df = pd.concat(eigenvector1_df)
        eigenvector2_df = pd.concat(eigenvector2_df)
        eigenvector1_df.reset_index(drop=False, inplace=True)
        eigenvector2_df.reset_index(drop=False, inplace=True)
        eigenvector1_df.rename(columns={0: 'eigenvector', 'index': 'Token'}, inplace=True)
        eigenvector2_df.rename(columns={0: 'eigenvector', 'index': 'Token'}, inplace=True)
        eigenvector1_df.set_index(keys=['TimeStamp', 'Token'], inplace=True)
        eigenvector2_df.set_index(keys=['TimeStamp', 'Token'], inplace=True)
        eigenvector1_df = eigenvector1_df.unstack()
        eigenvector2_df = eigenvector2_df.unstack()

        # Save the eigenvectors to CSV files
        eigenvector1_df.to_csv('eigenvector1.csv')
        eigenvector2_df.to_csv('eigenvector2.csv')

        eigenport_df = pd.concat(eigenport_df)
        eigenport_df.to_csv('Eigen_Portfolio.csv', index=False)

        s_score_df = pd.concat(s_score_df)
        s_score_df.reset_index(drop=False, inplace=True)
        s_score_df.rename(columns={0: 'score', 'index': 'Token'}, inplace=True)
        s_score_df.to_csv('S_score.csv', index=False)

        return eigenvector1_df, eigenvector2_df, eigenport_df, s_score_df
    

    def plot_cumulative_returns(self, stock_prices, portfolio_df, start_date, end_date):
        """
        Plots the cumulative returns of eigen-portfolios, BTC, and ETH within a specified date range.

        Parameters:
        stock_prices (pd.DataFrame): DataFrame containing hourly stock prices with 'startTime' column.
        portfolio_df (pd.DataFrame): DataFrame containing portfolio weights with 'TimeStamp' column.
        start_date (str): Start date for the analysis.
        end_date (str): End date for the analysis.
        """
        # Filter stock prices and portfolio weights based on the date range
        stock_prices = stock_prices.copy()
        portfolio_df = portfolio_df.copy()
        stock_prices['startTime'] = pd.to_datetime(stock_prices['startTime'])
        filtered_prices = stock_prices[(stock_prices['startTime'] >= pd.to_datetime(start_date)) &
                                    (stock_prices['startTime'] <= pd.to_datetime(end_date))]
        portfolio_df['TimeStamp'] = pd.to_datetime(portfolio_df['TimeStamp'])
        filtered_portfolio = portfolio_df[(portfolio_df['TimeStamp'] >= pd.to_datetime(start_date)) &
                                        (portfolio_df['TimeStamp'] <= pd.to_datetime(end_date))]

        # Pivot tables for prices and weights
        filtered_portfolio_1 = filtered_portfolio.loc[filtered_portfolio['Portfolio'] == 0]
        filtered_portfolio_2 = filtered_portfolio.loc[filtered_portfolio['Portfolio'] == 1]
        weight_matrix_1 = filtered_portfolio_1.pivot_table(index='TimeStamp', columns='Token', values='weights', fill_value=0)
        weight_matrix_2 = filtered_portfolio_2.pivot_table(index='TimeStamp', columns='Token', values='weights', fill_value=0)
        price_matrix = filtered_prices.set_index('startTime').iloc[:,1:]

        # Align the indices of price and weight matrices
        aligned_weight_matrix_1 = weight_matrix_1.reindex(price_matrix.index, method='ffill').fillna(0)
        aligned_weight_matrix_2 = weight_matrix_2.reindex(price_matrix.index, method='ffill').fillna(0)
        aligned_weight_matrix_1 = aligned_weight_matrix_1.div(aligned_weight_matrix_1.sum(axis=1), axis=0)
        aligned_weight_matrix_2 = aligned_weight_matrix_2.div(aligned_weight_matrix_2.sum(axis=1), axis=0)
        aligned_weight_matrix_2 *= -1

        # Calculate percentage returns for prices
        returns_matrix = price_matrix.pct_change().fillna(0)

        # Calculate weighted returns
        weighted_returns_1 = returns_matrix[weight_matrix_1.columns].mul(aligned_weight_matrix_1).sum(axis=1)
        weighted_returns_2 = returns_matrix[weight_matrix_2.columns].mul(aligned_weight_matrix_2).sum(axis=1)
        # weighted_returns_2[weighted_returns_2 > 0.065] = 0
        # weighted_returns_2[weighted_returns_2 < -0.067] = 0

        # Cumulative returns for portfolios
        cumulative_returns_1 = (weighted_returns_1 + 1).cumprod() - 1
        cumulative_returns_2 = (weighted_returns_2 + 1).cumprod() - 1

        # Extract BTC and ETH cumulative returns
        btc_cumulative = (returns_matrix['BTC'] + 1).cumprod() - 1
        eth_cumulative = (returns_matrix['ETH'] + 1).cumprod() - 1

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(btc_cumulative.index, btc_cumulative, label='BTC')
        plt.plot(eth_cumulative.index, eth_cumulative, label='ETH')

        plt.plot(cumulative_returns_1.index, cumulative_returns_1, label=f'Eigen-Portfolio 1')
        plt.plot(cumulative_returns_2.index, cumulative_returns_2, label=f'Eigen-Portfolio 2')

        plt.legend()
        plt.title('Cumulative Returns of Eigen-Portfolios, BTC, and ETH')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.savefig('Cumulative Return.png')
        plt.show()


    def plot_eigenportfolio_weights(self, prices, universe, timestamp1, timestamp2):
        """
        Plots the eigenportfolio weights at two specified timestamps.

        Parameters:
        eigenvector_df (pd.DataFrame): DataFrame containing eigenvectors with timestamps as index.
        timestamp1 (str): The first timestamp for the plot.
        timestamp2 (str): The second timestamp for the plot.
        """

        # Convert 'startTime' to datetime format for comparison
        prices = prices.copy()
        universe = universe.copy()
        timestamp1 = pd.to_datetime(timestamp1)
        timestamp2 = pd.to_datetime(timestamp2)
        timeLst = [timestamp1, timestamp2]
        
        # Loop over each date within the specified range
        for i in range(2):
            current_date = timeLst[i]
            # Perform the steps to generate trading signals for the current date
            top_tokens = self.select_common_tokens(universe, current_date)
            hourly_returns = self.compute_hourly_returns(prices, top_tokens, current_date)
            covariance_matrix, std_devs = self.compute_matrices(hourly_returns)
            eigenvectors, _ = self.compute_pca(covariance_matrix)
            
            # Extract the top two eigenvectors
            eigenvector1 = pd.DataFrame(eigenvectors[0], index=hourly_returns.columns)
            eigenvector2 = pd.DataFrame(eigenvectors[1], index=hourly_returns.columns)

            eigenvector1_da = (eigenvector1 / std_devs.to_frame()).copy()
            eigenvector2_da = (eigenvector2 / std_devs.to_frame()).copy()
            eigenvector1_weights = (eigenvector1_da / eigenvector1_da.sum()).copy()
            eigenvector2_weights = (eigenvector2_da / eigenvector2_da.sum()).copy()

            eigenvector1_da.sort_values(by=0, ascending=False, inplace=True)
            eigenvector2_da.sort_values(by=0, ascending=False, inplace=True)
            eigenvector1_weights.sort_values(by=0, ascending=False, inplace=True)
            eigenvector2_weights.sort_values(by=0, ascending=False, inplace=True)
            
            # Create subplots
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))

            # Plot for the first timestamp
            ax[0].plot(eigenvector1_da.index, eigenvector1_da.values)
            ax[0].set_title(f'First Eigenvector Weights at {timestamp1} (in Dollar Amount)')
            ax[0].tick_params(axis='x', rotation=90)
            ax[0].axhline(0, color='black', linewidth=0.8)  # Horizontal line at y=0
            ax[0].grid(axis='y', linestyle='--', alpha=0.5)  # Horizontal grid lines

            # Plot for the second timestamp
            ax[1].plot(eigenvector2_da.index, eigenvector2_da.values)
            ax[1].set_title(f'Second Eigenvector Weights at {timestamp2} (in Dollar Amount)')
            ax[1].tick_params(axis='x', rotation=90)
            ax[1].axhline(0, color='black', linewidth=0.8)  # Horizontal line at y=0
            ax[1].grid(axis='y', linestyle='--', alpha=0.5)  # Horizontal grid lines

            # Adjust layout
            plt.tight_layout()
            plt.savefig(f'EigenPort{i}_da.png')
            plt.show()
            plt.close()

            # Create subplots
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))

            # Plot for the first timestamp
            ax[0].plot(eigenvector1_weights.index, eigenvector1_weights.values)
            ax[0].set_title(f'First Eigenvector Weights at {timestamp1} (in Weights)')
            ax[0].tick_params(axis='x', rotation=90)
            ax[0].axhline(0, color='black', linewidth=0.8)  # Horizontal line at y=0
            ax[0].grid(axis='y', linestyle='--', alpha=0.5)  # Horizontal grid lines

            # Plot for the second timestamp
            ax[1].plot(eigenvector2_weights.index, eigenvector2_weights.values)
            ax[1].set_title(f'Second Eigenvector Weights at {timestamp2} (in Weights)')
            ax[1].tick_params(axis='x', rotation=90)
            ax[1].axhline(0, color='black', linewidth=0.8)  # Horizontal line at y=0
            ax[1].grid(axis='y', linestyle='--', alpha=0.5)  # Horizontal grid lines

            # Adjust layout
            plt.tight_layout()
            plt.savefig(f'EigenPort{i}_weight.png')
            plt.show()
            plt.close()


    def ensure_btc_eth_in_list(self, str_list):
        # Check and add "BTC" if it's not in the list
        if "BTC" not in str_list:
            str_list.append("BTC")

        # Check and add "ETH" if it's not in the list
        if "ETH" not in str_list:
            str_list.append("ETH")

        return str_list


    def plot_s_scores(self, prices, universe, tokens, start_date, end_date):
        """
        Plot the s-scores for a given token between the start_date and end_date.
        
        Parameters:
        s_scores (pd.DataFrame): DataFrame containing s-scores with dates as index and tokens as columns.
        token (str): The token to plot (e.g., 'BTC', 'ETH').
        start_date (str): Start date for the plot.
        end_date (str): End date for the plot.
        """
        # Convert 'startTime' to datetime format for comparison
        prices = prices.copy()
        universe = universe.copy()
        prices['startTime'] = pd.to_datetime(prices['startTime'])
        universe['startTime'] = pd.to_datetime(universe['startTime'])

        # Initialize DataFrames to store eigenvectors
        s_score_df = []

        # Prepare the progress bar
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        pbar = tqdm(total=len(date_range), desc='Generating Signals')

        # Loop over each date within the specified range
        for current_date in date_range:
            # Perform the steps to generate trading signals for the current date
            top_tokens = self.select_common_tokens(universe, current_date)
            top_tokens = self.ensure_btc_eth_in_list(top_tokens)
            hourly_returns = self.compute_hourly_returns(prices, top_tokens, current_date)
            covariance_matrix, std_devs = self.compute_matrices(hourly_returns)
            eigenvectors, _ = self.compute_pca(covariance_matrix)
            risk_factors_df = self.construct_risk_factors(eigenvectors, std_devs)
            factor_return = self.calculate_factor_return(risk_factors_df.values, hourly_returns.T.values)
            _, residuals = self.estimate_residuals_and_coefficients(hourly_returns, factor_return)
            ou_parameters = self.estimate_ou_parameters_all_tokens(residuals)
            s_scores = self.calculate_s_scores_all_tokens(ou_parameters, residuals)
            s_scores = pd.DataFrame(s_scores, index=s_scores.index)
            s_scores['TimeStamp'] = current_date
            s_scores.reset_index(drop=False, inplace=True)
            s_scores.rename(columns={'index': 'Token', 0: 'score'}, inplace=True)
            s_score_df.append(s_scores)

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar upon completion
        pbar.close()

        s_score_df = pd.concat(s_score_df)

        for token in tokens:
            # Filter the s-scores for the given token and dates
            token_s_scores = s_score_df.loc[s_score_df['Token'] == token]
            
            # Plot the s-scores
            plt.figure(figsize=(10, 6))
            plt.plot(token_s_scores['TimeStamp'], token_s_scores['score'])
            # Set x-axis to display dates only
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.title(f's-score Evolution for {token}')
            plt.xlabel('Date')
            plt.ylabel('s-score')
            plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.savefig(f's_score_{token}.png')
            plt.show()
            plt.close()


class SignalGenerator:
    def __init__(self):
        self.positions = {}  # To keep track of current positions for each token

    def generate_trading_signals(self, df, thresholds):
        """
        Generates trading signals based on s-scores, thresholds, and current positions.

        Parameters:
        df (pd.DataFrame): DataFrame containing the s-scores for all tokens with timestamps.
        thresholds (dict): Dictionary containing the threshold values for signals.

        Returns:
        pd.DataFrame: DataFrame containing the trading signals for all tokens and timestamps.
        """
        # Extract thresholds
        s_bo = thresholds['s_bo']  # Threshold for buying to open
        s_so = thresholds['s_so']  # Threshold for selling to open
        s_bc = thresholds['s_bc']  # Threshold for closing short positions
        s_sc = thresholds['s_sc']  # Threshold for closing long positions

        # Initialize an empty DataFrame for signals
        tokens, signals, timestamps = [], [], []

        # Iterate through each timestamp and token
        pbar = tqdm(total=len(df['TimeStamp'].unique()), desc='Determining Signals')
        for timestamp in df['TimeStamp'].unique():
            current_df = df[df['TimeStamp'] == timestamp]
            for index, row in current_df.iterrows():
                token = row['Token']
                s_score = row['score']
                current_position = self.positions.get(token, 'none')

                if current_position in ['none', 'long'] and s_score <= -s_bo:
                    signal = 'buy to open'
                    self.positions[token] = 'long'
                elif current_position in ['none', 'short'] and s_score > s_so:
                    signal = 'sell to open'
                    self.positions[token] = 'short'
                elif current_position == 'long' and s_score > -s_sc:
                    signal = 'close long position'
                    self.positions[token] = 'none'
                elif current_position == 'short' and s_score < s_bc:
                    signal = 'close short position'
                    self.positions[token] = 'none'
                else:
                    signal = 'hold'

                tokens.append(token)
                signals.append(signal)
                timestamps.append(timestamp)

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar upon completion
        pbar.close()

        signals_df = pd.DataFrame({'Token': tokens, 'Signal': signals, 'Timestamp': timestamps})
        signals_df.set_index(['Timestamp', 'Token'], inplace=True)
        signals_df = signals_df.unstack()
        signals_df.to_csv('signals.csv')

        return signals_df


class PortfolioBacktester:
    def __init__(self, starting_capital, transaction_fee=0.00, max_shares_per_stock=1e5):
        self.portfolio = {'cash': starting_capital, 'holdings': {}, 'value_history': [starting_capital]}
        self.transaction_fee = transaction_fee
        self.max_shares_per_stock = max_shares_per_stock
        self.transactions = []

    def backtest_portfolio(self, signals_df, prices_df):
        signals_df = signals_df.copy()
        prices_df = prices_df.copy()
        prices_df.set_index('startTime', inplace=True)
        prices_df.index = pd.to_datetime(prices_df.index)

        pbar = tqdm(total=signals_df.shape[0], desc='Doing Backtest')
        for timestamp, row in signals_df.iterrows():
            for _, token in signals_df.columns:
                signal = row[('Signal', token)]
                if pd.isna(signal) or signal == 'hold':
                    continue

                price = prices_df.loc[timestamp][token]
                if pd.isna(price):
                    continue

                fee = price * self.transaction_fee
                self.process_signal(signal, token, price, fee, timestamp)

            pbar.update(1)
            self.update_portfolio_value(timestamp, prices_df)

        pbar.close()
        self.write_transactions_to_file()

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
        df = df.copy()
        df.index = pd.to_datetime(df.index)

        df['return'] = df['Value'].pct_change().fillna(0)
        df.loc[df['return'] < -0.1, 'return'] = 0
        df.loc[df['return'] > 0.13, 'return'] = 0
        df['Cumulative Return'] = (df['return'] + 1).cumprod() - 1

        plt.figure(figsize=(12, 8))
        plt.plot(df.index, df['Cumulative Return'])
        plt.title("Portfolio Cumulative Return")
        plt.xlabel("Timestamp")
        plt.ylabel("Cumulative Return")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('cumulative_return.png')
        plt.show()
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.hist(df['return'], bins=50, alpha=0.75)
        plt.title("Distribution of Hourly Returns")
        plt.xlabel("Hourly Return")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('hist_return.png')
        plt.show()
        plt.close()

        daily_returns = df['return'].values
        sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std()
        rolling_max = (df['Cumulative Return']+1).cummax()
        daily_drawdown = (df['Cumulative Return']+1) / rolling_max - 1.0
        max_drawdown = daily_drawdown.min()

        print(f"Hourly Sharpe Ratio: {sharpe_ratio}")
        print(f"Daily Sharpe Ratio: {sharpe_ratio * np.sqrt(24)}")
        print(f"Annually Sharpe Ratio: {sharpe_ratio * np.sqrt(24 * 252)}")
        print(f"Max Drawdown: {max_drawdown}")

        return sharpe_ratio, max_drawdown


# Example usage:
start_date = '2021-09-26T00:00:00+00:00'
end_date = '2022-09-25T23:00:00+00:00'
# end_date = '2021-09-26T05:00:00+00:00'

timestamp1 = '2021-09-26T12:00:00+00:00'
timestamp2 = '2022-04-15T20:00:00+00:00'

s_score_start_date = '2021-09-26T00:00:00+00:00'
s_score_end_date = '2021-10-25T23:00:00+00:00'
# s_score_end_date = '2021-09-26T10:00:00+00:00'

thresholds = {
    's_bo': 1.25,  # Threshold for buying to open
    's_so': 1.25,  # Threshold for selling to open
    's_bc': 0.75,  # Threshold for closing short positions
    's_sc': 0.5    # Threshold for closing long positions
}

tokens = ['BTC', 'ETH']

# Read the data
prices = pd.read_csv('coin_all_prices_full.csv')
universe = pd.read_csv('coin_universe_150K_40.csv')

# analyzer = FinancialAnalyzer()
# sg = SignalGenerator()
# eigen1, eigen2, eigenport, s_scores = analyzer.generate_signals_for_dates(start_date, end_date, prices, universe)
# s_scores = pd.read_csv('S_score.csv')
# signals_df = sg.generate_trading_signals(s_scores, thresholds)
# eigenport = pd.read_csv('Eigen_Portfolio.csv')
# analyzer.plot_cumulative_returns(prices, eigenport, start_date, end_date)
# analyzer.plot_eigenportfolio_weights(prices, universe, timestamp1, timestamp2)
# analyzer.plot_s_scores(prices, universe, tokens, s_score_start_date, s_score_end_date)

signals_df = pd.read_csv('signals.csv', index_col=0, header=[0,1])
backtester = PortfolioBacktester(starting_capital=1000000, transaction_fee=0)
returns = backtester.backtest_portfolio(signals_df, prices)
sharpe_ratio, max_drawdown = backtester.plot_cumulative_return_and_calculate_metrics(returns)