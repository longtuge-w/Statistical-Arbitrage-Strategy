import os
import sys
import pandas as pd
from tqdm import tqdm


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
        signals_df.to_csv('trading_signal.csv')

        return signals_df