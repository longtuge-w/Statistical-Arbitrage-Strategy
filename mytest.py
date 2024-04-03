# from datetime import datetime, timedelta

# # Define start and end dates
# start_date = datetime.fromisoformat('2021-09-26T00:00:00+00:00')
# end_date = datetime.fromisoformat('2022-09-25T23:00:00+00:00')

# # Calculate the total number of data points
# total_data_points = 8760

# # Calculate the time interval between each data point
# time_interval = (end_date - start_date) / total_data_points

# # Find the time for the 388th data point
# data_point_388_time = start_date + time_interval * (322 - 1)  # Subtracting 1 because we start counting from 0
# print(data_point_388_time.isoformat())


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.decomposition import PCA
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')


class PCA_Factor:
    def __init__(self, coin_data, coin_name, date):
        self.time_win = 240
        self.coin_data = coin_data.set_index('startTime')
        self.coin_name = coin_name.set_index('startTime')
        self.date = pd.to_datetime(date)

        self.coin_pool = self.coin_name.loc[date].iloc[1:].tolist()
        self.current_market = None
        self.rets, self.norm_rets = self.cal_ret()
        self.pca = PCA(n_components=2)
        self.Q = None
        self.factor_matrix = self.cal_pca()

    def cal_ret(self):
        date_idx = self.coin_data.index.get_loc(self.date)
        start_idx = max(0, date_idx - self.time_win) # Ensure that idx is not negative
        self.current_market = self.coin_data.iloc[start_idx:date_idx + 1][self.coin_pool]
        self.current_market = self.current_market.replace([np.nan, np.Inf, -np.Inf], 0)

        # The tokens must have 80% valid data within the time window
        valid = (self.current_market != 0).mean()
        self.coin_pool = valid[valid >= 0.8].index.tolist()
        self.current_market = self.current_market[self.coin_pool]

        R_ik = self.current_market.pct_change()
        R_ik = R_ik.iloc[1:]
        R_ik = R_ik.replace([np.nan, np.Inf, -np.Inf], 0)
        Y_ik = zscore(R_ik, nan_policy = 'omit')
        Y_ik = Y_ik.replace([np.nan, np.Inf, -np.Inf], 0)
        
        return R_ik, Y_ik
    
    def cal_pca(self):
        self.pca.fit(self.rets.corr())
        
        price_std_devs = np.sqrt(np.array(self.norm_rets.var()))
        self.Q = self.pca.components_ / price_std_devs
        factor_matrix = self.Q @ self.rets.T
        return factor_matrix

    ## Question 3.1
    def get_largest_eigenvector(self):
        return self.pca.explained_variance_[0], self.pca.components_[0]

    def get_second_largest_eigenvector(self):
        return self.pca.explained_variance_[1], self.pca.components_[1]