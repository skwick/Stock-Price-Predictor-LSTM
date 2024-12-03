import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Utils:
    @staticmethod
    def df_to_windowed_df(dataframe, first_date, last_date, n):
        """
        Converts a dataframe to a windowed dataframe suitable for time series forecasting.
        """
        first_date = pd.Timestamp(first_date)
        last_date = pd.Timestamp(last_date)
        
        # Ensure the dataframe index is a datetime index
        if not isinstance(dataframe.index, pd.DatetimeIndex):
            dataframe.index = pd.to_datetime(dataframe.index)

        # Adjust the start and end dates to be within the dataframe's range
        first_date = max(first_date, dataframe.index.min())
        last_date = min(last_date, dataframe.index.max())

        X, Y, dates = [], [], []

        # Iterate over the date range from first_date to last_date
        for target_date in pd.date_range(first_date, last_date):
            df_subset = dataframe.loc[:target_date].tail(n + 1)
            if len(df_subset) < n + 1:
                continue

            # Append the last n-1 days of 'Close' prices and the last day's 'Close' price
            X.append(df_subset.iloc[:-1]['Close'].values) 
            Y.append(df_subset.iloc[-1]['Close'])
            dates.append(target_date)

        # Convert lists to numpy arrays
        X_array = np.array(X)
        Y_array = np.array(Y)

        return X_array, Y_array, dates

    @staticmethod
    def windowed_df_to_date_X_y(windowed_dataframe):
        """
        Converts a windowed dataframe to a date, X, and y array.
        """
        df_as_np = windowed_dataframe.to_numpy()
        dates = windowed_dataframe.index.to_numpy()

        # Select all columns after the first one
        middle_matrix = df_as_np[:, 1:]  
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        Y = df_as_np[:, -1]

        return dates, X.astype(np.float32), Y.astype(np.float32)

