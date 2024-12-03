import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessing:
    """
    Class for preprocessing stock data. Handles downloading, normalizing, and 
    splitting the data into train, validation, and test sets.
    """
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def download_data(self):
        """
        Downloads the stock closing price data from Yahoo Finance.
        """
        stock = yf.Ticker(self.ticker)
        self.stock_data = stock.history(start=self.start_date, end=self.end_date)
        self.stock_data.reset_index(inplace=True)

        # Convert 'Date' to datetime and extract the date part
        self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date']).dt.date
        self.stock_data = self.stock_data[['Date', 'Close']]
        self.stock_data.set_index('Date', inplace=True)
        
    def normalize_data(self):
        """
        Normalizes the data using MinMaxScaler, and splits the data into train, 
        validation, and test sets.
        """
        q_80 = int(len(self.stock_data) * 0.8)
        q_90 = int(len(self.stock_data) * 0.9)
        
        # Split data into train, validation, test
        self.train_data = self.stock_data.iloc[:q_80].copy()
        self.val_data = self.stock_data.iloc[q_80:q_90].copy()
        self.test_data = self.stock_data.iloc[q_90:].copy()

        # Fit scaler on training data
        self.train_data['Close'] = self.scaler.fit_transform(self.train_data[['Close']])
        self.val_data['Close'] = self.scaler.transform(self.val_data[['Close']])
        self.test_data['Close'] = self.scaler.transform(self.test_data[['Close']])
