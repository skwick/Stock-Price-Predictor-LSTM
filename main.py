from data_preprocessing import DataPreprocessing
from model import StockPricePredictor
from utils import Utils
import yfinance as yf
from datetime import datetime
from ui import UserInterface

def get_ticker():
    """
    Prompt the user to enter a validstock ticker.
    Continuously request input until a valid ticker is entered.
    """
    while True: 
        try:
            ticker = input("Please enter a stock ticker: ")
            if not ticker.strip():
                raise ValueError("Ticker cannot be empty.")
        
            stock = yf.Ticker(ticker)
            # Check if the ticker is valid
            if stock.history(period="1d").empty:
                raise ValueError("Invalid ticker symbol. Please enter a valid stock ticker.")
                
            return ticker
        except ValueError as e:
            print(f"Invalid input. {e}. Please try again.")
                
def get_date_range():
    """
    Prompt the user to enter a valid start and end date.
    Continuously request input until valid dates are entered.
    """
    while True:
        try:
            start_date = input("Please enter the start date (YYYY-MM-DD): ")
            end_date = input("Please enter the end date (YYYY-MM-DD): ")
                
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
                
            return start_date, end_date
        except ValueError:
            print("Invalid date format. Please enter dates in the format YYYY-MM-DD.")

def main():
    """
    The main function that controls the flow of the program.
    1. Get the stock ticker and date range from the user.
    2. Download and preprocess the data.
    3. Build and train the model.
    4. Initialize the UI and run it.
    """
    # Get Stock Ticker and Date Range
    ticker = get_ticker()
    start_date, end_date = get_date_range()

    # Data Preprocessing
    data_processor = DataPreprocessing(ticker, start_date, end_date)
    data_processor.download_data()
    data_processor.normalize_data()

    # Get the train, validation, and test data from the data processor
    train_data = data_processor.train_data
    val_data = data_processor.val_data
    test_data = data_processor.test_data

    # Create windowed datasets for train, validation, and test
    n = 3 # Window size
    train_windowed, train_targets, train_dates = Utils.df_to_windowed_df(train_data, start_date, end_date, n)
    val_windowed, val_targets, val_dates = Utils.df_to_windowed_df(val_data, start_date, end_date, n)
    test_windowed, test_targets, test_dates = Utils.df_to_windowed_df(test_data, start_date, end_date, n)

    # Reshape the windowed datasets to be compatible with the model
    train_windowed = train_windowed.reshape((train_windowed.shape[0], train_windowed.shape[1], 1))
    val_windowed = val_windowed.reshape((val_windowed.shape[0], val_windowed.shape[1], 1))
    test_windowed = test_windowed.reshape((test_windowed.shape[0], test_windowed.shape[1], 1))
   
    # Build and train the model
    model = StockPricePredictor()
    model.build_model()
    model.train_model(train_windowed, train_targets, val_windowed, val_targets, epochs=50)

    # Generate predictions for train, validation, and test data sets
    train_predictions = model.predict_price(train_windowed)
    val_predictions = model.predict_price(val_windowed)
    test_predictions = model.predict_price(test_windowed)

    # Initialize the UI
    ui = UserInterface(
        model=model,
        train_predictions=train_predictions,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
        scaler=data_processor.scaler,
        stock_data=data_processor.stock_data,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        train_dates=train_dates,
        val_dates=val_dates,
        test_dates=test_dates
    )
    
    # Run the UI for user interaction
    ui.run()

if __name__ == "__main__":
    main()
               