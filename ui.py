from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class UserInterface:
    """
    Handles all user interactions with the program.
    Includes methods for calculating metrics, plotting results,
    plotting technical indicators, and predicting future prices.
    """
    def __init__(self, model, train_predictions, val_predictions, test_predictions, train_targets, val_targets, 
                 test_targets, scaler, stock_data, train_data, val_data, test_data, train_dates, val_dates, test_dates):
        self.model = model
        self.train_predictions = train_predictions
        self.val_predictions = val_predictions
        self.test_predictions = test_predictions
        self.train_targets = train_targets
        self.val_targets = val_targets
        self.test_targets = test_targets
        self.scaler = scaler
        self.stock_data = stock_data
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_dates = train_dates
        self.val_dates = val_dates
        self.test_dates = test_dates
    
    def run(self): 
        """
        Main loop that displays the main menu and handles user choices.
        """
        while True:
            print("\n----WELCOME TO THE STOCK PRICE PREDICTION APP----")
            print("1. Show performance metrics")
            print("2. Show loss graph")
            print("3. Plot stock data")              
            print("4. Plot stock data with technical indicators")
            print("5. Predict stock price")
            print("6. Exit")
            
            choice = input("Please choose a number (1-6): ")
            
            if choice == "1":
                print("\nSelect dataset for metrics:")
                print("a. Training set")
                print("b. Validation set")
                print("c. Test set")
                print("d. Metrics across all datasets combined")
                
                dataset_choice = input("Please choose an option (a-d): ").lower()
                
                if dataset_choice == "a":
                    # Calculate and display performance metrics for the training set
                    print("\n--- Training Set Metrics ---")
                    metrics = self.calculate_metrics(self.train_targets, self.train_predictions)
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
                        
                elif dataset_choice == "b":
                    # Calculate and display performance metrics for the validation set
                    print("\n--- Validation Set Metrics ---")
                    metrics = self.calculate_metrics(self.val_targets, self.val_predictions)
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
                        
                elif dataset_choice == "c":
                    # Calculate and display performance metrics for the test set
                    print("\n--- Test Set Metrics ---")
                    metrics = self.calculate_metrics(self.test_targets, self.test_predictions)
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
                        
                elif dataset_choice == "d":
                    # Combine all targets and predictions
                    combined_targets = np.concatenate([self.train_targets, self.val_targets, self.test_targets])
                    combined_predictions = np.concatenate([self.train_predictions, self.val_predictions, self.test_predictions])
                    
                    # Calculate metrics on the combined data
                    print("\n--- Combined Datasets Metrics ---")
                    metrics = self.calculate_metrics(combined_targets, combined_predictions)
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
                    
                else:
                    print("Invalid choice. Please try again")
                                        
            elif choice == "2":
                # Display a training and validation loss graph
                self.show_loss_graph(self.model.train_loss, self.model.val_loss)
                
            elif choice == "3":
                # Display the results of the model without technical indicators
                self.plot_results(
                    self.train_predictions, 
                    self.train_targets, 
                    self.val_predictions, 
                    self.val_targets, 
                    self.test_predictions, 
                    self.test_targets, 
                    self.scaler,
                    with_indicators=False,
                    stock_data=self.stock_data
                )
                
            elif choice == "4":
                # Display the results of the model with technical indicators
                self.plot_results(
                    self.train_predictions, 
                    self.train_targets, 
                    self.val_predictions, 
                    self.val_targets, 
                    self.test_predictions, 
                    self.test_targets, 
                    self.scaler,
                    with_indicators=True,
                    stock_data=self.stock_data
                )
                
            elif choice == "5":
                # Predict and display the future prices for the next x days
                self.predict_next_x_days()
            
            elif choice == "6":
                # Exit the program
                print("Exiting the program...")
                break
            
            else:
                print("Invalid choice. Please try again.")
        
            input("Press enter to return to the main menu.")
    
    def calculate_metrics(self, actual, predicted):
        """
        Calculate and return a dictionary of performance metrics.
        """
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        metrics = {
            "MAE (Mean Absolute Error)": round(mae, 4),
            "MSE (Mean Squared Error)": round(mse, 4),
            "RMSE (Root Mean Squared Error)": round(rmse, 4),
            "R2 (R-Squared)": round(r2, 4)
        }
        
        return metrics
    
    def show_loss_graph(self, train_loss, val_loss):
        """
        Display a training and validation loss graph.
        """
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()       
    
    def plot_results(self, train_predictions, train_targets, val_predictions, val_targets, test_predictions, test_targets, scaler, with_indicators=False, stock_data=None):
        """
        Plot the actual and predicted stock prices for training, validation, and test data.
        """
        plt.figure(figsize=(14, 8))

        # Plot training data
        if train_predictions is not None and train_targets is not None:
            train_actual = scaler.inverse_transform(train_targets.reshape(-1, 1)).flatten()
            train_pred = scaler.inverse_transform(train_predictions.reshape(-1, 1)).flatten()

            train_index = self.train_dates  # Use windowed train_dates
            assert len(train_index) == len(train_actual), f"Train index length ({len(train_index)}) does not match train_actual length ({len(train_actual)})."

            plt.plot(train_index, train_actual, label="Train Actual", color="blue")
            plt.plot(train_index, train_pred, label="Train Predicted", linestyle="dashed", color="cyan")

        # Plot validation data
        if val_predictions is not None and val_targets is not None:
            val_actual = scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            val_pred = scaler.inverse_transform(val_predictions.reshape(-1, 1)).flatten()

            val_index = self.val_dates  # Use windowed val_dates
            assert len(val_index) == len(val_actual), f"Validation index length ({len(val_index)}) does not match val_actual length ({len(val_actual)})."
        
            plt.plot(val_index, val_actual, label="Validation Actual", color="green")
            plt.plot(val_index, val_pred, label="Validation Predicted", linestyle="dashed", color="lightgreen")
        
        # Plot test data
        if test_predictions is not None and test_targets is not None:
            test_actual = scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()
            test_pred = scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()

            test_index = self.test_dates  # Use windowed test_dates
            assert len(test_index) == len(test_actual), f"Test index length ({len(test_index)}) does not match test_actual length ({len(test_actual)})."
        
            plt.plot(test_index, test_actual, label="Test Actual", color="red")
            plt.plot(test_index, test_pred, label="Test Predicted", linestyle="dashed", color="orange")

        # If the user chooses to plot with technical indicators, add them
        if with_indicators and stock_data is not None:
            self.plot_technical_indicators(stock_data)

        # Format x-axis to display dates correctly
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Formatting
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title("Stock Price Predictions")
        plt.grid(True)
        plt.tight_layout()

        # Show the plot
        plt.show()
        
    def plot_technical_indicators(self, stock_data):
        """
        Calculates and plots 25-Day SMA, 25-Day EMA, and RSI on the stock data.
        """
        # Calculate and plot 25 Day simple moving average
        sma = stock_data['Close'].rolling(window=25).mean()
        plt.plot(stock_data['Close'].index, sma, label='25-Day SMA', color='black', linestyle='dashed')
        
        # Calculate and plot 25 Day exponential moving average
        ema = stock_data['Close'].ewm(span=25, adjust=False).mean()
        plt.plot(stock_data['Close'].index, ema, label='25-Day EMA', color='gray', linestyle='dashed')

        # Calculate and plot RSI (Relative Strength Index)
        diff = stock_data['Close'].diff()
        gain = diff.where(diff > 0, 0)  
        loss = -diff.where(diff < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        plt.plot(stock_data['Close'].index, rsi, label='RSI', color='red', linestyle='dashed')

        # Calculate and plot MACD (Moving Average Convergence/Divergence)
        macd = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
        signal = macd.ewm(span=9, adjust=False).mean()
        plt.plot(stock_data['Close'].index, macd, label='MACD', color='blue', linestyle='dashed')
        plt.plot(stock_data['Close'].index, signal, label='Signal', color='green', linestyle='dashed')
   
    def predict_next_x_days(self):
        """
        Predicts and displays the future prices for the next x days.
        """
        try:
            x = int(input("Please enter the number of future days you want to predict: "))
            if x <= 0:
                raise ValueError("Number of days must be greater than 0.")
        
            predictions = self.generate_predictions(x)
        
            # Inverse transform the predictions to original scale
            future_prices = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
            # Get the last date from the test set
            last_date = self.test_dates[-1]
            future_dates = []
            current_date = pd.to_datetime(last_date)
        
            # Generate future dates, skipping weekends
            for _ in range(x):
                current_date += pd.Timedelta(days=1)
                while current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    current_date += pd.Timedelta(days=1)
                future_dates.append(current_date)
        
            print("\n--- Predicted Future Prices ---")
            for date, price in zip(future_dates, future_prices):
                print(f"{date.strftime('%Y-%m-%d')}: {price:.2f}")
    
        except ValueError as e:
            print(f"Invalid input. {e}. Please try again.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    
    def generate_predictions(self, x):
        """
        Generates predictions for the next x days.
        """
        model = self.model.model
        scaler = self.scaler

        # Get the last 3 days of data from the test set
        last_window = self.test_data.tail(3)['Close'].values

        if len(last_window) < 3:
            print("Insufficient data to form the initial prediction window.")
            return []

        # Reshape the last window to match the model's input shape
        last_window = last_window.astype(np.float32).reshape((1, 3, 1))

        predictions = []

        # Generate predictions for the next x days
        for day in range(x):
            try:
                # Predict the next day's price
                next_pred_array = model.predict(last_window, batch_size=1)

                next_pred = next_pred_array[0][0]
                predictions.append(next_pred)

                # Update the window with the new prediction
                last_window = np.append(last_window.flatten()[1:], next_pred).reshape((1, 3, 1)).astype(np.float32)

            except Exception as e:
                print(f"Error during prediction on day {day + 1}: {e}")
                break

        return predictions
    
    