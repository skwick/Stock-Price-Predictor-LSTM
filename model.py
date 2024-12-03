from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np

class StockPricePredictor:
    """
    Class for building, training, and predicting stock prices using an LSTM model.
    """
    def __init__(self):
        self.model = None
        
    def build_model(self):
        """
        Builds and compilesthe LSTM model.
        """
        self.model = Sequential([layers.Input(shape=(3, 1)),    # Input layer expecting a window of 3 previous prices
                                 layers.LSTM(64),               # LSTM layer with 64 units
                                 layers.Dense(32, activation='relu'), # First hidden dense layer with 32 units and ReLU activation
                                 layers.Dense(32, activation='relu'), # Second hidden dense layer with 32 units and ReLU activation
                                 layers.Dense(1)])                  # Output layer for predicting the next price
        
        # Compile the model with mean squared error loss, Adam optimizer, and mean absolute error metric
        self.model.compile(loss='mse', 
                           optimizer=Adam(learning_rate=0.001),
                           metrics=['mean_absolute_error'])
        
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Trains the model on the training data and validates on the validation data.
        """
        history = self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))
        
        # Store the training and validation loss for the optional loss plot
        self.train_loss = history.history['loss']
        self.val_loss = history.history['val_loss']
        
    def predict_price(self, X_test):
        """
        Predicts the stock price for the given test data.
        """
        predictions = self.model.predict(X_test)
        return predictions.flatten()
