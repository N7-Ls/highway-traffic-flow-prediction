# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:24:08 2024

@author: bryan
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data(filepath, sequence_length):
    data = pd.read_csv(filepath)  # Adjust file reading method as needed
    values = data['target_column'].values  # Replace 'target_column' with actual column
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))

    X, y = [], []
    for i in range(sequence_length, len(scaled_values)):
        X.append(scaled_values[i - sequence_length:i, 0])
        y.append(scaled_values[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for RNN input
    return X, y, scaler

# Define LSTM model
def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Define GRU model
def create_gru_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train model
def train_model(model, X_train, y_train, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

# Evaluate model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    plt.plot(y_test, label='True Values')
    plt.plot(predictions, label='Predictions')
    plt.legend()
    plt.show()

    return mse

# Usage Example
filepath = 'data.csv'  # Replace with actual file path
sequence_length = 60  # Length of time steps
X, y, scaler = load_data(filepath, sequence_length)

# Split into train and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM
lstm_model = create_lstm_model((X_train.shape[1], 1))
lstm_history = train_model(lstm_model, X_train, y_train, epochs=50, batch_size=32)
lstm_mse = evaluate_model(lstm_model, X_test, y_test, scaler)

# GRU
gru_model = create_gru_model((X_train.shape[1], 1))
gru_history = train_model(gru_model, X_train, y_train, epochs=50, batch_size=32)
gru_mse = evaluate_model(gru_model, X_test, y_test, scaler)
