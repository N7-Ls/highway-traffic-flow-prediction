# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 11:51:08 2025

@author: bryan
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

#%% Function to load and preprocess data for a specific month
def load_month_data(year, month):
    file_path = f"merged/{year}_{month}_merged.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, encoding='latin1')
    data.columns = [
        "Year", "Month", "Day", "Hour", "Minute", "Route", "Vehicle_Type", "Average_Speed", "Traffic_Volume", "Rainfall"
    ]
    data['datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    data = data.sort_values(by='datetime')

    # Filter out rows where Vehicle_Type != 31 and Minute != 0
    data = data[(data['Minute'] == 0) & (data['Average_Speed'] > 65)]

    # Create new features for day and hour
    data['DayOfWeek'] = data['datetime'].dt.dayofweek
    data['HourOfDay'] = data['Hour']

    # One-hot encode categorical features
    encoder_route = OneHotEncoder(sparse_output=False)
    route_encoded = encoder_route.fit_transform(data[['Route']])

    encoder_day = OneHotEncoder(sparse_output=False)
    day_encoded = encoder_day.fit_transform(data[['DayOfWeek']])

    encoder_hour = OneHotEncoder(sparse_output=False)
    hour_encoded = encoder_hour.fit_transform(data[['HourOfDay']])

    # Combine all features
    numerical_features = data[['Average_Speed', 'Rainfall']].values
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    # Add Traffic_Volume to features
    traffic_volume = data[['Traffic_Volume']].values
    traffic_volume = scaler.fit_transform(traffic_volume)

    features = np.hstack([
        route_encoded,
        day_encoded,
        hour_encoded,
        numerical_features,
        traffic_volume
    ])

    # Define target variables
    speed_scaler = StandardScaler()
    targets = speed_scaler.fit_transform(data[['Average_Speed']].values)

    return features, targets, features.shape[1], speed_scaler

#%% Sliding window generator function
def sliding_window_generator(features, targets, window_size, prediction_size, batch_size):
    num_samples = len(features) - window_size - prediction_size + 1
    for i in range(0, num_samples, batch_size):
        X, y = [], []
        for j in range(i, min(i + batch_size, num_samples)):
            X.append(features[j:j + window_size])
            y.append(targets[j + window_size:j + window_size + prediction_size])
        yield np.array(X), np.array(y)

#%% Generator for batch data processing
def data_generator(years, months, window_size, prediction_size, batch_size):
    for year in years:
        for month in months:
            try:
                # Load data for the current month
                features, targets, feature_dim, speed_scaler = load_month_data(year, month)

                # Generate sliding window batches
                for X_batch, y_batch in sliding_window_generator(features, targets, window_size, prediction_size, batch_size):
                    yield X_batch, y_batch, feature_dim, speed_scaler
            except FileNotFoundError:
                print(f"Skipping missing file for {year}-{month:02d}")

#%% Initialize parameters
window_size = 31 * 15  # Past 2 weeks (hourly data)
prediction_size = 3 * 15  # Predict the next 3 days (hourly data)
batch_size = 32

#%% Determine feature_dim dynamically
train_years = [2023, 2024]
train_months = list(range(1, 11))  # Use months 1-10 for training

# Fetch the first batch to determine feature_dim
data_iter = data_generator(train_years, train_months, window_size, prediction_size, batch_size)
X_sample, y_sample, feature_dim, speed_scaler = next(data_iter)

#%% Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(256, activation='tanh', return_sequences=True, input_shape=(window_size, feature_dim)),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(64, activation='tanh'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, activation='tanh'),
    tf.keras.layers.Dense(prediction_size),
    tf.keras.layers.Reshape((prediction_size, 1))
])

# Configure optimizer with fixed learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, clipnorm=1.0)
model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=1.0))

#%% Prepare data generators
val_months = [11]  # Use month 11 for validation
test_months = [12]  # Use month 12 for testing

train_gen = ((X, y) for X, y, _, _ in data_generator(train_years, train_months, window_size, prediction_size, batch_size))
val_gen = ((X, y) for X, y, _, _ in data_generator([2023], val_months, window_size, prediction_size, batch_size))

#%% Train the model
# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_gen,
    steps_per_epoch=100,
    validation_data=val_gen,
    validation_steps=20,
    epochs=20,
    callbacks=[early_stopping]
)

#%% Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%% Predict and evaluate results on test data
test_gen = ((X, y, speed_scaler) for X, y, _, speed_scaler in data_generator([2023], test_months, window_size, prediction_size, batch_size))

mse_speed_list = []
for X_batch, y_batch, speed_scaler in test_gen:
    y_pred_batch = model.predict(X_batch)

    # Flatten and inverse transform
    y_pred_batch_2d = y_pred_batch.reshape(-1, 1)
    y_true_batch_2d = y_batch.reshape(-1, 1)

    y_pred_batch_2d = speed_scaler.inverse_transform(y_pred_batch_2d)
    y_true_batch_2d = speed_scaler.inverse_transform(y_true_batch_2d)

    # Reshape back to original shape
    y_pred_batch = y_pred_batch_2d.reshape(y_pred_batch.shape)
    y_true_batch = y_true_batch_2d.reshape(y_batch.shape)

    mse_speed = mean_squared_error(y_true_batch.flatten(), y_pred_batch.flatten())
    mse_speed_list.append(mse_speed)

print(f"Mean Squared Error for Average Speed: {np.mean(mse_speed_list):.4f}")

#%% Plot results with enhanced visualization
plt.figure(figsize=(14, 8))

# Define range, select a smaller region
zoom_start, zoom_end = 200, 300

# Average Speed
plt.subplot(2, 1, 1)
plt.plot(y_true_batch.flatten()[zoom_start:zoom_end], label='True Average Speed', color='blue', alpha=0.7)
plt.plot(y_pred_batch.flatten()[zoom_start:zoom_end], label='Predicted Average Speed', color='orange', alpha=0.7)
plt.title("Average Speed Prediction (Zoomed In)")
plt.xlabel("Time (Zoomed Range)")
plt.ylabel("Speed")
plt.legend()

plt.tight_layout()
plt.show()
