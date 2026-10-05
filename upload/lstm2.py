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
    file_path = f"merged\\{year}_{month}_merged.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, encoding='latin1')
    data.columns = [
        "Year", "Month", "Day", "Hour", "Minute", "Route", "Vehicle_Type", "Average_Speed", "Traffic_Volume", "Rainfall"
    ]
    data['datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    data = data.sort_values(by='datetime')

    # One-hot encode categorical features
    encoder_route = OneHotEncoder(sparse_output=False)
    route_encoded = encoder_route.fit_transform(data[['Route']])

    encoder_vehicle = OneHotEncoder(sparse_output=False)
    vehicle_type_encoded = encoder_vehicle.fit_transform(data[['Vehicle_Type']])

    # Combine all features
    numerical_features = data[['Average_Speed', 'Traffic_Volume', 'Rainfall']].values
    scaler = StandardScaler()
    numerical_features = scaler.fit_transform(numerical_features)

    features = np.hstack([
        route_encoded,
        vehicle_type_encoded,
        numerical_features
    ])

    # Define target variables
    targets = data[['Average_Speed', 'Traffic_Volume']].values

    return features, targets, features.shape[1]

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
                features, targets, feature_dim = load_month_data(year, month)

                # Generate sliding window batches
                for X_batch, y_batch in sliding_window_generator(features, targets, window_size, prediction_size, batch_size):
                    yield X_batch, y_batch, feature_dim
            except FileNotFoundError:
                print(f"Skipping missing file for {year}-{month:02d}")

#%% Initialize parameters
window_size = 14 * 24  # Past 2 weeks (hourly data)
prediction_size = 3 * 24  # Predict the next 3 days (hourly data)
batch_size = 32

#%% Determine feature_dim dynamically
train_years = [2022, 2023]
train_months = list(range(1, 11))  # Use months 1-10 for training

# Fetch the first batch to determine feature_dim
data_iter = data_generator(train_years, train_months, window_size, prediction_size, batch_size)
X_sample, y_sample, feature_dim = next(data_iter)

#%% Build Bidirectional LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, activation='tanh', return_sequences=True),
        input_shape=(window_size, feature_dim)
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='tanh')),
    tf.keras.layers.Dense(prediction_size * 2),
    tf.keras.layers.Reshape((prediction_size, 2))
])

# 調整學習率和梯度裁剪
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='mse')

#%% Prepare data generators
val_months = [11]  # Use month 11 for validation
test_months = [12]  # Use month 12 for testing

train_gen = ((X, y) for X, y, _ in data_generator(train_years, train_months, window_size, prediction_size, batch_size))
val_gen = ((X, y) for X, y, _ in data_generator([2023], val_months, window_size, prediction_size, batch_size))

#%% Train the model
# 訓練過程中加入學習率調整
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_gen,
    steps_per_epoch=50,
    validation_data=val_gen,
    validation_steps=10,
    epochs=20,
    callbacks=[lr_schedule, early_stopping]
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
test_gen = ((X, y) for X, y, _ in data_generator([2023], test_months, window_size, prediction_size, batch_size))

mse_speed_list, mse_volume_list = [], []
for X_batch, y_batch in test_gen:
    y_pred_batch = model.predict(X_batch)
    mse_speed = mean_squared_error(y_batch[:, :, 0].flatten(), y_pred_batch[:, :, 0].flatten())
    mse_volume = mean_squared_error(y_batch[:, :, 1].flatten(), y_pred_batch[:, :, 1].flatten())
    mse_speed_list.append(mse_speed)
    mse_volume_list.append(mse_volume)

print(f"Mean Squared Error for Average Speed: {np.mean(mse_speed_list):.4f}")
print(f"Mean Squared Error for Traffic Volume: {np.mean(mse_volume_list):.4f}")

#%% Plot results with enhanced visualization
plt.figure(figsize=(14, 8))

# 定義範圍，選擇一小段區域
zoom_start, zoom_end = 200, 300

# Average Speed
plt.subplot(2, 1, 1)
plt.plot(y_batch[zoom_start:zoom_end, :, 0].flatten() * 100, label='True Average Speed', color='blue', alpha=0.7)
plt.plot(y_pred_batch[zoom_start:zoom_end, :, 0].flatten() * 100, label='Predicted Average Speed', color='orange', alpha=0.7)
plt.title("Average Speed Prediction (Scaled x100, Zoomed In)")
plt.xlabel("Time (Zoomed Range)")
plt.ylabel("Speed (scaled)")
plt.legend()

# Traffic Volume
plt.subplot(2, 1, 2)
plt.plot(y_batch[zoom_start:zoom_end, :, 1].flatten() * 100, label='True Traffic Volume', color='green', alpha=0.7)
plt.plot(y_pred_batch[zoom_start:zoom_end, :, 1].flatten() * 100, label='Predicted Traffic Volume', color='red', alpha=0.7)
plt.title("Traffic Volume Prediction (Scaled x100, Zoomed In)")
plt.xlabel("Time (Zoomed Range)")
plt.ylabel("Volume (scaled)")
plt.legend()

plt.tight_layout()
plt.show()
