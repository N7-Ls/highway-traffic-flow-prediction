# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 19:40:57 2024

@author: User
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

#%% Data loading
data = pd.read_csv('international-airline-passengers.csv')
data.rename(columns = {'Month':'Time' ,'International airline passengers: monthly totals in thousands. Jan 49 ? Dec 60':'Amount'}, inplace = True)

# Fix the error of time
data['Time'] = pd.to_datetime(data['Time'])
data['Time'] = data['Time'].apply(lambda x: x.replace(year=x.year - 100)) 

print(data.head())
data_to_use = data['Amount'].values

#%% Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_use.reshape(-1, 1))

#%% Set the x-axis
## Set the major formatter for the x-axis to display abbreviated month names
# plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))
## Set the major locator for the x-axis to locate ticks at the beginning of each month
# plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())

#%% Plot the original data
# plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
# plt.title('Monthly totals of passengers from 1949 to 1960')
# plt.xlabel('Month')
# plt.ylabel('Amount of passengers (thousands)')
# plt.plot(scaled_data, label='Data')
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()

#%% Slice window
def window_data(data, window_size):
    X = []
    y = []
    i = 0
    while (i + window_size) < len(data):
        X.append(data[i : i + window_size])
        y.append(data[i + window_size])     
        i += 1
    assert len(X) ==  len(y)
    return X, y

window_size = 6
X, y = window_data(scaled_data, window_size)

#%% Distribute train/test data
X_train = np.array(X[:114])
y_train = np.array(y[:114])
X_test = np.array(X[114:])
y_test = np.array(y[114:])

#%% Parameter setting
batch_size = 8
hidden_layer = 256
clip_margin = 4
learning_rate = 0.0015
epochs = 200

inputs = tf.placeholder(tf.float32, [batch_size, window_size, 1])
targets = tf.placeholder(tf.float32, [batch_size, 1])

#Weights for the input gate
weights_input_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev = 0.05))
weights_input_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
bias_input = tf.Variable(tf.zeros([hidden_layer]))

#weights for the forgot gate
weights_forget_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev = 0.05))
weights_forget_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
bias_forget = tf.Variable(tf.zeros([hidden_layer]))

#weights for the output gate
weights_output_gate = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev = 0.05))
weights_output_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
bias_output = tf.Variable(tf.zeros([hidden_layer]))

#weights for the memory cell
weights_memory_cell = tf.Variable(tf.truncated_normal([1, hidden_layer], stddev = 0.05))
weights_memory_cell_hidden = tf.Variable(tf.truncated_normal([hidden_layer, hidden_layer], stddev = 0.05))
bias_memory_cell = tf.Variable(tf.zeros([hidden_layer]))

# Output layer weigts
weights_output = tf.Variable(tf.truncated_normal([hidden_layer, 1], stddev = 0.05))
bias_output_layer = tf.Variable(tf.zeros([1]))

#%% Define LSTM model
def LSTM_cell(input, state, output):
    
    input_gate = tf.sigmoid(tf.matmul(input, weights_input_gate) + tf.matmul(output, weights_input_hidden) + bias_input)
    forget_gate = tf.sigmoid(tf.matmul(input, weights_forget_gate) + tf.matmul(output, weights_forget_hidden) + bias_forget)
    output_gate = tf.sigmoid(tf.matmul(input, weights_output_gate) + tf.matmul(output, weights_output_hidden) + bias_output)
    memory_cell = tf.tanh(tf.matmul(input, weights_memory_cell) + tf.matmul(output, weights_memory_cell_hidden) + bias_memory_cell)
    state = state * forget_gate + input_gate * memory_cell
    output = output_gate * tf.tanh(state)
    return state, output

#%% List the output
outputs = []
for i in range(batch_size):
    batch_state = np.zeros([1, hidden_layer], dtype = np.float32) 
    batch_output = np.zeros([1, hidden_layer], dtype = np.float32)
    for j in range(window_size):
        batch_state, batch_output = LSTM_cell(tf.reshape(inputs[i][j], (-1, 1)), batch_state, batch_output)
    outputs.append(tf.matmul(batch_output, weights_output) + bias_output_layer)
 
#%% Define loss  
losses = []
for i in range(len(outputs)):
    losses.append(tf.losses.mean_squared_error(tf.reshape(targets[i], (-1, 1)), outputs[i]))
    
loss = tf.reduce_mean(losses)

#%% Define optimizer with gradient clipping
gradients = tf.gradients(loss, tf.trainable_variables())
clipped, _ = tf.clip_by_global_norm(gradients, clip_margin)
optimizer = tf.train.AdamOptimizer(learning_rate)
trained_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))

#%% Train the network
session = tf.Session()
session.run(tf.global_variables_initializer())
for i in range(epochs):
    traind_scores = []
    ii = 0
    epoch_loss = []
    while(ii + batch_size) <= len(X_train):
        X_batch = X_train[ii:ii+batch_size]
        y_batch = y_train[ii:ii+batch_size]
        
        o, c, _ = session.run([outputs, loss, trained_optimizer], feed_dict={inputs:X_batch, targets:y_batch})
        
        epoch_loss.append(c)
        traind_scores.append(o)
        ii += batch_size
    if (i % 30) == 0:
        print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))

sup =[]
for i in range(len(traind_scores)):
    for j in range(len(traind_scores[i])):
        sup.append(traind_scores[i][j][0])

#%% Testing        
tests = []
i = 0
while i+batch_size <= len(X_test):
    
    o = session.run([outputs], feed_dict={inputs:X_test[i:i+batch_size]})
    i += batch_size
    tests.append(o)
    
tests_new = []
for i in range(len(tests)):
    for j in range(len(tests[i][0])):
        tests_new.append(tests[i][0][j])
        
test_results = []
for i in range(138):
    if i >= 114:
        test_results.append(tests_new[i-114])
    else:
        test_results.append(None)

test_results = np.asarray(test_results, dtype="object")
        
#%% Plot predictions
plt.figure(figsize=(16, 7))
plt.plot(scaled_data, label='Original data')
plt.plot(sup, label='Training data')
plt.plot(test_results, label='Testing data')
plt.legend()
plt.show()