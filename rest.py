#!flask/bin/python
from flask import Flask, request

app = Flask(__name__)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from pandas_datareader import data
# Importing the training set
dataset_train = data.DataReader("GOOG", "yahoo", "2013-08-01", "2018-08-01")
training_set = dataset_train[["Adj Close"]].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

num_days = 10

# Creating a data structure with 10 timesteps and 1 output
X_train = []
y_train = []
for i in range(num_days, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-num_days:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
#regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
#regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 500, batch_size = 128)

@app.route('/')
def index():

    prices = request.args.get("prices").split(",")
    prices = list(map(float, prices))
    X = []
    X.append(prices)
    X = np.array(X)

    X = sc.transform(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    predicted_stock_price = regressor.predict(X)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    return str(predicted_stock_price[0, 0])
        

if __name__ == '__main__':
    app.run(debug=False)
