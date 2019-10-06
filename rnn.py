# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from pandas_datareader import data
import joblib

# Importing the training set
dataset_train = data.DataReader("KO", "yahoo", "2013-08-01", "2018-08-01")
training_set = dataset_train[["Adj Close"]].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

joblib.dump(sc, "scaler_cocacola.pkl")

num_days = 20

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

regressor.save("regressor_cocacola.h5")

from keras.models import load_model

sc = joblib.load("scaler_cocacola.pkl")
regressor = load_model("regressor_cocacola.h5")

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = data.DataReader("KO", "yahoo", "2018-08-02", "2019-08-02")
real_stock_price = dataset_test[["Adj Close"]].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Adj Close'], dataset_test['Adj Close']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - num_days:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(num_days, len(inputs)):
    X_test.append(inputs[i-num_days:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

percent_error = np.mean(np.abs(real_stock_price - predicted_stock_price) / real_stock_price * 100)

correct = 0
for i in range(1, len(predicted_stock_price)):
    predicted_diff = predicted_stock_price[i, 0] - real_stock_price[i - 1, 0]
    true_diff = real_stock_price[i, 0] - real_stock_price[i - 1, 0]
    
    predicted_trend = "up" if predicted_diff > 0 else "down"
    true_trend = "up" if true_diff > 0 else "down"
    
    if predicted_trend == true_trend:
        correct += 1

accuracy = correct / (len(predicted_stock_price) - 1)
