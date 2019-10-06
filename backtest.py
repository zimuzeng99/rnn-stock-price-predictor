from zipline.api import order_target_percent, record, symbol, schedule_function, date_rules, time_rules

import numpy as np
import pandas as pd

import joblib
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def initialize(context):
    context.regressor = load_model("regressor_google.h5")
    context.sc = joblib.load("scaler_google.pkl")
    context.asset = symbol('F')
    context.position = None
    schedule_function(predict_price, date_rules.every_day(), time_rules.market_close(minutes=30))

def predict_price(context, data):
    print(data.current(context.asset, "price"))
    prices = data.history(context.asset, 'close', bar_count=10, frequency="1d")
    print(prices)
    prices = prices.values
    prices = np.reshape(prices, (prices.shape[0], prices.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    if predicted_stock_price > data.current(context.asset, "price") and context.position != "long":
        order_target_percent(context.asset, 1)
        context.position = "long"
    elif predicted_stock_price < data.current(context.asset, "price") and context.position != "short":
        order_target_percent(context.asset, -1)
        context.position = "short"
