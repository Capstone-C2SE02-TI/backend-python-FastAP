# from IPython.display import clear_output
from api.trading_services.trading_straregy_final_version import process_test_data
import json
import pprint
from sklearn.metrics import confusion_matrix
import pytz
import datetime
from xgboost import XGBClassifier
import warnings

# import shap
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv, set_option
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import os
import numpy as np
import pandas as pd
import seaborn as sns
from api.trading_services.technical_handler import *
from api.trading_services.dimRed import *
import pickle
from fastapi import FastAPI
app = FastAPI()


modelDir = 'models'


class Trader:
    def __init__(self, money, buy_amount):
        self.money = money
        self.baseMoney = money
        self.amount = 0
        self.buy_amount = buy_amount
        self.history = [0]
        self.rawMoney = [0]

    def calcResult(self, price):
        before = self.baseMoney
        after = self.money + self.amount*price

        print( max(self.history), after / before * 100)
        return {
            "maxAsset": max(self.history),
            "history": self.history[1:],
            "rawMoney": self.rawMoney[1:],
            # "After" : self.money + self.amount*price,
            "profit": after / before * 100
        }

    def buy(self, price):
        if self.money < self.buy_amount * price:
            self.history.append(self.money + self.amount*price)
            self.rawMoney.append(self.money)
            return False
        
        self.money -= self.buy_amount * price
        self.amount += self.buy_amount

        self.history.append(self.money + self.amount*price)
        self.rawMoney.append(self.money)
#         print("buy",self.money,self.amount)
        return True

    def sell(self, price):

        self.money += self.amount*price
        self.amount = 0
        self.history.append(self.money + self.amount*price)
        self.rawMoney.append(self.money)
        return True
    
    # def get_sell_amount(self):
        
    def stand(self, price):
        self.history.append(self.money + self.amount*price)
        self.rawMoney.append(self.money)


class prediction_service:
    def __init__(self):
        self.models = self.config_models(modelDir)

    def config_models(self, modelDir):

        models = []
        for model in os.listdir(modelDir):

            modelPath = os.path.join(modelDir, model)

            if not modelPath.endswith('.pkl'):
                continue
            with open(modelPath, 'rb') as f:
                clf2 = pickle.load(f)
                models.append(clf2)
                print(clf2)

        return models

    def plot_result(self, data_predict):
        preds = data_predict.to_dict('records')
        timestamps = []
        prices = []
        signals = []
        histories = []
        colors = []
        rawMoneys = []
        # Extract the timestamp, price, and signal values from the data
        for item in preds:
            timestamp = item['timestamp']
            price = item["price"]
            signal = item["signal"]
            history = item["history"]
            rawMoney = item["rawMoney"]
            timestamps.append(int(timestamp))
            prices.append(price)
            signals.append(signal)
            histories.append(history)
            rawMoneys.append(rawMoney)
            if signal == 0:
                colors.append('green')
            elif signal == 1:
                colors.append('red')
            else:
                colors.append('none')

        # Plotting the data
        plt.figure(figsize=(15, 5))  # Set the figure size to make the plot bigger
        plt.plot(timestamps, prices, color='blue', linewidth=0.2)
        # plt.scatter(timestamps, signals, marker='o', color='red')
        plt.plot(timestamps, histories, color='black', linewidth=0.25)
        plt.plot(timestamps, rawMoneys, color='pink', linewidth=0.25)
        plt.scatter(timestamps, prices, c=colors, cmap='cool', linewidths=0.2)
        plt.xlabel('Timestamp')
        plt.ylabel('Price')
        plt.title('Bitcoin Price and Signals')
        plt.show()
    def get_predict(self, data_source):

        data_source = self.pre_process_data_source(data_source)
        maxProfit = 0
        best_data = None
        for model in self.models:
            data_predict = self.process_data_source(data_source, model)

            trader = Trader(10000, 0.05)
            print(data_predict['signal'].value_counts().to_dict())
            
            for index, row in data_predict.iterrows():
                rsignal = row['signal']
                rprice = row['price']

                latestPrice = rprice
                if rsignal == 2:
                    trader.stand(rprice)
                    continue

                if rsignal == 0:
                    trader.buy(rprice)
                else:
                    trader.sell(rprice)

            data = trader.calcResult(latestPrice)
            data_predict['history'] = data['history']
            data_predict['rawMoney'] = data['rawMoney']
            
            if maxProfit < data['profit']:
                maxProfit = data['profit']
                best_data = data_predict

            # self.plot_result(data_predict)
            

        return best_data

    def pre_process_data_source(self, data_source):

        sliceIndex = 0
        timestamp = [int(t) for t in data_source.keys()][-sliceIndex:]
        price = list(data_source.values())[-sliceIndex:]

        df = pd.DataFrame({'timestamp': timestamp, 'price': price})

        return df


    def process_data_source(self, data_source, model):
        df = process_test_data(data_source, model)

        return df


predict_service = prediction_service()
# a = prediction_service()
# with open('./BTC.json') as data:
#     data_source = json.load(data)
#     data_source = data_source['data']['prices']['year']

# preds = a.get_predict(data_source)

# trader1 = Trader(1000000, 0.1)
# latestPrice = 0
# for index, row in preds.iterrows():

#     rsignal = row['signal']
#     rprice = row['price']

#     latestPrice = rprice
#     if rsignal == 2:
#         continue

#     if rsignal == 0:
#         trader1.buy(rprice)
#     else:
#         trader1.sell(rprice)

# data = trader1.calcResult(latestPrice)

# print(data)
