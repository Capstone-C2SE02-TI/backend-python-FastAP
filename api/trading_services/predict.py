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

    def calcResult(self, price):

        before = self.baseMoney
#         print("Asset peak:", max(self.history))
        # print("Before", self.baseMoney)

        after = self.money + self.amount*price
#         print("After", self.money + self.amount*price)

#         print("profit ratio",100 - before / after * 100)

        return {
            "asset": max(self.history),
            # "Before": self.baseMoney,
            # "After" : self.money + self.amount*price,
            "profit": 100 - before / after * 100
        }

    def buy(self, price):
        if self.money < self.buy_amount * price:
            return False
        self.money -= self.buy_amount * price
        self.amount += self.buy_amount

        self.history.append(self.money + self.amount*price)
#         print("buy",self.money,self.amount)
        return True

    def sell(self, price):

        self.money += self.amount*price
        self.amount = 0

        return True


class prediction_service:
    def __init__(self):
        self.models = self.config_models(modelDir)
        print(self.models)

    def config_models(self, modelDir):

        models = []
        for model in os.listdir(modelDir):

            modelPath = os.path.join(modelDir, model)

            if not modelPath.endswith('.pkl'):
                continue
            with open(modelPath, 'rb') as f:
                clf2 = pickle.load(f)
                models.append(clf2)

        return models

    def get_predict(self, data_source):

        data_source = self.pre_process_data_source(data_source)
        data_predict = self.process_data_source(data_source, self.models[0])

        return data_predict

    def pre_process_data_source(self, data_source):
        timestamp = []
        price = []

        for t, p in data_source.items():
            timestamp.append(int(t))
            price.append(p)

        df = pd.DataFrame(list(zip(timestamp, price)),
                          columns=['timestamp', 'price'])

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
