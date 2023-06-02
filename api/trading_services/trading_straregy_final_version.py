# from IPython.display import clear_output
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

sns.set()


# for dirname, _, filenames in os.walk('/kaggle/'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')
# %matplotlib inline

# time series cross validation
# https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/

''' FUNCTIONS '''

# One plot type


# plot n verticle subplots

colours = ['tab:blue', 'tab:red', 'tab:green']


cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Split for TimeSeries


def dateparse(time_in_secs):
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs) // 1000))


def create_target(ldf, tr_id=False, rows=1, priceChange=1):
    # calculate percent change over next 10 rows
    ldf['pct_change'] = ldf['price'].pct_change(rows).shift(-rows) * 100
    
    ldf['signal'] = np.select(
        [ldf['pct_change'] > priceChange, ldf['pct_change'] < -priceChange],
        [0, 1],
        default=2
    )
    # if (tr_id is not True):
    #     print(ldf['signal'].value_counts())


models = []
# Lightweight Models
models.append(('LDA', LinearDiscriminantAnalysis()))  # Unsupervised Model
models.append(('KNN', KNeighborsClassifier()))  # Unsupervised Model
models.append(('TREE', DecisionTreeClassifier()))  # Supervised Model
models.append(('NB', GaussianNB()))  # Unsupervised Model

# More Advanced Models
models.append(('GBM', GradientBoostingClassifier(n_estimators=25)))
models.append(('XGB', XGBClassifier(n_estimators=25, eval_metric='logloss')))
models.append(('CAT', CatBoostClassifier(silent=True,
                                         n_estimators=25)))
models.append(('RF', RandomForestClassifier(n_estimators=25)))


def model_predict(index, data_source):

    feature = 'signal'
    y_train = data_source[feature]
    X_train = data_source.loc[:, data_source.columns != feature]

    res = models[index][1].fit(X_train, y_train)

    return res


def show_confusion_matrix(y_eval, preds):
    cm = confusion_matrix(y_eval, preds)

    # Create a heatmap to visualize the confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


# define a conversion function for the native timestamps in the csv file


def dateparse(time_in_secs):
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs) // 1000))


# Data Periods used in Notebook
# Selectio Plot Period for visualisation only
plot_period = slice('2014-7-7 0:00', '2023-4-4 8:00')
# Select Data Period for Analysis
data_period = slice('2014-7-7 0:00', '2023-4-4 8:00')

# Path to CSV
path = './price_data.csv'

# path = 'bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv'


def process_data(data_source, createSignal=True, rows=1, priceChange=1):
    data_source = data_source.dropna()

    if createSignal:
        create_target(data_source, False, rows, priceChange)

    tech_indi(data_source, False)
    data_source = data_source.dropna()

    # Already remove signal inside
    period_test_1_processed = dimRed(data_source,
                                     split_id=[0.2, None],
                                     model_id='fastica',
                                     n_comp=5,
                                     scaler_id=[True, 3], isSignal=createSignal, plot_id=False)

    if not createSignal:
        period_test_1_processed['price'] = data_source['price']
        period_test_1_processed['timestamp'] = data_source['timestamp']

    return period_test_1_processed


def train_flow(model_id, data_source):
    model = model_predict(model_id, data_source)
    feature = 'signal'
    y_train = data_source[feature]
    X_train = data_source.loc[:, data_source.columns != feature]

    y_pred = model.predict(X_train)

    return y_pred, y_train, model


def eval_result(y_pred, y_train):
    eval_res = accuracy_score(y_pred, y_train)
    print('Acc', eval_res)

    # show_confusion_matrix(y_train, y_pred)


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

#         print("sell",self.money, self.amount)


# for r in range(1,2, 1):
#   for priceChange in range(1, 2,1):
df = pd.read_csv(path, parse_dates=[0],
                 date_parser=dateparse,
                 index_col='timestamp')
# Select Data Period for Analysis
data_period = slice('2013-4-28 0:00', '2020-3-1 0:00')
period_train_1 = df[data_period]

# Select Data Period for Analysis
data_test_period = slice('2020-4-1 0:00', '2021-4-1 0:00')
period_test_1 = df[data_test_period]


def process_test_data(data_source, model):
    period_test_1_processed = process_data(
        data_source, createSignal=False)

    # print("test", period_test_1)
    # print("ptest", period_test_1_processed)
    price = period_test_1_processed['price']
    timestamp = period_test_1_processed['timestamp']
    period_test_1_processed = period_test_1_processed.loc[:,
                                                          period_test_1_processed.columns != 'price']
    period_test_1_processed = period_test_1_processed.loc[:,
                                                          period_test_1_processed.columns != 'timestamp']
    period_test_1_processed = period_test_1_processed.loc[:,
                                                          period_test_1_processed.columns != 'signal']

    preds_test = model.predict(period_test_1_processed)
    period_test_1_processed['price'] = price
    period_test_1_processed['signal'] = preds_test
    period_test_1_processed['timestamp'] = timestamp
    period_test_1_processed = period_test_1_processed.reset_index()

    return period_test_1_processed


def traintrain_flow(r: int, priceChange: int) -> list[object]:
    data_prices = []

    for i in range(len(models)):
        print([r, priceChange, i])

        # if count % 10 == 0:
        #     clear_output(wait=True)

        period_train_1_processed = process_data(
            period_train_1, True, rows=r, priceChange=priceChange)
        # print("train", period_train_1)
        # print("ptrain", period_train_1_processed)
        signalCount = period_train_1_processed['signal'].value_counts(
        ).to_dict()

        if signalCount.get(0, 0) <= 20 or signalCount.get(1, 0) <= 20:
            print(signalCount.get(0), signalCount.get(0), "Continue")
            continueTimes += 1
            continue
        y_pred, y_train, model = train_flow(i, period_train_1_processed)

        signalCount = y_train.value_counts().to_dict()

        if signalCount.get(0, 0) <= 20 or signalCount.get(1, 0) <= 20:
            print(signalCount.get(0), signalCount.get(0), "Continue")
            continueTimes += 1
            continue

        eval_result(y_pred, y_train)
        # print("rows", r)
        # print("price change", priceChange)

        period_test_1_processed = process_data(
            period_test_1, createSignal=False)

        # print("test", period_test_1)
        # print("ptest", period_test_1_processed)
        price = period_test_1_processed['price']
        period_test_1_processed = period_test_1_processed.loc[:,
                                                              period_test_1_processed.columns != 'price']
        period_test_1_processed = period_test_1_processed.loc[:,
                                                              period_test_1_processed.columns != 'signal']

        preds_test = model.predict(period_test_1_processed)
        period_test_1_processed['price'] = price
        period_test_1_processed['signal'] = preds_test
        period_test_1_processed = period_test_1_processed.reset_index()

        trader1 = Trader(1000000, 0.1)
        latestPrice = 0
        for index, row in period_test_1_processed.iterrows():

            rsignal = row['signal']
            rprice = row['price']

            latestPrice = rprice
            if rsignal == 2:
                continue

            if rsignal == 0:
                trader1.buy(rprice)
            else:
                trader1.sell(rprice)

        data = trader1.calcResult(latestPrice)
        data['row'] = r
        data['priceChange'] = priceChange
        data['model'] = i

        data_prices.append(data)

        if (data['profit'] >= 10 or data['asset'] >= 1100000) or i == 0:
            print(data)
        else:
            print("useless")

        print('---------------'*5)
    return data_prices


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(traintrain_flow(53, 37))
