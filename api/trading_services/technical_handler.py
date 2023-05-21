# from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import pytz
import datetime
from xgboost import plot_importance, XGBClassifier, XGBRegressor
import warnings
import time
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.decomposition import FastICA
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
# import shap
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from pandas import read_csv, set_option
from matplotlib import cm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
import pandas as pd
import seaborn as sns


def ma(df, n):
    return pd.Series(df['price'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))

# exponentially weighted moving average


def ema(df, n):
    return pd.Series(df['price'].ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))

# Calculation of price momentum


def mom(df, n):
    return pd.Series(df.diff(n), name='Momentum_' + str(n))

# rate of change


def roc(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    return pd.Series(((M / N) * 100), name='ROC_' + str(n))

# relative strength index


def rsi(df, period):
    delta = df.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    # first value is sum of avg gains
    u[u.index[period-1]] = np.mean(u[:period])
    u = u.drop(u.index[:(period-1)])
    # first value is sum of avg losses
    d[d.index[period-1]] = np.mean(d[:period])
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / \
        d.ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + rs)

# stochastic oscillators slow & fast


def sto(close, low, high, n, id):
    stok = ((close - low.rolling(n).min()) /
            (high.rolling(n).max() - low.rolling(n).min())) * 100
    if (id == 0):
        return stok
    else:
        return stok.rolling(3).mean()


def tech_indi(ldf, tr_id=True):
    ''' Moving Average '''
    ldf['MA21'] = ma(ldf, 10)
    ldf['MA63'] = ma(ldf, 30)
    ldf['MA252'] = ma(ldf, 200)
    lst_MA = ['MA21', 'MA63', 'MA252']

    ''' Exponentially Weighted Moving Average '''
    ldf['EMA10'] = ema(ldf, 10)
    ldf['EMA30'] = ema(ldf, 30)
    ldf['EMA200'] = ema(ldf, 200)
    lst_EMA = ['EMA10', 'EMA30', 'EMA200']

    ''' Momentum '''
    ldf['MOM10'] = mom(ldf['price'], 10)
    ldf['MOM30'] = mom(ldf['price'], 30)
    lst_MOM = ['MOM10', 'MOM30']

    ''' Relative Strength Index '''
    ldf['RSI10'] = rsi(ldf['price'], 10)
    ldf['RSI30'] = rsi(ldf['price'], 30)
    ldf['RSI200'] = rsi(ldf['price'], 200)
    lst_RSI = ['RSI10', 'RSI30', 'RSI200']

    ''' Slow Stochastic Oscillators '''
    ldf['%K10'] = sto(ldf['price'], ldf['price'], ldf['price'], 5, 0)
    ldf['%K30'] = sto(ldf['price'], ldf['price'], ldf['price'], 10, 0)
    ldf['%K200'] = sto(ldf['price'], ldf['price'], ldf['price'], 20, 0)
    lst_pK = ['%K10', '%K30', '%K200']

    ''' Fast Stochastic Oscillators '''
    ldf['%D10'] = sto(ldf['price'], ldf['price'], ldf['price'], 10, 1)
    ldf['%D30'] = sto(ldf['price'], ldf['price'], ldf['price'], 30, 1)
    ldf['%D200'] = sto(ldf['price'], ldf['price'], ldf['price'], 200, 1)
    lst_pD = ['%D10', '%D30', '%D200']

    # Plot Training Data
    if (tr_id):
        plot_line(ldf.loc[plot_period, lst_MA], lst_MA,
                  title='Moving Average (window=21,63,252)')
        plot_line(ldf.loc[plot_period, lst_EMA], lst_EMA,
                  title='Exponential Moving Average (window=10,30,200)')
        plot_line(ldf.loc[plot_period, lst_MOM], lst_MOM, title='Momentum')
        plot_line(ldf.loc[plot_period, lst_RSI], lst_RSI,
                  title='Relative Strength Index')
        plot_line(ldf.loc[plot_period, lst_pK], lst_pK,
                  title='Stochastic Oscillators (slow)')
        plot_line(ldf.loc[plot_period, lst_pD], lst_pD,
                  title='Stochastic Oscillators (Fast)')


def plot_line(ldf, lst, title='', sec_id=None, size=[350, 1000]):

    # sec_id - list of [False,False,True] values of when to activate supblots; same length as lst

    if (sec_id != None):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()

    if (len(lst) != 1):
        ii = -1
        for i in lst:
            ii += 1
            if (sec_id != None):
                fig.add_trace(go.Scatter(x=ldf.index, y=ldf[lst[ii]], mode='lines', name=lst[ii], line=dict(
                    width=2.0)), secondary_y=sec_id[ii])
            else:
                fig.add_trace(go.Scatter(
                    x=ldf.index, y=ldf[lst[ii]], mode='lines', name=lst[ii], line=dict(width=2.0)))
    else:
        fig.add_trace(go.Scatter(
            x=ldf.index, y=ldf[lst[0]], mode='lines', name=lst[0], line=dict(width=2.0)))

    fig.update_layout(height=size[0], width=size[1], template='plotly_white', title=title,
                      margin=dict(l=50, r=80, t=50, b=40))
    fig.show()


plot_period = slice('2014-7-7 0:00', '2023-4-4 8:00')
