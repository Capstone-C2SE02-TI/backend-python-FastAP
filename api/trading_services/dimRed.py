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


def scatterPlot(xDF, yDF, algoName):

    sns.set_style('whitegrid')
    fig, ax = plt.subplots()
    tempDF = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join="inner")
    tempDF.columns = ["Component 1", "Component 2", "Label"]
    g = sns.scatterplot(x="Component 1", y="Component 2", data=tempDF, hue="Label",
                        linewidth=0.5, alpha=0.5, s=15, edgecolor='k')
    plt.title(algoName)
    plt.legend()

    for i in ['top', 'right', 'bottom', 'left']:
        ax.spines[i].set_color('black')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='both', ls='--', alpha=0.9)
    plt.show()


def dimRed(ldf, feature='signal', split_id=[None, None], n_comp=5, plot_id=True,
           model_id='sparserandomprojection', scaler_id=[False, None], isSignal=True):

    # Given a dataframe, split feature/target variable
    X = ldf.copy()
    if isSignal:
        y = ldf[feature].copy()
        del X[feature]

    n_jobs = -1
    rs = 32

    if (model_id is 'pca'):
        whiten = False
        model = PCA(n_components=n_comp, whiten=whiten, random_state=rs)
    if (model_id is 'sparsepca'):
        alpha = 1
        model = SparsePCA(n_components=n_comp, alpha=alpha,
                          random_state=rs, n_jobs=n_jobs)
    elif (model_id is 'kernelpca'):
        kernel = 'rbf'
        gamma = None
        model = KernelPCA(n_components=n_comp, kernel=kernel,
                          gamma=gamma, n_jobs=n_jobs, random_state=rs)
    elif (model_id is 'incrementalpca'):
        batch_size = None
        model = IncrementalPCA(n_components=n_comp, batch_size=batch_size)
    elif (model_id is 'truncatedsvd'):
        algorithm = 'randomized'
        n_iter = 5
        model = TruncatedSVD(
            n_components=n_comp, algorithm=algorithm, n_iter=n_iter, random_state=rs)
    elif (model_id is 'gaussianrandomprojection'):
        eps = 0.5
        model = GaussianRandomProjection(
            n_components=n_comp, eps=eps, random_state=rs)
    elif (model_id is 'sparserandomprojection'):
        density = 'auto'
        eps = 0.5
        dense_output = True
        model = SparseRandomProjection(n_components=n_comp, density=density,
                                       eps=eps, dense_output=dense_output, random_state=rs)
    if (model_id is 'isomap'):
        n_neigh = 2
        model = Isomap(n_neighbors=n_neigh, n_components=n_comp, n_jobs=n_jobs)
    elif (model_id is 'mds'):
        n_init = 1
        max_iter = 50
        metric = False
        model = MDS(n_components=n_comp, n_init=n_init, max_iter=max_iter, metric=True,
                    n_jobs=n_jobs, random_state=rs)
    elif (model_id is 'locallylinearembedding'):
        n_neigh = 10
        method = 'modified'
        model = LocallyLinearEmbedding(n_neighbors=n_neigh, n_components=n_comp, method=method,
                                       random_state=rs, n_jobs=n_jobs)
    elif (model_id is 'tsne'):
        learning_rate = 300
        perplexity = 30
        early_exaggeration = 12
        init = 'random'
        model = TSNE(n_components=n_comp, learning_rate=learning_rate,
                     perplexity=perplexity, early_exaggeration=early_exaggeration,
                     init=init, random_state=rs)
    elif (model_id is 'minibatchdictionarylearning'):
        alpha = 1
        batch_size = 200
        n_iter = 25
        model = MiniBatchDictionaryLearning(n_components=n_comp, alpha=alpha,
                                            batch_size=batch_size, n_iter=n_iter, random_state=rs)
    elif (model_id is 'fastica'):
        algorithm = 'parallel'
        whiten = True
        max_iter = 100
        model = FastICA(n_components=n_comp, algorithm=algorithm, whiten=whiten,
                        max_iter=max_iter, random_state=rs)

    # Scaling
    if (scaler_id[0]):

        opts = [StandardScaler(), RobustScaler(), MinMaxScaler(),
                Normalizer(norm='l2')]
        scaler = opts[scaler_id[1]].fit(X)
        X_sca = pd.DataFrame(scaler.fit_transform(X),
                             columns=X.columns,
                             index=X.index)  # summarize transformed data

    # Unsupervised Dimension Reduction
    if (scaler_id[0]):
        X_red = model.fit_transform(X_sca)
    else:
        X_red = model.fit_transform(X)
    X_red = pd.DataFrame(data=X_red, index=X.index)

    if isSignal:
        if (plot_id):
            scatterPlot(X_red, y, model_id)
        X_red[feature] = y

    return X_red  # return new feature matrix
