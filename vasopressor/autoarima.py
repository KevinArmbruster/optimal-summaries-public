# %%
from darts.datasets import ETTh1Dataset
from darts.models import AutoARIMA, ARIMA
from darts.metrics.metrics import mae, mse
from darts.dataprocessing.transformers import Scaler
import numpy as np
import pandas as pd
import torch
import random
import csv
import datetime
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_timeline

from models import CBM, TaskType
from preprocess_helpers import *
from helper import *
from param_initializations import *
from optimization_strategy import greedy_selection

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
device

# %%
series = ETTh1Dataset().load()


# %%
train_series, test_series = series.split_before(0.6)
val_series, test_series = test_series.split_before(0.5)


# %%
scaler = StandardScaler()
scaler_wrapper = Scaler(scaler)
train_series = scaler_wrapper.fit_transform(train_series)
val_series = scaler_wrapper.transform(val_series)
test_series = scaler_wrapper.transform(test_series)


# %%
def preprocess_data():
    series = ETTh1Dataset().load()
    scaler = StandardScaler()
    
    train_series, test_series = series.split_before(0.6)
    val_series, test_series = test_series.split_before(0.5)
    
    scaler_wrapper = Scaler(scaler)
    train_series = scaler_wrapper.fit_transform(train_series)
    val_series = scaler_wrapper.transform(val_series)
    test_series = scaler_wrapper.transform(test_series)

    return train_series, val_series, test_series


# %%
batch_size = 32
learning_rate = 0.005
epochs = 10
seq_len = 336
pred_len = 96


# %%
data = series.pd_dataframe()
print ("\nMissing values :  ", data.isnull().any())
data.plot(subplots=True)

# %%
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

data = series.pd_dataframe()

for col in data.columns:
    plot_acf(data[col], lags=40, alpha=0.05)
    plt.title(f"ACF for {col}")
    plt.show()

    plot_pacf(data[col], lags=40, alpha=0.05)
    plt.title(f"PACF for {col}")
    plt.show()


# %%
train_series, val_series, test_series = preprocess_data()

model = AutoARIMA(#start_p=0, d=None, start_q=0, max_p=5, max_d=5, max_q=5, 
                  #start_P=0, D=None, start_Q=0, max_P=3, max_D=5, max_Q=3, 
                  max_order=None, m=24, # hourly, so 24 
                  stepwise=True, maxiter=100,
                  error_action='ignore', suppress_warnings=True, trace=True)
model.fit(series=train_series["OT"])#, future_covariates=train_series[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]])
model.save("./aa-without-covariats.pt")


# %%
train_series, val_series, test_series = preprocess_data()

model = AutoARIMA(#start_p=0, d=None, start_q=0, max_p=5, max_d=5, max_q=5, 
                  #start_P=0, D=None, start_Q=0, max_P=3, max_D=5, max_Q=3, 
                  max_order=None, m=24, # hourly, so 24 
                  stepwise=True, maxiter=100,
                  error_action='ignore', suppress_warnings=True, trace=True)
model.fit(series=train_series["OT"], future_covariates=train_series[["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]])
model.save("./aa-with-covariats.pt")

