# %%

from darts.datasets import ETTh1Dataset
from darts.models import NLinearModel
from darts.metrics.metrics import mae, mse
import numpy as np
import pandas as pd
import torch
import random
import csv
import datetime
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_timeline

import models
import models_3d_concepts_on_time
import models_3d_atomics_on_variate_to_concepts
from preprocess_helpers import *
from helper import *
from param_initializations import *
from optimization_strategy import greedy_selection

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
device


# %%
series = ETTh1Dataset().load()


# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, T, window_stride=1, pred_len=1):
        self.data = data
        self.targets = targets
        assert targets.size(0) == data.size(0)
        self.T = T # time window
        self.window_stride = window_stride
        self.pred_len = pred_len
        self.N, self.V = data.shape

    def __len__(self):
        return len(range(0, self.N - self.T - self.pred_len + 1, self.window_stride))

    def __getitem__(self, idx):
        start = idx * self.window_stride
        end = start + self.T

        X = self.data[start:end]
        # if mode == "S": # predict only target
        y = self.targets[end:end + self.pred_len].flatten()
        # elif mode == "MS": # predict all variables
        #   y = self.data[end:end + self.pred_len, :7].flatten()
        return X, y


# %%
def preprocess_data(series, seq_len, window_stride=1, pred_len=1, batch_size = 512):
    scaler = StandardScaler()
    
    train, test = series.split_before(0.6)
    val, test = test.split_before(0.5)
    
    print("Train/Val/Test", len(train), len(val), len(test))
    
    train_og = train.pd_dataframe()
    train = scaler.fit_transform(train_og)
    train = pd.DataFrame(train, columns=train_og.columns)
    X_train = train
    y_train = train[["OT"]]
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_train)
    X_train = torch.cat([X_train, indicators], axis=1)
    
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len, window_stride, pred_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val_og = val.pd_dataframe()
    val = scaler.transform(val_og)
    val = pd.DataFrame(val, columns=val_og.columns)
    X_val = val
    y_val = val[["OT"]]
    X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_val)
    X_val = torch.cat([X_val, indicators], axis=1)
    
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len, window_stride, pred_len)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test_og = test.pd_dataframe()
    test = scaler.transform(test_og)
    test = pd.DataFrame(test, columns=test_og.columns)
    X_test = test
    y_test = test[["OT"]]
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_test)
    X_test = torch.cat([X_test, indicators], axis=1)
    
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len, window_stride, pred_len)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, scaler


# %%
random_seed = 1
set_seed(random_seed)


# %%
seq_len = 336
pred_len = 96
n_atomics_list = list(range(2,11,2))
n_concepts_list = list(range(2,11,2))
changing_dim = len(series.columns)
input_dim = 2 * changing_dim


# %%
def initializeModel_with_atomics(n_atomics, n_concepts, input_dim, changing_dim, seq_len, output_dim, use_summaries_for_atomics, top_k=''):
    model = models_3d_atomics_on_variate_to_concepts.CBM(input_dim = input_dim, 
                            changing_dim = changing_dim, 
                            seq_len = seq_len,
                            num_concepts = n_concepts,
                            num_atomics = n_atomics,
                            use_summaries_for_atomics = use_summaries_for_atomics,
                            opt_lr = 3e-3, # trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                            opt_weight_decay = 1e-05, # trial.suggest_float("wd", 1e-5, 1e-1, log=True),
                            l1_lambda=0.001,
                            cos_sim_lambda=0.01,
                            output_dim = output_dim,
                            top_k=top_k,
                            task_type=models_3d_atomics_on_variate_to_concepts.TaskType.REGRESSION,
                            )
    model = model.to(device)
    return model


def get_activation(arg):
    switch = {
        'sigmoid': torch.nn.Sigmoid(),
        'relu': torch.nn.ReLU(),
    }
    return switch.get(arg, None)


# %%
def objective(trial):
    n_atomics = trial.suggest_int("n_atomics", 1, 100)
    n_concepts = trial.suggest_int("n_concepts", 1, 100)
    use_summaries_for_atomics = trial.suggest_categorical('use_summaries_for_atomics', [False, True])
    
    model = initializeModel_with_atomics(n_atomics, n_concepts, input_dim, changing_dim, seq_len, output_dim=pred_len, use_summaries_for_atomics=use_summaries_for_atomics)

    model.atomic_activation_func = get_activation(trial.suggest_categorical("atomic_activation_func", ["sigmoid", "relu"]))
    model.concept_activation_func = get_activation(trial.suggest_categorical("concept_activation_func", ["sigmoid", "relu"]))
    
    # Get the FashionMNIST dataset.
    train_loader, val_loader, test_loader, scaler = preprocess_data(series, seq_len, pred_len=pred_len)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=model.optimizer, patience=5)
    val_loss = model.fit(train_loader, val_loader, None, 
              save_model_path=None, 
              max_epochs=5000,
              scheduler=scheduler,
              trial=trial,
              )
    
    del model, train_loader, val_loader, test_loader, scaler
    gc.collect()
    torch.cuda.empty_cache()

    return val_loss


# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_jobs=10, n_trials=100, timeout=60 * 60 * 12)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

# Retrieve top k trials
k = 50
print("Top k trials:", k)
top_trials = sorted(study.trials, key=lambda trial: trial.value)[:k]

for i, trial in enumerate(top_trials, 1):
    print(f"{i}: Value = {trial.value}, Params = {trial.params}")

# Plots
fig = plot_optimization_history(study)
fig.write_image("plot_optimization_history.png") 
fig = plot_param_importances(study)
fig.write_image("plot_param_importances.png") 
fig = plot_timeline(study)
fig.write_image("plot_timeline.png") 

# nohup python hyperparameter_optim.py &> hyp.log.out &
