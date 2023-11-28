# %%

from darts.datasets import ETTh1Dataset
import numpy as np
import pandas as pd
import torch
import random
import csv
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.autograd import Variable
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from sklearn.preprocessing import StandardScaler

import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_timeline

from models import LogisticRegressionWithSummariesAndBottleneck_Wrapper, TaskType
from preprocess_helpers import *
from helper import *
from param_initializations import *
from optimization_strategy import greedy_selection

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
device

# %%
series = ETTh1Dataset().load()

print(series.start_time())
print(series.end_time())


# %%
class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, T, window_stride=1, target_steps_ahead=1):
        self.data = data
        self.targets = targets
        assert targets.size(0) == data.size(0)
        self.T = T # time window
        self.window_stride = window_stride
        self.target_steps_ahead = target_steps_ahead
        self.N, self.V = data.shape

    def __len__(self):
        return len(range(0, self.N - self.T - self.target_steps_ahead + 1, self.window_stride))

    def __getitem__(self, idx):
        start = idx * self.window_stride
        end = start + self.T

        X = self.data[start:end]
        y = self.targets[end:end + self.target_steps_ahead].squeeze(-1)
        return X, y


# %%
def preprocess_data(series, time_len, window_stride=1, target_steps_ahead=1, batch_size = 1024, scaler = StandardScaler(with_std=False)):
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
    
    train_dataset = TimeSeriesDataset(X_train, y_train, time_len, window_stride, target_steps_ahead)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True)

    val_og = val.pd_dataframe()
    val = scaler.transform(val_og)
    val = pd.DataFrame(val, columns=val_og.columns)
    X_val = val
    y_val = val[["OT"]]
    X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_val)
    X_val = torch.cat([X_val, indicators], axis=1)
    
    val_dataset = TimeSeriesDataset(X_val, y_val, time_len, window_stride, target_steps_ahead)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True)

    test_og = test.pd_dataframe()
    test = scaler.transform(test_og)
    test = pd.DataFrame(test, columns=test_og.columns)
    X_test = test
    y_test = test[["OT"]]
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_test)
    X_test = torch.cat([X_test, indicators], axis=1)
    
    test_dataset = TimeSeriesDataset(X_test, y_test, time_len, window_stride, target_steps_ahead)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, scaler


# %%
time_len = 10
train_loader, val_loader, test_loader, scaler = preprocess_data(series, time_len, target_steps_ahead=24)

for X,y in train_loader:
    print(X.shape)
    print(y.shape)
    break

print("Batches", len(train_loader), len(val_loader), len(test_loader))

# %%
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, color="black", label="Train")
    plt.plot(val_losses, color="green", label="Val")
    plt.legend()
    plt.show()

# %% [markdown]
# ## Regression

# %%
time_len = 96
target_steps_ahead = 96


# %%
experiment_folder = f"/workdir/optimal-summaries-public/vasopressor/models/etth1/forecasting-L{time_len}-T{target_steps_ahead}/"
model_path = experiment_folder + "forecasting_c{}.pt"
random_seed = 1

if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)

# %%

mae_metric = MeanAbsoluteError().to(device)
mse_metric = MeanSquaredError().to(device)


# %%
def initializeModel(trial, n_concepts):
#     vals_to_init = init_cutoffs_randomly(changing_dim * 9)
    logregbottleneck = LogisticRegressionWithSummariesAndBottleneck_Wrapper(input_dim = input_dim, 
                                                changing_dim = changing_dim,
                                                num_concepts = n_concepts,
                                                time_len = time_len,
                                                opt_lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                                                opt_weight_decay = trial.suggest_float("wd", 1e-5, 1e-1, log=True),
                                                l1_lambda = 0.001, #trial.suggest_float("l1", 1e-5, 1e-1, log=True),
                                                cos_sim_lambda = 0.01, #trial.suggest_float("cossim", 1e-5, 1e-1, log=True),
                                                output_dim = target_steps_ahead,
                                                task_type = TaskType.REGRESSION,
                                                )
    logregbottleneck.cuda()
    return logregbottleneck


# %%
def objective(trial):
    train_loader, val_loader, test_loader, scaler = preprocess_data(series, time_len, target_steps_ahead=target_steps_ahead)
    model = initializeModel(trial, n_concepts = 18)
    model.fit(train_loader, val_loader, None, None, 2000, trial=trial)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (Xb, yb) in enumerate(test_loader):
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model.forward(Xb)
            
            # mae = mae_metric(preds, yb).item()
            mse = mse_metric(preds, yb).item()
        # mae = mae_metric.compute().item()
        mse = mse_metric.compute().item()
        # mae_metric.reset()
        mse_metric.reset()
    
    return mse



# %%
set_seed(random_seed)

changing_dim = len(series.columns)
input_dim = 2 * changing_dim

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_jobs=1, n_trials=100, timeout=60 * 60 * 18)

fig = plot_optimization_history(study)
fig.write_image("plot_optimization_history.png") 
fig = plot_param_importances(study)
fig.write_image("plot_param_importances.png") 
fig = plot_timeline(study)
fig.write_image("plot_timeline.png") 

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

