# %%
import sys
sys.path.append('..')
from darts.datasets import ETTh1Dataset
from darts.models import NLinearModel
from darts.metrics.metrics import mae, mse
import numpy as np
import pandas as pd
import torch
import random
import csv
import os, subprocess, gc, time, datetime
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
from optuna.visualization import *

import models.models_original as models_original
import models.models_3d_atomics as new_models
from vasopressor.preprocess_helpers import *
from models.helper import *
from models.param_initializations import *
from models.optimization_strategy import greedy_forward_selection

gpu_id = int(subprocess.check_output('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs', shell=True, text=True))
device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available else torch.device('cpu')
print("current device", device)


# %%
def line(x, w=0.3, b=5):
    return w*x + b

time = np.linspace(0, 1000, 5000).reshape(-1,1)

trend_series = line(time)
trend_series = np.array(trend_series).reshape(-1,1)

def sine_wave(time, frequency = 1, amplitude = 1, phase = 0):
    return amplitude * np.sin(frequency * time + phase)

components_series = sine_wave(time, frequency = 1, amplitude = 5, phase = np.pi) \
                + sine_wave(time, frequency = 0.1, amplitude = 10, phase = 0) \
                + trend_series

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
    
    train_end = int(len(series) * 0.6)
    val_end = int(train_end + len(series) * 0.2)
    
    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]
    
    # train, test = series.split_before(0.6)
    # val, test = test.split_before(0.5)
    
    print("Train/Val/Test", len(train), len(val), len(test))
    
    train = scaler.fit_transform(train)
    X_train = pd.DataFrame(train)
    y_train = X_train
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_train)
    X_train = torch.cat([X_train, indicators], axis=1)
    
    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len, window_stride, pred_len)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

    val = scaler.transform(val)
    X_val = pd.DataFrame(val)
    y_val = X_val
    X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    
    indicators = torch.isfinite(X_val)
    X_val = torch.cat([X_val, indicators], axis=1)
    
    val_dataset = TimeSeriesDataset(X_val, y_val, seq_len, window_stride, pred_len)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4, pin_memory=True)

    test = scaler.transform(test)
    X_test = pd.DataFrame(test)
    y_test = X_test
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
changing_dim = 1 # len(series.columns)
input_dim = 2 * changing_dim


# %%

def initializeModel_with_atomics(trial, n_atomics, n_concepts, input_dim, changing_dim, seq_len, output_dim, use_summaries_for_atomics, use_indicators, top_k=''):
    model = new_models.CBM(input_dim = input_dim, 
                            changing_dim = changing_dim, 
                            seq_len = seq_len,
                            num_concepts = n_concepts,
                            num_atomics = n_atomics,
                            use_summaries_for_atomics = use_summaries_for_atomics,
                            use_indicators = use_indicators,
                            opt_lr = trial.suggest_float("opt_lr", 1e-5, 1e-2, log=True), # 1e-3, # 2e-4
                            opt_weight_decay = trial.suggest_float("opt_weight_decay", 1e-8, 1e-2, log=True), # 1e-5, # 1e-05,
                            l1_lambda = trial.suggest_float("l1_lambda", 1e-8, 1e-2, log=True), # 1e-5, # 0.001,
                            cos_sim_lambda = trial.suggest_float("cos_sim_lambda", 1e-5, 1e-2, log=True), # 1e-5, # 0.01,
                            output_dim = output_dim,
                            top_k=top_k,
                            task_type=new_models.TaskType.REGRESSION,
                            device=device
                            )
    model = model.to(device)
    return model


# %%
def objective(trial: TrialState):
    n_atomics = trial.suggest_int("n_atomics", 8, 1024, step=8)
    n_concepts = trial.suggest_int("n_concepts", 8, 1024, step=8)
    use_summaries_for_atomics = trial.suggest_categorical('use_summaries_for_atomics', [True, False])
    use_indicators = trial.suggest_categorical('use_indicators', [True, False])
    
    train_loader, val_loader, test_loader, scaler = preprocess_data(components_series, seq_len, pred_len=pred_len)
    
    model = initializeModel_with_atomics(trial, n_atomics, n_concepts, input_dim, changing_dim, seq_len, output_dim=pred_len, use_summaries_for_atomics=use_summaries_for_atomics, use_indicators=use_indicators)
    
    model.activation_func = torch.nn.ReLU() if trial.suggest_categorical('use_relu', [True, False]) else torch.nn.Sigmoid()
    scheduler = None # torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=model.optimizer, patience=10, factor=0.7)
    
    
    try:
        val_loss = model.fit(train_loader, val_loader, None, save_model_path=None, max_epochs=10000, scheduler=scheduler, patience=100, trial=trial)

        mse_metric = MeanSquaredError().to(device)
        model.eval()
        with torch.inference_mode():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model.forward(Xb)
                
                mse = mse_metric(preds, yb).item()
            mse = mse_metric.compute().item()
            mse_metric.reset()
        
        
        del model, train_loader, val_loader, test_loader, scaler
        gc.collect()
        torch.cuda.empty_cache()

        trial.set_user_attr('val_loss', val_loss)
        return mse
    
    except RuntimeError as e:
        
        del model, train_loader, val_loader, test_loader, scaler
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"RuntimeError occurred: {e}")
        return 1e6


# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_jobs=4, gc_after_trial=True, n_trials=500) #, timeout=60 * 60 * 10) #60 * 60 * 12)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

# Retrieve top k trials
# k = 200
# print("Top k trials:", k)
top_trials = sorted(study.trials, key=lambda trial: trial.value)

for i, trial in enumerate(top_trials, 1):
    print(f"{i}: Criteria = {trial.value}, User Attrs = {trial.user_attrs}, Params = {trial.params}")

time.sleep(10) # wait for printing

# Plots
fig = plot_optimization_history(study)
fig.write_image("plot_optimization_history.png") 
fig = plot_param_importances(study)
fig.write_image("plot_param_importances.png") 
fig = plot_timeline(study)
fig.write_image("plot_timeline.png") 
fig = plot_intermediate_values(study)
fig.write_image("plot_intermediate_values.png") 
fig = plot_parallel_coordinate(study, params=["n_atomics", "n_concepts"])
fig.write_image("plot_parallel_coordinate.png") 
fig = plot_contour(study)
fig.write_image("plot_contour.png") 
fig = plot_param_importances(study, target=lambda t: t.duration.total_seconds(), target_name="duration")
fig.write_image("plot_param_importances_duration.png") 

# nohup python3 hyperparameter_optim.py > log.txt 2>&1 &
# ps -ef | grep "KA_Time" | awk '$0 !~ /grep/ {print $2}' | xargs kill
# ps -ef | grep "hyperparameter_optim" | awk '$0 !~ /grep/ {print $2}' | xargs kill
