# %%
import sys
sys.path.append('..')
from aeon.datasets import load_classification
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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import Accuracy, AUROC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import optuna
from optuna.trial import TrialState
from optuna.visualization import *

import models.original_models as original_models
import models.models_3d_atomics_on_variate_to_concepts as new_models
from vasopressor.preprocess_helpers import *
from models.helper import *
from models.param_initializations import *
from models.optimization_strategy import greedy_selection

gpu_id = int(subprocess.check_output('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs', shell=True, text=True))
device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available else torch.device('cpu')
print("current device", device)


# %%
X, y = load_classification("SpokenArabicDigits", extract_path="/workdir/data")
y = y.astype(int)


# %%
lengths = []
for i, x in enumerate(X):
    lengths.append(x.shape[1])
lengths.sort()

MIN_TS_SIZE = min(lengths)
MAX_TS_SIZE = max(lengths)

print(MIN_TS_SIZE, MAX_TS_SIZE)


# %%
def preprocess_data_multiclass(_X, _y):
    equi_length_X = []
    for x in _X:
        pad_width = ((0, 0), (0, MAX_TS_SIZE - x.shape[1]))
        padded = np.pad(x, pad_width, mode='constant', constant_values=0)
        equi_length_X.append(padded)

    equi_length_X = np.array(equi_length_X)
    equi_length_X = equi_length_X.swapaxes(1,2)
    
    indicators_3d = ~np.isnan(equi_length_X)

    data = np.concatenate([equi_length_X, indicators_3d], axis=-1) # (N x seq_len x 2*changing_dim)
    
    ## target
    _y = _y - 1
    y_unique = np.unique(_y)
    num_classes = len(y_unique)
    
    # initiazing datasets
    weights = compute_class_weight(class_weight='balanced', classes=y_unique, y=_y)
    weights = torch.Tensor(weights).to(device)
    
    return data, _y, num_classes, weights


def initialize_data(r, _X, _y, batch_size = 256, multiclass = False):   
    
    # train-test-split
    torch.set_printoptions(sci_mode=False)
    X_train, X_test, y_train, y_test = train_test_split(_X, _y, test_size = 0.15, random_state = r, stratify = _y)

    # train-val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = r, stratify = y_train)

    X_train_pt = torch.tensor(X_train)
    y_train_pt = torch.tensor(y_train, torch.FloatTensor)

    X_val_pt = torch.tensor(X_val)
    y_val_pt = torch.tensor(y_val, torch.FloatTensor)

    X_test_pt = torch.tensor(X_test)
    y_test_pt = torch.tensor(y_test, torch.FloatTensor)
    if multiclass:
        y_test_pt = torch.argmax(y_test_pt, dim=1)

    train_dataset = TensorDataset(X_train_pt, y_train_pt)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = TensorDataset(X_val_pt, y_val_pt)
    val_loader = DataLoader(val_dataset, batch_size = X_val_pt.shape[0], shuffle=False, num_workers=4, pin_memory=True)

    test_dataset = TensorDataset(X_test_pt, y_test_pt)
    test_loader = DataLoader(test_dataset, batch_size = X_test_pt.shape[0], shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader



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
    
    data, _y, num_classes, weights = preprocess_data_multiclass(X, y)
    train_loader, val_loader, test_loader = initialize_data(1, data, _y, multiclass=True)
    
    input_dim = data.shape[2]
    changing_dim = X[0].shape[0]
    seq_len = data.shape[1]

    auroc_metric = AUROC(task="multiclass", num_classes=num_classes).to(device)
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

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
