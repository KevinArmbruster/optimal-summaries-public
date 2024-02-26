import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, RobustScaler

from aeon.datasets import load_classification

from einops import rearrange


MIMIC_changing_vars = [
 'dbp',
 'fio2',
 'GCS',
 'hr',
 'map',
 'sbp',
 'spontaneousrr',
 'spo2',
 'temp',
# 'urine', # not found in mimic
 'bun',
 'magnesium',
 'platelets',
 'sodium',
 'alt',
 'hct',
 'po2',
 'ast',
 'potassium',
 'wbc',
 'bicarbonate',
 'creatinine',
 'lactate',
 'pco2',
 'glucose',
 'inr',
 'hgb',
 'bilirubin_total']


def load_MIMIC_data():
    X_time = np.load("/workdir/data/mimic-iii/vasopressor-X_time.npy")
    X_ind = np.load("/workdir/data/mimic-iii/vasopressor-X_ind.npy")
    X_static = np.load("/workdir/data/mimic-iii/vasopressor-X_static.npy")
    y = np.load("/workdir/data/mimic-iii/vasopressor-Ylogits.npy")
    column_names = np.load("/workdir/data/mimic-iii/vasopressor-column_names.npy", allow_pickle=True)
    static_names = list(column_names[-8:])
    return X_time, X_ind, X_static, y, MIMIC_changing_vars, static_names


def create_MIMIC_datasets(X_time, X_ind, X_static, _Y_logits, output_dim = 2, batch_size = 512, random_state = 1):
    
    # ## target
    y = _Y_logits
    if output_dim == 1:
        y = _Y_logits[:, 1, None]
    
    y_unique = np.unique(y)
    num_classes = len(y_unique)
    
    # # class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=y_unique, y=_Y_logits[:, 1])
    class_weights = torch.tensor(class_weights)
    
    if output_dim == 1:
        class_weights = class_weights[1]/class_weights[0] # == pos / neg   # get ONLY positive sample weights
    
    
    # split 60/20/20 %    
    X_time_train, X_time_test, X_ind_train, X_ind_test, X_static_train, X_static_test, y_train, y_test = train_test_split(X_time, X_ind, X_static, y, test_size = 0.40, random_state = random_state, stratify = y)
    X_time_test, X_time_val, X_ind_test, X_ind_val, X_static_test, X_static_val, y_test, y_val = train_test_split(X_time_test, X_ind_test, X_static_test, y_test, test_size = 0.50, random_state = random_state, stratify = y_test)


    # Datasets
    X_time_train = torch.tensor(X_time_train, dtype=torch.float32)
    X_ind_train = torch.tensor(X_ind_train, dtype=torch.float32)
    X_static_train = torch.tensor(X_static_train, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    X_time_val = torch.tensor(X_time_val, dtype=torch.float32)
    X_ind_val = torch.tensor(X_ind_val, dtype=torch.float32)
    X_static_val = torch.tensor(X_static_val, dtype=torch.float32)
    y_val = torch.tensor(y_val)

    X_time_test = torch.tensor(X_time_test, dtype=torch.float32)
    X_ind_test = torch.tensor(X_ind_test, dtype=torch.float32)
    X_static_test = torch.tensor(X_static_test, dtype=torch.float32)
    y_test = torch.tensor(y_test)
    
    
    # Dataloaders
    train_dataset = TensorDataset(X_time_train, X_ind_train, X_static_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)

    val_dataset = TensorDataset(X_time_val, X_ind_val, X_static_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4)

    test_dataset = TensorDataset(X_time_test, X_ind_test, X_static_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, class_weights, num_classes


def get_MIMIC_dataloader(output_dim = 2, batch_size = 512, random_state = 1):
    X_time, X_ind, X_static, Y_logits, changing_vars, static_names = load_MIMIC_data()
    seq_len = X_time.shape[1]
    train_loader, val_loader, test_loader, class_weights, num_classes = create_MIMIC_datasets(X_time, X_ind, X_static, Y_logits, output_dim, batch_size, random_state)
    return train_loader, val_loader, test_loader, class_weights, num_classes, len(changing_vars), len(static_names), seq_len


def get_tiselac_dataloader(batch_size = 512, random_state = 1):
    X, y = load_classification("Tiselac", extract_path="/workdir/data")
    seq_len = X.shape[2]
    changing_dim = X.shape[1]
    static_dim = 0
    train_loader, val_loader, test_loader, class_weights, num_classes = preprocess_tiselac(X, y, batch_size, random_state)
    return train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len


def preprocess_tiselac(X_time, _y, batch_size = 512, random_state = 1):
    # X
    X_time = rearrange(X_time, "b v t -> b t v")
    
    X_ind = ~np.isnan(X_time)
    X_time = np.nan_to_num(X_time, nan=0.0)
    
    # Target
    _y = _y.astype(int)
    _y = _y - 1
    y_unique = np.unique(_y)
    num_classes = len(y_unique)
    
    
    # Class weights
    weights = compute_class_weight(class_weight='balanced', classes=y_unique, y=_y)
    weights = torch.Tensor(weights)
    
    
    # Split
    X_time_train, X_time_test, X_ind_train, X_ind_test, y_train, y_test = train_test_split(X_time, X_ind, _y, test_size = 0.40, random_state = random_state, stratify = _y)
    X_time_test, X_time_val, X_ind_test, X_ind_val, y_test, y_val = train_test_split(X_time_test, X_ind_test, y_test, test_size = 0.50, random_state = random_state, stratify = y_test)


    # Normalize
    X_time_train, X_time_val, X_time_test = normalize_across_time(X_time_train, X_time_val, X_time_test, X_time.shape[2])
    
    
    # Datasets
    X_time_train = torch.tensor(X_time_train, dtype=torch.float32)
    X_ind_train = torch.tensor(X_ind_train, dtype=torch.float32)
    y_train = torch.tensor(y_train)

    X_time_val = torch.tensor(X_time_val, dtype=torch.float32)
    X_ind_val = torch.tensor(X_ind_val, dtype=torch.float32)
    y_val = torch.tensor(y_val)

    X_time_test = torch.tensor(X_time_test, dtype=torch.float32)
    X_ind_test = torch.tensor(X_ind_test, dtype=torch.float32)
    y_test = torch.tensor(y_test)


    # Dataloaders
    train_dataset = TensorDataset(X_time_train, X_ind_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=4)

    val_dataset = TensorDataset(X_time_val, X_ind_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=4)

    test_dataset = TensorDataset(X_time_test, X_ind_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=4)
    
    
    return train_loader, val_loader, test_loader, weights, num_classes


def normalize_across_time(X_train, X_val, X_test, n_variables):
    scaler = StandardScaler()
    
    # N x T x V => N*T x V
    X_train_reshaped = X_train.reshape(-1, n_variables)
    X_val_reshaped = X_val.reshape(-1, n_variables)
    X_test_reshaped = X_test.reshape(-1, n_variables)

    nX_train = scaler.fit_transform(X_train_reshaped)
    nX_val = scaler.transform(X_val_reshaped)
    nX_test = scaler.transform(X_test_reshaped)
    
    # revert shape
    nX_train = nX_train.reshape(X_train.shape)
    nX_val = nX_val.reshape(X_val.shape)
    nX_test = nX_test.reshape(X_test.shape)
    
    return nX_train, nX_val, nX_test
