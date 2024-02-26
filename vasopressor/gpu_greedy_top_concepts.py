
import sys
sys.path.append('..')


import argparse
import numpy as np
import pandas as pd
import torch
import random
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torchmetrics.classification import AUROC, Accuracy, ConfusionMatrix, F1Score
import os, subprocess, gc, time, datetime
from itertools import product

import models.models_original as models_original
import models.models_3d_atomics as models_3d_atomics
import models.models_3d as models_3d
from vasopressor.preprocess_helpers import load_and_create_MIMIC_dataloader
from models.helper import *
from models.param_initializations import *
from models.optimization_strategy import *

device = get_free_gpu()


parser = argparse.ArgumentParser()
parser.add_argument('--split_random_state', type=int, default=1)
parser.add_argument('--dir', type=str, default='/workdir/optimal-summaries-public/_models/vasopressor/original')
parser.add_argument('--n_concepts', type=int, default=4)
FLAGS = parser.parse_args()

random_seed = FLAGS.split_random_state

# prep data
train_loader, val_loader, test_loader, class_weights, num_classes, changing_vars, static_names, seq_len = load_and_create_MIMIC_dataloader(output_dim = 2, batch_size = 512, random_state = random_seed)


experiment_folder = FLAGS.dir
experiment_top_k_folder = add_subfolder(experiment_folder)
makedir(experiment_top_k_folder)

config_original = {
    "n_concepts": 4,
    "use_indicators": True,
}

model_path = get_filename_from_dict(experiment_folder, config_original)

set_seed(random_seed)

seq_len = seq_len
changing_dim = len(changing_vars)
static_dim = len(static_names)

model = models_original.CBM(**config_original, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=2, device=device)
model.fit(train_loader, val_loader, p_weight=class_weights.to(device), save_model_path=model_path.format(**config_original, seed = random_seed), max_epochs=10000)

evaluate_classification(model, test_loader, num_classes=num_classes, device=device)


results = greedy_forward_selection(auroc_metric, test_loader, topkinds, logregbottleneck, track_metrics=[acc_metric])


filename = experiment_top_k_folder + "bottleneck_r{}_c{}_topkinds.csv".format(FLAGS.split_random_state, n_concepts)

write