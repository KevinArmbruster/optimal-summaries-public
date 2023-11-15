import os
import argparse
import csv
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from torchmetrics import AUROC

from models import LogisticRegressionWithSummariesAndBottleneck_Wrapper
from param_initializations import *
from preprocess_helpers import myPreprocessed
from optimization_strategy import greedy_selection

def tensor_wrap(x, klass=torch.Tensor):
    return x if 'torch' in str(type(x)) else klass(x)

parser = argparse.ArgumentParser()
parser.add_argument('--split_random_state', type=int, default=1)
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--n_concepts', type=int, default=4)
FLAGS = parser.parse_args()


# device = torch.device("cuda:0")  # Uncomment this to run on GPU
torch.cuda.get_device_name(0)
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# prep data
X_np, Y_logits, changing_vars, _ = myPreprocessed("../vasopressor-Xdata.npy", "../vasopressor-Ylogits.npy")

# train-test-split
torch.set_printoptions(sci_mode=False)
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = FLAGS.split_random_state, stratify = Y_logits)

# train-val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = FLAGS.split_random_state, stratify = y_train)

# X_pt = Variable(tensor_wrap(X_np)).cuda()

pos_prop = np.mean(np.array(Y_logits)[:, 1])

p_weight = torch.Tensor([1 / (1 - pos_prop), 1 / pos_prop]).cuda()

X_train_pt = Variable(tensor_wrap(X_train)).cuda()
y_train_pt = Variable(tensor_wrap(y_train, torch.FloatTensor)).cuda()

X_val_pt = Variable(tensor_wrap(X_val)).cuda()
y_val_pt = Variable(tensor_wrap(y_val, torch.FloatTensor)).cuda()

X_test_pt = Variable(tensor_wrap(X_test)).cuda()
y_test_pt = Variable(tensor_wrap(y_test, torch.FloatTensor)).cuda()

batch_size = 256

train_dataset = TensorDataset(X_train_pt, y_train_pt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

val_dataset = TensorDataset(X_val_pt, y_val_pt)
val_loader = DataLoader(val_dataset, batch_size = X_val_pt.shape[0], shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

test_dataset = TensorDataset(X_test_pt, y_test_pt)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

input_dim = X_np[0].shape[1]
changing_dim = len(changing_vars)


experiment_folder = FLAGS.dir or "/workdir/optimal-summaries-public/vasopressor/models/mimic-iii/vasopressor/"
experiment_top_k_folder = os.path.join(experiment_folder, "top-k/")
if not os.path.exists(experiment_top_k_folder):
    os.makedirs(experiment_top_k_folder)


n_concepts = FLAGS.n_concepts

# get top features for a set number of concepts
topkinds = []
with open(experiment_top_k_folder + 'topkindsr{}c{}.csv'.format(FLAGS.split_random_state, n_concepts), mode ='r')as file:
    # reading the CSV file
    csvFile = csv.reader(file)
    for row in csvFile:
        topkinds.append(np.array(list(map(int, row))))


# run experiment
file = open(experiment_folder + 'bottleneck_r{}_c{}_gridsearch.csv'.format(FLAGS.split_random_state, n_concepts))
csvreader = csv.reader(file)
header = next(csvreader)
bottleneck_row = []
for row in csvreader:
    if row[3]=="0.001" and row[4]=="0.01":
        bottleneck_row = np.array(row).astype(float)  
# format hyperparameters for csv reader
row=[int(el) if el >= 1 else el for el in bottleneck_row]
row=[0 if el == 0 else el for el in bottleneck_row]


set_seed(FLAGS.split_random_state)
logregbottleneck = LogisticRegressionWithSummariesAndBottleneck_Wrapper(input_dim, 
                                                                            changing_dim,
                                                                            9,                     
                                                                            n_concepts,
                                                                            True,
                                                                            init_cutoffs_to_zero, 
                                                                            init_rand_lower_thresholds, 
                                                                            init_rand_upper_thresholds,
                                                                            cutoff_times_temperature=1.0,
                                                                            cutoff_times_init_values=None,
                                                                            opt_lr = row[1],
                                                                            opt_weight_decay = row[2],
                                                                            l1_lambda=row[3],
                                                                            cos_sim_lambda = row[4]
                                                                            )
logregbottleneck.cuda()

logregbottleneck.fit(train_loader, val_loader, p_weight, 
                     save_model_path = experiment_folder + "/bottleneck_r{}_c{}_optlr_{}_optwd_{}_l1lambda_{}_cossimlambda_{}.pt".format(FLAGS.split_random_state,int(row[0]),row[1],row[2],row[3],row[4]), 
                     epochs=10, 
                     save_every_n_epochs=10)


auroc_metric = AUROC(task="binary").cuda()
best_aucs, best_auc_inds, best_auc_concepts = greedy_selection(auroc_metric, X_test_pt, y_test, n_concepts, topkinds, logregbottleneck)


filename = experiment_top_k_folder + "bottleneck_r{}_c{}_topkinds.csv".format(FLAGS.split_random_state, n_concepts)

# writing to csv file
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Best AUC", "Best AUC Concept #", "Best AUC ind #"])
    # writing the data rows 
    for row in zip(best_aucs, best_auc_concepts, best_auc_inds):
        csvwriter.writerow(list(row))
