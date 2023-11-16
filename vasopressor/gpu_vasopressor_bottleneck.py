import os
import argparse
import time
import csv

import numpy as np
import pandas as pd
import pickle
import torch
from torchmetrics import AUROC
from sklearn.model_selection import train_test_split

from models import LogisticRegressionWithSummariesAndBottleneck_Wrapper
from param_initializations import *
from preprocess_helpers import myPreprocessed

X_np, Y_logits, changing_vars, _ = myPreprocessed()

parser = argparse.ArgumentParser()

parser.add_argument('--split_random_state', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_concepts', type=int, default='4')
# parser.add_argument('--zero_weight',type=bool, default=False)
parser.add_argument('--top_k',type=str, default='')
parser.add_argument('--top_k_num',type=int, default=0)

parser.add_argument('--init_cutoffs', type=str, default='init_cutoffs_to_zero')
parser.add_argument('--cutoff_times_init_values_filepath', type=str, default='')
parser.add_argument('--init_thresholds', type=str, default='init_rand')
parser.add_argument('--cutoff_times_temperature', type=float, default=1.0)
parser.add_argument('--thresholds_temperature', type=float, default=0.1)
parser.add_argument('--ever_measured_temperature', type=float, default=0.1)
parser.add_argument('--switch_temperature', type=float, default=0.1)

parser.add_argument('--opt_lr', type=float, default=1e-3)
parser.add_argument('--opt_weight_decay', type=float, default=1e-5)
parser.add_argument('--l1_lambda', type=float, default=1e-3)
parser.add_argument('--cos_sim_lambda', type=float, default=1e-2)

parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--model_output_name', type=str, default='')
parser.add_argument('--num_epochs', type=int, default=1000)
parser.add_argument('--save_every', type=int, default=100)


FLAGS = parser.parse_args()

directory = FLAGS.output_dir or "/workdir/optimal-summaries-public/vasopressor/models/mimic-iii/vasopressor/"
if not os.path.exists(directory):
    os.makedirs(directory)


# device = torch.device("cuda:0")  # Uncomment this to run on GPU
torch.cuda.get_device_name(0)
torch.cuda.is_available()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# train-test-split
torch.set_printoptions(sci_mode=False)
X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = FLAGS.split_random_state, stratify = Y_logits)

# train-val split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = FLAGS.split_random_state, stratify = y_train)

from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable

def tensor_wrap(x, klass=torch.Tensor):
    return x if 'torch' in str(type(x)) else klass(x)

X_pt = Variable(tensor_wrap(X_np)).cuda()

pos_prop = np.mean(np.array(Y_logits)[:, 1])

p_weight = torch.Tensor([1 / (1 - pos_prop), 1 / pos_prop]).cuda()

X_train_pt = Variable(tensor_wrap(X_train)).cuda()
y_train_pt = Variable(tensor_wrap(y_train, torch.FloatTensor)).cuda()

X_val_pt = Variable(tensor_wrap(X_val)).cuda()
y_val_pt = Variable(tensor_wrap(y_val, torch.FloatTensor)).cuda()

X_test_pt = Variable(tensor_wrap(X_test)).cuda()
y_test_pt = Variable(tensor_wrap(y_test, torch.FloatTensor)).cuda()

batch_size = FLAGS.batch_size

train_dataset = TensorDataset(X_train_pt, y_train_pt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

val_dataset = TensorDataset(X_val_pt, y_val_pt)
val_loader = DataLoader(val_dataset, batch_size = X_val_pt.shape[0], shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

test_dataset = TensorDataset(X_test_pt, y_test_pt)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))

input_dim = X_np[0].shape[1]
changing_dim = len(changing_vars)

cutoff_init_fn = init_cutoffs_to_zero

if FLAGS.init_cutoffs == 'init_cutoffs_to_twelve':
    cutoff_init_fn = init_cutoffs_to_twelve

elif FLAGS.init_cutoffs == 'init_cutoffs_to_twentyfour':
    cutoff_init_fn = init_cutoffs_to_twentyfour
    
elif FLAGS.init_cutoffs == 'init_cutoffs_randomly':
    cutoff_init_fn = init_cutoffs_randomly
    
lower_thresh_init_fn = init_rand_lower_thresholds
upper_thresh_init_fn = init_rand_upper_thresholds

if FLAGS.init_thresholds == 'zeros':
    lower_thresh_init_fn = init_zeros
    upper_thresh_init_fn = init_zeros
    
    
cutoff_times_init_values = None
if len(FLAGS.cutoff_times_init_values_filepath) > 0:
    # Load the numpy array from its filepath.
    cutoff_times_init_values = pickle.load( open( FLAGS.cutoff_times_init_values_filepath, "rb" ) )


set_seed(FLAGS.split_random_state)

# initialize model
logregbottleneck = LogisticRegressionWithSummariesAndBottleneck_Wrapper(input_dim, 
                                                 changing_dim, 
                                                 9,
                                                 FLAGS.num_concepts,
                                                 True,
                                                 cutoff_init_fn, 
                                                 lower_thresh_init_fn, 
                                                 upper_thresh_init_fn,
                                                 cutoff_times_temperature=FLAGS.cutoff_times_temperature,
                                                 cutoff_times_init_values=cutoff_times_init_values,
                                                 opt_lr = FLAGS.opt_lr,
                                                 opt_weight_decay = FLAGS.opt_weight_decay,
                                                 l1_lambda = FLAGS.l1_lambda,
                                                 cos_sim_lambda = FLAGS.cos_sim_lambda,
                                                 top_k = FLAGS.top_k,
                                                 top_k_num = FLAGS.top_k_num
                                                )
logregbottleneck.cuda()


# train model
logregbottleneck.fit(train_loader, val_loader, p_weight,
         save_model_path = directory + FLAGS.model_output_name,
         epochs=FLAGS.num_epochs,
         save_every_n_epochs=FLAGS.save_every)

torch.set_printoptions(precision=10)


# get AUC
auroc_metric = AUROC(task="binary").cuda()
y_pred = logregbottleneck.forward_probabilities(X_test_pt)
score = auroc_metric(y_pred, y_test_pt).item()


# write results to csv
filename = "bottleneck_r{}_c{}_gridsearch".format(FLAGS.split_random_state,FLAGS.num_concepts)
with open('{file_path}.csv'.format(file_path=os.path.join(directory, filename)), 'a+') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
    # csvwriter.writerow([FLAGS.num_concepts, FLAGS.opt_lr, FLAGS.opt_weight_decay, score]) 
    csvwriter.writerow([FLAGS.num_concepts, FLAGS.opt_lr, FLAGS.opt_weight_decay, FLAGS.l1_lambda, FLAGS.cos_sim_lambda, score]) 
