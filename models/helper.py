from typing import List
import numpy as np
import pandas as pd
import csv
import os, subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torchmetrics.classification import AUROC, Accuracy, ConfusionMatrix, F1Score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from einops import rearrange
from enum import Enum
from rtpt import RTPT

from models.weights_parser import WeightsParser


summary_dict = {0:'mean', 1:'var', 2:'ever measured', 3:'mean of indicators', 4: 'var of indicators', 5:'# switches', 6:'slope', 7:'slope std err', 8:'first time measured', 9:'last time measured', 10:'hours above threshold', 11:'hours below threshold'}

def get_name_of_feature(ind: int, changing_variables_names: List[str], seq_len: int = 1, static_variable_names: List[str] = []):
    # model input row to bottleneck:
    # Changing_vars per T
    # Indicators per T
    # Static features
    # Summary features per changing_vars
    
    variable_name_list = []
    indicator_names = [f"{name}_ind" for name in changing_variables_names]
    
    # result is = V1_T1, V2_T1, V3_T1, ..., V1_T2, V2_T2, V3_T2, ... repeat V for T times
    changing_vars_per_time = [f"{name}_time_{t}" for t in range(1, seq_len + 1) for name in changing_variables_names]
    indicators_per_time = [f"{name}_time_{t}" for t in range(1, seq_len + 1) for name in indicator_names]
    
    # create final name list
    variable_name_list = changing_vars_per_time + indicators_per_time + static_variable_names
    non_summary_dim = len(variable_name_list)
    
    # get feature name from lists and summary
    if ind < non_summary_dim:
        # raw feature
        return variable_name_list[ind], 'raw'
    else:
        # summary statistic of feature
        ind = ind - non_summary_dim
        summary = ind // len(changing_variables_names)
        feature = ind % len(changing_variables_names)
        return changing_variables_names[feature], summary_dict[summary]


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        print(n, torch.isnan(p.grad).any())
        if(p.requires_grad): #and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(right=len(ave_grads)) # left=0, 
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
    plt.show()


def visualize_top100_weights_per_channel(layer, top_k=100):
    abs_weight = layer.weight.detach().cpu().numpy()
    abs_weight = np.abs(abs_weight)
    
    top_k = min(top_k, abs_weight.shape[1])
    
    max_y = np.max(abs_weight)

    for c in range(abs_weight.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        inds = np.argsort(-abs_weight[c])[:top_k]
        ax.bar(np.arange(1,top_k+1), abs_weight[c][inds])
        ax.set_xlabel(f"Top {top_k} features")
        ax.set_ylabel("abs value of feature coefficient")
        ax.set_ylim(0, max_y)
        plt.show()


def jaccard_similarity(*lists):
    sets = [set(lst) for lst in lists]
    
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    
    similarity = len(intersection) / len(union)
    
    return similarity


def extract_to(batch, device):
    if len(batch) == 3:
        X_time, X_ind, y = [tensor.to(device=device) for tensor in batch]
        X_static = torch.empty(0, device=device)
    else:
        X_time, X_ind, X_static, y = [tensor.to(device=device) for tensor in batch]
    
    return X_time, X_ind, X_static, y


def create_3d_input_as_b_v_t(time_dependent_vars, indicators, static_vars, summaries, use_indicators):
    time = 1
    variate = 2
    
    if use_indicators:
        input = torch.cat([time_dependent_vars, indicators], axis=time)
    else:
        input = time_dependent_vars
    
    if not torch.equal(static_vars, torch.empty(0, device=static_vars.device)):
        static_vars = static_vars.unsqueeze(variate) # b x s1 x 1
        static_vars = static_vars.expand(-1, -1, time_dependent_vars.size(variate)) # b x s1 x v
    
    if summaries != None: # b x s2 x v
        input = torch.cat([input, static_vars, summaries], axis=time)
    else:
        input = torch.cat([input, static_vars], axis=time)
    
    input = rearrange(input, "b t v -> b v t")
    return input


def get_time_feat_2d(time_dependent_vars, indicators, static_vars, use_only_last_timestep, use_indicators):
    seq_len = time_dependent_vars.size(1)
    
    if use_only_last_timestep:
        time_feats_2d = time_dependent_vars[:, seq_len-1, :]
        
        if use_indicators:
            ind2d = indicators[:, seq_len-1, :]
            time_feats_2d = torch.cat((time_feats_2d, ind2d), dim=-1)
            
        time_feats_2d = torch.cat((time_feats_2d, static_vars), dim=-1)
    else:
        # use full timeseries, reshape 3d to 2d, keep sample size N and merge Time x Variables (N x T x V) => (N x T*V)
        time_feats_2d = time_dependent_vars.reshape(time_dependent_vars.shape[0], -1) # result is V1_T1, V2_T1, V3_T1, ..., V1_T2, V2_T2, V3_T2, ... repeat V, T times
        
        if use_indicators:
            indicators_2d = indicators.reshape(indicators.shape[0], -1)
            time_feats_2d = torch.cat((time_feats_2d, indicators_2d), dim=1)
        
        time_feats_2d = torch.cat((time_feats_2d, static_vars), dim=1)
    
    return time_feats_2d
    

def normalize_gradient_(params, norm_type, p_norm_type=2):
    for param in params:
        if norm_type == "FULL" or torch.squeeze(param.grad).dim() < 2:
            param.grad = normalize(param.grad, p=p_norm_type, dim=None)
        elif norm_type == "COMPONENT_WISE":
            param.grad = normalize(param.grad, p=p_norm_type, dim=0)
        
    return


def evaluate_classification(model, dataloader, num_classes = 2, average = "macro"):
    device = model.device
    num_classes = model.output_dim
    
    if num_classes == 2:
        auroc_metric = AUROC(task="binary").to(device)
        accuracy_metric = Accuracy(task="binary").to(device)
        f1_metric = F1Score(task="binary").to(device)
        # conf_matrix = ConfusionMatrix(task="binary").to(device)
    else:
        auroc_metric = AUROC(task="multiclass", num_classes=num_classes, average = average).to(device)
        accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average = average).to(device)
        f1_metric = F1Score(task="multiclass", num_classes=num_classes, top_k=1, average = average).to(device)
        # conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    for batch in dataloader:
        *data, target = extract_to(batch, device)
        preds = model(*data)

        _ = auroc_metric(preds, target).item()
        _ = accuracy_metric(preds, target).item()
        _ = f1_metric(preds, target).item()

    auc = auroc_metric.compute().item()
    acc = accuracy_metric.compute().item()
    f1 = f1_metric.compute().item()

    auroc_metric.reset()
    accuracy_metric.reset()
    f1_metric.reset()
    
    print(f"AUC macro {auc:.3f}")
    print(f"ACC macro {acc:.3f}")
    print(f" F1 macro {f1:.3f}")
    
    return auc, acc, f1


class TaskType(Enum):
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"


def create_state_dict(model, epoch):
    state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': model.optimizer.state_dict(),
                    'train_losses': model.train_losses,
                    'val_losses': model.val_losses,
                    }
        
    return state

def add_all_parsers(parser:WeightsParser, changing_dim, static_dim = 0, seq_len = 1, use_indicators = True, str_type = 'linear'):
    if str_type == 'linear':
        time_feat_dim = 2 * changing_dim * seq_len + static_dim
        parser.add_shape(str(str_type) + '_time_', time_feat_dim)
        
    parser.add_shape(str(str_type) + '_mean_', changing_dim)
    parser.add_shape(str(str_type) + '_var_', changing_dim)
    if use_indicators:
        parser.add_shape(str(str_type) + '_ever_measured_', changing_dim)
        parser.add_shape(str(str_type) + '_mean_indicators_', changing_dim)
        parser.add_shape(str(str_type) + '_var_indicators_', changing_dim)
        parser.add_shape(str(str_type) + '_switches_', changing_dim)
    
    # slope_indicators are the same weights for all of the slope features.
    parser.add_shape(str(str_type) + '_slope_', changing_dim)
    
    if str_type == 'linear':
        parser.add_shape(str(str_type) + '_slope_stderr_', changing_dim)
        if use_indicators:
            parser.add_shape(str(str_type) + '_first_time_measured_', changing_dim)
            parser.add_shape(str(str_type) + '_last_time_measured_', changing_dim)
        
    parser.add_shape(str(str_type) + '_hours_above_threshold_', changing_dim)
    parser.add_shape(str(str_type) + '_hours_below_threshold_', changing_dim)
    return

def makedir(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def add_subfolder(path, name="/top-k/"):
    dir, file = os.path.split(path)
    path = dir + name + file
    return path

def write_df_2_csv(path: str, df: pd.DataFrame):
    makedir(path)
    df.to_csv(path, header=True, index=False)

def read_df_from_csv(path):
    return pd.read_csv(path)

def get_free_gpu():
    gpu_id = int(subprocess.check_output('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs', shell=True, text=True))
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available else torch.device('cpu')
    print("current device", device)
    return device

def get_filename_from_dict(folder, config):
    model_path = folder + "".join([f"{key}_{{{key}}}_" for key in config.keys()]) + "seed_{seed}.pt"
    return model_path

def visualize_optimization_results(model, val_loader, test_loader, greedy_results):
    model.clear_all_weight_masks()

    plt.plot(greedy_results["auc"], label = f"AUC {greedy_results['auc'].values[-1]:.3f}")
    plt.plot(greedy_results["acc"], label = f"ACC {greedy_results['acc'].values[-1]:.3f}")
    plt.plot(greedy_results["f1"], label = f"F1 {greedy_results['f1'].values[-1]:.3f}")

    auc, acc, f1 = evaluate_classification(model, val_loader)
    plt.axhline(y=auc, color='blue', linestyle='--', label=f"AUC (before) {auc:.3f}")
    plt.axhline(y=acc, color='orange', linestyle='--', label=f"ACC (before) {acc:.3f}")
    plt.axhline(y=f1, color='green', linestyle='--', label=f"F1 (before) {f1:.3f}")

    auc, acc, f1 = evaluate_classification(model, test_loader)
    # plt.axhline(y=auc, color='blue', linestyle=':', label=f"AUC (pre, test) {auc:.3f}")
    # plt.axhline(y=acc, color='orange', linestyle=':', label=f"ACC (pre, test) {acc:.3f}")
    # plt.axhline(y=f1, color='green', linestyle=':', label=f"F1 (pre, test) {f1:.3f}")

    model.deactivate_bottleneck_weights_if_top_k(greedy_results)

    auc, acc, f1 = evaluate_classification(model, test_loader)
    plt.axhline(y=auc, color='blue', linestyle='-.', label=f"AUC (after, test set) {auc:.3f}")
    plt.axhline(y=acc, color='orange', linestyle='-.', label=f"ACC (after, test set) {acc:.3f}")
    plt.axhline(y=f1, color='green', linestyle='-.', label=f"F1 (after, test set) {f1:.3f}")

    plt.xlabel('Num Features')
    plt.ylabel('Metrics')
    plt.title('Greedy Selection')

    plt.legend()
    plt.show()



class LazyLinearWithMask(nn.LazyLinear):
    def __init__(self, out_features, bias=True, weight_mask=None, ema_decay=0.9):
        super().__init__(out_features, bias)
        self.cls_to_become = LazyLinearWithMask
        self.weight_mask = weight_mask
        self.ema_gradient = None
        self.ema_decay = ema_decay
    
    def set_weight_mask(self, weight_mask):
        assert self.weight_mask is None or self.weight_mask.shape == self.weight.shape
        assert self.weight_mask is None or self.weight_mask.device == self.weight.device
        self.weight_mask = weight_mask
    
    def clear_weight_mask(self):
        self.weight_mask = None
    
    def update_ema_gradient(self):
        if self.ema_gradient == None:
            self.ema_gradient = self.weight.grad.detach()
        else:
            self.ema_gradients = self.ema_decay * self.ema_gradients + (1 - self.ema_decay) * self.weight.grad.detach()
    
    def create_weight_mask_from_ema_gradient(self):
        pass
    
    def forward(self, input):
        if self.weight_mask is None:
            return F.linear(input, self.weight, self.bias)
        else:
            masked_weight = self.weight * self.weight_mask
            return F.linear(input, masked_weight, self.bias)
