from typing import List
import numpy as np
import pandas as pd
import random
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


def jaccard_similarity(list_of_lists):
    sets = [set(lst) for lst in list_of_lists]
    
    intersection = set.intersection(*sets)
    union = set.union(*sets)
    
    if len(union) == 0:
        similarity = 0
    else:
        similarity = len(intersection) / len(union)
    
    return similarity


def score_pruning_stability(models):
    indice_set_per_model = []

    for model in models:
        first_layer = model.regularized_layers[0]
        mask = first_layer.weight_mask.detach()
        
        selected_indices = []
        for concept_id in range(mask.shape[0]):
            selected_indices.extend(torch.nonzero(mask[concept_id], as_tuple=False).flatten().tolist())
        selected_indices = set(selected_indices)
        
        indice_set_per_model.append(selected_indices)
    
    jac_sim = jaccard_similarity(indice_set_per_model)
    return jac_sim


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

def add_subfolder(path, name="top-k"):
    dir, file = os.path.split(path)
    path = os.path.join(dir, name, file)
    return path

def write_df_2_csv(path: str, df: pd.DataFrame):
    makedir(path)
    df.to_csv(path, header=True, index=False)

def read_df_from_csv(path):
    return pd.read_csv(path)

def get_free_gpu():
    # gpu_id = int(subprocess.check_output('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1 | head -n 1 | xargs', shell=True, text=True))
    gpu_id = 15
    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available else torch.device('cpu')
    print("current device", device)
    return device

    
def visualize_optimization_results(model, val_loader, test_loader, greedy_results):
    model.clear_all_weight_masks()

    plt.plot(greedy_results["auc"], label = f"AUC {greedy_results['auc'].values[-1]:.3f}")
    plt.plot(greedy_results["acc"], label = f"ACC {greedy_results['acc'].values[-1]:.3f}")
    plt.plot(greedy_results["f1"], label = f"F1 {greedy_results['f1'].values[-1]:.3f}")

    print("Validation set - Before Pruning")
    auc, acc, f1 = evaluate_classification(model, val_loader)
    plt.axhline(y=auc, color='blue', linestyle='--', label=f"AUC (before) {auc:.3f}")
    plt.axhline(y=acc, color='orange', linestyle='--', label=f"ACC (before) {acc:.3f}")
    plt.axhline(y=f1, color='green', linestyle='--', label=f"F1 (before) {f1:.3f}")

    print("Test set - Before Pruning")
    auc, acc, f1 = evaluate_classification(model, test_loader)
    # plt.axhline(y=auc, color='blue', linestyle=':', label=f"AUC (pre, test) {auc:.3f}")
    # plt.axhline(y=acc, color='orange', linestyle=':', label=f"ACC (pre, test) {acc:.3f}")
    # plt.axhline(y=f1, color='green', linestyle=':', label=f"F1 (pre, test) {f1:.3f}")

    model.deactivate_bottleneck_weights_if_top_k(greedy_results)

    print("Test set - After Pruning")
    auc, acc, f1 = evaluate_classification(model, test_loader)
    plt.axhline(y=auc, color='blue', linestyle='-.', label=f"AUC (after, test set) {auc:.3f}")
    plt.axhline(y=acc, color='orange', linestyle='-.', label=f"ACC (after, test set) {acc:.3f}")
    plt.axhline(y=f1, color='green', linestyle='-.', label=f"F1 (after, test set) {f1:.3f}")

    plt.xlabel('Num Features')
    plt.ylabel('Metrics')
    plt.title('Greedy Selection')

    plt.legend()
    plt.show()


def get_total_and_remaining_parameters_from_masks(layers):
    total = sum([layer.weight.numel() + layer.bias.numel() for layer in layers])
    remaining = sum([
        (layer.weight_mask.sum().item() if layer.weight_mask is not None else layer.weight.numel()) +
        (layer.bias_mask.sum().item() if layer.bias_mask is not None else layer.bias.numel())
        for layer in layers
    ])
    return total, remaining


def evaluate_greedy_selection(models, results, get_dataloader, dataset, random_states:List[int] = [1,2,3]):
    metrics_df = pd.DataFrame(columns=["Model", "Dataset", "Seed", "Split", "Pruning", "Finetuned", "AUC", "ACC", "F1", "Total parameter", "Remaining parameter"])

    for model, greedy_results, random_state in zip(models, results, random_states):
        set_seed(random_state)
        
        train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state = random_state)
        
        model.clear_all_weight_masks()
        total, remaining = get_total_and_remaining_parameters_from_masks(model.regularized_layers)
        
        metrics = evaluate_classification(model, val_loader)
        metrics_df.loc[len(metrics_df)] = {"Model": model.get_short_model_name(), "Dataset": dataset, "Seed": random_state, "Split": "val", "Pruning": "Before", "Finetuned": False, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        metrics = evaluate_classification(model, test_loader)
        metrics_df.loc[len(metrics_df)] = {"Model": model.get_short_model_name(), "Dataset": dataset, "Seed": random_state, "Split": "test", "Pruning": "Before", "Finetuned": False, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        
        model.deactivate_bottleneck_weights_if_top_k(greedy_results)
        total, remaining = get_total_and_remaining_parameters_from_masks(model.regularized_layers)
        
        metrics = evaluate_classification(model, val_loader)
        metrics_df.loc[len(metrics_df)] = {"Model": model.get_short_model_name(), "Dataset": dataset, "Seed": random_state, "Split": "val", "Pruning": "Greedy", "Finetuned": False, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        metrics = evaluate_classification(model, test_loader)
        metrics_df.loc[len(metrics_df)] = {"Model": model.get_short_model_name(), "Dataset": dataset, "Seed": random_state, "Split": "test", "Pruning": "Greedy", "Finetuned": False, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        
        save_model_path = add_subfolder(model.save_model_path, "finetuned")
        makedir(save_model_path)
        model.try_load_else_fit(train_loader, val_loader, p_weight=class_weights, save_model_path=save_model_path, max_epochs=10000, patience=10)
        
        metrics = evaluate_classification(model, val_loader)
        metrics_df.loc[len(metrics_df)] = {"Model": model.get_short_model_name(), "Dataset": dataset, "Seed": random_state, "Split": "val", "Pruning": "Greedy", "Finetuned": True, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        metrics = evaluate_classification(model, test_loader)
        metrics_df.loc[len(metrics_df)] = {"Model": model.get_short_model_name(), "Dataset": dataset, "Seed": random_state, "Split": "test", "Pruning": "Greedy", "Finetuned": True, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        
    return metrics_df


def plot_selected_weights(weight, top_k_inds, greedy_results, top_k=None, sorted=True, log_scale=True):
    abs_weight = weight.detach().cpu().numpy()
    abs_weight = np.abs(abs_weight)
    
    n_concepts = abs_weight.shape[0]
    max_y = np.max(abs_weight)
    
    fig, axs = plt.subplots(n_concepts, figsize=(8, 2 * n_concepts))
    
    for c in range(n_concepts):
        ax = axs[c]
        
        init_features_idx = top_k_inds[c]
        selected_features_idx = greedy_results[greedy_results["Concept"] == c]["Feature"].to_list()
        
        min_weight = np.min(abs_weight[c][selected_features_idx])
        max_weight = np.max(abs_weight[c][selected_features_idx])
        
        if top_k is None:
            if sorted:
                weight_idx = np.argsort(-abs_weight[c])
            else:
                weight_idx = range(abs_weight.shape[1])
            
            weight_idx = weight_idx[abs_weight[c][weight_idx] >= min_weight]
        else:
            if sorted:
                weight_idx = np.argsort(-abs_weight[c])
            else:
                weight_idx = range(abs_weight.shape[1])
            
            top_k = max(top_k, abs_weight.shape[1])
            weight_idx = weight_idx[:top_k]
        
        n_rel_feat = len(weight_idx)
        
        def getColor(idx):
            if idx in selected_features_idx:
                return "red"
            elif idx in init_features_idx:
                return "blue"
            else:
                return "gray"
        
        colors = [getColor(idx) for idx in weight_idx]
        ax.bar(np.arange(1, len(weight_idx)+1), abs_weight[c][weight_idx], color=colors)
        
        ax.set_title(f"Initialized with {len(init_features_idx)}; selected {len(selected_features_idx)}; of total {len(abs_weight[c])})")
        ax.set_xlabel(f"Top {n_rel_feat} features (descending by weight) (min={min_weight:.3f}) (max={max_weight:.3f})")
        ax.set_ylabel(f"Concept {c}")
        ax.set_ylim(0, max_y)
        if log_scale:
            ax.set_yscale('log')
        
        leg_handles = [plt.Rectangle((0,0),1,1, color=color) for color in ['red', 'blue', 'gray']]
        leg_labels = ["Selected", "Initialization", "Neither"]
        ax.legend(leg_handles, leg_labels)
    
    # plt.ylabel("abs value of feature coefficient")
    plt.tight_layout()
    plt.show()
    
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, color="black", label="Train")
    plt.plot(val_losses, color="green", label="Val")
    plt.yscale("log")
    plt.legend()
    plt.show()
    
    
class LazyLinearWithMask(nn.LazyLinear):
    def __init__(self, out_features, bias=True, ema_decay=0.9):
        super().__init__(out_features, bias)
        self.cls_to_become = LazyLinearWithMask
        self.weight_mask = None
        self.bias_mask = None
        self.ema_gradient = None
        self.ema_decay = ema_decay
    
    def set_weight_mask(self, weight_mask, bias_mask = None):
        if weight_mask is None:
            return
        
        assert weight_mask.shape == self.weight.shape
        self.weight_mask = weight_mask.to(self.weight.device)
        
        if bias_mask is not None:
            assert bias_mask.shape == self.bias.shape
            self.bias_mask = bias_mask.to(self.weight.device)
    
    def set_weight_mask_from_weight(self):
        self.weight_mask = (self.weight != 0).int()
        self.bias_mask = (self.bias != 0).int()
    
    def clear_weight_mask(self):
        self.weight_mask = None
        self.bias_mask = None
    
    def update_ema_gradient(self):
        if self.ema_gradient is None:
            self.ema_gradient = self.weight.grad.detach()
        else:
            self.ema_gradient = self.ema_decay * self.ema_gradient + (1 - self.ema_decay) * self.weight.grad.detach()
    
    def forward(self, input):
        if self.weight_mask is None:
            return F.linear(input, self.weight, self.bias)
        else:
            masked_weight = self.weight * self.weight_mask
            
            if self.bias_mask is None:
                masked_bias = self.bias
            else:
                masked_bias = self.bias * self.bias_mask
            
            return F.linear(input, masked_weight, masked_bias)
    

def mask_smallest_magnitude(weight_tensor, remain_active=5):
    flattened_weights = weight_tensor.flatten()
    _, sorted_indices = torch.sort(flattened_weights)

    if remain_active < 1:
        idx = int(len(sorted_indices) * remain_active)
    else:
        idx = remain_active
    thresholded_idx = sorted_indices[:idx]

    binary_mask = torch.zeros_like(flattened_weights)
    binary_mask[thresholded_idx] = 1

    weight_mask = binary_mask.reshape(weight_tensor.shape)
    bias_mask = torch.sum(weight_mask, dim=1)
    bias_mask[bias_mask > 0] = 1

    return weight_mask, bias_mask

def mask_shrinking_weights(layer):
    weight_mask = layer.ema_gradient * layer.weight.detach()
    weight_mask = weight_mask > 0

    bias_mask = torch.sum(weight_mask, dim=1)
    bias_mask[bias_mask > 0] = 1

    return weight_mask, bias_mask


def set_seed(r):
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    random.seed(r)
    torch.manual_seed(r)
    torch.cuda.manual_seed(r)
    np.random.seed(r)
