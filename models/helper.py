from typing import List
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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


def extract_to(batch, device):
    if len(batch) == 3:
        X_time, X_ind, y = [tensor.to(device=device) for tensor in batch]
        X_static = torch.empty(0, device=device)
    else:
        X_time, X_ind, X_static, y = [tensor.to(device=device) for tensor in batch]
    
    return X_time, X_ind, X_static, y
