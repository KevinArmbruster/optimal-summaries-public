
import sys
sys.path.append('..')

from typing import List
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, cosine_similarity
from models.helper import TaskType


def compute_loss(y_true: torch.Tensor, y_pred: torch.Tensor, p_weight: torch.Tensor, l1_lambda: float, cos_sim_lambda: float, regularized_layers: List[torch.nn.Module], task_type: TaskType.CLASSIFICATION):
    output_dim = 1 if y_true.dim() == 1 else y_true.size(1)
    
    if task_type == TaskType.CLASSIFICATION and output_dim <= 2:
        task_loss = binary_cross_entropy_with_logits(y_pred, y_true.float(), pos_weight = p_weight)
    elif task_type == TaskType.CLASSIFICATION and output_dim > 2:
        task_loss = cross_entropy(y_pred, y_true, weight = p_weight)
    elif task_type == TaskType.REGRESSION:
        task_loss = mse_loss(y_pred, y_true)
    else:
        raise NotImplementedError("Loss not defined!")
    
    # Lasso regularization
    l1_loss = 0
    if l1_lambda != 0:
        L1_norm = torch.sum([torch.norm(layer.weight, p=1) for layer in regularized_layers])
        l1_loss = l1_lambda * L1_norm
    
    # Cosine_similarity regularization
    cos_sim_loss = 0
    if cos_sim_lambda != 0:
        cos_sim = torch.sum([cos_sim(layer) for layer in regularized_layers])
        cos_sim_loss = cos_sim_lambda * cos_sim
    
    return task_loss + l1_loss + cos_sim_loss

def cos_sim(layer: torch.nn.Module):
    if layer.out_features == 1:
        return 0
    
    concepts = torch.arange(layer.out_features, device=layer.weight.device)
    indices = torch.combinations(concepts, 2)
    
    weights = layer.weight[indices]
    cos_sim = torch.abs(cosine_similarity(weights[:, 0], weights[:, 1], dim=1)).sum()
    
    return cos_sim
