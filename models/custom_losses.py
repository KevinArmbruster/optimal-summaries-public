
import sys
sys.path.append('..')

from typing import List
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, cosine_similarity
from models.helper import TaskType


def compute_loss(y_true: torch.Tensor, y_pred: torch.Tensor, p_weight: torch.Tensor, l1_lambda: float, cos_sim_lambda: float, regularized_layers: List[torch.nn.Module], task_type: TaskType.CLASSIFICATION):
    output_dim = 1 if y_pred.dim() == 1 else y_pred.size(1)
    
    if task_type == TaskType.CLASSIFICATION and output_dim <= 2:
        task_loss = binary_cross_entropy_with_logits(y_pred, y_true.float(), pos_weight = p_weight)
    elif task_type == TaskType.CLASSIFICATION and output_dim > 2:
        task_loss = cross_entropy(y_pred, y_true, weight = p_weight)
    elif task_type == TaskType.REGRESSION:
        task_loss = mse_loss(y_pred, y_true)
    else:
        print("task_type", task_type, task_type == TaskType.CLASSIFICATION, task_type == TaskType.REGRESSION)
        print("output_dim", output_dim, output_dim <= 2, output_dim > 2)
        print("y_true", y_true.shape)
        print("y_pred", y_pred.shape)
        raise NotImplementedError("Loss not defined!")
    
    # Lasso regularization
    l1_loss = 0
    if l1_lambda != 0:
        l1_norm = sum([torch.norm(layer.weight, p=1) for layer in regularized_layers])
        l1_loss = l1_lambda * l1_norm
    
    # Cosine_similarity regularization
    cos_sim_loss = 0
    if cos_sim_lambda != 0:
        cos_sim = sum([cosine_similarity_over_out_channels(layer) for layer in regularized_layers])
        cos_sim_loss = cos_sim_lambda * cos_sim
    
    return task_loss + l1_loss + cos_sim_loss

def cosine_similarity_over_out_channels(layer: torch.nn.Module):
    if layer.out_features == 1:
        return 0
    
    concepts = torch.arange(layer.out_features, device=layer.weight.device)
    indices = torch.combinations(concepts, 2)
    
    weights = layer.weight[indices]
    cos_sim = torch.abs(cosine_similarity(weights[:, 0], weights[:, 1], dim=1)).sum()
    
    return cos_sim
