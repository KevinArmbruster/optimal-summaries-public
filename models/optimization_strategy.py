
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
from typing import List
from models.original_models import CBM
from tqdm import tqdm
from torchmetrics import Metric
from collections import defaultdict
from models.helper import extract_to

def get_top_features_per_concept(layer) -> List[List[int]]:
    abs_bottleneck_weight = layer.weight.detach().cpu().numpy()
    abs_bottleneck_weight = np.abs(abs_bottleneck_weight)

    # get 90th percentile of feature weight per concept
    sum90p = np.sum(abs_bottleneck_weight, axis=-1) * 0.90
    
    # get top K indices
    top_k_inds = []
    for c in range(abs_bottleneck_weight.shape[0]):
        topkinds_conc = []
        curr_sum = 0
        inds = np.argsort(-abs_bottleneck_weight[c]) #desc
        sorted_weight = abs_bottleneck_weight[c][inds]
        
        for ind, weight in zip(inds, sorted_weight):
            curr_sum += abs(weight)
            if curr_sum <= sum90p[c]:
                topkinds_conc.append(ind)
            else:
                break
        
        # if selects less than 10, choose 10 best
        if len(topkinds_conc) < 10:
            topkinds_conc = np.argsort(-abs_bottleneck_weight[c])[:10].tolist()
        
        top_k_inds.append(topkinds_conc)

    print("Found", len(top_k_inds), "Concepts")
    print("90th percentile per concept", sum90p)
    print([f"Concept {i} len: {len(x)}" for i, x in enumerate(top_k_inds)])
    
    return top_k_inds


def greedy_forward_selection(model: CBM, layers_to_prune: List[torch.nn.Module], top_k_inds: List[List[List[int]]], val_loader: DataLoader, optimize_metric: Metric, device, track_metrics: dict[str, Metric] = None):
    optimize_metric.reset()
    FEATURE_BUDGET = np.sum([10 * layer.out_features for layer in layers_to_prune])
    
    def setup(layer, top_k):
        condition = torch.zeros(layer.weight.shape, dtype=torch.bool).to(device)
        original_weight = layer.weight.clone().detach()
        return {"condition": condition, "original_weight": original_weight, "layer": layer, "top_k": top_k}
    
    layer_info_list = [setup(layer, top_k) for layer, top_k in zip(layers_to_prune, top_k_inds)]
    results = defaultdict(list)
    
    model.eval()
    with torch.no_grad():
        with tqdm(total=FEATURE_BUDGET) as pbar:
            for _ in range(FEATURE_BUDGET):
                best_score = 0

                for layer_id, layer_info in enumerate(layer_info_list):
                    condition, original_weight, layer, top_k = layer_info.values()
                    
                    for concept_id in range(layer.out_features):
                        for feat_id in top_k[concept_id]:
                            
                            # add 1 feature to test the score
                            if not condition[concept_id][feat_id]:
                                condition[concept_id][feat_id] = True
                                layer.weight = Parameter(layer.weight.where(condition, torch.tensor(0.0).to(device))) # input if condition, else other

                                # get score with added feature
                                for batch in val_loader:
                                    *data, y_true = extract_to(batch, device)
                                    y_pred = model(*data)
                                    _ = optimize_metric(y_pred, y_true)
                                
                                curr_score = optimize_metric.compute().item()
                                optimize_metric.reset()
                                # print("Layer", layer_id, "Concept", concept_id, "Feat", feat_id, "Score", curr_score)
                                
                                if (curr_score > best_score):
                                    best_score = curr_score
                                    best_score_layer_id = layer_id
                                    best_score_concept_id = concept_id
                                    best_score_feat_id = feat_id

                                # remove feature
                                condition[concept_id][feat_id] = False
                                layer.weight = Parameter(original_weight)

                # keep best scoring concept-feature
                layer_info_list[best_score_layer_id]["condition"][best_score_concept_id][best_score_feat_id] = True
                # print("Total 0 values are", (layer_info_list[best_score_layer_id]["condition"] == 0).sum())
                # print("Total NON 0 values are", (layer_info_list[best_score_layer_id]["condition"] != 0).sum())
            
                results["Score"].append(best_score)
                results["Layer"].append(best_score_layer_id)
                results["Concept"].append(best_score_concept_id)
                results["Feature"].append(best_score_feat_id)
                
                # print(best_score, best_score_layer_id, best_score_concept_id, best_score_feat_id)
                pbar_postfix = {'Score': f'{best_score:.5f}'}
                
                # track additional metrics for current concept-feature selection
                if track_metrics:
                    for layer_info in layer_info_list:
                        layer_info["layer"].weight = Parameter(layer_info["layer"].weight.where(layer_info["condition"], torch.tensor(0.0).to(device)))
                    
                    for name, metric in track_metrics.items():
                        metric.reset()
                        metric = metric.to(device=device)
                        for batch in val_loader:
                            *X, y_true = extract_to(batch, device)
                            y_pred = model(*X)
                            curr_score = metric(y_pred, y_true)
                        curr_score = metric.compute().item()
                        results[name].append(curr_score)
                        pbar_postfix[name] = curr_score
                        metric.reset()
                        
                    for layer_info in layer_info_list:
                        layer_info["layer"].weight = Parameter(layer_info["original_weight"])

                pbar.set_postfix(pbar_postfix)
                pbar.update()
                
    return pd.DataFrame(results)
