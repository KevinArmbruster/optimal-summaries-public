import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List
from models import LogisticRegressionWithSummariesAndBottleneck_Wrapper
from tqdm import tqdm
from torchmetrics import Metric
from collections import defaultdict

def greedy_selection(optimize_metric: Metric, test_loader: DataLoader, top_k_inds: List[List[int]], wrapper: LogisticRegressionWithSummariesAndBottleneck_Wrapper, device = 'cuda', track_metrics: dict[str, Metric] = None):
    optimize_metric.reset()
    num_concepts = wrapper.get_num_concepts()
    FEATURE_BUDGET = 10 * num_concepts
    
    condition = torch.zeros(wrapper.model.bottleneck.weight.shape, dtype=torch.bool).to(device)
    original_state = torch.nn.Parameter(wrapper.model.bottleneck.weight.clone().detach())
    state = defaultdict(list)
    
    wrapper.eval()
    with torch.no_grad():
        for i in tqdm(range(FEATURE_BUDGET)):
            best_score = 0
            best_score_ind = -1
            best_score_concept = -1

            for c in range(num_concepts):
                for ind in top_k_inds[c]:
                    # add 1 feature to test score
                    if not condition[c][ind]:
                        condition[c][ind]=True
                        wrapper.model.bottleneck.weight = torch.nn.Parameter(wrapper.model.bottleneck.weight.where(condition, torch.tensor(0.0).to(device)))

                        # get score with added feature
                        for i, (X_test, y_test) in enumerate(test_loader):
                            X_test, y_test = X_test.to(device), y_test.to(device)
                            y_pred = wrapper.forward_probabilities(X_test)
                            curr_score = optimize_metric(y_pred, y_test)
                        curr_score = optimize_metric.compute().item()
                        optimize_metric.reset()
                        
                        if (curr_score > best_score):
                            best_score = curr_score
                            best_score_ind = ind
                            best_score_concept = c

                        # remove feature
                        condition[c][ind]=False
                        wrapper.model.bottleneck.weight = original_state

            condition[best_score_concept][best_score_ind] = True
            
            state["Score"].append(best_score)
            state["ID"].append(best_score_ind)
            state["Concept"].append(best_score_concept)
            
            # track additional metrics
            if track_metrics:
                wrapper.model.bottleneck.weight = torch.nn.Parameter(wrapper.model.bottleneck.weight.where(condition, torch.tensor(0.0).to(device)))
                for name, metric in track_metrics.items():
                    for i, (X_test, y_test) in enumerate(test_loader):
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        y_pred = wrapper.forward_probabilities(X_test)
                        curr_score = metric(y_pred, y_test)
                    curr_score = metric.compute().item()
                    state[name].append(curr_score)
                    metric.reset()
                wrapper.model.bottleneck.weight = original_state # probably not necessary

    return pd.DataFrame(state)
