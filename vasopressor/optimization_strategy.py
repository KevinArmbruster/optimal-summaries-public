import numpy as np
import torch
from typing import List
from models import LogisticRegressionWithSummariesAndBottleneck_Wrapper
from tqdm import tqdm
from torchmetrics import Metric

def greedy_selection(X_test_pt: torch.Tensor, y_test: torch.Tensor, top_k_inds: List[List[int]], wrapper: LogisticRegressionWithSummariesAndBottleneck_Wrapper, score_func: Metric):
    num_concepts = wrapper.num_concepts
    feature_budget = 10 * num_concepts
    
    condition = torch.zeros(wrapper.model.bottleneck.weight.shape, dtype=torch.bool).cuda()
    best_scores = []
    best_score_inds = []
    best_score_concepts = []
    
    for i in tqdm(range(feature_budget)):
        best_score = 0
        best_score_ind = -1
        best_score_concept = -1
        temp = torch.nn.Parameter(wrapper.model.bottleneck.weight.clone().detach())
    
        for c in range(num_concepts):
            for ind in top_k_inds[c]:
            # add 1 feature to test score
                if not condition[c][ind]:
                    condition[c][ind]=True
                    wrapper.model.bottleneck.weight = torch.nn.Parameter(wrapper.model.bottleneck.weight.where(condition, torch.tensor(0.0).cuda()))

                # get score with added feature
                    y_pred = wrapper.forward_probabilities(X_test_pt)
                    curr_score = score_func(y_pred, y_test)
                    if (curr_score > best_score):
                        best_score = curr_score
                        best_score_ind = ind
                        best_score_concept = c

                # remove feature
                    condition[c][ind]=False
                    wrapper.model.bottleneck.weight = temp

        condition[best_score_concept][best_score_ind] = True
        best_scores.append(best_score)
        best_score_inds.append(best_score_ind)
        best_score_concepts.append(best_score_concept)
    return best_scores, best_score_inds, best_score_concepts
