import numpy as np
import torch
import random

def init_cutoffs_to_zero(d):
    return np.zeros(d)

def init_cutoffs_to_small(d):
    return np.zeros(d) + 0.1

def init_cutoffs_to_50perc(d):
    return np.zeros(d) + 0.5

def init_rand_upper_thresholds(d):
    return np.random.rand(d)

def init_rand_lower_thresholds(d):
    return np.random.rand(d) - 1
