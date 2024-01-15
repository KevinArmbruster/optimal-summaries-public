import numpy as np
import torch
import random

def init_cutoffs_to_twelve(d):
    return np.zeros(d) + 12

def init_cutoffs_to_zero(d):
    return np.zeros(d)

def init_cutoffs_to_small(d):
    return np.zeros(d) + 0.1

def init_cutoffs_to_50perc(d):
    return np.zeros(d) + 0.5

def init_cutoffs_to_twentyfour(d):
    return np.zeros(d) + 24

# Initialize all cutoffs to a uniform random integer between 0 and (x - 1)
def init_cutoffs_randomly(d):
    return np.random.randint(24, size=d).astype('float')

def init_rand_upper_thresholds(d):
    return np.random.rand(d)

def init_rand_lower_thresholds(d):
    return np.random.rand(d) - 1

def init_zeros(d):
    return np.zeros(d)

def set_seed(r):
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    random.seed(r)
    torch.manual_seed(r)
    torch.cuda.manual_seed(r)
    np.random.seed(r)
