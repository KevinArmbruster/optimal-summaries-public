
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import csv
from tqdm import tqdm
from time import sleep
import optuna
from rtpt import RTPT
from einops import rearrange

from models.param_initializations import *
from models.EarlyStopping import EarlyStopping
from models.weights_parser import WeightsParser
from models.differentiable_summaries import calculate_summaries
from models.helper import *
from models.custom_losses import compute_loss
from models.BaseModel import BaseCBM


class CBM(BaseCBM):
    def __init__(self, 
                static_dim, 
                changing_dim, 
                seq_len,
                n_concepts,
                output_dim,
                use_indicators = True,
                use_only_last_timestep = False,
                use_grad_norm = False,
                use_summaries = True,
                differentiate_cutoffs = True,
                init_cutoffs_f = init_cutoffs_to_zero,
                init_lower_thresholds_f = init_rand_lower_thresholds, 
                init_upper_thresholds_f = init_rand_upper_thresholds,
                temperature = 0.1,
                opt_lr = 1e-3,
                opt_weight_decay = 1e-5,
                l1_lambda=1e-3,
                cos_sim_lambda=1e-2,
                ema_decay=0.9,
                top_k = None,
                top_k_num = np.inf,
                task_type = TaskType.CLASSIFICATION,
                device = 'cuda',
                ):
        """Initializes the LogisticRegressionWithSummaries with training hyperparameters.
        
        Args:
            input_dim (int): number of input dimensions
            changing_dim (int): number of non-static input dimensions
            init_cutoffs (function): function to initialize cutoff-time parameters
            init_lower_thresholds (function): function to initialize lower threshold parameters
            init_upper_thresholds (function): function to initialize upper threshold parameters
            seq_len (int): number of time-steps in each trajectory
            -- 
            opt_lr (float): learning rate for the optimizer
            opt_weight_decay (float): weight decay for the optimizer
            num_concepts (int): number of concepts in bottleneck layer
            l1_lambda (float): lambda value for L1 regularization
            cos_sim_lambda (float): lambda value for cosine similarity regularization
            
        """
        super(CBM, self).__init__()
        self.architecture = "original"
        
        self.static_dim = static_dim
        self.changing_dim = changing_dim
        self.seq_len = seq_len
        self.num_concepts = n_concepts
        self.num_summaries = 12 # number of calculated summaries
        
        self.use_indicators = use_indicators
        self.use_only_last_timestep = use_only_last_timestep
        self.use_grad_norm = use_grad_norm
        self.use_summaries = use_summaries
        
        self.differentiate_cutoffs = differentiate_cutoffs
        self.init_cutoffs_f = init_cutoffs_f 
        self.init_lower_thresholds_f = init_lower_thresholds_f
        self.init_upper_thresholds_f = init_upper_thresholds_f
        self.temperature = temperature
        
        self.opt_lr = opt_lr
        self.opt_weight_decay = opt_weight_decay
        self.l1_lambda = l1_lambda
        self.cos_sim_lambda = cos_sim_lambda
        self.ema_decay = ema_decay
        
        self.top_k = top_k
        self.top_k_num = top_k_num
        self.output_dim = output_dim
        self.task_type = task_type
        self.device = device
        
        self.create_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = opt_lr, weight_decay = opt_weight_decay)

    def get_model_name(self):
        return f"{self.architecture}_num_concepts_{self.num_concepts}_use_indicators_{self.use_indicators}_use_only_last_timestep_{self.use_only_last_timestep}"
    
    def get_short_model_name(self):
        return f"{self.architecture}"
    
    def create_model(self):
        self.sigmoid_layer = nn.Sigmoid()
        
        # activation function to convert output into probabilities
        # not needed during training as pytorch losses are optimized and include sigmoid / softmax
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim == 1:
            self.output_af = nn.Sigmoid()
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 1:
            self.output_af = nn.Softmax(dim=1)
        elif self.task_type == TaskType.REGRESSION:
            self.output_af = nn.Identity()
        else:
            raise NotImplementedError("Config not defined!")


        self.weight_parser = WeightsParser()
        self.cs_parser = WeightsParser()
        add_all_parsers(self.weight_parser, self.changing_dim, self.static_dim, self.seq_len)
        add_all_parsers(self.cs_parser, self.changing_dim, str_type = 'cs')
        
        
        # Initialize cutoff_times to by default use all of the timesteps.
        self.cutoff_percentage = -torch.ones(1, self.cs_parser.num_weights, device=self.device)
        
        if self.differentiate_cutoffs:
            cutoff_vals = self.init_cutoffs_f(self.cs_parser.num_weights)
            self.cutoff_percentage = nn.Parameter(torch.tensor(cutoff_vals, requires_grad=True, device=self.device).reshape(1, self.cs_parser.num_weights))

        self.times = torch.tensor(np.transpose(np.tile(range(self.seq_len), (self.cs_parser.num_weights, 1))), device=self.device)
        
        
        self.lower_thresholds = nn.Parameter(torch.tensor(self.init_lower_thresholds_f(self.changing_dim), requires_grad=True, device=self.device))
        self.upper_thresholds = nn.Parameter(torch.tensor(self.init_upper_thresholds_f(self.changing_dim), requires_grad=True, device=self.device))
        
        self.thresh_temperature = self.temperature
        self.cutoff_percentage_temperature = self.temperature
        self.ever_measured_temperature = self.temperature
        
        
        # bottleneck layer
        # in B x T x V
        self.bottleneck = LazyLinearWithMask(self.num_concepts) # self.weight_parser.num_weights
        # -> B x T x C
        
        # prediction task
        self.layer_output = nn.LazyLinear(self.output_dim)
        
        self.regularized_layers = nn.ModuleList(self.bottleneck)
        
        self.to(device=self.device)
        # self.deactivate_bottleneck_weights_if_top_k()
        return
    
    def forward(self, time_dependent_vars, indicators, static_vars):
        assert time_dependent_vars.dim() == 3 and time_dependent_vars.size(1) == self.seq_len and time_dependent_vars.size(2) == self.changing_dim
        assert indicators.shape == time_dependent_vars.shape
        assert torch.equal(static_vars, torch.empty(0, device=self.device)) or (static_vars.dim() == 2 and static_vars.size(1) == self.static_dim)
        
        # Encodes the patient_batch, then computes the forward.
        summaries = calculate_summaries(self, time_dependent_vars, indicators, self.use_indicators)
        
        input = get_time_feat_2d(time_dependent_vars, indicators, static_vars, self.use_only_last_timestep, self.use_indicators)
        
        if self.use_summaries:
            summaries2d = summaries.reshape(summaries.size(0), -1)
            input = torch.cat([input, summaries2d], axis=1)
        
        bottleneck = self.bottleneck(input)
        
        activation = self.sigmoid_layer(bottleneck)
        return self.layer_output(activation)
    