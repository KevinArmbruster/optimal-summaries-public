
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
                n_atomics,
                n_concepts,
                output_dim,
                use_summaries_for_atomics,
                use_indicators = True,
                use_grad_norm = False,
                use_summaries = True,
                differentiate_cutoffs = True,
                init_cutoffs_f = init_cutoffs_to_100perc,
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
                device = "cuda",
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
        self.architecture = "atomics"
        
        self.static_dim = static_dim
        self.changing_dim = changing_dim
        self.seq_len = seq_len
        self.num_concepts = n_concepts
        self.num_atomics = n_atomics
        self.num_summaries = 12 # number of calculated summaries
        
        self.use_summaries_for_atomics = use_summaries_for_atomics
        self.use_indicators = use_indicators
        self.use_grad_norm = use_grad_norm
        self.use_summaries  = use_summaries 
        
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
        return f"{self.architecture}_num_concepts_{self.num_concepts}_num_atomics_{self.num_atomics}_use_summaries_for_atomics_{self.use_summaries_for_atomics}_use_indicators_{self.use_indicators}_use_summaries_{self.use_summaries}"
    
    def get_short_model_name(self):
        return f"{self.architecture}_sum2atomics_{self.use_summaries_for_atomics}"
    
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
        add_all_parsers(self.weight_parser, self.changing_dim, self.static_dim, self.seq_len, self.use_indicators)
        add_all_parsers(self.cs_parser, self.changing_dim, self.use_indicators, str_type = 'cs')
        
        
        # Initialize cutoff_times to by default use all of the timesteps.
        if self.differentiate_cutoffs:
            cutoff_vals = self.init_cutoffs_f(self.cs_parser.num_weights)
            self.cutoff_percentage = nn.Parameter(torch.tensor(cutoff_vals, requires_grad=True, dtype=torch.float32, device=self.device).reshape(1, self.cs_parser.num_weights))
        else:
            self.cutoff_percentage = torch.zeros(1, self.cs_parser.num_weights, dtype=torch.float32, device=self.device)
        
        
        # times is tensor of size (seq_len x num_weights)
        self.times = torch.tensor(np.transpose(np.tile(range(self.seq_len), (self.cs_parser.num_weights, 1))), device=self.device)
        
        
        self.lower_thresholds = nn.Parameter(torch.tensor(self.init_lower_thresholds_f(self.changing_dim), requires_grad=True, dtype=torch.float32, device=self.device))
        self.upper_thresholds = nn.Parameter(torch.tensor(self.init_upper_thresholds_f(self.changing_dim), requires_grad=True, dtype=torch.float32, device=self.device))
        
        self.thresh_temperature = self.temperature
        self.cutoff_percentage_temperature = self.temperature
        self.ever_measured_temperature = self.temperature
        
        self.activation_func = nn.Sigmoid()
        
        if self.use_summaries_for_atomics:
            # concat summaries to patient_batch during forward
            # in B x V x (T + Summaries)
            self.layer_time_to_atomics = LazyLinearWithMask(self.num_atomics) # in T_and_summaries
            # -> B x V x A
            self.flatten = nn.Flatten()
            # -> B x V*A
            self.layer_to_concepts = LazyLinearWithMask(self.num_concepts)
            # -> B x C
        
        elif not self.use_summaries_for_atomics:
            # in B x V x T
            self.layer_time_to_atomics = LazyLinearWithMask(self.num_atomics)
            # -> B x V x A
            self.flatten = nn.Flatten()
            # concat summaries to atomics during forward
            # -> B x (V*A + Summaries)
            self.layer_to_concepts = LazyLinearWithMask(self.num_concepts)
            # -> B x C
        
        self.layer_output = nn.Linear(self.num_concepts, self.output_dim)
        # B x Out
        
        self.regularized_layers = nn.ModuleList([self.layer_time_to_atomics, self.layer_to_concepts])
        
        self.ema_gradient = {}
        for name, param in self.named_parameters():
            self.ema_gradient[name] = None
        
        self.to(device=self.device)
        # self.deactivate_bottleneck_weights_if_top_k()
        return
    
    def forward(self, time_dependent_vars, indicators, static_vars):
        assert time_dependent_vars.dim() == 3 and time_dependent_vars.size(1) == self.seq_len and time_dependent_vars.size(2) == self.changing_dim
        assert indicators.shape == time_dependent_vars.shape
        assert torch.equal(static_vars, torch.empty(0, device=self.device)) or (static_vars.dim() == 2 and static_vars.size(1) == self.static_dim)
        
        if self.use_summaries:
            summaries = calculate_summaries(model=self, time_dependent_vars=time_dependent_vars, indicators=indicators, use_indicators=self.use_indicators)
        else:
            summaries = None
        
        if self.use_summaries_for_atomics:
            input = create_3d_input_as_b_v_t(time_dependent_vars=time_dependent_vars, indicators=indicators, static_vars=static_vars, summaries=summaries, use_indicators=self.use_indicators)
            
            atomics = self.layer_time_to_atomics(input)
            atomics = self.activation_func(atomics)
            flat = self.flatten(atomics)
            
            concepts = self.layer_to_concepts(flat)
            concepts = self.activation_func(concepts)
        
        elif not self.use_summaries_for_atomics:
            if self.use_indicators:
                input = torch.cat([time_dependent_vars, indicators], axis=1) # cat along time instead of var
            else:
                input = time_dependent_vars
            input = rearrange(input, "b t v -> b v t")
            
            atomics = self.layer_time_to_atomics(input)
            atomics = self.activation_func(atomics)
            flat = self.flatten(atomics)
            
            # concat activation, statics and summaries
            if self.use_summaries:
                summaries_2d = summaries.reshape(summaries.size(0), -1)
                atomics_statics_summaries = torch.cat([flat, static_vars, summaries_2d], axis=-1)
            else:
                atomics_statics_summaries = torch.cat([flat, static_vars], axis=-1)
            
            concepts = self.layer_to_concepts(atomics_statics_summaries)
            concepts = self.activation_func(concepts)
        
        input = self.layer_output(concepts)
        return input
    