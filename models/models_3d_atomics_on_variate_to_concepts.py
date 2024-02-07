
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import torch
import csv

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, cosine_similarity, normalize
from models.param_initializations import *
from models.EarlyStopping import EarlyStopping
from models.weights_parser import WeightsParser
from models.differentiable_summaries import calculate_summaries
from models.helper import plot_grad_flow, extract_to

from tqdm import tqdm
from time import sleep

import optuna
from optuna.trial import TrialState
from rtpt import RTPT
import random
import time
from enum import Enum
from einops import rearrange

class TaskType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


def add_all_parsers(parser:WeightsParser, changing_dim, static_dim = 0, seq_len = 1, use_indicators = True, str_type = 'linear'):
    if str_type == 'linear':
        time_feat_dim = 2 * changing_dim * seq_len + static_dim
        parser.add_shape(str(str_type) + '_time_', time_feat_dim)
        
    parser.add_shape(str(str_type) + '_mean_', changing_dim)
    parser.add_shape(str(str_type) + '_var_', changing_dim)
    if use_indicators:
        parser.add_shape(str(str_type) + '_ever_measured_', changing_dim)
        parser.add_shape(str(str_type) + '_mean_indicators_', changing_dim)
        parser.add_shape(str(str_type) + '_var_indicators_', changing_dim)
        parser.add_shape(str(str_type) + '_switches_', changing_dim)
    
    # slope_indicators are the same weights for all of the slope features.
    parser.add_shape(str(str_type) + '_slope_', changing_dim)
    
    if str_type == 'linear':
        parser.add_shape(str(str_type) + '_slope_stderr_', changing_dim)
        if use_indicators:
            parser.add_shape(str(str_type) + '_first_time_measured_', changing_dim)
            parser.add_shape(str(str_type) + '_last_time_measured_', changing_dim)
        
    parser.add_shape(str(str_type) + '_hours_above_threshold_', changing_dim)
    parser.add_shape(str(str_type) + '_hours_below_threshold_', changing_dim)
        

def create_state_dict(self, epoch):
    state = {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    }
        
    return state


class CBM(nn.Module):
    def __init__(self,
                static_dim,
                changing_dim, 
                seq_len,
                num_atomics,
                num_concepts,
                output_dim,
                use_summaries_for_atomics,
                use_indicators,
                use_fixes,
                use_grad_norm,
                noise_std,
                use_summaries,
                differentiate_cutoffs = True,
                init_cutoffs_f = init_cutoffs_to_50perc,
                init_lower_thresholds_f = init_rand_lower_thresholds, 
                init_upper_thresholds_f = init_rand_upper_thresholds,
                temperature = 0.1,
                opt_lr = 1e-4,
                opt_weight_decay = 0.,
                l1_lambda=0.,
                cos_sim_lambda=0.,
                top_k = '',
                top_k_num = 0,
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
        
        self.static_dim = static_dim
        self.changing_dim = changing_dim
        self.seq_len = seq_len
        self.num_concepts = num_concepts
        self.num_atomics = num_atomics
        self.num_summaries = 12 # number of calculated summaries
        
        self.use_summaries_for_atomics = use_summaries_for_atomics
        self.use_indicators = use_indicators
        self.use_fixes = use_fixes
        self.use_grad_norm = use_grad_norm
        self.noise_std = noise_std
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
        
        self.top_k = top_k
        self.top_k_num = top_k_num
        self.output_dim = output_dim
        self.task_type = task_type
        self.device = device
        
        self.create_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = opt_lr, weight_decay = opt_weight_decay)

    
    def create_model(self):
        self.sigmoid_layer = nn.Sigmoid()
        
        # activation function to convert output into probabilities
        # not needed during training as pytorch losses are optimized and include sigmoid / softmax
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim <= 2:
            self.output_af = nn.Sigmoid()
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 2:
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
            self.layer_time_to_atomics = nn.LazyLinear(self.num_atomics) # in T_and_summaries
            # -> B x V x A
            self.flatten = nn.Flatten()
            # -> B x V*A
            self.layer_to_concepts = nn.LazyLinear(self.num_concepts)
            # -> B x C
        
        elif not self.use_summaries_for_atomics:
            # in B x V x T
            self.layer_time_to_atomics = nn.LazyLinear(self.num_atomics)
            # -> B x V x A
            self.flatten = nn.Flatten()
            # concat summaries to atomics during forward
            # -> B x (V*A + Summaries)
            self.layer_to_concepts = nn.LazyLinear(self.num_concepts)
            # -> B x C
        
        self.layer_output = nn.Linear(self.num_concepts, self.output_dim)
        # B x Out
        
        self.deactivate_bottleneck_weights_if_top_k()
        return
        
    def deactivate_bottleneck_weights_if_top_k(self):
        if (self.top_k != ''):
            file = open(self.top_k)
            csvreader = csv.reader(file)
            header = next(csvreader)
            
            top_k_concepts = []
            top_k_inds = []
            i = 0
            
            layers_to_prune = [self.layer_time_to_atomics, self.layer_to_concepts] # layer id / seq needs to be the same as during greedy selection
            
            for row in csvreader:
                if (i < self.top_k_num):
                    top_k_concepts.append(int(row[1]))
                    top_k_concepts.append(int(row[2]))
                    top_k_inds.append(int(row[3]))
                    i+=1
                else:
                    break
            
            for i, layer in enumerate(layers_to_prune):
                condition = torch.zeros(layer.weight.shape, dtype=torch.bool)
                
                for concept_id, feat_id in zip(top_k_concepts, top_k_inds):
                    condition[concept_id][feat_id] = True
                    
                layer.weight = torch.nn.Parameter(layer.weight.where(condition, torch.tensor(0.0)))
        return
    
    def forward(self, time_dependent_vars, indicators, static_vars):
        assert time_dependent_vars.dim() == 3 and time_dependent_vars.size(1) == self.seq_len and time_dependent_vars.size(2) == self.changing_dim
        assert indicators.shape == time_dependent_vars.shape
        assert torch.equal(static_vars, torch.empty(0, device=self.device)) or (static_vars.dim() == 2 and static_vars.size(1) == self.static_dim)
        
        if self.use_indicators:
            cat = torch.cat([time_dependent_vars, indicators], axis=1) # cat along time instead of var
        else:
            cat = time_dependent_vars
        rearranged = rearrange(cat, "b t v -> b v t")
        
        if self.use_summaries_for_atomics:
            if not torch.equal(static_vars, torch.empty(0, device=self.device)):
                # static vars are concatinated along time, for each var, similar to summaries
                static_vars = static_vars.unsqueeze(1) # b x 1 x 8
                static_vars = static_vars.expand(-1, rearranged.size(1), -1) # -1 => unchanged
            
            if self.use_summaries:
                summaries = calculate_summaries(self, time_dependent_vars, indicators, self.use_indicators, self.use_fixes)
                summaries = [tensor.unsqueeze(-1) for tensor in summaries] # add time dim for cat
                
                patient_and_summaries = torch.cat([rearranged, static_vars] + summaries, axis=-1)
            else:
                patient_and_summaries = torch.cat([rearranged, static_vars], axis=-1)
            # print("patient_and_summaries", patient_and_summaries.shape)
            
            atomics = self.layer_time_to_atomics(patient_and_summaries)
            atomics = self.activation_func(atomics)
            # print("after atomics", atomics.shape)
            flat = self.flatten(atomics)
            # print("after flatten", flat.shape)
            
            concepts = self.layer_to_concepts(flat)
            concepts = self.activation_func(concepts)
            # print("after concepts", concepts.shape)
        
        elif not self.use_summaries_for_atomics:
            atomics = self.layer_time_to_atomics(rearranged)
            atomics = self.activation_func(atomics)
            # print("after atomics", atomics.shape)
            flat = self.flatten(atomics)
            # print("after flatten", flat.shape)
            
            # concat activation and summaries
            if self.use_summaries:
                summaries = calculate_summaries(self, time_dependent_vars, indicators, self.use_indicators, self.use_fixes)
                atmomics_and_summaries = torch.cat([flat, static_vars] + summaries, axis=-1)
            else:
                atmomics_and_summaries = torch.cat([flat, static_vars], axis=-1)
            # print("atmomics_and_summaries", atmomics_and_summaries.shape)
            
            concepts = self.layer_to_concepts(atmomics_and_summaries)
            concepts = self.activation_func(concepts)
            # print("after concepts", concepts.shape)
        
        out = self.layer_output(concepts)
        return out
    
    def forward_probabilities(self, *data):
        output = self(*data)
        return self.output_af(output)
    
    def predict(self, *data):
        probs = self.forward_probabilities(*data)
        return torch.argmax(probs, dim=1)
    
    def forward_one_step_ahead(self, *data, steps_ahead):
        predictions = []
        time_dependent_vars, indicators, static_vars = data # b t v
        
        for _ in range(steps_ahead):
            output = self(time_dependent_vars, indicators, static_vars) # b x v
            output = output.unsqueeze(1) # inflate to 3d
            predictions.append(output)
            
            # create indicators
            ind = torch.isfinite(output)
            
            # replace previous first with new last entry, to maintain seq length
            time_dependent_vars = torch.cat([time_dependent_vars[:, 1:, :], output], dim=1)
            indicators = torch.cat([indicators[:, 1:, :], ind], dim=1)
        
        return torch.cat(predictions, axis=1)
    
    def argmax_to_preds(self, y_probs):
        return torch.argmax(y_probs, dim=1)
        
    def _load_model(self, path, print_=True):
        """
        Args:
            path (str): filepath to the model
        """
        try:
            checkpoint = torch.load(path)
        except:
            return
        
        self.load_state_dict(checkpoint['model_state_dict'])
        if print_:
            print("Loaded model from " + path)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.curr_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        self.earlyStopping.best_state = checkpoint
        self.earlyStopping.min_max_criterion = min(checkpoint['val_losses'])
        
        if (self.top_k != ''):
            file = open(self.top_k)
            csvreader = csv.reader(file)
            header = next(csvreader)
            top_k_inds = []
            top_k_concepts = []
            i = 0
            for row in csvreader:
                if (i<self.top_k_num):
                    top_k_inds.append(int(row[2]))
                    top_k_concepts.append(int(row[1]))
                    i+=1
                else:
                    break
            condition = torch.zeros(self.bottleneck.weight.shape, dtype=torch.bool, device=self.device)
            for i in range(len(top_k_inds)):
                condition[top_k_concepts[i]][top_k_inds[i]]=True
            self.bottleneck.weight = torch.nn.Parameter(self.bottleneck.weight.where(condition, torch.tensor(0.0, device=self.device)))
        sleep(0.5)
        
        return checkpoint.get("early_stopping", False)

    def fit(self, train_loader, val_loader, p_weight, save_model_path, max_epochs=10000, save_every_n_epochs=10, patience=10, warmup_epochs=0, scheduler=None, trial=None, show_grad=False):
        """
        Args:
            train_loader (torch.DataLoader): 
            val_tensor (torch.DataLoader):
            p_weight (tensor): weight parameter used to calculate BCE loss 
            save_model_path (str): filepath to save the model progress
            epochs (int): number of epochs to train
        """
        
        rtpt = RTPT(name_initials='KA', experiment_name='TimeSeriesCBM', max_iterations=max_epochs)
        rtpt.start()
        
        self.train_losses = []
        self.val_losses = []
        self.curr_epoch = -1
        
        self.earlyStopping = EarlyStopping(patience=patience, warmup_epochs=warmup_epochs)
        early_stopped = self._load_model(save_model_path)
        
        if early_stopped:
            return
        
        epochs = range(self.curr_epoch+1, max_epochs)
        
        with tqdm(total=len(epochs), unit=' epoch') as pbar:
            
            for epoch in epochs:
                self.train()
                train_loss = 0
            
                ### Train loop
                for batch in train_loader:
                    X_time, X_ind, X_static, y = extract_to(batch, self.device)
                    
                    if self.noise_std:
                        X_time = X_time + torch.normal(mean=0, std=self.noise_std, size=X_time.shape).to(device=self.device)
                        
                    y_pred = self(X_time, X_ind, X_static)

                    loss = self.compute_loss(y, y_pred, p_weight)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    if self.use_grad_norm:
                        self.normalize_gradient_(self.parameters(), self.use_grad_norm)

                    train_loss += loss * X_time.size(0)
                    
                    
                    # for name, param in self.named_parameters():
                    #     isnan = param.grad is None or torch.isnan(param.grad).any()
                    #     if isnan:
                    #         print(f"Parameter: {name}, Gradient NAN? {isnan}")
                    #         return
                    
                    if show_grad:
                        plot_grad_flow(self.named_parameters())
                    
                    # if (self.top_k != ''): # freeze parameters
                    #     self.bottleneck.weight.grad.fill_(0.)
            
                    self.optimizer.step()
                
                train_loss = train_loss / len(train_loader.sampler)
                
                
                if (epoch % save_every_n_epochs) == 0:
                    self.eval()
                    with torch.no_grad():
                        
                        self.train_losses.append(train_loss.item())
                        
                        ### Validation loop
                        val_loss = 0
                        for batch in val_loader:
                            X_time, X_ind, X_static, y = extract_to(batch, self.device)
                            
                            y_pred = self(X_time, X_ind, X_static)

                            val_loss += self.compute_loss(y, y_pred, p_weight) * X_time.size(0)
                        
                        val_loss = val_loss / len(val_loader.sampler)
                        self.val_losses.append(val_loss.item())
                        
                        
                        ### Auxilliary stuff
                        state = create_state_dict(self, epoch)
                        
                        if self.earlyStopping.check_improvement(val_loss, state):
                            break
                        
                        if scheduler:
                            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                scheduler.step(val_loss)
                            else:
                                scheduler.step()
                        
                        if save_model_path:
                            torch.save(state, save_model_path)
                        
                        if trial:
                            trial.report(val_loss, epoch)
                            
                            if trial.should_prune():
                                raise optuna.exceptions.TrialPruned()
                
                pbar.set_postfix({'Train Loss': f'{train_loss.item():.5f}', 'Val Loss': f'{self.val_losses[-1]:.5f}', "Best Val Loss": f'{self.earlyStopping.min_max_criterion:.5f}'})
                pbar.update()
                rtpt.step(subtitle=f"loss={train_loss:2.2f}")
            
        if save_model_path and self.earlyStopping.best_state:
            torch.save(self.earlyStopping.best_state, save_model_path)
        
        # print("cutoff_percentage after", self.cutoff_percentage.round(decimals=3))
        # print("lower_thresholds after", self.lower_thresholds.round(decimals=3))
        # print("upper_thresholds after", self.upper_thresholds.round(decimals=3))
        
        return self.val_losses[-1]
    
    def normalize_gradient_(self, params, norm_type, p_norm_type=2):
        for param in params:
            if norm_type == "FULL" or torch.squeeze(param.grad).dim() < 2:
                param.grad = normalize(param.grad, p=p_norm_type, dim=None)
            elif norm_type == "COMPONENT_WISE":
                param.grad = normalize(param.grad, p=p_norm_type, dim=0)
            
        return

    def compute_loss(self, yb, y_pred, p_weight):
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim <= 2:
            task_loss = binary_cross_entropy_with_logits(y_pred, yb.float(), pos_weight = p_weight)
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 2:
            task_loss = cross_entropy(y_pred, yb, weight = p_weight)
        elif self.task_type == TaskType.REGRESSION:
            task_loss = mse_loss(y_pred, yb)
        else:
            raise NotImplementedError("Loss not defined!")
        
        # Lasso regularization
        l1_loss = 0
        if self.l1_lambda != 0:
            L1_norm = torch.norm(self.layer_to_concepts.weight, 1) + torch.norm(self.layer_time_to_atomics.weight, 1)
            l1_loss = self.l1_lambda * L1_norm
        
        # Cosine_similarity regularization
        cos_sim_loss = 0
        if self.cos_sim_lambda != 0:
            cos_sim = self.cos_sim(self.layer_to_concepts) + self.cos_sim(self.layer_time_to_atomics)
            cos_sim_loss = self.cos_sim_lambda * cos_sim
        
        return task_loss + l1_loss + cos_sim_loss

    def cos_sim(self, layer):
        if layer.out_features == 1:
            return 0
        
        concepts = torch.arange(layer.out_features, device=self.device)
        indices = torch.combinations(concepts, 2)
        
        weights = layer.weight[indices]
        cos_sim = torch.abs(cosine_similarity(weights[:, 0], weights[:, 1], dim=1)).sum()
        
        return cos_sim
