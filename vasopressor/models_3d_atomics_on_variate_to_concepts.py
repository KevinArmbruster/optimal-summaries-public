from weights_parser import WeightsParser

import numpy as np
import pandas as pd
import pickle
import torch
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, cosine_similarity
from custom_losses import LSTM_compound_loss, custom_bce_horseshoe
from param_initializations import set_seed, init_cutoffs_to_small, init_cutoffs_to_50perc, init_rand_lower_thresholds, init_rand_upper_thresholds
from torchmetrics import AUROC, Accuracy, MeanSquaredError

from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from EarlyStopping import EarlyStopping

from tqdm import tqdm
from time import sleep

from itertools import combinations

import optuna
from optuna.trial import TrialState
from rtpt import RTPT
import random
import time
from enum import Enum
from einops import rearrange

import debug_grad_graph

class TaskType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


def add_all_parsers(parser:WeightsParser, changing_dim, static_dim = 0, seq_len = 1, str_type = 'linear'):
    if str_type == 'linear':
        time_feat_dim = 2 * changing_dim * seq_len + static_dim
        parser.add_shape(str(str_type) + '_time_', time_feat_dim)
        
    parser.add_shape(str(str_type) + '_mean_', changing_dim)
    parser.add_shape(str(str_type) + '_var_', changing_dim)
    parser.add_shape(str(str_type) + '_ever_measured_', changing_dim)
    parser.add_shape(str(str_type) + '_mean_indicators_', changing_dim)
    parser.add_shape(str(str_type) + '_var_indicators_', changing_dim)
    parser.add_shape(str(str_type) + '_switches_', changing_dim)
    
    # slope_indicators are the same weights for all of the slope features.
    parser.add_shape(str(str_type) + '_slope_', changing_dim)
    
    if str_type == 'linear':
        parser.add_shape(str(str_type) + '_slope_stderr_', changing_dim)
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
                 input_dim, 
                 changing_dim, 
                 seq_len,
                 num_atomics,
                 use_summaries_for_atomics,
                 num_concepts,
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
                 output_dim = 2,
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
        
        self.input_dim = input_dim
        self.changing_dim = changing_dim
        self.static_dim = input_dim - 2 * changing_dim
        self.seq_len = seq_len
        self.num_concepts = num_concepts
        self.num_atomics = num_atomics
        self.use_summaries_for_atomics = use_summaries_for_atomics
        self.num_summaries = 12 # number of calculated summaries in function encode_patient_batch
        
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
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim == 2:
            self.output_af = nn.Sigmoid()
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 2:
            self.output_af = nn.Softmax(dim=1)
        elif self.task_type == TaskType.REGRESSION:
            self.output_af = nn.Identity()


        self.weight_parser = WeightsParser()
        self.cs_parser = WeightsParser()
        add_all_parsers(self.weight_parser, self.changing_dim, self.static_dim, self.seq_len)
        add_all_parsers(self.cs_parser, self.changing_dim, str_type = 'cs')
        
        
        # Initialize cutoff_times to by default use all of the timesteps.
        self.cutoff_percentage = torch.zeros(1, self.cs_parser.num_weights, device=self.device)
        
        if self.differentiate_cutoffs:
            cutoff_vals = self.init_cutoffs_f(self.cs_parser.num_weights)
            self.cutoff_percentage = nn.Parameter(torch.tensor(cutoff_vals, requires_grad=True, device=self.device).reshape(1, self.cs_parser.num_weights))

        # times is tensor of size (seq_len x num_weights)
        self.times = torch.tensor(np.transpose(np.tile(range(self.seq_len), (self.cs_parser.num_weights, 1))), device=self.device)
        
        
        self.lower_thresholds = nn.Parameter(torch.tensor(self.init_lower_thresholds_f(self.changing_dim), requires_grad=True, device=self.device))
        self.upper_thresholds = nn.Parameter(torch.tensor(self.init_upper_thresholds_f(self.changing_dim), requires_grad=True, device=self.device))
        
        self.thresh_temperature = self.temperature
        self.cutoff_percentage_temperature = self.temperature
        self.ever_measured_temperature = self.temperature
        
        self.atomic_activation_func = nn.Sigmoid()
        self.concept_activation_func = nn.Sigmoid()
        
        if self.use_summaries_for_atomics:
            # concat summaries to patient_batch during forward
            # in B x V x (T + Summaries)
            T_and_summaries = 2 * self.seq_len + self.num_summaries
            self.layer_time_to_atomics = nn.Linear(T_and_summaries, self.num_atomics)
            # -> B x V x A
            self.flatten = nn.Flatten()
            # -> B x V*A
            self.layer_to_concepts = nn.Linear(self.changing_dim * self.num_atomics, self.num_concepts)
            # -> B x C
        
        elif not self.use_summaries_for_atomics:
            # in B x V x T
            self.layer_time_to_atomics = nn.Linear(2 * self.seq_len, self.num_atomics)
            # -> B x V x A
            self.flatten = nn.Flatten()
            # concat summaries to atomics during forward
            # -> B x (V*A + Summaries)
            self.layer_to_concepts = nn.Linear(self.num_atomics * self.changing_dim + self.num_summaries * self.changing_dim, self.num_concepts)
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
            top_k_inds = []
            top_k_concepts = []
            i = 0
            for row in csvreader:
                if (i <self.top_k_num):
                    top_k_inds.append(int(row[2]))
                    top_k_concepts.append(int(row[1]))
                    i+=1
                else:
                    break
            condition = torch.zeros(self.bottleneck.weight.shape, dtype=torch.bool) #device=self.device # during init still on cpu
            for i in range(len(top_k_inds)):
                condition[top_k_concepts[i]][top_k_inds[i]]=True
            self.bottleneck.weight = torch.nn.Parameter(self.bottleneck.weight.where(condition, torch.tensor(0.0))) #device=self.device # during init still on cpu
        return
    
    def calculate_summaries(self, patient_batch, epsilon_denom=1e-8):
        # Computes the encoding (s, x) + (weighted_summaries) in the order defined in weight_parser.
        # Returns pre-sigmoid P(Y = 1 | patient_batch)
        temperatures = torch.tensor(np.full((1, self.cs_parser.num_weights), self.cutoff_percentage_temperature), device=self.device)
        
        # Get changing variables
        batch_changing_vars = patient_batch[:, :, :self.changing_dim]
        batch_measurement_ind = patient_batch[:, :, self.changing_dim: self.changing_dim * 2]
        batch_static_vars = patient_batch[:, 0, self.changing_dim * 2:] # static is the same accross time
        
        cutoff = ((self.seq_len+1) * torch.clip(self.cutoff_percentage, 0, 1)) - 1 # range [-1, seq_len], enables flexibility to use full or no time series values, without big bias
        weight_vector = self.sigmoid_layer((self.times - cutoff) / temperatures).reshape(1, self.seq_len, self.cs_parser.num_weights)
        
        
        # MEAN FEATURES
        # Calculate \sum_t (w_t * x_t * m_t)
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_mean_']
        mean_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_average = torch.sum(mean_weight_vector * (batch_changing_vars * batch_measurement_ind), dim=1)

        mean_feats = weighted_average / (torch.sum(mean_weight_vector, dim=1) + epsilon_denom)
        # TODO denom forgot batch_measurement_ind
        # mean_feats = weighted_average / (torch.sum(mean_weight_vector * batch_measurement_ind, dim=1) + epsilon_denom)
        
        
        # VARIANCE FEATURES
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_var_']
        var_weight_vector = weight_vector[:, :, start_i : end_i]
        
        x_mean = torch.mean(batch_measurement_ind * batch_changing_vars, dim=1, keepdim=True)
        # TODO x bar is not normal mean, but divide by sum of M
        # x_mean = torch.sum(batch_measurement_ind * batch_changing_vars, dim=1, keepdim=True) / torch.sum(batch_measurement_ind, dim=1, keepdim=True)

        weighted_variance = torch.sum(batch_measurement_ind * var_weight_vector * (batch_changing_vars - x_mean)**2, dim=1)
        
        squared_sum_of_weights = torch.sum(batch_measurement_ind * var_weight_vector, dim=1)**2
        sum_of_squared_weights = torch.sum(batch_measurement_ind * var_weight_vector ** 2, dim=1)
        
        normalizing_term = squared_sum_of_weights / (squared_sum_of_weights + sum_of_squared_weights + epsilon_denom)
        var_feats = weighted_variance / (normalizing_term + epsilon_denom)
        # TODO should be * not /
        # var_feats = normalizing_term * weighted_variance
        
    	
        # INDICATOR FOR EVER BEING MEASURED
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_ever_measured_']
        ever_measured_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_ind_average = torch.sum(ever_measured_weight_vector * batch_measurement_ind, dim=1)
        pre_sigmoid = weighted_ind_average / (self.ever_measured_temperature * torch.sum(ever_measured_weight_vector, dim=1) + epsilon_denom)
        ever_measured_feats = self.sigmoid_layer(pre_sigmoid) - 0.5
        
        
        # MEAN OF INDICATOR SEQUENCE
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_mean_indicators_']
        mean_ind_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_ind_average = torch.sum(mean_ind_weight_vector * batch_measurement_ind, dim=1)
        mean_ind_feats = weighted_ind_average / (torch.sum(mean_ind_weight_vector, dim=1) + epsilon_denom)
        
        
        # VARIANCE OF INDICATOR SEQUENCE
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_var_indicators_']
        var_ind_weight_vector = weight_vector[:, :, start_i : end_i]
                
        x_mean_ind = torch.mean(batch_measurement_ind, dim=1, keepdim=True)
        weighted_variance_ind = torch.sum(var_ind_weight_vector * (batch_measurement_ind - x_mean_ind)**2, dim=1)     
           
        squared_sum_of_weights = torch.sum(var_ind_weight_vector, dim=1)**2
        sum_of_squared_weights = torch.sum(var_ind_weight_vector ** 2, dim=1)
        normalizing_term = squared_sum_of_weights / (squared_sum_of_weights + sum_of_squared_weights + epsilon_denom)
        
        var_ind_feats = weighted_variance_ind / (normalizing_term + epsilon_denom)
        # TODO should be * not /
        # var_ind_feats = normalizing_term * weighted_variance_ind
        
        
        # COUNT OF SWITCHES
        # Compute the number of times the indicators switch from missing to measured, or vice-versa.
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_switches_']
        switches_weight_vector = weight_vector[:, :, start_i : end_i][:, :-1, :]
        
        # Calculate m_{n t + 1} - m_{ n t}
        # Sum w_t + sigmoids of each difference
        later_times = batch_changing_vars[:, 1:, :]
        earlier_times = batch_changing_vars[:, :-1, :]
        
        switch_feats = torch.sum(switches_weight_vector * torch.abs(later_times - earlier_times), dim=1) / (torch.sum(switches_weight_vector, dim=1) + epsilon_denom)
        
        
        # FIRST TIME MEASURED
        # LAST TIME MEASURED
        
        # For each variable in the batch, compute the first time it was measured.
        # Set equal to -1 if never measured.
        
        # For each feature, calculate the first time it was measured
        # Index of the second dimension of the indicators

        mask_max_values, mask_max_indices = torch.max(batch_measurement_ind, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = -1
        
        first_time_feats = mask_max_indices / float(batch_measurement_ind.shape[1])
        
        # Last time measured is the last index of the max.
        # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
        flipped_batch_measurement_indicators = torch.flip(batch_measurement_ind, [1])
        
        mask_max_values, mask_max_indices = torch.max(flipped_batch_measurement_indicators, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = batch_measurement_ind.shape[1]
        
        last_time_feats = (float(batch_measurement_ind.shape[1]) - mask_max_indices) / float(batch_measurement_ind.shape[1])
        
        # SLOPE OF L2
        # STANDARD ERROR OF L2     
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_slope_']
        slope_weight_vector = weight_vector[:, :, start_i : end_i]
        
        # Zero out the batch_changing_vars so that they are zero if the features are not measured.
        linreg_y = batch_changing_vars * batch_measurement_ind
        
        # The x-values for this linear regression are the times.
        # Zero them out so that they are zero if the features are not measured.
        linreg_x = torch.tensor(np.transpose(np.tile(range(self.seq_len), (self.changing_dim, 1))), device=self.device)
        linreg_x = linreg_x.repeat(linreg_y.shape[0], 1, 1) * batch_measurement_ind
        
        # Now, compute the slope and standard error.
        weighted_x = torch.unsqueeze(torch.sum(slope_weight_vector * linreg_x, dim = 1) / (torch.sum(slope_weight_vector, dim = 1) + epsilon_denom), 1)
        weighted_y = torch.unsqueeze(torch.sum(slope_weight_vector * linreg_y, dim = 1) / (torch.sum(slope_weight_vector, dim = 1) + epsilon_denom), 1)
        
        slope_num = torch.sum(slope_weight_vector * (linreg_x - weighted_x) * (linreg_y - weighted_y), dim=1)
        slope_den = torch.sum(slope_weight_vector * (linreg_x - weighted_x)**2, dim =1)
        
        slope_feats = slope_num / (slope_den + epsilon_denom)
        
        # If the denominator is zero, set the feature equal to 0.
        var_denom = torch.sum(slope_weight_vector * (linreg_x - weighted_x)**2, dim=1)
        slope_stderr_feats = 1 / (var_denom + epsilon_denom)
        
        slope_stderr_feats = torch.where(var_denom > 0, slope_stderr_feats, var_denom)
        # TODO slope_weight_vector could be negative, just check for 0
        # slope_stderr_feats = torch.where(var_denom == 0, slope_stderr_feats, var_denom)
        
        
        # HOURS ABOVE THRESHOLD
        # HOURS BELOW THRESHOLD
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_hours_above_threshold_']
        above_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_hours_below_threshold_']
        below_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
        
        upper_features = self.sigmoid_layer((batch_changing_vars - self.upper_thresholds) / self.thresh_temperature)
        lower_features = self.sigmoid_layer((self.lower_thresholds - batch_changing_vars) / self.thresh_temperature)
        
        # sum upper_features and lower_features across timesteps
        above_tmp = batch_measurement_ind * above_thresh_weight_vector
        above_threshold_feats = torch.sum(above_tmp * upper_features, dim=1) / (torch.sum(above_tmp, dim=1) + epsilon_denom)
        
        below_tmp = batch_measurement_ind * below_thresh_weight_vector
        below_threshold_feats = torch.sum(below_tmp * lower_features, dim=1) / (torch.sum(below_tmp, dim=1) + epsilon_denom)
        
        
        # return summaries
        summaries = [mean_feats.float(), var_feats.float(), 
                     ever_measured_feats.float(), mean_ind_feats.float(), var_ind_feats.float(), switch_feats.float(), 
                     slope_feats.float(), slope_stderr_feats.float(), 
                     first_time_feats.float(), last_time_feats.float(), 
                     above_threshold_feats.float(), below_threshold_feats.float()]
        # print("summaries", len(summaries))
        
        # print("mean_feats", summaries[0].shape) # torch.Size([512, 7])
        # print("var_feats", summaries[1].shape) # torch.Size([512, 7])
        # print("cat", cat.shape)

        return batch_changing_vars, batch_measurement_ind, batch_static_vars, summaries
    
    def forward(self, patient_batch, epsilon_denom=1e-8):
        
        batch_changing_vars, batch_measurement_indicators, batch_static_vars, summaries = self.calculate_summaries(patient_batch, epsilon_denom)
        
        if self.use_summaries_for_atomics:
            # concat summaries along time
            cat = torch.cat([batch_changing_vars, batch_measurement_indicators], axis=1) # cat along time instead of var
            rearranged = rearrange(cat, "b t v -> b v t")
            summaries = [tensor.unsqueeze(-1) for tensor in summaries] # add time dim for cat
            
            patient_and_summaries = torch.cat([rearranged] + summaries, axis=-1)
            # print("patient_and_summaries", patient_and_summaries.shape)
            
            atomics = self.layer_time_to_atomics(patient_and_summaries)
            atomics = self.atomic_activation_func(atomics) # relu?
            # print("after atomics", atomics.shape)
            flat = self.flatten(atomics)
            # print("after flatten", flat.shape)
            
            concepts = self.layer_to_concepts(flat)
            concepts = self.concept_activation_func(concepts)
            # print("after concepts", concepts.shape)
        
        elif not self.use_summaries_for_atomics:
            cat = torch.cat([batch_changing_vars, batch_measurement_indicators], axis=1) # cat along time instead of var
            rearranged = rearrange(cat, "b t v -> b v t")
            # print("rearranged", rearranged.shape)
            
            atomics = self.layer_time_to_atomics(rearranged)
            atomics = self.atomic_activation_func(atomics) # relu?
            # print("after atomics", atomics.shape)
            flat = self.flatten(atomics)
            # print("after flatten", flat.shape)
            
            # concat activation and summaries
            atmomics_and_summaries = torch.cat([flat] + summaries, axis=-1)
            # print("atmomics_and_summaries", atmomics_and_summaries.shape)
            
            concepts = self.layer_to_concepts(atmomics_and_summaries)
            concepts = self.concept_activation_func(concepts)
            # print("after concepts", concepts.shape)
        
        out = self.layer_output(concepts)
        return out
    
    def forward_probabilities(self, patient_batch):
        output = self(patient_batch)
        return self.output_af(output)
    
    def predict(self, patient_batch):
        probs = self.forward_probabilities(patient_batch)
        return torch.argmax(probs, dim=1)
    
    def get_num_concepts(self):
        return self.num_concepts
    
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
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(self.device), yb.to(self.device)
                    y_pred = self(Xb)

                    loss = self.compute_loss(yb, y_pred, p_weight)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    train_loss += loss * Xb.size(0)
                    
                    
                    if (self.top_k != ''):
                        self.bottleneck.weight.grad.fill_(0.)
            
                    self.optimizer.step()
                    
                    if show_grad:
                        plot_grad_flow(self.named_parameters())
                
                train_loss = train_loss / len(train_loader.sampler)
                
                
                if (epoch % save_every_n_epochs) == 0:
                    self.eval()
                    with torch.no_grad():
                        
                        self.train_losses.append(train_loss.item())
                        
                        ### Validation loop
                        val_loss = 0
                        for Xb, yb in val_loader:
                            Xb, yb = Xb.to(self.device), yb.to(self.device)
                
                            # Forward pass.
                            y_pred = self(Xb)

                            val_loss += self.compute_loss(yb, y_pred, p_weight) * Xb.size(0)
                        
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
        
        print("cutoff_percentage after", self.cutoff_percentage.round(decimals=3))
        print("lower_thresholds after", self.lower_thresholds.round(decimals=3))
        print("upper_thresholds after", self.upper_thresholds.round(decimals=3))
        
        return self.val_losses[-1]

    def compute_loss(self, yb, y_pred, p_weight):                
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim == 2:
            task_loss = binary_cross_entropy_with_logits(y_pred, yb, pos_weight = p_weight)
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 2:
            task_loss = cross_entropy(y_pred, yb, weight = p_weight)
        elif self.task_type == TaskType.REGRESSION:
            task_loss = mse_loss(y_pred, yb)
        else:
            print("Loss not defined!")
            exit()
        
        # Lasso regularization
        L1_norm = torch.norm(self.layer_to_concepts.weight, 1)
        l1_loss = self.l1_lambda * L1_norm
        
        # Cosine_similarity regularization
        cos_sim_loss = 0
        if self.num_concepts != 1:
            concepts = torch.arange(self.num_concepts, device=self.device)
            indices = torch.combinations(concepts, 2)  # Generate all combinations of concept indices

            weights = self.layer_to_concepts.weight[indices]  # Extract corresponding weight vectors
            cos_sim = torch.abs(cosine_similarity(weights[:, 0], weights[:, 1], dim=1)).sum()
            
            cos_sim_loss = self.cos_sim_lambda * cos_sim
        
        return task_loss + l1_loss + cos_sim_loss

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad): #and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(right=len(ave_grads)) # left=0, 
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.yscale("log")
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
    plt.show()
