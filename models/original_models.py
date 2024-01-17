
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import pickle
import torch
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, cosine_similarity
from models.custom_losses import LSTM_compound_loss, custom_bce_horseshoe
from models.param_initializations import set_seed, init_cutoffs_to_zero, init_rand_lower_thresholds, init_rand_upper_thresholds
from torchmetrics import AUROC, Accuracy, MeanSquaredError

from sklearn.metrics import roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from models.EarlyStopping import EarlyStopping
from models.weights_parser import WeightsParser

from tqdm import tqdm
from time import sleep

import optuna
from optuna.trial import TrialState
from rtpt import RTPT
import random
import time
from enum import Enum
from einops import rearrange

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
        
        
class LSTM_NextState(nn.Module):
    def __init__(self, input_dim, changing_dim, lstm_hidden_dim, hidden_next_state_dim, hidden_outcome_dim):        
        super(LSTM_NextState, self).__init__()
        
        self.input_dim = input_dim
        self.changing_dim = changing_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.static_dim = self.input_dim - (2 * self.changing_dim)
        
        self.hidden_next_state_dim = hidden_next_state_dim
        self.hidden_outcome_dim = hidden_outcome_dim
        
        self.sigmoid = nn.Sigmoid()
        
        # The LSTM takes patient histories as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.changing_dim, self.lstm_hidden_dim)
        
        # Two NN layers that map from hidden state space to prediction space
        # Obj: Predict next state.
        self.hidden2tag1 = nn.Linear(self.lstm_hidden_dim, self.hidden_next_state_dim)
        self.relu1 = nn.ReLU()
        self.hidden2tag2 = nn.Linear(self.hidden_next_state_dim, self.changing_dim)
        
        # Two NN layers that map from hidden state space to outcome space
        # Obj: Predict a binary outcome (Logit 2D)
        self.hidden2outcome1 = nn.Linear(self.lstm_hidden_dim + self.input_dim, self.hidden_outcome_dim)
        self.relu2 = nn.ReLU()
        self.hidden2outcome2 = nn.Linear(self.hidden_outcome_dim, 2)
            
    def forward(self, patient_batch):
        # Get changing variables.
        feats_time_5 = patient_batch[:, 5, :]
        
        batch_changing_vars = patient_batch[:, :5, :self.changing_dim]
        batch_measurement_indicators = patient_batch[:, :5, self.changing_dim: self.changing_dim * 2]
        
        lstm_out, _ = self.lstm(batch_changing_vars)   
        
        # Use all of the hidden states to predict the next state.
        tag_space_linear = self.hidden2tag1(lstm_out)
        re_tag = self.relu1(tag_space_linear)
        tag_space = self.hidden2tag2(re_tag)
        
        # Use the last LSTM hidden state, concatenated with the static 
        # variables to predict the outcomes.
        
        outcome_cov = torch.cat((lstm_out[:, -1, :].float(), feats_time_5), axis=1)
        
        out_space_linear = self.hidden2outcome1(outcome_cov)
        re_out = self.relu2(out_space_linear)
        out_space = self.hidden2outcome2(re_out)
        
        # return next state predictions, outcome logits, hidden state, and pre-outcome layer activations
        return tag_space, out_space, lstm_out, out_space_linear
 

class LSTM_Baseline(nn.Module):
    """
    Class to track training for a baseline LSTM.
    """

    def __init__(self, input_dim, changing_dim, lstm_hidden_dim, hidden_next_state_dim, hidden_outcome_dim, alpha = 1e-4, beta = 10., opt_lr = 1e-4, opt_weight_decay = 1e-4, batch_size=256):
        """Initializes the LSTM.
        
        Args:
            input_dim (int): number of input dimensions
            changing_dim (int): number of non-static input dimensions
            lstm_hidden_dim (int): number of hidden states for LSTM output
            hidden_next_state_dim (int): number of nodes in the linear layer mapping to predict the next state
            hidden_outcome_dim (int): number of nodes in the linear layer mapping to predict the outcome of interest
            alpha (float): coefficient of the classification loss term in the compound loss function
            beta (float): coefficient of the L2 regularization penalty in the compound loss function
            opt_lr (float): learning rate for the optimizer
            opt_weight_decay (float): weight decay for the optimizer
            batch_size (int): batch size for the optimizer
        """
        super(LSTM_Baseline, self).__init__()
        
        self.input_dim = input_dim
        self.changing_dim = changing_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_next_state_dim = hidden_next_state_dim
        self.hidden_outcome_dim = hidden_outcome_dim
        
        self.model = LSTM_NextState(input_dim, 
                                    changing_dim, 
                                    lstm_hidden_dim, 
                                    hidden_next_state_dim, 
                                    hidden_outcome_dim)
        
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= opt_lr, weight_decay = opt_weight_decay)
        
    def _load_model(self, path, print_=True):
        """Loads pre-trained model from checkpoint
        
        Args:
            path (str): location of the model
        """
        try:
            checkpoint = torch.load(path)
        except:
            return
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if print_:
            print("Loaded model from " + path)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.curr_epoch = checkpoint['epoch']
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
        if 'train_aucs' in checkpoint:
            self.train_aucs = checkpoint['train_aucs']
            self.val_aucs = checkpoint['val_aucs']
        sleep(0.5)
        
    def calculate_loss(self, dataset, p_weight):
        for batch, data in enumerate(dataset):
            Xb, yb = data

            X_t1_true = Xb[:, 1:, :self.changing_dim]

            # Forward pass.
            self.model.zero_grad()
            X_t1_pred, y_pred, _, _ = self.model(Xb)

            # Get the mask from Xb.
            # The mask is for X_t1_true.
            mask = Xb[:, 1:, self.changing_dim: self.changing_dim * 2] > 0

        return (self.loss_func(X_t1_true, X_t1_pred, yb, y_pred, mask, self.model, p_weight, self.beta, self.alpha) / Xb.shape[0]).item()
    
    def calculate_auc(self, dataset):
        auc = -1 
        for batch, data in enumerate(dataset):
            Xb, yb = data

            X_t1_true = Xb[:, 1:, :self.changing_dim]

            # Forward pass.
            self.model.zero_grad()
            X_t1_pred, y_pred, _, _ = self.model(Xb)

            y_hats = self.model.sigmoid(y_pred)[:,1].detach().cpu().numpy()

            if np.isnan(y_hats).any() == False:
                auc = roc_auc_score(yb.detach().cpu().numpy()[:, 1], y_hats)
            
        return auc
        

    def fit(self, train_dataset, val_dataset, p_weight, save_model_path, epochs=10000, save_every_n_epochs=100):
        """
        
        Args:
            train_dataset (torch.utils.data.TensorDataset): 
            val_dataset (torch.utils.data.TensorDataset):
            p_weight (tensor): weight parameter used to calculate BCE loss 
            save_model_path (str): filepath to save the model progress
            epochs (int): number of epochs to train
        """
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        train_full_data_loader = DataLoader(train_dataset, batch_size=train_dataset.__len__(), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size = val_dataset.__len__(), shuffle=True, num_workers=0)
        
        self.loss_func = LSTM_compound_loss
        
        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []
        
        self.curr_epoch = -1
        
        self._load_model(save_model_path)
        
        epoch_loss = 0
        
        for epoch in tqdm(range(self.curr_epoch+1, epochs)):
            if (epoch % save_every_n_epochs) == (-1 % save_every_n_epochs):
                # Append loss to loss array
                
                # Calculate train loss.
                self.train_losses.append(self.calculate_loss(train_full_data_loader, p_weight))
                self.train_aucs.append(self.calculate_auc(train_full_data_loader))
                
                self.val_losses.append(self.calculate_loss(val_loader, p_weight))
                self.val_aucs.append(self.calculate_auc(val_loader))
                
                # Save train and test set AUCs.
                y_hat_train = []

                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_losses': self.train_losses,
                            'val_losses': self.val_losses,
                            'train_aucs': self.train_aucs,
                            'val_aucs': self.val_aucs
                            }, save_model_path)
            
            epoch_loss = 0
            num_points = 0
            for batch, data in enumerate(train_loader):

                Xb, yb = data
                num_points += Xb.shape[0]
                # Get the changing dimensions for time-steps 1 through 6.
                X_t1_true = Xb[:, 1:, :self.changing_dim]

                # Only include time-steps 0 through 5 for the forward pass.
                # Forward pass.
                self.model.zero_grad()
                X_t1_pred, y_pred, _, _ = self.model(Xb)

                # Get the mask from Xb.
                # The mask is for X_t1_true.
                mask = Xb[:, 1:, self.changing_dim: self.changing_dim * 2] > 0

                loss = self.loss_func(X_t1_true, X_t1_pred, yb, y_pred, mask, self.model, p_weight, self.beta, self.alpha)
                loss.backward()

                epoch_loss += loss.item()

                # update all parameters
                self.optimizer.step()
                
############################

        
class LogisticRegressionWithSummaries(nn.Module):
    def __init__(self, 
                 input_dim, 
                 changing_dim, 
                 num_cutoff_times, 
                 differentiate_cutoffs,
                 init_cutoffs, 
                 init_lower_thresholds, 
                 init_upper_thresholds,
                 cutoff_times_temperature = 1.0,
                 cutoff_times_init_values = None,
                 thresholds_temperature = 0.1,
                 ever_measured_temperature = 0.1,
                 switch_temperature = 0.1,
                 seq_len = 6,
                 top_k = ''
                 ):
        """Initializes the LogisticRegressionWithSummaries.
        
        Args:
            input_dim (int): number of input dimensions
            changing_dim (int): number of non-static input dimensions
            num_cutoff_tines (int): number of cutoff-time parameters
            differentiate_cutoffs (bool): indicator for whether cutoff-time parameters are learned
            init_cutoffs (function): function to initialize cutoff-time parameters
            init_lower_thresholds (function): function to initialize lower threshold parameters
            init_upper_thresholds (function): function to initialize upper threshold parameters
            cutoff_times_temperature (float): temperature used to calculate cutoff-time parameters
            thresholds_temperature (float): temperature used to calculate threshold summaries
            ever_measured_temperature (float): temperature used to calculate measurement indicator summaries
            switch_temperature (float): temperature used to calculate switch summaries
            seq_len (int): number of time-steps in each trajectory
        """
        super(LogisticRegressionWithSummaries, self).__init__()

        self.seq_len = seq_len
        self.changing_dim = changing_dim
        self.num_cutoff_times = num_cutoff_times
        self.top_k = top_k
        
        self.sigmoid_for_weights = nn.Sigmoid()
        self.sigmoid_for_ever_measured = nn.Sigmoid()
        self.sigmoid_for_switches = nn.Sigmoid()
        
        self.upper_thresh_sigmoid = nn.Sigmoid()
        self.lower_thresh_sigmoid = nn.Sigmoid()
        
        self.sigmoid = nn.Sigmoid()
        
        num_total_c_weights = changing_dim * num_cutoff_times
        
        # Initialize cutoff_times to by default use all of the timesteps.
        self.cutoff_percentage = - 12 * torch.ones(1, num_total_c_weights).to(self.device)
      
            
        if differentiate_cutoffs: 
            cutoff_vals = init_cutoffs(num_total_c_weights)
            
            if cutoff_times_init_values is not None:
                cutoff_vals = cutoff_times_init_values
                
            self.cutoff_percentage = nn.Parameter(torch.tensor(cutoff_vals, requires_grad=True).reshape(1, num_total_c_weights).to(self.device))
            
        self.times = torch.tensor(np.transpose(np.tile(range(seq_len), (changing_dim, 1)))).to(self.device)
        self.times = self.times.repeat(1, num_cutoff_times).to(self.device)

        self.weight_parser = WeightsParser()
        self.cs_parser = WeightsParser()
        add_all_parsers(self.weight_parser, input_dim, self.changing_dim)
        add_all_parsers(self.cs_parser, input_dim, self.changing_dim, 'cs')
        
        self.lower_thresholds = nn.Parameter(torch.tensor(init_lower_thresholds(changing_dim)).to(self.device))
        self.upper_thresholds = nn.Parameter(torch.tensor(init_upper_thresholds(changing_dim)).to(self.device))
        
        self.lower_thresholds.retain_grad()
        self.upper_thresholds.retain_grad()
        
        self.thresh_temperature = thresholds_temperature
        self.cutoff_percentage_temperature = cutoff_times_temperature
        self.ever_measured_temperature = ever_measured_temperature
        self.switch_temperature = switch_temperature
        
        self.linear = nn.Linear(self.weight_parser.num_weights, 2)
        
        if (self.top_k != ''):
            file = open(self.top_k)
            csvreader = csv.reader(file)
            header = next(csvreader)
            top_k_inds = []
            for row in csvreader:
                top_k_inds.append(int(row[1]))
            condition = torch.zeros(self.linear.weight.shape, dtype=torch.bool).to(self.device)
            for i in range(len(top_k_inds)):
                condition[:,top_k_inds[i]]=True
            self.linear.weight = torch.nn.Parameter(self.linear.weight.where(condition, torch.tensor(0.0).to(self.device)))
    
    def encode_patient_batch(self, patient_batch, epsilon_denom=1e-8):
	# Computes the encoding (s, x) + (weighted_summaries) in the order defined in weight_parser.
        # Returns pre-sigmoid P(Y = 1 | patient_batch)
        temperatures = torch.tensor(np.full((1, self.cs_parser.num_weights), self.cutoff_percentage_temperature), device=self.device)
        
        # Get changing variables
        batch_changing_vars = patient_batch[:, :, :self.changing_dim]
        batch_measurement_indicators = patient_batch[:, :, self.changing_dim: self.changing_dim * 2]
        # batch_measurement_repeat = batch_measurement_indicators.repeat(1, 1, self.num_cutoff_times)
        
        weight_vector = self.sigmoid_for_weights((self.times - self.cutoff_percentage) / temperatures).reshape(1, self.seq_len, self.cs_parser.num_weights)
        # Calculate weighted mean features
        
        # Sum of all weights across time-steps
        # weight_norm = torch.sum(weight_vector * batch_measurement_repeat, dim=1)
        # weight_mask = torch.sum(batch_measurement_indicators, dim=1)
        
        # MEAN FEATURES

        # Calculate \sum_t (w_t * x_t * m_t)
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_mean_']
        mean_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_average = torch.sum(mean_weight_vector * (batch_changing_vars * batch_measurement_indicators), dim=1)

        mean_feats = (weighted_average / (torch.sum(mean_weight_vector, dim=1) + epsilon_denom))
        
        # VARIANCE FEATURES
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_var_']
        var_weight_vector = weight_vector[:, :, start_i : end_i]
        
        x_mean = torch.mean(batch_measurement_indicators * batch_changing_vars, dim=1, keepdim=True)
        weighted_variance = torch.sum(batch_measurement_indicators * var_weight_vector * (batch_changing_vars - x_mean)**2, dim=1)        
        normalizing_term = torch.sum(batch_measurement_indicators * var_weight_vector, dim=1)**2 / (torch.sum(batch_measurement_indicators * var_weight_vector, dim=1)**2 + torch.sum(batch_measurement_indicators * var_weight_vector ** 2, dim=1) + epsilon_denom)
        
        var_feats = weighted_variance / (normalizing_term + epsilon_denom)
        

        # INDICATOR FOR EVER BEING MEASURED
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_ever_measured_']
        ever_measured_weight_vector = weight_vector[:, :, start_i : end_i]
        
        ever_measured_feats = self.sigmoid_for_ever_measured( torch.sum(ever_measured_weight_vector * batch_measurement_indicators, dim=1) / (self.ever_measured_temperature * torch.sum(ever_measured_weight_vector, dim=1) + epsilon_denom)) - 0.5
        
        
        # MEAN OF INDICATOR SEQUENCE
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_mean_indicators_']
        mean_ind_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_ind_average = torch.sum(mean_ind_weight_vector * batch_measurement_indicators, dim=1)
        mean_ind_feats = weighted_ind_average / (torch.sum(mean_ind_weight_vector, dim=1) + epsilon_denom)
        
        # VARIANCE OF INDICATOR SEQUENCE
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_var_indicators_']
        var_ind_weight_vector = weight_vector[:, :, start_i : end_i]
                
        x_mean_ind = torch.mean(batch_measurement_indicators, dim=1, keepdim=True)
        weighted_variance_ind = torch.sum(var_ind_weight_vector * (batch_measurement_indicators - x_mean_ind)**2, dim=1)        
        normalizing_term = torch.sum(var_ind_weight_vector, dim=1)**2 / (torch.sum(var_ind_weight_vector, dim=1)**2 + torch.sum(var_ind_weight_vector ** 2, dim=1) + epsilon_denom)
        
        var_ind_feats = weighted_variance_ind / (normalizing_term + epsilon_denom)
        
        
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

        mask_max_values, mask_max_indices = torch.max(batch_measurement_indicators, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = -1
        
        first_time_feats = mask_max_indices / float(batch_measurement_indicators.shape[1])
        
        # Last time measured is the last index of the max.
        # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
        flipped_batch_measurement_indicators = torch.flip(batch_measurement_indicators, [1])
        
        mask_max_values, mask_max_indices = torch.max(flipped_batch_measurement_indicators, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = batch_measurement_indicators.shape[1]
        
        last_time_feats = (float(batch_measurement_indicators.shape[1]) - mask_max_indices) / float(batch_measurement_indicators.shape[1])
        
        # SLOPE OF L2
        # STANDARD ERROR OF L2     
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_slope_']
        slope_weight_vector = weight_vector[:, :, start_i : end_i]
        
        # Zero out the batch_changing_vars so that they are zero if the features are not measured.
        linreg_y = batch_changing_vars * batch_measurement_indicators
        
        # The x-values for this linear regression are the times.
        # Zero them out so that they are zero if the features are not measured.
        linreg_x = torch.tensor(np.transpose(np.tile(range(self.seq_len), (self.changing_dim, 1)))).to(self.device)
        linreg_x = linreg_x.repeat(linreg_y.shape[0], 1, 1) * batch_measurement_indicators
        
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
        
        # HOURS ABOVE THRESHOLD
        # HOURS BELOW THRESHOLD
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_hours_above_threshold_']
        above_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_hours_below_threshold_']
        below_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
            
        upper_features = self.upper_thresh_sigmoid((batch_changing_vars - self.upper_thresholds)/self.thresh_temperature)
        lower_features = self.lower_thresh_sigmoid((self.lower_thresholds - batch_changing_vars)/self.thresh_temperature)
                
        # (batch, timestep, features)
        # sum upper_features and lower_features across timesteps
        above_threshold_feats = torch.sum(batch_measurement_indicators * above_thresh_weight_vector * upper_features, dim=1) / (torch.sum(batch_measurement_indicators * above_thresh_weight_vector, dim=1) + epsilon_denom)
        below_threshold_feats = torch.sum(batch_measurement_indicators * below_thresh_weight_vector * lower_features, dim=1) / (torch.sum(batch_measurement_indicators * below_thresh_weight_vector, dim=1) + epsilon_denom)
        
        feats_time_5 = patient_batch[:, 5, :]

        cat = torch.cat((feats_time_5.float(), mean_feats.float(), var_feats.float(), ever_measured_feats.float(), mean_ind_feats.float(), var_ind_feats.float(), switch_feats.float(), slope_feats.float(), slope_stderr_feats.float(), first_time_feats.float(), last_time_feats.float(), above_threshold_feats.float(), below_threshold_feats.float()), axis=1)
        
        # print('mean')
        # print(mean_feats[0, :])
        # print('var')
        # print(var_feats[0, :])
        # print('ever meas')
        # print(ever_measured_feats[0, :])
        # print('mean indicat')
        # print(mean_ind_feats[0, :])
        # print('var ind')
        # print(var_ind_feats[0, :])
        # print('switch')
        # print(switch_feats[0, :])
        # print('slope')
        # print(slope_feats[0, :])
        # print('slope stderr')
        # print(slope_stderr_feats[0, :])
        # print('first time')
        # print(first_time_feats[0, :])
        # print('last time')
        # print(last_time_feats[0, :])
        # print('above thresh')
        # print(above_threshold_feats[0, :])
        # print('below thresh')
        # print(below_threshold_feats[0, :])
        
        return cat
    
    def forward(self, patient_batch, epsilon_denom=1e-8):
        # Encodes the patient_batch, then computes the forward.
        return self.linear(self.encode_patient_batch(patient_batch, epsilon_denom))
    
    def forward_encoded(self, encoded):
        # Computes forward on already-encoded vector "encoded".
        return self.linear(encoded)
    
    def forward_probabilities(self, patient_batch):
        return self.sigmoid(self.forward(patient_batch))[:,1].item()


class LogisticRegressionWithSummaries_Wrapper(nn.Module):
    """
    Wrapper class to track training for a LogisticRegressionWithSummaries.
    """

    def __init__(self, 
                 input_dim, 
                 changing_dim, 
                 num_cutoff_times, 
                 differentiate_cutoffs,
                 init_cutoffs, 
                 init_lower_thresholds, 
                 init_upper_thresholds, 
                 cutoff_times_temperature = 1.0,
                 alpha = 1e-4,
                 tau = 10.,
                 opt_lr = 1e-4,
                 opt_weight_decay = 0.,
                 cutoff_times_init_values = None,
                 top_k = ''
                ):
        """Initializes the LogisticRegressionWithSummaries with training hyperparameters.
        
        Args:
            input_dim (int): number of input dimensions
            changing_dim (int): number of non-static input dimensions
            num_cutoff_tines (int): number of cutoff-time parameters
            init_cutoffs (function): function to initialize cutoff-time parameters
            init_lower_thresholds (function): function to initialize lower threshold parameters
            init_upper_thresholds (function): function to initialize upper threshold parameters
            seq_len (int): number of time-steps in each trajectory
            -- 
            alpha (float): multiplicative coefficient of the horseshoe loss term
            tau (float): constant used to calculate horseshoe loss term
            opt_lr (float): learning rate for the optimizer
            opt_weight_decay (float): weight decay for the optimizer
            
        """
        super(LogisticRegressionWithSummaries_Wrapper, self).__init__()
        
        self.input_dim = input_dim
        self.changing_dim = changing_dim
        self.num_cutoff_times = num_cutoff_times
        self.differentiate_cutoffs = differentiate_cutoffs
        self.init_cutoffs = init_cutoffs 
        self.init_lower_thresholds = init_lower_thresholds
        self.init_upper_thresholds = init_upper_thresholds
        self.top_k = top_k
        
        self.model = LogisticRegressionWithSummaries(input_dim, 
                                                        changing_dim, 
                                                        num_cutoff_times,
                                                        differentiate_cutoffs,                         
                                                        init_cutoffs, 
                                                        init_lower_thresholds, 
                                                        init_upper_thresholds,
                                                        cutoff_times_temperature,
                                                        cutoff_times_init_values,
                                                        top_k=top_k)
        
        self.alpha = alpha
        self.tau = tau
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= opt_lr, weight_decay = opt_weight_decay)
        
    def _load_model(self, path, print_=True):
        """
        Args:
            path (str): filepath to the model
        """
        try:
            checkpoint = torch.load(path)
        except:
            return
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if print_:
            print("Loaded model from " + path)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.curr_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        sleep(0.5)

    def fit(self, train_loader, val_loader, p_weight, save_model_path, epochs=10000, save_every_n_epochs=100):
        """
        
        Args:
            train_loader (torch.DataLoader): 
            val_tensor (torch.DataLoader):
            p_weight (tensor): weight parameter used to calculate BCE loss 
            save_model_path (str): filepath to save the model progress
            epochs (int): number of epochs to train
        """
        
        rtpt = RTPT(name_initials='KA', experiment_name='TimeSeriesCBM', max_iterations=epochs)
        rtpt.start()
        
        self.loss_func = custom_bce_horseshoe
        self.train_losses = []
        self.val_losses = []
        self.curr_epoch = -1
        
        self._load_model(save_model_path)
        
        epoch_loss = 0
        
        for epoch in tqdm(range(self.curr_epoch+1, epochs)):
            if (epoch % save_every_n_epochs) == (-1 % save_every_n_epochs):
                # Append loss to loss array
                epoch_loss = epoch_loss / num_batches
                
                self.train_losses.append(epoch_loss)
                
                # Calculate validation set loss
                val_loss = 0
                num_batches_val = 0
                for batch, data in enumerate(val_loader):
                    Xb, yb = data
                    num_batches_val +=1
           
                    # Forward pass.
                    self.model.zero_grad()
                    output = self.model(Xb)

                    val_loss += self.loss_func(output, yb, self.model.linear.weight, p_weight)
              
                self.val_losses.append(val_loss.item()/num_batches_val)
                
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_losses': self.train_losses,
                            'val_losses': self.val_losses,
                            }, save_model_path)
            
            epoch_loss = 0
            num_batches = 0
            for batch, data in enumerate(train_loader):

                Xb, yb = data
                num_batches += 1

                # Forward pass.
                self.model.zero_grad()
                output = self.model(Xb)

                loss = self.loss_func(output, yb, self.model.linear.weight, p_weight)
                loss.backward()

                epoch_loss += loss.item()
                
                if (self.top_k != ''):
                    file = open(self.top_k)
                    csvreader = csv.reader(file)
                    header = next(csvreader)
                    top_k_inds = []
                    for row in csvreader:
                        top_k_inds.append(int(row[1]))
                    condition = torch.zeros(self.model.linear.weight.grad.shape, dtype=torch.bool, device=self.device)
                    for i in range(len(top_k_inds)):
                        condition[:,top_k_inds[i]]=True
                    self.model.linear.weight.grad = torch.nn.Parameter(self.model.linear.weight.grad.where(condition, torch.tensor(0.0, device=self.device)))

                # update all parameters
                self.optimizer.step()
        
            rtpt.step(subtitle=f"loss={loss:2.2f}")


class CBM(nn.Module):
    def __init__(self, 
                 input_dim, 
                 changing_dim, 
                 seq_len,
                 num_concepts,
                 differentiate_cutoffs = True,
                 init_cutoffs_f = init_cutoffs_to_zero,
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
        self.sigmoid = nn.Sigmoid()
        
        # activation function to convert output into probabilities
        # not needed during training as pytorch losses are optimized and include sigmoid / softmax
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim < 2:
            self.output_af = nn.Sigmoid()
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 2:
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
        self.bottleneck = nn.Linear(self.weight_parser.num_weights, self.num_concepts)
        # -> B x T x C
                
        # prediction task
        self.linear = nn.LazyLinear(self.output_dim)
        
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
    
    def encode_patient_batch(self, patient_batch, epsilon_denom=1e-8):
        # Computes the encoding (s, x) + (weighted_summaries) in the order defined in weight_parser.
        # Returns pre-sigmoid P(Y = 1 | patient_batch)
        temperatures = torch.tensor(np.full((1, self.cs_parser.num_weights), self.cutoff_percentage_temperature), device=self.device)
        
        # Get changing variables
        batch_changing_vars = patient_batch[:, :, :self.changing_dim]
        batch_measurement_indicators = patient_batch[:, :, self.changing_dim: self.changing_dim * 2]
        batch_static_vars = patient_batch[:, 0, self.changing_dim * 2:] # static is the same accross time
        
        weight_vector = self.sigmoid((self.times - self.cutoff_percentage) / temperatures).reshape(1, self.seq_len, self.cs_parser.num_weights)
        
        # MEAN FEATURES
        # Calculate \sum_t (w_t * x_t * m_t)
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_mean_']
        mean_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_average = torch.sum(mean_weight_vector * (batch_changing_vars * batch_measurement_indicators), dim=1)

        mean_feats = (weighted_average / (torch.sum(mean_weight_vector, dim=1) + epsilon_denom))
        
        # VARIANCE FEATURES
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_var_']
        var_weight_vector = weight_vector[:, :, start_i : end_i]
        
        x_mean = torch.mean(batch_measurement_indicators * batch_changing_vars, dim=1, keepdim=True)
        weighted_variance = torch.sum(batch_measurement_indicators * var_weight_vector * (batch_changing_vars - x_mean)**2, dim=1)        
        normalizing_term = torch.sum(batch_measurement_indicators * var_weight_vector, dim=1)**2 / (torch.sum(batch_measurement_indicators * var_weight_vector, dim=1)**2 + torch.sum(batch_measurement_indicators * var_weight_vector ** 2, dim=1) + epsilon_denom)
        
        var_feats = weighted_variance / (normalizing_term + epsilon_denom)
        

        # INDICATOR FOR EVER BEING MEASURED
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_ever_measured_']
        ever_measured_weight_vector = weight_vector[:, :, start_i : end_i]
        
        ever_measured_feats = self.sigmoid( torch.sum(ever_measured_weight_vector * batch_measurement_indicators, dim=1) / (self.ever_measured_temperature * torch.sum(ever_measured_weight_vector, dim=1) + epsilon_denom)) - 0.5
        
        
        # MEAN OF INDICATOR SEQUENCE
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_mean_indicators_']
        mean_ind_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_ind_average = torch.sum(mean_ind_weight_vector * batch_measurement_indicators, dim=1)
        mean_ind_feats = weighted_ind_average / (torch.sum(mean_ind_weight_vector, dim=1) + epsilon_denom)
        
        # VARIANCE OF INDICATOR SEQUENCE
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_var_indicators_']
        var_ind_weight_vector = weight_vector[:, :, start_i : end_i]
                
        x_mean_ind = torch.mean(batch_measurement_indicators, dim=1, keepdim=True)
        weighted_variance_ind = torch.sum(var_ind_weight_vector * (batch_measurement_indicators - x_mean_ind)**2, dim=1)        
        normalizing_term = torch.sum(var_ind_weight_vector, dim=1)**2 / (torch.sum(var_ind_weight_vector, dim=1)**2 + torch.sum(var_ind_weight_vector ** 2, dim=1) + epsilon_denom)
        
        var_ind_feats = weighted_variance_ind / (normalizing_term + epsilon_denom)
        
        
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

        mask_max_values, mask_max_indices = torch.max(batch_measurement_indicators, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = -1
        
        first_time_feats = mask_max_indices / float(batch_measurement_indicators.shape[1])
        
        # Last time measured is the last index of the max.
        # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
        flipped_batch_measurement_indicators = torch.flip(batch_measurement_indicators, [1])
        
        mask_max_values, mask_max_indices = torch.max(flipped_batch_measurement_indicators, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = batch_measurement_indicators.shape[1]
        
        last_time_feats = (float(batch_measurement_indicators.shape[1]) - mask_max_indices) / float(batch_measurement_indicators.shape[1])
        
        # SLOPE OF L2
        # STANDARD ERROR OF L2     
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_slope_']
        slope_weight_vector = weight_vector[:, :, start_i : end_i]
        
        # Zero out the batch_changing_vars so that they are zero if the features are not measured.
        linreg_y = batch_changing_vars * batch_measurement_indicators
        
        # The x-values for this linear regression are the times.
        # Zero them out so that they are zero if the features are not measured.
        linreg_x = torch.tensor(np.transpose(np.tile(range(self.seq_len), (self.changing_dim, 1))), device=self.device)
        linreg_x = linreg_x.repeat(linreg_y.shape[0], 1, 1) * batch_measurement_indicators
        
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
        
        # HOURS ABOVE THRESHOLD
        # HOURS BELOW THRESHOLD
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_hours_above_threshold_']
        above_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
        
        start_i, end_i = self.cs_parser.idxs_and_shapes['cs_hours_below_threshold_']
        below_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
            
        upper_features = self.sigmoid((batch_changing_vars - self.upper_thresholds)/self.thresh_temperature)
        lower_features = self.sigmoid((self.lower_thresholds - batch_changing_vars)/self.thresh_temperature)
                
        # (batch, timestep, features)
        # sum upper_features and lower_features across timesteps
        above_threshold_feats = torch.sum(batch_measurement_indicators * above_thresh_weight_vector * upper_features, dim=1) / (torch.sum(batch_measurement_indicators * above_thresh_weight_vector, dim=1) + epsilon_denom)
        below_threshold_feats = torch.sum(batch_measurement_indicators * below_thresh_weight_vector * lower_features, dim=1) / (torch.sum(batch_measurement_indicators * below_thresh_weight_vector, dim=1) + epsilon_denom)
        
        # feats_time_23 = patient_batch[:, 23, :]     
        # time_feats = patient_batch[:, self.seq_len-1, :] # use last timestamp
        
        # use full timeseries, reshape 3d to 2d, keep sample size N and merge Time x Variables (N x T x V) => (N x T*V)
        changing_vars_2d = batch_changing_vars.reshape(batch_changing_vars.shape[0], -1) # result is V1_T1, V2_T1, V3_T1, ..., V1_T2, V2_T2, V3_T2, ... repeat V, T times
        indicators_2d = batch_measurement_indicators.reshape(batch_measurement_indicators.shape[0], -1)
        time_feats = torch.cat((changing_vars_2d, indicators_2d, batch_static_vars), dim=1)
        
        # print("patient_batch", patient_batch.shape)
        # print("time_feats", time_feats.shape)

        cat = torch.cat((time_feats.float(), mean_feats.float(), var_feats.float(), ever_measured_feats.float(), mean_ind_feats.float(), var_ind_feats.float(), switch_feats.float(), slope_feats.float(), slope_stderr_feats.float(), first_time_feats.float(), last_time_feats.float(), above_threshold_feats.float(), below_threshold_feats.float()), axis=1)

        # print("mean_feats", mean_feats.shape)
        # print("var_feats", var_feats.shape)
        # print("cat", cat.shape)

        return cat
    
    def forward(self, patient_batch, epsilon_denom=1e-8):
        # Encodes the patient_batch, then computes the forward.
        encoded = self.encode_patient_batch(patient_batch, epsilon_denom)
        bottleneck = self.bottleneck(encoded)
        activation = self.sigmoid(bottleneck)
        return self.linear(activation)
    
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

    def fit(self, train_loader, val_loader, p_weight, save_model_path, max_epochs=10000, save_every_n_epochs=20, patience=10, warmup_epochs=0, scheduler=None, trial=None, show_grad=False):
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
        
        self.earlyStopping = EarlyStopping(patience=patience, min_delta=0, mode=EarlyStopping.Mode.MIN)
        early_stopped = self._load_model(save_model_path)
        
        if early_stopped:
            return
        
        for epoch in tqdm(range(self.curr_epoch+1, max_epochs)):
            self.train()
            train_loss = 0
            
            for batch_idx, (Xb, yb) in enumerate(train_loader):
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                y_pred = self(Xb)

                loss = self.compute_loss(yb, y_pred, p_weight)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                train_loss += loss * Xb.size(0)
                
                
                for name, param in self.named_parameters():
                    isnan = param.grad is None or torch.isnan(param.grad).any()
                    if isnan:
                        print(f"Parameter: {name}, Gradient NAN? {isnan}")
                        return
                
                if show_grad:
                    plot_grad_flow(self.named_parameters())
                    return
                
                if (self.top_k != ''):
                    self.bottleneck.weight.grad.fill_(0.)
        
                self.optimizer.step()
            
            train_loss = train_loss / len(train_loader.sampler)
            
            if (epoch % save_every_n_epochs) == (-1 % save_every_n_epochs):
                self.eval()
                with torch.no_grad():
                    
                    self.train_losses.append(train_loss.item())
                    
                    # Calculate validation set loss
                    val_loss = 0
                    for batch_idx, (Xb, yb) in enumerate(val_loader):
                        Xb, yb = Xb.to(self.device), yb.to(self.device)
            
                        # Forward pass.
                        y_pred = self(Xb)

                        val_loss += self.compute_loss(yb, y_pred, p_weight) * Xb.size(0)
                    
                    val_loss = val_loss / len(val_loader.sampler)
                    self.val_losses.append(val_loss.item())
                    
                    if scheduler:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        else:
                            scheduler.step()
                    
                    state = create_state_dict(self, epoch)
                    
                    if self.earlyStopping.check_improvement(val_loss, state):
                        break
                    
                    if save_model_path:
                        torch.save(state, save_model_path)
                    
                    if trial:
                        trial.report(val_loss, epoch)
                        
                        if trial.should_prune():
                            raise optuna.exceptions.TrialPruned()
            
            rtpt.step(subtitle=f"loss={train_loss:2.2f}")
            
        if save_model_path and self.earlyStopping.best_state:
            torch.save(self.earlyStopping.best_state, save_model_path)
        
        return self.val_losses[-1]

    def compute_loss(self, yb, y_pred, p_weight):                
        if self.task_type == TaskType.CLASSIFICATION and self.output_dim < 2:
            task_loss = binary_cross_entropy_with_logits(y_pred, yb.float(), pos_weight = p_weight)
        elif self.task_type == TaskType.CLASSIFICATION and self.output_dim > 2:
            task_loss = cross_entropy(y_pred, yb, weight = p_weight)
        elif self.task_type == TaskType.REGRESSION:
            task_loss = mse_loss(y_pred, yb)
        else:
            raise NotImplementedError("Config not defined!")
        
        # Lasso regularization
        L1_norm = torch.norm(self.bottleneck.weight, 1)
        l1_loss = self.l1_lambda * L1_norm 
        
        # Cosine_similarity regularization
        cos_sim_loss = 0
        if self.num_concepts != 1:
            concepts = torch.arange(self.num_concepts, device=self.device)
            indices = torch.combinations(concepts, 2)  # Generate all combinations of concept indices

            weights = self.bottleneck.weight[indices]  # Extract corresponding weight vectors
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
        print(n, torch.isnan(p.grad).any())
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
