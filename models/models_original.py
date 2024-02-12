
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, mse_loss, cosine_similarity
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


class MultiplicativeInteractionLayer(nn.Module):
    def __init__(self, in_features, in_features2, out_features):
        super(MultiplicativeInteractionLayer, self).__init__()

        self.bilinear = nn.Bilinear(in_features, in_features2, out_features)
        self.linear1 = nn.Linear(in_features, out_features, bias=False)
        self.linear2 = nn.Linear(in_features2, out_features, bias=False)

    def forward(self, x, z):
        output = self.bilinear(x, z) + self.linear1(x) + self.linear2(z)
        return output


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x): # needs 3d mat, b t v
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        # print("SA weighted, out:", weighted.shape, "in:", x.shape)
        return weighted


class CBM(nn.Module):
    def __init__(self, 
                static_dim, 
                changing_dim, 
                seq_len,
                num_concepts,
                output_dim,
                use_indicators,
                use_only_last_timestep,
                use_grad_norm,
                use_multiplicative_interactions,
                use_summaries,
                differentiate_cutoffs = True,
                init_cutoffs_f = init_cutoffs_to_zero,
                init_lower_thresholds_f = init_rand_lower_thresholds, 
                init_upper_thresholds_f = init_rand_upper_thresholds,
                temperature = 0.1,
                opt_lr = 1e-4,
                opt_weight_decay = 0.,
                l1_lambda=0.,
                cos_sim_lambda=0.,
                top_k = "",
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
        
        self.static_dim = static_dim
        self.changing_dim = changing_dim
        self.seq_len = seq_len
        self.num_concepts = num_concepts
        self.num_summaries = 12 # number of calculated summaries
        
        self.use_indicators = use_indicators
        self.use_only_last_timestep = use_only_last_timestep
        self.use_grad_norm = use_grad_norm
        self.use_multiplicative_interactions = use_multiplicative_interactions
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
        if self.use_multiplicative_interactions == "MI":
            dim = self.changing_dim * self.seq_len * 2 + self.static_dim + self.num_summaries * self.changing_dim
            self.bottleneck = MultiplicativeInteractionLayer(dim, dim, self.num_concepts)
            
        elif self.use_multiplicative_interactions == "SA":
            self._self_attention = SelfAttention(2 * self.seq_len + self.num_summaries + self.static_dim)
            self._linear = nn.LazyLinear(self.num_concepts)
            self.bottleneck = nn.Sequential(self._self_attention, nn.Flatten(), self._linear)
            
        else:
            self.bottleneck = nn.LazyLinear(self.num_concepts) # self.weight_parser.num_weights
        # -> B x T x C
        
        # prediction task
        self.layer_output = nn.LazyLinear(self.output_dim)
        
        self.regularized_layers = [self.bottleneck]
        
        self.to(device=self.device)
        # self.deactivate_bottleneck_weights_if_top_k()
        return
        
    def deactivate_bottleneck_weights_if_top_k(self):
        if (self.top_k != ''):
            # init weights, needed with lazy layers
            self(torch.zeros(2, self.seq_len, self.changing_dim, device=self.device), torch.zeros(2, self.seq_len, self.changing_dim, device=self.device), torch.zeros(2, self.static_dim, device=self.device))
            
            file = open(self.top_k)
            csvreader = csv.reader(file)
            header = next(csvreader)
            
            top_k_concepts = []
            top_k_inds = []
            i = 0
            
            for row in csvreader:
                if (i < self.top_k_num):
                    top_k_concepts.append(int(row[1]))
                    top_k_concepts.append(int(row[2]))
                    top_k_inds.append(int(row[3]))
                    i+=1
                else:
                    break
            
            self.weight_masks = []
            for i, layer in enumerate(self.regularized_layers):
                weight_mask = torch.zeros(layer.weight.shape, dtype=torch.bool, device=self.device)
                
                for concept_id, feat_id in zip(top_k_concepts, top_k_inds):
                    weight_mask[concept_id][feat_id] = True
                
                self.weight_masks.append(weight_mask)
                layer.weight = torch.nn.Parameter(layer.weight.where(weight_mask, torch.tensor(0.0, device=self.device)))
        return
    
    def forward(self, time_dependent_vars, indicators, static_vars):
        assert time_dependent_vars.dim() == 3 and time_dependent_vars.size(1) == self.seq_len and time_dependent_vars.size(2) == self.changing_dim
        assert indicators.shape == time_dependent_vars.shape
        assert torch.equal(static_vars, torch.empty(0, device=self.device)) or (static_vars.dim() == 2 and static_vars.size(1) == self.static_dim)
        
        # Encodes the patient_batch, then computes the forward.
        summaries = calculate_summaries(self, time_dependent_vars, indicators, self.use_indicators)
        
        if self.use_multiplicative_interactions == "MI":
            input = get_time_feat_2d(time_dependent_vars, indicators, static_vars, self.use_only_last_timestep, self.use_indicators)
            summaries2d = summaries.reshape(summaries.size(0), -1)
            input = torch.cat([input, summaries2d], axis=1)
            
            bottleneck = self.bottleneck(input, input)
            
        elif self.use_multiplicative_interactions == "SA":
            if self.use_indicators:
                input = torch.cat([time_dependent_vars, indicators], axis=1) # cat along time instead of var
            else:
                input = time_dependent_vars
            input = rearrange(input, "b t v -> b v t")
            
            if not torch.equal(static_vars, torch.empty(0, device=self.device)):
                # static vars are concatinated along time, for each var, similar to summaries
                static_vars = static_vars.unsqueeze(1) # b x 1 x 8
                static_vars = static_vars.expand(-1, input.size(1), -1) # -1 => unchanged
            
            input = torch.cat([input, static_vars, summaries], axis=-1)
            
            bottleneck = self.bottleneck(input)
            
        else:
            input = get_time_feat_2d(time_dependent_vars, indicators, static_vars, self.use_only_last_timestep, self.use_indicators)
            
            if self.use_summaries:
                summaries2d = summaries.reshape(summaries.size(0), -1)
                input = torch.cat([input, summaries2d], axis=1)
            
            bottleneck = self.bottleneck(input)
        
        activation = self.sigmoid_layer(bottleneck)
        return self.layer_output(activation)
    
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
        
    def _load_model(self, path, print=True):
        """
        Args:
            path (str): filepath to the model
        """
        try:
            checkpoint = torch.load(path)
        except:
            return False
        
        if print:
            print("Loaded model from " + path)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.curr_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
                
        self.deactivate_bottleneck_weights_if_top_k()
        sleep(0.5)
        
        return True
    
    def try_load_else_fit(self, *args, **kwargs):
        if self._load_model(kwargs.get('save_model_path')):
            return
        else:
            return self.fit(*args, **kwargs)
    
    def fit(self, train_loader, val_loader, p_weight, save_model_path, max_epochs=10000, save_every_n_epochs=10, patience=10, scheduler=None, trial=None):
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
        
        self.earlyStopping = EarlyStopping(patience=patience)
        self._load_model(save_model_path)
        
        epochs = range(self.curr_epoch+1, max_epochs)
        
        with tqdm(total=len(epochs), unit=' epoch') as pbar:
            
            for epoch in epochs:
                self.train()
                train_loss = 0
            
                ### Train loop
                for batch in train_loader:
                    X_time, X_ind, X_static, y_true = extract_to(batch, self.device)
                    
                    y_pred = self(X_time, X_ind, X_static)

                    loss = compute_loss(y_true=y_true, y_pred=y_pred, p_weight=p_weight, l1_lambda=self.l1_lambda, cos_sim_lambda=self.cos_sim_lambda, regularized_layers=self.regularized_layers, task_type=self.task_type)
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    
                    if self.use_grad_norm:
                        normalize_gradient_(self.parameters(), self.use_grad_norm)

                    train_loss += loss * X_time.size(0)
                    
                    # for name, param in self.named_parameters():
                    #     isnan = param.grad is None or torch.isnan(param.grad).any()
                    #     if isnan:
                    #         print(f"Parameter: {name}, Gradient NAN? {isnan}")
                    #         return
                    
                    if (self.top_k != ''):
                        for layer, weight_mask in zip(self.regularized_layers, self.weight_masks):
                            layer.weight.grad = torch.nn.Parameter(layer.weight.grad.where(weight_mask, torch.tensor(0.0, device=self.device)))

                    self.optimizer.step()
                
                train_loss = train_loss / len(train_loader.sampler)
                
                
                if (epoch % save_every_n_epochs) == 0:
                    self.eval()
                    with torch.no_grad():
                        
                        self.train_losses.append(train_loss.item())
                        
                        ### Validation loop
                        val_loss = 0
                        for batch in val_loader:
                            X_time, X_ind, X_static, y_true = extract_to(batch, self.device)
                            
                            y_pred = self(X_time, X_ind, X_static)

                            vloss = compute_loss(y_true=y_true, y_pred=y_pred, p_weight=p_weight, l1_lambda=self.l1_lambda, cos_sim_lambda=self.cos_sim_lambda, regularized_layers=self.regularized_layers, task_type=self.task_type)
                            val_loss += vloss * X_time.size(0)
                        
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
        
        return self.val_losses[-1]
