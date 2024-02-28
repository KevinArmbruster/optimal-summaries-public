import torch
from rtpt import RTPT
from tqdm import tqdm
import optuna
from sparselearning.core import CosineDecay, Masking
from time import sleep

from models.custom_losses import compute_loss
from models.EarlyStopping import EarlyStopping
from models.helper import *


class BaseCBM(nn.Module):
    def __init__(self):
        super(BaseCBM, self).__init__()
        self.save_model_path = None
    
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
        
    def update_ema_gradient(self):
        with torch.no_grad():
            for layer in self.regularized_layers:
                layer.update_ema_gradient()
    
    def clear_ema_gradient(self):
        with torch.no_grad():
            for layer in self.regularized_layers:
                layer.ema_gradient = None
    
    def mask_by_weight_magnitude(self, remain_active_list):
        with torch.no_grad():
            for layer, remain_active in zip(self.regularized_layers, remain_active_list):
                weight_mask, bias_mask = mask_smallest_magnitude(layer.weight, remain_active)
                layer.set_weight_mask(weight_mask, bias_mask)
    
    def mask_by_gradient_magnitude(self, remain_active_list):
        with torch.no_grad():
            for layer, remain_active in zip(self.regularized_layers, remain_active_list):
                weight_mask, bias_mask = mask_smallest_magnitude(layer.ema_gradient, remain_active)
                layer.set_weight_mask(weight_mask, bias_mask)
    
    def mask_shrinking_weights(self):
        with torch.no_grad():
            for layer in self.regularized_layers:
                weight_mask, bias_mask = mask_shrinking_weights(layer)
                layer.set_weight_mask(weight_mask, bias_mask)
    
    def clear_all_weight_masks(self):
        for layer in self.regularized_layers:
            layer.clear_weight_mask()
    
    def deactivate_bottleneck_weights_if_top_k(self, top_k = None, top_k_num = np.inf):
        if top_k is not None:
            self.top_k = top_k
            self.top_k_num = top_k_num
        
        if self.top_k is not None:
            if isinstance(self.top_k, str):
                # if path, load df
                self.top_k = read_df_from_csv(self.top_k)
            
            # init weights, needed with lazy layers
            self(torch.zeros(2, self.seq_len, self.changing_dim, device=self.device), torch.zeros(2, self.seq_len, self.changing_dim, device=self.device), torch.zeros(2, self.static_dim, device=self.device))
            
            for i, layer in enumerate(self.regularized_layers):
                layer_config = self.top_k[self.top_k["Layer"] == i]
                weight_mask = torch.zeros(layer.weight.shape, dtype=torch.bool, device=layer.weight.device)
                
                for j, (concept_id, feat_id) in enumerate(zip(layer_config["Concept"], layer_config["Feature"]), 1):
                    weight_mask[concept_id][feat_id] = True
                    if j == self.top_k_num:
                        break
                
                layer.set_weight_mask(weight_mask)
        return
        
    def _load_model(self, path_or_checkpoint, print_=True):
        """
        Args:
            path (str): filepath to the model
        """
        try:
            if isinstance(path_or_checkpoint, str):
                checkpoint = torch.load(path_or_checkpoint)
            else:
                checkpoint = path_or_checkpoint
                
            if print_:
                print("Loaded model from " + path_or_checkpoint)
        except:
            return False
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.curr_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        sleep(0.5)
        
        return True
    
    def try_load_else_fit(self, *args, **kwargs):
        if self._load_model(kwargs.get('save_model_path')):
            self.save_model_path = kwargs.get('save_model_path')
            return
        else:
            return self.fit(*args, **kwargs)
    
    def fit(self, train_loader, val_loader, p_weight=None, save_model_path=None, max_epochs=1000, save_every_n_epochs=10, patience=10, scheduler=None, trial=None, sparse_fit=False):
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
        self.save_model_path = save_model_path
        p_weight = p_weight.to(self.device)
        
        self.earlyStopping = EarlyStopping(patience=patience)
        self._load_model(save_model_path)
        
        if sparse_fit:
            T_max = len(train_loader) * max_epochs
            decay = CosineDecay(prune_rate=0.2, T_max=T_max)
            mask = Masking(self.optimizer, prune_rate_decay=decay)
            mask.add_module(self.bottleneck, density=0.05)
        
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
                    
                    self.update_ema_gradient()
                    
                    if sparse_fit:
                        mask.step()
                    else:
                        self.optimizer.step()
                
                if sparse_fit:
                    mask.at_end_of_epoch()
                
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
        if self.earlyStopping.best_state:
            self._load_model(self.earlyStopping.best_state)
        
        return self.val_losses[-1]

