import torch
import numpy as np

def calculate_summaries(model, patient_batch, use_indicators, use_fixes, use_only_last_timestep, epsilon_denom=1e-8):
    summaries = []
    
    # Computes the encoding (s, x) + (weighted_summaries) in the order defined in weight_parser.
    # Returns pre-sigmoid P(Y = 1 | patient_batch)
    temperatures = torch.tensor(np.full((1, model.cs_parser.num_weights), model.cutoff_percentage_temperature), device=model.device)
    
    # Get changing variables
    batch_changing_vars = patient_batch[:, :, :model.changing_dim]
    batch_measurement_inds = patient_batch[:, :, model.changing_dim: model.changing_dim * 2]
    batch_static_vars = patient_batch[:, 0, model.changing_dim * 2:] # static is the same accross time
    
    cutoff = ((model.seq_len+1) * torch.clip(model.cutoff_percentage, 0, 1)) - 1 # range [-1, seq_len], enables flexibility to use full or no time series values, without big bias
    weight_vector = model.sigmoid_layer((model.times - cutoff) / temperatures).reshape(1, model.seq_len, model.cs_parser.num_weights)
    
    
    # MEAN FEATURES
    # Calculate \sum_t (w_t * x_t * m_t)
    start_i, end_i = model.cs_parser.idxs_and_shapes['cs_mean_']
    mean_weight_vector = weight_vector[:, :, start_i : end_i]
    
    weighted_average = torch.sum(mean_weight_vector * (batch_changing_vars * batch_measurement_inds), dim=1)

    if not use_fixes:
        mean_feats = weighted_average / (torch.sum(mean_weight_vector, dim=1) + epsilon_denom)
    else:
        # TODO denom forgot batch_measurement_inds
        mean_feats = weighted_average / (torch.sum(mean_weight_vector * batch_measurement_inds, dim=1) + epsilon_denom)
    summaries.append(mean_feats.float())
    
    # VARIANCE FEATURES
    start_i, end_i = model.cs_parser.idxs_and_shapes['cs_var_']
    var_weight_vector = weight_vector[:, :, start_i : end_i]
    
    if not use_fixes:
        x_mean = torch.mean(batch_measurement_inds * batch_changing_vars, dim=1, keepdim=True)
    else:
        # TODO x bar is not normal mean, but divide by sum of M
        x_mean = torch.sum(batch_measurement_inds * batch_changing_vars, dim=1, keepdim=True) / (torch.sum(batch_measurement_inds, dim=1, keepdim=True) + epsilon_denom)

    weighted_variance = torch.sum(batch_measurement_inds * var_weight_vector * (batch_changing_vars - x_mean)**2, dim=1)
    
    squared_sum_of_weights = torch.sum(batch_measurement_inds * var_weight_vector, dim=1)**2
    sum_of_squared_weights = torch.sum(batch_measurement_inds * var_weight_vector ** 2, dim=1)
    
    normalizing_term = squared_sum_of_weights / (squared_sum_of_weights + sum_of_squared_weights + epsilon_denom)
    if not use_fixes:
        var_feats = weighted_variance / (normalizing_term + epsilon_denom)
    else:
        # TODO should be * not /
        var_feats = normalizing_term * weighted_variance
    summaries.append(var_feats.float())
    
    
    if use_indicators:
        # INDICATOR FOR EVER BEING MEASURED
        start_i, end_i = model.cs_parser.idxs_and_shapes['cs_ever_measured_']
        ever_measured_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_ind_average = torch.sum(ever_measured_weight_vector * batch_measurement_inds, dim=1)
        pre_sigmoid = weighted_ind_average / (model.ever_measured_temperature * torch.sum(ever_measured_weight_vector, dim=1) + epsilon_denom)
        ever_measured_feats = model.sigmoid_layer(pre_sigmoid) - 0.5
        summaries.append(ever_measured_feats.float())
    
    
        # MEAN OF INDICATOR SEQUENCE
        start_i, end_i = model.cs_parser.idxs_and_shapes['cs_mean_indicators_']
        mean_ind_weight_vector = weight_vector[:, :, start_i : end_i]
        
        weighted_ind_average = torch.sum(mean_ind_weight_vector * batch_measurement_inds, dim=1)
        mean_ind_feats = weighted_ind_average / (torch.sum(mean_ind_weight_vector, dim=1) + epsilon_denom)
        summaries.append(mean_ind_feats.float())
    
    
        # VARIANCE OF INDICATOR SEQUENCE
        start_i, end_i = model.cs_parser.idxs_and_shapes['cs_var_indicators_']
        var_ind_weight_vector = weight_vector[:, :, start_i : end_i]
                
        x_mean_ind = torch.mean(batch_measurement_inds, dim=1, keepdim=True)
        weighted_variance_ind = torch.sum(var_ind_weight_vector * (batch_measurement_inds - x_mean_ind)**2, dim=1)     
        
        squared_sum_of_weights = torch.sum(var_ind_weight_vector, dim=1)**2
        sum_of_squared_weights = torch.sum(var_ind_weight_vector ** 2, dim=1)
        normalizing_term = squared_sum_of_weights / (squared_sum_of_weights + sum_of_squared_weights + epsilon_denom)
    
        if not use_fixes:
            var_ind_feats = weighted_variance_ind / (normalizing_term + epsilon_denom)
        else:
            # TODO should be * not /
            var_ind_feats = normalizing_term * weighted_variance_ind
        summaries.append(var_ind_feats.float())
    
    
        # COUNT OF SWITCHES
        # Compute the number of times the indicators switch from missing to measured, or vice-versa.
        start_i, end_i = model.cs_parser.idxs_and_shapes['cs_switches_']
        switches_weight_vector = weight_vector[:, :, start_i : end_i][:, :-1, :]
        
        # Calculate m_{n t + 1} - m_{ n t}
        # Sum w_t + sigmoids of each difference
        later_times = batch_changing_vars[:, 1:, :]
        earlier_times = batch_changing_vars[:, :-1, :]
        
        switch_feats = torch.sum(switches_weight_vector * torch.abs(later_times - earlier_times), dim=1) / (torch.sum(switches_weight_vector, dim=1) + epsilon_denom)
        summaries.append(switch_feats.float())
        
        # FIRST TIME MEASURED
        # LAST TIME MEASURED
        
        # For each variable in the batch, compute the first time it was measured.
        # Set equal to -1 if never measured.
        
        # For each feature, calculate the first time it was measured
        # Index of the second dimension of the indicators

        mask_max_values, mask_max_indices = torch.max(batch_measurement_inds, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = -1
        
        first_time_feats = mask_max_indices / float(batch_measurement_inds.shape[1])
        summaries.append(first_time_feats.float())

        
        # Last time measured is the last index of the max.
        # https://discuss.pytorch.org/t/how-to-reverse-a-torch-tensor/382
        flipped_batch_measurement_inds = torch.flip(batch_measurement_inds, [1])
        
        mask_max_values, mask_max_indices = torch.max(flipped_batch_measurement_inds, dim=1)
        # if the max-mask is zero, there is no nonzero value in the row
        mask_max_indices[mask_max_values == 0] = batch_measurement_inds.shape[1]
        
        last_time_feats = (float(batch_measurement_inds.shape[1]) - mask_max_indices) / float(batch_measurement_inds.shape[1])
        summaries.append(last_time_feats.float())

    
    # SLOPE OF L2
    # STANDARD ERROR OF L2     
    start_i, end_i = model.cs_parser.idxs_and_shapes['cs_slope_']
    slope_weight_vector = weight_vector[:, :, start_i : end_i]
    
    # Zero out the batch_changing_vars so that they are zero if the features are not measured.
    linreg_y = batch_changing_vars * batch_measurement_inds
    
    # The x-values for this linear regression are the times.
    # Zero them out so that they are zero if the features are not measured.
    linreg_x = torch.tensor(np.transpose(np.tile(range(model.seq_len), (model.changing_dim, 1))), device=model.device)
    linreg_x = linreg_x.repeat(linreg_y.shape[0], 1, 1) * batch_measurement_inds
    
    # Now, compute the slope and standard error.
    weighted_x = torch.unsqueeze(torch.sum(slope_weight_vector * linreg_x, dim = 1) / (torch.sum(slope_weight_vector, dim = 1) + epsilon_denom), 1)
    weighted_y = torch.unsqueeze(torch.sum(slope_weight_vector * linreg_y, dim = 1) / (torch.sum(slope_weight_vector, dim = 1) + epsilon_denom), 1)
    
    slope_num = torch.sum(slope_weight_vector * (linreg_x - weighted_x) * (linreg_y - weighted_y), dim=1)
    slope_den = torch.sum(slope_weight_vector * (linreg_x - weighted_x)**2, dim =1)
    
    slope_feats = slope_num / (slope_den + epsilon_denom)
    summaries.append(slope_feats.float())
    
    
    # If the denominator is zero, set the feature equal to 0.
    var_denom = torch.sum(slope_weight_vector * (linreg_x - weighted_x)**2, dim=1)
    slope_stderr_feats = 1 / (var_denom + epsilon_denom)
    
    slope_stderr_feats = torch.where(var_denom > 0, slope_stderr_feats, var_denom)
    # TODO slope_weight_vector could be negative, just check for 0
    # slope_stderr_feats = torch.where(var_denom == 0, slope_stderr_feats, var_denom)
    summaries.append(slope_stderr_feats.float())
    
    
    # HOURS ABOVE THRESHOLD
    # HOURS BELOW THRESHOLD
    start_i, end_i = model.cs_parser.idxs_and_shapes['cs_hours_above_threshold_']
    above_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
    
    start_i, end_i = model.cs_parser.idxs_and_shapes['cs_hours_below_threshold_']
    below_thresh_weight_vector = weight_vector[:, :, start_i : end_i]
    
    upper_features = model.sigmoid_layer((batch_changing_vars - model.upper_thresholds) / model.thresh_temperature)
    lower_features = model.sigmoid_layer((model.lower_thresholds - batch_changing_vars) / model.thresh_temperature)
    
    # sum upper_features and lower_features across timesteps
    above_tmp = batch_measurement_inds * above_thresh_weight_vector
    above_threshold_feats = torch.sum(above_tmp * upper_features, dim=1) / (torch.sum(above_tmp, dim=1) + epsilon_denom)
    summaries.append(above_threshold_feats.float())
    
    below_tmp = batch_measurement_inds * below_thresh_weight_vector
    below_threshold_feats = torch.sum(below_tmp * lower_features, dim=1) / (torch.sum(below_tmp, dim=1) + epsilon_denom)
    summaries.append(below_threshold_feats.float())
    
    
    ### 
    if use_only_last_timestep:
        time_feats_2d = patient_batch[:, model.seq_len-1, :]
        
        # # b t v
        # batch_changing_vars = batch_changing_vars[:, model.seq_len-1, :]
        # batch_measurement_inds = batch_measurement_inds[:, model.seq_len-1, :]
    else:
        # use full timeseries, reshape 3d to 2d, keep sample size N and merge Time x Variables (N x T x V) => (N x T*V)
        changing_vars_2d = batch_changing_vars.reshape(batch_changing_vars.shape[0], -1) # result is V1_T1, V2_T1, V3_T1, ..., V1_T2, V2_T2, V3_T2, ... repeat V, T times
        indicators_2d = batch_measurement_inds.reshape(batch_measurement_inds.shape[0], -1)
        time_feats_2d = torch.cat((changing_vars_2d, indicators_2d, batch_static_vars), dim=1)
    
    # print("summaries", len(summaries))
    # print("mean_feats", summaries[0].shape) # torch.Size([512, 7])
    # print("var_feats", summaries[1].shape) # torch.Size([512, 7])
    
    # print("batch_changing_vars", batch_changing_vars.shape, (batch_changing_vars is None or torch.isnan(batch_changing_vars).any()))
    # print("batch_measurement_inds", batch_measurement_inds.shape, (batch_measurement_inds is None or torch.isnan(batch_measurement_inds).any()))
    # print("batch_static_vars", batch_static_vars.shape, (batch_static_vars is None or torch.isnan(batch_static_vars).any()))
    # print("summaries", len(summaries), [(summ is None or torch.isnan(summ).any()).item() for summ in summaries])

    return batch_changing_vars, batch_measurement_inds, batch_static_vars, summaries, time_feats_2d
