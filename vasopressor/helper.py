from typing import List

summary_dict = {0:'mean', 1:'var', 2:'ever measured', 3:'mean of indicators', 4: 'var of indicators', 5:'# switches', 6:'slope', 7:'slope std err', 8:'first time measured', 9:'last time measured', 10:'hours above threshold', 11:'hours below threshold'}

def get_name_of_feature(ind: int, changing_variables_names: List[str], time_len: int = 1, static_variable_names: List[str] = []):
    # model input row to bottleneck:
    # Changing_vars per T
    # Indicators per T
    # Static features
    # Summary features per changing_vars
    
    variable_name_list = []
    indicator_names = [f"{name}_ind" for name in changing_variables_names]
    
    # result is = V1_T1, V2_T1, V3_T1, ..., V1_T2, V2_T2, V3_T2, ... repeat V for T times
    changing_vars_per_time = [f"{name}_time_{t}" for t in range(1, time_len + 1) for name in changing_variables_names]
    indicators_per_time = [f"{name}_time_{t}" for t in range(1, time_len + 1) for name in indicator_names]
    
    # create final name list
    variable_name_list = changing_vars_per_time + indicators_per_time + static_variable_names
    non_summary_dim = len(variable_name_list)
    
    # get feature name from lists and summary
    if ind < non_summary_dim:
        # raw feature
        return variable_name_list[ind], 'raw'
    else:
        # summary statistic of feature
        ind = ind - non_summary_dim
        summary = ind // len(changing_variables_names)
        feature = ind % len(changing_variables_names)
        return changing_variables_names[feature], summary_dict[summary]
