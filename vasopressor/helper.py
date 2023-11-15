
summary_dict = {0:'mean', 1:'var', 2:'ever measured', 3:'mean of indicators', 4: 'var of indicators', 5:'# switches', 6:'slope', 7:'slope std err', 8:'first time measured', 9:'last time measured', 10:'hours above threshold', 11:'hours below threshold'}

def getConcept(data_cols, input_dim, changing_dim, ind):
    if ind < input_dim:
        # raw feature
        return data_cols[ind], 'raw'
    else:
        # summary statistic of feature
        ind = ind - input_dim
        summary = ind // changing_dim
        feature = ind % changing_dim
        return data_cols[feature], summary_dict[summary]
