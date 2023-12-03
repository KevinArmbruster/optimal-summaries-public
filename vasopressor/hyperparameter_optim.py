# %%
import os
import numpy as np
import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_timeline
import torch
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from preprocess_helpers import myPreprocessed
from models import CBM


# %%
X_np, Y_logits, changing_vars, data_cols = myPreprocessed()
input_dim = X_np.shape[2]
changing_dim = len(changing_vars)

# %%
DEVICE = torch.device("cuda")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 1000


# %%
def tensor_wrap(x, klass=torch.Tensor):
    return x if 'torch' in str(type(x)) else klass(x)

def initializeData(rnd):
    # train-test-split
    torch.set_printoptions(sci_mode=False)
    X_train, X_test, y_train, y_test = train_test_split(X_np, Y_logits, test_size = 0.15, random_state = rnd, stratify = Y_logits)

    # train-val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.20, random_state = rnd, stratify = y_train)

    # X_pt = Variable(tensor_wrap(X_np)).cuda()
    
    # print("y_test information")
    # print(sum(np.array(y_test)[:, 1]==1))
    # print([i for i, x in enumerate(np.array(y_test)[:, 1]) if x==1])
    
    # initiazing datasets
    pos_prop = np.mean(np.array(Y_logits)[:, 1])

    p_weight = torch.Tensor([1 / (1 - pos_prop), 1 / pos_prop]).cuda()

    X_train_pt = Variable(tensor_wrap(X_train)).cuda()
    y_train_pt = Variable(tensor_wrap(y_train, torch.FloatTensor)).cuda()

    X_val_pt = Variable(tensor_wrap(X_val)).cuda()
    y_val_pt = Variable(tensor_wrap(y_val, torch.FloatTensor)).cuda()

    X_test_pt = Variable(tensor_wrap(X_test)).cuda()
    y_test_pt = Variable(tensor_wrap(y_test, torch.FloatTensor)).cuda()

    train_dataset = TensorDataset(X_train_pt, y_train_pt)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)

    val_dataset = TensorDataset(X_val_pt, y_val_pt)
    val_loader = DataLoader(val_dataset, batch_size = X_val_pt.shape[0], shuffle=True, num_workers=0)

    test_dataset = TensorDataset(X_test_pt, y_test_pt)
    test_loader = DataLoader(test_dataset, batch_size=X_test_pt.shape[0], shuffle=True, num_workers=0)
    
    return train_loader, val_loader, X_test, y_test, p_weight


# %%

def init_cutoffs_to_zero(d):
    return np.zeros(d)

# init the upper and lower thresholds to random values
def init_rand_upper_thresholds(d):
    return np.random.rand(d)

def init_rand_lower_thresholds(d):
    return np.random.rand(d) - 1

def init_zeros(d):
    return np.zeros(d)

# %%
def initializeModel(trial, num_concepts = 4):
#     vals_to_init = init_cutoffs_randomly(changing_dim * 9)
    logregbottleneck = CBM(input_dim, 
                                                changing_dim, 
                                                9,                     
                                                num_concepts,
                                                True,
                                                init_cutoffs_to_zero, 
                                                init_rand_lower_thresholds, 
                                                init_rand_upper_thresholds,
                                                cutoff_times_temperature=0.1,
                                                cutoff_times_init_values=None,
                                                opt_lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
                                                opt_weight_decay=trial.suggest_float("wd", 1e-5, 1e-1, log=True),
                                                l1_lambda=0.001, #trial.suggest_float("l1", 1e-5, 1e-1, log=True),
                                                cos_sim_lambda=0.01, #trial.suggest_float("cossim", 1e-5, 1e-1, log=True),
                                                )
    logregbottleneck.cuda()
    return logregbottleneck


# %%
def objective(trial):
    # Generate the model.
    model = initializeModel(trial)

    # Get the FashionMNIST dataset.
    train_loader, val_loader, X_test, y_test, p_weight  = initializeData(rnd=1)
    
    train_loss, val_loss = model.fit(train_loader, val_loader, p_weight,
         save_model_path = "",
         epochs=EPOCHS,
         save_every_n_epochs=100)
    return val_loss


# %%

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_jobs=10, n_trials=100, timeout=60 * 60 * 19)

fig = plot_optimization_history(study)
fig.write_image("plot_optimization_history.png") 
fig = plot_param_importances(study)
fig.write_image("plot_param_importances.png") 
fig = plot_timeline(study)
fig.write_image("plot_timeline.png") 

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
