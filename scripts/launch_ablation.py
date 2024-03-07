
import sys
sys.path.append('..')

import argparse
from itertools import product

import models.models_original as models_original
import models.models_3d_atomics as models_3d_atomics
import models.models_3d as models_3d
from models.data import *
from models.helper import *
from models.param_initializations import *
from models.optimization_strategy import *


parser = argparse.ArgumentParser()
parser.add_argument('--random_states', type=int, nargs="+", default=[1,2,3])
parser.add_argument('--dataset', type=str, choices=['mimic', 'tiselac', 'spoken_arabic_digits'])
parser.add_argument('--model', type=str, choices=['original', 'shared', 'atomics'])
# parser.add_argument('--pruning', type=str, choices=['greedy', 'mixed_greedy', 'weight_magnitude', "gradient_magnitude", "weight_gradient_magnitude", 'sparse_learning'])
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--save_load_path', type=str, default="/workdir/optimal-summaries-public/_models_ablation/")

# configurable default model options
parser.add_argument('--n_concepts', type=int, default=4)
parser.add_argument('--n_atomics', type=int, default=10)
parser.add_argument('--switch_encode_dim', action='store_false', default=True)
parser.add_argument('--switch_summaries_layer', action='store_false', default=True)
parser.add_argument('--switch_use_only_last_timestep', action='store_true', default=False)

args = parser.parse_args()

print("All arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")


def get_dataloader(random_state):
    set_seed(random_state)

    if args.dataset == "mimic":
        return get_MIMIC_dataloader(random_state = random_state)
    elif args.dataset == "tiselac":
        return get_tiselac_dataloader(random_state = random_state)
    elif args.dataset == "spoken_arabic_digits":
        return get_arabic_spoken_digits_dataloader(random_state = random_state)
    else:
        print("No known dataset selected")
        sys.exit(1)


def get_model(random_state, config):
    set_seed(random_state)
    
    train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state)
    
    if args.model == "original":
        model = models_original.CBM(**config, n_concepts=args.n_concepts, use_only_last_timestep=args.switch_use_only_last_timestep, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=num_classes, device=args.device)
    elif args.model == "shared":
        model = models_3d.CBM(**config, n_concepts=args.n_concepts, encode_time_dim=args.switch_encode_dim, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=num_classes, device=args.device)
    elif args.model == "atomics":
        model = models_3d_atomics.CBM(**config, n_concepts=args.n_concepts, n_atomics=args.n_atomics, use_summaries_for_atomics=args.switch_summaries_layer, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=num_classes, device=args.device)
    else:
        print("No known model selected")
        sys.exit(1)
    return model


def get_metrics(num_classes):
    if num_classes == 2:
        auroc_metric = AUROC(task="binary").to(args.device)
        accuracy_metric = Accuracy(task="binary").to(args.device)
        f1_metric = F1Score(task="binary").to(args.device)
        # conf_matrix = ConfusionMatrix(task="binary").to(args.device)
    else:
        average = "macro"
        auroc_metric = AUROC(task="multiclass", num_classes=num_classes, average = average).to(args.device)
        accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1, average = average).to(args.device)
        f1_metric = F1Score(task="multiclass", num_classes=num_classes, top_k=1, average = average).to(args.device)
        # conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(args.device)
    
    return {"acc": accuracy_metric, "f1": f1_metric, "auc": auroc_metric}



makedir(args.save_load_path)


permutations = {'use_summaries': [True, False],
                'use_indicators': [True, False], }
permutations = [dict(zip(permutations.keys(), values)) for values in product(*permutations.values())]


result_df = pd.DataFrame(columns=["Model", "Indicators", "Summaries", "Dataset", "Seed", "Split", "AUC", "ACC", "F1", "Cutoff", "Lower threshold", "Upper threshold"])

for config in permutations:
    print("Config", config)
    
    for random_state in args.random_states:
        set_seed(random_state)
        
        train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state)
        
        model = get_model(random_state, config)
        model_path = model.get_model_path(base_path=args.save_load_path, dataset=args.dataset, seed=random_state)
        model.try_load_else_fit(train_loader, val_loader, p_weight=class_weights, save_model_path=model_path, max_epochs=10000, save_every_n_epochs=10, patience=10, sparse_fit=False)
        
        
        cutoff_percentage_mean_per_var = model.cutoff_percentage.detach().view(model.changing_dim, -1).round(decimals=2).mean(dim=1).squeeze().tolist()
        lower_thresholds = model.lower_thresholds.detach().round(decimals=2).squeeze().tolist()
        upper_thresholds = model.upper_thresholds.detach().round(decimals=2).squeeze().tolist()
        
        
        metrics = evaluate_classification(model, val_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Indicators": model.use_indicators, "Summaries": model.use_summaries, "Dataset": args.dataset, "Seed": random_state, "Split": "val", "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Cutoff": cutoff_percentage_mean_per_var, "Lower threshold": lower_thresholds, "Upper threshold": lower_thresholds}
        metrics = evaluate_classification(model, test_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Indicators": model.use_indicators, "Summaries": model.use_summaries, "Dataset": args.dataset, "Seed": random_state, "Split": "test", "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Cutoff": cutoff_percentage_mean_per_var, "Lower threshold": lower_thresholds, "Upper threshold": lower_thresholds}
        

results_path = model.get_model_path(base_path=args.save_load_path, dataset=args.dataset, ending="_results.csv")
results_path = add_subfolder(results_path, "results")
write_df_2_csv(results_path, result_df)



print("Done")
