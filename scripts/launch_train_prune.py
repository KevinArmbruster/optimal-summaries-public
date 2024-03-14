
import sys
sys.path.append('..')

import argparse
from itertools import permutations

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
parser.add_argument('--pruning', type=str, choices=['greedy', 'mixed_greedy', 'weight_magnitude', "gradient_magnitude", "weight_gradient_magnitude", 'sparse_learning'])
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--save_load_path', type=str, default="/workdir/optimal-summaries-public/_models_train_prune/")

# configurable default model options
parser.add_argument('--n_concepts', type=int, default=4)
parser.add_argument('--n_atomics', type=int, default=10)
parser.add_argument('--switch_encode_dim', action='store_false', default=True)
parser.add_argument('--switch_summaries_layer', action='store_false', default=True)
parser.add_argument('--switch_indicators', action='store_false', default=True)
parser.add_argument('--switch_use_only_last_timestep', action='store_true', default=False)
parser.add_argument('--switch_use_summaries', action='store_false', default=True)

args = parser.parse_args()

print("All arguments:")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")


### Helper functions

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


def get_model(random_state):
    set_seed(random_state)
    
    train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state)
    
    if args.model == "original":
        model = models_original.CBM(n_concepts=args.n_concepts, use_indicators=args.switch_indicators, use_only_last_timestep=args.switch_use_only_last_timestep, use_summaries=args.switch_use_summaries, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=num_classes, device=args.device)
    elif args.model == "shared":
        model = models_3d.CBM(n_concepts=args.n_concepts, encode_time_dim=args.switch_encode_dim, use_indicators=args.switch_indicators, use_summaries=args.switch_use_summaries, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=num_classes, device=args.device)
    elif args.model == "atomics":
        model = models_3d_atomics.CBM(n_concepts=args.n_concepts, n_atomics=args.n_atomics, use_summaries_for_atomics=args.switch_summaries_layer, use_indicators=args.switch_indicators, use_summaries=args.switch_use_summaries, static_dim=static_dim, changing_dim=changing_dim, seq_len=seq_len, output_dim=num_classes, device=args.device)
    else:
        print("No known model selected")
        sys.exit(1)
    return model


def get_trained_model(random_state):
    set_seed(random_state)

    train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state)
    
    model = get_model(random_state)
    model_path = model.get_model_path(base_path=args.save_load_path, dataset=args.dataset, pruning=args.pruning, seed=random_state)
    model.try_load_else_fit(train_loader, val_loader, p_weight=class_weights, save_model_path=model_path, max_epochs=10000, save_every_n_epochs=10, patience=10, sparse_fit=False)

    evaluate_classification(model=model, dataloader=val_loader, num_classes=num_classes)
    
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



### Individual experiments

makedir(args.save_load_path)
models = []


if args.pruning == "greedy":
    
    results = []
    
    for random_state in args.random_states:
        model = get_trained_model(random_state)
        models.append(model)
        train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state)
        
        # greedy search
        track_metrics = get_metrics(num_classes)
        top_k_inds = [get_top_features_per_concept(layer) for layer in model.regularized_layers]
        
        greedy_path = add_subfolder(model.save_model_path, "top-k") + ".csv"
        makedir(greedy_path)
        
        greedy_results = greedy_forward_selection(model=model, layers_to_prune=model.regularized_layers, top_k_inds=top_k_inds, val_loader=val_loader, optimize_metric=track_metrics["auc"], track_metrics=track_metrics, save_path=greedy_path)
        results.append(greedy_results)
    
    result_df = evaluate_greedy_selection(models, results, get_dataloader, dataset=args.dataset, random_states=args.random_states)
    
    
elif args.pruning == "mixed_greedy" and args.model == "atomics":
    
    # TODO not finished, replace greedy search of shared layer by importance pruning, then greedy search 2nd
    pass
    
    
elif args.pruning in ('weight_magnitude', "gradient_magnitude", "weight_gradient_magnitude"):
    
    result_df = pd.DataFrame(columns=["Model", "Dataset", "Seed", "Split", "Pruning", "Finetuned", "AUC", "ACC", "F1", "Total parameter", "Remaining parameter"])
    
    for random_state in args.random_states:
        model = get_trained_model(random_state)
        models.append(model)
        train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state = random_state)
        
        # base
        total, remaining = get_total_and_remaining_parameters_from_masks(model.regularized_layers)
        
        metrics = evaluate_classification(model, val_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Dataset": args.dataset, "Seed": random_state, "Split": "val", "Pruning": "Before", "Finetuned": False, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        metrics = evaluate_classification(model, test_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Dataset": args.dataset, "Seed": random_state, "Split": "test", "Pruning": "Before", "Finetuned": False, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        
        
        # prune and finetune
        new_model_path = add_subfolder(model.save_model_path, "finetuned")
        makedir(new_model_path)
        
        start_n_weights = [layer.weight.numel() for layer in model.regularized_layers]
        end_n_weights = [layer.weight.shape[0] * 10 for layer in model.regularized_layers] # feature budget
        
        amount_of_pruning_steps = 16
        iterative_steps = [list(np.linspace(start, end, amount_of_pruning_steps, dtype=int))[1:] for start, end in zip(start_n_weights, end_n_weights)]
        
        # fill ema gradient by fit -> repeat: mask, clear, fit, evaluate
        model.fit(train_loader, val_loader, p_weight=class_weights, save_model_path=new_model_path, max_epochs=1, save_every_n_epochs=1, patience=1)
        
        for step in zip(*iterative_steps):
            
            if args.pruning == "weight_magnitude":
                model.mask_by_weight_magnitude(step)
                
            elif args.pruning == "gradient_magnitude":
                model.mask_by_gradient_magnitude(step)
                
            elif args.pruning == "weight_gradient_magnitude":
                model.mask_by_weight_gradient_magnitude(step)
                
            model.clear_ema_gradient()
            model.fit(train_loader, val_loader, p_weight=class_weights, save_model_path=new_model_path, max_epochs=10000, save_every_n_epochs=1, patience=10)
                    
        total, remaining = get_total_and_remaining_parameters_from_masks(model.regularized_layers)
        
        metrics = evaluate_classification(model, val_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Dataset": args.dataset, "Seed": random_state, "Split": "val", "Pruning": args.pruning, "Finetuned": True, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        metrics = evaluate_classification(model, test_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Dataset": args.dataset, "Seed": random_state, "Split": "test", "Pruning": args.pruning, "Finetuned": True, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}


elif args.pruning == "sparse_learning":
    
    result_df = pd.DataFrame(columns=["Model", "Dataset", "Seed", "Split", "Pruning", "Finetuned", "AUC", "ACC", "F1", "Total parameter", "Remaining parameter"])
    
    for random_state in args.random_states:
        
        model = get_model(random_state)
        models.append(model)
        train_loader, val_loader, test_loader, class_weights, num_classes, changing_dim, static_dim, seq_len = get_dataloader(random_state)
        model_path = model.get_model_path(base_path=args.save_load_path, dataset=args.dataset, pruning=args.pruning, seed=random_state)
        model.try_load_else_fit(train_loader, val_loader, p_weight=class_weights, save_model_path=model_path, max_epochs=10000, save_every_n_epochs=10, patience=10, sparse_fit=True)
        
        # sparse learning does not create weight masks, but sets weights to 0, just for identical models
        for layer in model.regularized_layers:
            layer.set_weight_mask_from_weight()
        
        total, remaining = get_total_and_remaining_parameters_from_masks(model.regularized_layers)
        
        metrics = evaluate_classification(model, val_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Dataset": args.dataset, "Seed": random_state, "Split": "val", "Pruning": "sparse_learning", "Finetuned": True, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        metrics = evaluate_classification(model, test_loader)
        result_df.loc[len(result_df)] = {"Model": model.get_short_model_name(), "Dataset": args.dataset, "Seed": random_state, "Split": "test", "Pruning": "sparse_learning", "Finetuned": True, "AUC": metrics[0], "ACC": metrics[1], "F1": metrics[2], "Total parameter": total, "Remaining parameter": remaining}
        
    
else:
    print("No known pruning method selected")
    sys.exit(1)


### Compute stability

jac_sim = score_pruning_stability(models)
result_df['Jaccard Similarity'] = jac_sim


### Write results

results_path = model.get_model_path(base_path=args.save_load_path, dataset=args.dataset, pruning=args.pruning, ending="_results.csv")
results_path = add_subfolder(results_path, "results")
write_df_2_csv(results_path, result_df)

print("Done")
