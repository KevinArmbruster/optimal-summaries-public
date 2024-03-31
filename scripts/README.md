
# Scripts

This folder contains all scripts used to execute the final experiments.

[Top-level README](../README.md)

## Scripts

* Backbone script that launchs one experiment [launch_train_prune.py](/optimal-summaries-public/scripts/launch_train_prune.py)
* Notebook to debug ``launch_train_prune.py`` [notebook_launch_train_prune.ipynb](/optimal-summaries-public/scripts/notebook_launch_train_prune.ipynb)
* Backbone script that launchs one ablation setting [launch_ablation.py](/optimal-summaries-public/scripts/launch_ablation.py)
* Run all experiments of a pruning algorithm ``run*.sh``

## Helpful commands to find or delete model directories

* ```find /workdir/optimal-summaries-public/_models -type d -name "weight_magnitude" -print```
* ```find /workdir/optimal-summaries-public/_models -type d -name "weight_magnitude" -exec rm -rf {} +```
