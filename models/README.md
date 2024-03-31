
# Models

Folder contains all models, losses, dataloader, early stopping, etc.

[Top-level README](../README.md)

## Models

* All models inherit from [Base Model](/optimal-summaries-public/models/BaseModel.py)
* [Original model](/optimal-summaries-public/models/models_original.py)
* [Shared models](/optimal-summaries-public/models/models_3d.py)
    * SharedFeature ``encode_time_dim = True``
    * SharedTime ``encode_time_dim = False``
* [2Layer models](/optimal-summaries-public/models/models_3d_atomics.py)
    * 2Layers H2A ``use_summaries_for_atomics = True``
    * 2Layers H2C ``use_summaries_for_atomics = False``

## Pruning

* [Greedy Wrapper](/optimal-summaries-public/models/optimization_strategy.py)
* Other pruning algorithms directly in [launch_train_prune.py](/optimal-summaries-public/scripts/launch_train_prune.py)

## Other

* [Differentiable Summaries](/optimal-summaries-public/models/differentiable_summaries.py)
* [Various helper functions](/optimal-summaries-public/models/helper.py)
