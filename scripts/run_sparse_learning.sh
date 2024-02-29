#!/bin/bash

mkdir -p ./logs/sparse_learning/

# Run commands for mimic dataset
nohup python3 launch.py --dataset mimic --model original --pruning sparse_learning --device "cuda" > ./logs/sparse_learning/mimic_original_sparse_learning.log 2>&1 &
nohup python3 launch.py --dataset mimic --model shared --pruning sparse_learning --device "cuda" > ./logs/sparse_learning/mimic_shared_sparse_learning.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning sparse_learning --device "cuda" > ./logs/sparse_learning/mimic_atomics_sparse_learning.log 2>&1 &

nohup python3 launch.py --dataset mimic --model shared --pruning sparse_learning --switch_encode_dim --device "cuda" > ./logs/sparse_learning/mimic_shared_sparse_learning2.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning sparse_learning --switch_summaries_layer --device "cuda" > ./logs/sparse_learning/mimic_atomics_sparse_learning2.log 2>&1 &

# Run commands for tiselac dataset
nohup python3 launch.py --dataset tiselac --model original --pruning sparse_learning --device "cuda:1" > ./logs/sparse_learning/tiselac_original_sparse_learning.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model shared --pruning sparse_learning --device "cuda:1" > ./logs/sparse_learning/tiselac_shared_sparse_learning.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning sparse_learning --device "cuda:1" > ./logs/sparse_learning/tiselac_atomics_sparse_learning.log 2>&1 &

nohup python3 launch.py --dataset tiselac --model shared --pruning sparse_learning --switch_encode_dim --device "cuda:1" > ./logs/sparse_learning/tiselac_shared_sparse_learning2.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning sparse_learning --switch_summaries_layer --device "cuda:1" > ./logs/sparse_learning/tiselac_atomics_sparse_learning2.log 2>&1 &

# Run commands for spoken_arabic_digits dataset
nohup python3 launch.py --dataset spoken_arabic_digits --model original --pruning sparse_learning --device "cuda:2" > ./logs/sparse_learning/spoken_arabic_digits_original_sparse_learning.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning sparse_learning --device "cuda:2" > ./logs/sparse_learning/spoken_arabic_digits_shared_sparse_learning.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning sparse_learning --device "cuda:2" > ./logs/sparse_learning/spoken_arabic_digits_atomics_sparse_learning.log 2>&1 &

nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning sparse_learning --switch_encode_dim --device "cuda:2" > ./logs/sparse_learning/spoken_arabic_digits_shared_sparse_learning2.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning sparse_learning --switch_summaries_layer --device "cuda:2" > ./logs/sparse_learning/spoken_arabic_digits_atomics_sparse_learning2.log 2>&1 &
