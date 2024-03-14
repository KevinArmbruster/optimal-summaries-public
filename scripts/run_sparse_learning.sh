#!/bin/bash

log_path="./logs/sparse_learning"

mkdir -p $log_path

# Run commands for mimic dataset
device="cuda:6"
nohup python3 launch_train_prune.py --dataset mimic --model original --pruning sparse_learning --device "$device" > $log_path/mimic_original_sparse_learning.log 2>&1 &
sleep 1
nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning sparse_learning --device "$device" > $log_path/mimic_shared_sparse_learning.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning sparse_learning --device "$device" > $log_path/mimic_atomics_sparse_learning.log 2>&1 &

nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning sparse_learning --switch_encode_dim --device "$device" > $log_path/mimic_shared_sparse_learning2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning sparse_learning --switch_summaries_layer --device "$device" > $log_path/mimic_atomics_sparse_learning2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
device="cuda:7"
nohup python3 launch_train_prune.py --dataset tiselac --model original --pruning sparse_learning --device "$device" > $log_path/tiselac_original_sparse_learning.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning sparse_learning --device "$device" > $log_path/tiselac_shared_sparse_learning.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning sparse_learning --device "$device" > $log_path/tiselac_atomics_sparse_learning.log 2>&1 &

nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning sparse_learning --switch_encode_dim --device "$device" > $log_path/tiselac_shared_sparse_learning2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning sparse_learning --switch_summaries_layer --device "$device" > $log_path/tiselac_atomics_sparse_learning2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
device="cuda:8"
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model original --pruning sparse_learning --device "$device" > $log_path/spoken_arabic_digits_original_sparse_learning.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning sparse_learning --device "$device" > $log_path/spoken_arabic_digits_shared_sparse_learning.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning sparse_learning --device "$device" > $log_path/spoken_arabic_digits_atomics_sparse_learning.log 2>&1 &

nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning sparse_learning --switch_encode_dim --device "$device" > $log_path/spoken_arabic_digits_shared_sparse_learning2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning sparse_learning --switch_summaries_layer --device "$device" > $log_path/spoken_arabic_digits_atomics_sparse_learning2.log 2>&1 &
