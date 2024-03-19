#!/bin/bash

log_path="./logs/importance"

mkdir -p $log_path

# Run commands for mimic dataset
device="cuda:3"
nohup python3 launch_train_prune.py --dataset mimic --model original --pruning importance --device "$device" > $log_path/mimic_original_importance.log 2>&1 &
sleep 1
nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning importance --device "$device" > $log_path/mimic_shared_importance.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning importance --device "$device" > $log_path/mimic_atomics_importance.log 2>&1 &

nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning importance --switch_encode_dim --device "$device" > $log_path/mimic_shared_importance2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning importance --switch_summaries_layer --device "$device" > $log_path/mimic_atomics_importance2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
device="cuda:4"
nohup python3 launch_train_prune.py --dataset tiselac --model original --pruning importance --device "$device" > $log_path/tiselac_original_importance.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning importance --device "$device" > $log_path/tiselac_shared_importance.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning importance --device "$device" > $log_path/tiselac_atomics_importance.log 2>&1 &

nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning importance --switch_encode_dim --device "$device" > $log_path/tiselac_shared_importance2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning importance --switch_summaries_layer --device "$device" > $log_path/tiselac_atomics_importance2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
device="cuda:5"
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model original --pruning importance --device "$device" > $log_path/spoken_arabic_digits_original_importance.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning importance --device "$device" > $log_path/spoken_arabic_digits_shared_importance.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning importance --device "$device" > $log_path/spoken_arabic_digits_atomics_importance.log 2>&1 &

nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning importance --switch_encode_dim --device "$device" > $log_path/spoken_arabic_digits_shared_importance2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning importance --switch_summaries_layer --device "$device" > $log_path/spoken_arabic_digits_atomics_importance2.log 2>&1 &
