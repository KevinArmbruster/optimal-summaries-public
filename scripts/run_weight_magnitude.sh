#!/bin/bash

log_path="./logs/weight_magnitude"

mkdir -p $log_path

# Run commands for mimic dataset
device="cuda:6"
nohup python3 launch_train_prune.py --dataset mimic --model original --pruning weight_magnitude --device "$device" > $log_path/mimic_original_weight_magnitude.log 2>&1 &
sleep 1
nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning weight_magnitude --device "$device" > $log_path/mimic_shared_weight_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning weight_magnitude --device "$device" > $log_path/mimic_atomics_weight_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning weight_magnitude --switch_encode_dim --device "$device" > $log_path/mimic_shared_weight_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning weight_magnitude --switch_summaries_layer --device "$device" > $log_path/mimic_atomics_weight_magnitude2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
device="cuda:9"
nohup python3 launch_train_prune.py --dataset tiselac --model original --pruning weight_magnitude --device "$device" > $log_path/tiselac_original_weight_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning weight_magnitude --device "$device" > $log_path/tiselac_shared_weight_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning weight_magnitude --device "$device" > $log_path/tiselac_atomics_weight_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning weight_magnitude --switch_encode_dim --device "$device" > $log_path/tiselac_shared_weight_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning weight_magnitude --switch_summaries_layer --device "$device" > $log_path/tiselac_atomics_weight_magnitude2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
device="cuda:8"
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model original --pruning weight_magnitude --device "$device" > $log_path/spoken_arabic_digits_original_weight_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning weight_magnitude --device "$device" > $log_path/spoken_arabic_digits_shared_weight_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning weight_magnitude --device "$device" > $log_path/spoken_arabic_digits_atomics_weight_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning weight_magnitude --switch_encode_dim --device "$device" > $log_path/spoken_arabic_digits_shared_weight_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning weight_magnitude --switch_summaries_layer --device "$device" > $log_path/spoken_arabic_digits_atomics_weight_magnitude2.log 2>&1 &
