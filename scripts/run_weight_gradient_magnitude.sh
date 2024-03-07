#!/bin/bash

log_path="./logs/weight_gradient_magnitude"

mkdir -p $log_path

# Run commands for mimic dataset
device="cuda:10"
nohup python3 launch_train_prune.py --dataset mimic --model original --pruning weight_gradient_magnitude --device "$device" > $log_path/mimic_original_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning weight_gradient_magnitude --device "$device" > $log_path/mimic_shared_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning weight_gradient_magnitude --device "$device" > $log_path/mimic_atomics_weight_gradient_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning weight_gradient_magnitude --switch_encode_dim --device "$device" > $log_path/mimic_shared_weight_gradient_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning weight_gradient_magnitude --switch_summaries_layer --device "$device" > $log_path/mimic_atomics_weight_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
device="cuda:11"
nohup python3 launch_train_prune.py --dataset tiselac --model original --pruning weight_gradient_magnitude --device "$device" > $log_path/tiselac_original_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning weight_gradient_magnitude --device "$device" > $log_path/tiselac_shared_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning weight_gradient_magnitude --device "$device" > $log_path/tiselac_atomics_weight_gradient_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning weight_gradient_magnitude --switch_encode_dim --device "$device" > $log_path/tiselac_shared_weight_gradient_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning weight_gradient_magnitude --switch_summaries_layer --device "$device" > $log_path/tiselac_atomics_weight_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
device="cuda:12"
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model original --pruning weight_gradient_magnitude --device "$device" > $log_path/spoken_arabic_digits_original_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning weight_gradient_magnitude --device "$device" > $log_path/spoken_arabic_digits_shared_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning weight_gradient_magnitude --device "$device" > $log_path/spoken_arabic_digits_atomics_weight_gradient_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning weight_gradient_magnitude --switch_encode_dim --device "$device" > $log_path/spoken_arabic_digits_shared_weight_gradient_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning weight_gradient_magnitude --switch_summaries_layer --device "$device" > $log_path/spoken_arabic_digits_atomics_weight_gradient_magnitude2.log 2>&1 &
