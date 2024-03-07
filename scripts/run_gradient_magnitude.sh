#!/bin/bash

log_path="./logs/gradient_magnitude"

mkdir -p $log_path

# Run commands for mimic dataset
device="cuda:13"
nohup python3 launch_train_prune.py --dataset mimic --model original --pruning gradient_magnitude --device "$device" > $log_path/mimic_original_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning gradient_magnitude --device "$device" > $log_path/mimic_shared_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning gradient_magnitude --device "$device" > $log_path/mimic_atomics_gradient_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset mimic --model shared --pruning gradient_magnitude --switch_encode_dim --device "$device" > $log_path/mimic_shared_gradient_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset mimic --model atomics --pruning gradient_magnitude --switch_summaries_layer --device "$device" > $log_path/mimic_atomics_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
device="cuda:14"
nohup python3 launch_train_prune.py --dataset tiselac --model original --pruning gradient_magnitude --device "$device" > $log_path/tiselac_original_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning gradient_magnitude --device "$device" > $log_path/tiselac_shared_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning gradient_magnitude --device "$device" > $log_path/tiselac_atomics_gradient_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset tiselac --model shared --pruning gradient_magnitude --switch_encode_dim --device "$device" > $log_path/tiselac_shared_gradient_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset tiselac --model atomics --pruning gradient_magnitude --switch_summaries_layer --device "$device" > $log_path/tiselac_atomics_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
device="cuda:15"
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model original --pruning gradient_magnitude --device "$device" > $log_path/spoken_arabic_digits_original_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning gradient_magnitude --device "$device" > $log_path/spoken_arabic_digits_shared_gradient_magnitude.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning gradient_magnitude --device "$device" > $log_path/spoken_arabic_digits_atomics_gradient_magnitude.log 2>&1 &

nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model shared --pruning gradient_magnitude --switch_encode_dim --device "$device" > $log_path/spoken_arabic_digits_shared_gradient_magnitude2.log 2>&1 &
nohup python3 launch_train_prune.py --dataset spoken_arabic_digits --model atomics --pruning gradient_magnitude --switch_summaries_layer --device "$device" > $log_path/spoken_arabic_digits_atomics_gradient_magnitude2.log 2>&1 &
