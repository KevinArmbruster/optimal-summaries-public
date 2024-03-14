#!/bin/bash

log_path="./logs/ablation"

mkdir -p "$log_path"

# Run commands for mimic dataset
device="cuda:3"
nohup python3 launch_ablation.py --dataset mimic --model original --device "$device" > "$log_path/mimic_original.log" 2>&1 &
sleep 1
nohup python3 launch_ablation.py --dataset mimic --model shared --device "$device" > "$log_path/mimic_shared.log" 2>&1 &
nohup python3 launch_ablation.py --dataset mimic --model atomics --device "$device" > "$log_path/mimic_atomics.log" 2>&1 &

nohup python3 launch_ablation.py --dataset mimic --model shared --switch_encode_dim --device "$device" > "$log_path/mimic_shared2.log" 2>&1 &
nohup python3 launch_ablation.py --dataset mimic --model atomics --switch_summaries_layer --device "$device" > "$log_path/mimic_atomics2.log" 2>&1 &

sleep 10

# Run commands for tiselac dataset
device="cuda:4"
nohup python3 launch_ablation.py --dataset tiselac --model original --device "$device" > "$log_path/tiselac_original.log" 2>&1 &
nohup python3 launch_ablation.py --dataset tiselac --model shared --device "$device" > "$log_path/tiselac_shared.log" 2>&1 &
nohup python3 launch_ablation.py --dataset tiselac --model atomics --device "$device" > "$log_path/tiselac_atomics.log" 2>&1 &

nohup python3 launch_ablation.py --dataset tiselac --model shared --switch_encode_dim --device "$device" > "$log_path/tiselac_shared2.log" 2>&1 &
nohup python3 launch_ablation.py --dataset tiselac --model atomics --switch_summaries_layer --device "$device" > "$log_path/tiselac_atomics2.log" 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
device="cuda:5"
nohup python3 launch_ablation.py --dataset spoken_arabic_digits --model original --device "$device" > "$log_path/spoken_arabic_digits_original.log" 2>&1 &
nohup python3 launch_ablation.py --dataset spoken_arabic_digits --model shared --device "$device" > "$log_path/spoken_arabic_digits_shared.log" 2>&1 &
nohup python3 launch_ablation.py --dataset spoken_arabic_digits --model atomics --device "$device" > "$log_path/spoken_arabic_digits_atomics.log" 2>&1 &

nohup python3 launch_ablation.py --dataset spoken_arabic_digits --model shared --switch_encode_dim --device "$device" > "$log_path/spoken_arabic_digits_shared2.log" 2>&1 &
nohup python3 launch_ablation.py --dataset spoken_arabic_digits --model atomics --switch_summaries_layer --device "$device" > "$log_path/spoken_arabic_digits_atomics2.log" 2>&1 &
