#!/bin/bash

mkdir -p ./logs/weight_gradient_magnitude/

# Run commands for mimic dataset
nohup python3 launch.py --dataset mimic --model original --pruning weight_gradient_magnitude --device "cuda:3" > ./logs/weight_gradient_magnitude/mimic_original_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset mimic --model shared --pruning weight_gradient_magnitude --device "cuda:3" > ./logs/weight_gradient_magnitude/mimic_shared_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning weight_gradient_magnitude --device "cuda:3" > ./logs/weight_gradient_magnitude/mimic_atomics_weight_gradient_magnitude.log 2>&1 &

nohup python3 launch.py --dataset mimic --model shared --pruning weight_gradient_magnitude --switch_encode_dim --device "cuda:3" > ./logs/weight_gradient_magnitude/mimic_shared_weight_gradient_magnitude2.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning weight_gradient_magnitude --switch_summaries_layer --device "cuda:3" > ./logs/weight_gradient_magnitude/mimic_atomics_weight_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
nohup python3 launch.py --dataset tiselac --model original --pruning weight_gradient_magnitude --device "cuda:4" > ./logs/weight_gradient_magnitude/tiselac_original_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model shared --pruning weight_gradient_magnitude --device "cuda:4" > ./logs/weight_gradient_magnitude/tiselac_shared_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning weight_gradient_magnitude --device "cuda:4" > ./logs/weight_gradient_magnitude/tiselac_atomics_weight_gradient_magnitude.log 2>&1 &

nohup python3 launch.py --dataset tiselac --model shared --pruning weight_gradient_magnitude --switch_encode_dim --device "cuda:4" > ./logs/weight_gradient_magnitude/tiselac_shared_weight_gradient_magnitude2.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning weight_gradient_magnitude --switch_summaries_layer --device "cuda:4" > ./logs/weight_gradient_magnitude/tiselac_atomics_weight_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
nohup python3 launch.py --dataset spoken_arabic_digits --model original --pruning weight_gradient_magnitude --device "cuda:5" > ./logs/weight_gradient_magnitude/spoken_arabic_digits_original_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning weight_gradient_magnitude --device "cuda:5" > ./logs/weight_gradient_magnitude/spoken_arabic_digits_shared_weight_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning weight_gradient_magnitude --device "cuda:5" > ./logs/weight_gradient_magnitude/spoken_arabic_digits_atomics_weight_gradient_magnitude.log 2>&1 &

nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning weight_gradient_magnitude --switch_encode_dim --device "cuda:5" > ./logs/weight_gradient_magnitude/spoken_arabic_digits_shared_weight_gradient_magnitude2.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning weight_gradient_magnitude --switch_summaries_layer --device "cuda:5" > ./logs/weight_gradient_magnitude/spoken_arabic_digits_atomics_weight_gradient_magnitude2.log 2>&1 &
