#!/bin/bash

mkdir -p ./logs/gradient_magnitude/

# Run commands for mimic dataset
nohup python3 launch.py --dataset mimic --model original --pruning gradient_magnitude --device "cuda:1" > ./logs/gradient_magnitude/mimic_original_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset mimic --model shared --pruning gradient_magnitude --device "cuda:1" > ./logs/gradient_magnitude/mimic_shared_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning gradient_magnitude --device "cuda:1" > ./logs/gradient_magnitude/mimic_atomics_gradient_magnitude.log 2>&1 &

nohup python3 launch.py --dataset mimic --model shared --pruning gradient_magnitude --switch_encode_dim --device "cuda:1" > ./logs/gradient_magnitude/mimic_shared_gradient_magnitude2.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning gradient_magnitude --switch_summaries_layer --device "cuda:1" > ./logs/gradient_magnitude/mimic_atomics_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for tiselac dataset
nohup python3 launch.py --dataset tiselac --model original --pruning gradient_magnitude --device "cuda:2" > ./logs/gradient_magnitude/tiselac_original_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model shared --pruning gradient_magnitude --device "cuda:2" > ./logs/gradient_magnitude/tiselac_shared_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning gradient_magnitude --device "cuda:2" > ./logs/gradient_magnitude/tiselac_atomics_gradient_magnitude.log 2>&1 &

nohup python3 launch.py --dataset tiselac --model shared --pruning gradient_magnitude --switch_encode_dim --device "cuda:2" > ./logs/gradient_magnitude/tiselac_shared_gradient_magnitude2.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning gradient_magnitude --switch_summaries_layer --device "cuda:2" > ./logs/gradient_magnitude/tiselac_atomics_gradient_magnitude2.log 2>&1 &

sleep 10

# Run commands for spoken_arabic_digits dataset
nohup python3 launch.py --dataset spoken_arabic_digits --model original --pruning gradient_magnitude --device "cuda:0" > ./logs/gradient_magnitude/spoken_arabic_digits_original_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning gradient_magnitude --device "cuda:0" > ./logs/gradient_magnitude/spoken_arabic_digits_shared_gradient_magnitude.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning gradient_magnitude --device "cuda:0" > ./logs/gradient_magnitude/spoken_arabic_digits_atomics_gradient_magnitude.log 2>&1 &

nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning gradient_magnitude --switch_encode_dim --device "cuda:0" > ./logs/gradient_magnitude/spoken_arabic_digits_shared_gradient_magnitude2.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning gradient_magnitude --switch_summaries_layer --device "cuda:0" > ./logs/gradient_magnitude/spoken_arabic_digits_atomics_gradient_magnitude2.log 2>&1 &
