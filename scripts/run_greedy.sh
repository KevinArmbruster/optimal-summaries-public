#!/bin/bash

mkdir -p ./logs/greedy/

# Run commands for mimic dataset
nohup python3 launch.py --dataset mimic --model original --pruning greedy --device "cuda" > ./logs/greedy/mimic_original_greedy.log 2>&1 &
nohup python3 launch.py --dataset mimic --model shared --pruning greedy --device "cuda" > ./logs/greedy/mimic_shared_greedy.log 2>&1 &
nohup python3 launch.py --dataset mimic --model shared --pruning greedy --encode_time_dim False --device "cuda:0" > ./logs/greedy/mimic_shared_greedy2.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning greedy --device "cuda" > ./logs/greedy/mimic_atomics_greedy.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning greedy --use_summaries_for_atomics False --device "cuda:0" > ./logs/greedy/mimic_atomics_greedy2.log 2>&1 &

# nohup python3 launch.py --dataset mimic --model atomics --pruning mixed_greedy --device "cuda" > ./logs/greedy/mimic_atomics_greedy.log 2>&1 &
# nohup python3 launch.py --dataset mimic --model atomics --pruning mixed_greedy --use_summaries_for_atomics False --device "cuda:0" > ./logs/greedy/mimic_atomics_greedy2.log 2>&1 &


# Run commands for tiselac dataset
nohup python3 launch.py --dataset tiselac --model original --pruning greedy --device "cuda:1" > ./logs/greedy/tiselac_original_greedy.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model shared --pruning greedy --device "cuda:1" > ./logs/greedy/tiselac_shared_greedy.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model shared --pruning greedy --encode_time_dim False --device "cuda:1" > ./logs/greedy/tiselac_shared_greedy2.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning greedy --device "cuda:1" > ./logs/greedy/tiselac_atomics_greedy.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning greedy --use_summaries_for_atomics False --device "cuda:1" > ./logs/greedy/tiselac_atomics_greedy2.log 2>&1 &

# nohup python3 launch.py --dataset tiselac --model atomics --pruning mixed_greedy --device "cuda:1" > ./logs/greedy/tiselac_atomics_greedy.log 2>&1 &
# nohup python3 launch.py --dataset tiselac --model atomics --pruning mixed_greedy --use_summaries_for_atomics False --device "cuda:1" > ./logs/greedy/tiselac_atomics_greedy2.log 2>&1 &


# Run commands for spoken_arabic_digits dataset
nohup python3 launch.py --dataset spoken_arabic_digits --model original --pruning greedy --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_original_greedy.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning greedy --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_shared_greedy.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning greedy --encode_time_dim False --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_shared_greedy2.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning greedy --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_atomics_greedy.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning greedy --use_summaries_for_atomics False --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_atomics_greedy2.log 2>&1 &

# nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning mixed_greedy --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_atomics_greedy.log 2>&1 &
# nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning mixed_greedy --use_summaries_for_atomics False --device "cuda:2" > ./logs/greedy/spoken_arabic_digits_atomics_greedy2.log 2>&1 &
