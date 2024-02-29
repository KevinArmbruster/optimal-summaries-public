#!/bin/bash

mkdir -p ./logs/importance/

# Run commands for mimic dataset
nohup python3 launch.py --dataset mimic --model original --pruning importance --device "cuda:0" > ./logs/importance/mimic_original_importance.log 2>&1 &
nohup python3 launch.py --dataset mimic --model shared --pruning importance --device "cuda:0" > ./logs/importance/mimic_shared_importance.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning importance --device "cuda:0" > ./logs/importance/mimic_atomics_importance.log 2>&1 &

nohup python3 launch.py --dataset mimic --model shared --pruning importance --switch_encode_dim --device "cuda:0" > ./logs/importance/mimic_shared_importance2.log 2>&1 &
nohup python3 launch.py --dataset mimic --model atomics --pruning importance --switch_encode_dim --device "cuda:0" > ./logs/importance/mimic_atomics_importance2.log 2>&1 &

# Run commands for tiselac dataset
nohup python3 launch.py --dataset tiselac --model original --pruning importance --device "cuda:3" > ./logs/importance/tiselac_original_importance.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model shared --pruning importance --device "cuda:3" > ./logs/importance/tiselac_shared_importance.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning importance --device "cuda:3" > ./logs/importance/tiselac_atomics_importance.log 2>&1 &

nohup python3 launch.py --dataset tiselac --model shared --pruning importance --switch_encode_dim --device "cuda:3" > ./logs/importance/tiselac_shared_importance2.log 2>&1 &
nohup python3 launch.py --dataset tiselac --model atomics --pruning importance --switch_encode_dim --device "cuda:3" > ./logs/importance/tiselac_atomics_importance2.log 2>&1 &

# Run commands for spoken_arabic_digits dataset
nohup python3 launch.py --dataset spoken_arabic_digits --model original --pruning importance --device "cuda:5" > ./logs/importance/spoken_arabic_digits_original_importance.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning importance --device "cuda:5" > ./logs/importance/spoken_arabic_digits_shared_importance.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning importance --device "cuda:5" > ./logs/importance/spoken_arabic_digits_atomics_importance.log 2>&1 &

nohup python3 launch.py --dataset spoken_arabic_digits --model shared --pruning importance --switch_encode_dim --device "cuda:5" > ./logs/importance/spoken_arabic_digits_shared_importance2.log 2>&1 &
nohup python3 launch.py --dataset spoken_arabic_digits --model atomics --pruning importance --switch_encode_dim --device "cuda:5" > ./logs/importance/spoken_arabic_digits_atomics_importance2.log 2>&1 &
