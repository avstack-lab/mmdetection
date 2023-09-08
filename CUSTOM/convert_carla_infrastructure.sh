#!/usr/bin/env bash

python3 convert_any_avstack_labels.py \
    --dataset carla \
    --subfolder infrastructure \
    --data_dir /data/spencer/CARLA/multi-agent-v1 \
    --n_skips 2
