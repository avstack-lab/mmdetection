#!/usr/bin/env bash

set -x

SKIPS=${1:?"missing arg 1 for SKIPS"}

python3 convert_any_avstack_labels.py \
    --dataset carla \
    --subfolder vehicle/skip-"$SKIPS" \
    --data_dir /data/spencer/CARLA/multi-sensor-v1 \
    --n_skips "$SKIPS"
