#!/bin/bash

MODEL_TYPE="gpt2"

DATA_TYPES="debates,sotu,campaign"

echo $DATA_TYPES $MODEL_TYPE

python results/plot_uniqueness.py \
    --model_type $MODEL_TYPE \
    --data_types $DATA_TYPES \
    --data_dir data/ \
    --figure_dir figures/ \
    --mask_ents true
