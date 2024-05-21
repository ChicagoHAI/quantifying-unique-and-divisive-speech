#!/bin/bash

MODEL_TYPE="gpt2"

for DATA_TYPE in sotu campaign debates 
do
    echo $DATA_TYPE
    CHECKPOINT=<path to model checkpoint>
    
    python results/score_uniqueness.py \
    --model_type $MODEL_TYPE \
    --data $DATA_TYPE \
    --data_dir data/ \
    --output_dir out/ \
    --cache_dir <cache_dir> \
    --mask_ents 1 \
    --window_size 512 \
    --model_checkpoint $CHECKPOINT \
    --batch_size 8 \
    --device cuda \
    --add_start_token 1 
    
done
