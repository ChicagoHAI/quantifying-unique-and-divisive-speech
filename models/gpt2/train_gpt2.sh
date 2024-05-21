#!/bin/bash -l

for DATA in debates sotu campaign
do

    OUT_DIR=quantifying-unique-and-divisive-speech/out/$DATA
    DATA_DIR=quantifying-unique-and-divisive-speech/data/$DATA
    CACHE_DIR=<path_to_cache_dir>

    # skip validation step w/ --limit-val-batches
    CUDA_VISIBLE_DEVICES=0 python -m models.main \
        --task lm \
        --max_epochs 10 \
        --limit_val_batches 0.0 \
        --max_seq_length 1024 \
        --output_dir $OUT_DIR \
        --data_dir $DATA_DIR \
        --model_name_or_path gpt2 \
        --warmup_steps 500 \
        --gpus 1 \
        --do_train \
        --cache_dir $CACHE_DIR \
        --train_batch_size 4 \
        --eval_batch_size 4 \
        --learning_rate 5e-5 \
        --overwrite_dir \
        --fp16 
    
done