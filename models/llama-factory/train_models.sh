#!/bin/bash

conda activate llama-factory
LF_DIR="LLaMA-Factory"
cd $LF_DIR

CONFIG_DIR="quantifying-unique-and-divisive-speech/models/llama-factory/train"

for DATA in debates sotu campaign
do
    MODEL_NAME="gemma2b"  

    OUT_DIR=$LF_DIR/saves/$MODEL_NAME"_"$DATA
    DATA_DIR=$LF_DIR/data
    mkdir -p $OUT_DIR

    YAML_FILE=$CONFIG_DIR/$MODEL_NAME"_"$DATA.yaml 
    echo $YAML_FILE

    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train $YAML_FILE
done