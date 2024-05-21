#!/bin/bash

DATA_TYPES="debates,sotu,campaign"

python results/plot_divisive_lexicon.sh \
    --data_types $DATA_TYPES \
    --data_dir data/ \
    --figure_dir figures/ \

