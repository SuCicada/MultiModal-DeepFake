#!/bin/bash
EXPID=checkpoint

HOST='127.0.0.1'
PORT='2'

NUM_GPU=1

python test.py \
--config 'configs/test.yaml' \
--output_dir 'results' \
--launcher pytorch \
--rank 0 \
--log_num ${EXPID} \
--dist-url tcp://${HOST}:1003${PORT} \
--token_momentum \
--world_size $NUM_GPU \
--test_epoch best \

