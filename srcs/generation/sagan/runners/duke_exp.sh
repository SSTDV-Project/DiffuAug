#!/bin/bash
/opt/conda/bin/python /workspace/DiffuAug/srcs/generation/sagan/runners/main.py \
    --model duke \
    --data_dir /data/duke_data/size_64/split_datalabel \
    --checkpoint_dir /data/results/generation/exps/sagan/models \
    --sampling_dir /data/results/generation/sampling/sampling_tests \
    --num_classes 2 \
    --img_size 64 \
    --channels 1 \
    --batch_size 32

