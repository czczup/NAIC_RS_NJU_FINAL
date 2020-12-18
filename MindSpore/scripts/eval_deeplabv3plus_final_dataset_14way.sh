#!/bin/bash

export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0


python eval.py --data_root="datasets/naicrs/final"  \
               --data_lst="datasets/naicrs/txt/final.txt"  \
               --batch_size=32  \
               --num_classes=14  \
               --mode="06"  \
               --backbone=resnext101  \
               --ckpt_path="checkpoints/resnext101_deeplabv3plus.ckpt"  \
               --model=deeplabv3plusv2