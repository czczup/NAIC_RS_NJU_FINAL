#!/bin/bash

export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0


python eval.py --data_root="datasets/naicrs/datasetA/trainval"  \
               --data_lst="datasets/naicrs/txt/valA.txt"  \
               --batch_size=32  \
               --backbone=resnext101  \
               --ckpt_path="checkpoints/resnext101_dunet.ckpt"  \
               --model=dunetv2