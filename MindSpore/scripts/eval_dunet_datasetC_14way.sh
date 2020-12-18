#!/bin/bash

export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0


python eval.py --data_root="datasets/naicrs/datasetC/trainval"  \
               --data_lst="datasets/naicrs/txt/valC.txt"  \
               --batch_size=32  \
               --num_classes=14  \
               --mode="06"  \
               --backbone=resnext101  \
               --ckpt_path="checkpoints/resnext101_dunet.ckpt"  \
               --model=dunetv2