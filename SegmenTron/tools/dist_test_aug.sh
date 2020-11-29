#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

export NCCL_LL_THRESHOLD=0

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/test.py --config-file $CONFIG --launcher pytorch --aug-test ${@:3}