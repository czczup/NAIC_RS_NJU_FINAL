#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

export DEVICE_ID=0
export SLOG_PRINT_TO_STDOUT=0


python eval.py --data_root="datasets/naicrs/datasetA/trainval"  \
               --data_lst="datasets/naicrs/txt/valA.txt"  \
               --batch_size=32  \
               --backbone=resnext101  \
               --ckpt_path="runs/checkpoints/resnext101_deeplabv3plus.ckpt"  \
               --model=deeplabv3plusv2