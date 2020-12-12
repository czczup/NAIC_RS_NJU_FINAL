#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

#http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
# ============================================================================

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DEVICE_ID=7
export SLOG_PRINT_TO_STDOUT=0

python trainv2.py --data_file="datasets/naicrs/mindrecordAC"  \
                    --train_dir="runs/checkpoints/01"  \
                    --train_epochs=30  \
                    --batch_size=24  \
                    --crop_size=256  \
                    --base_lr=0.001  \
                    --lr_type=poly  \
                    --min_scale=0.7  \
                    --max_scale=1.3  \
                    --ignore_label=-1  \
                    --model=deeplabv3plusv2  \
                    --aux  \
                    --pth_pretrained='runs/checkpoints/0046.pth'  \
                    --save_steps=10000 \
