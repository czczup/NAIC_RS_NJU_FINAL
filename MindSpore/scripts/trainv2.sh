export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DEVICE_ID=7
export SLOG_PRINT_TO_STDOUT=0

python trainv2.py --data_file="datasets/naicrs/mindrecordAC"  \
                    --train_dir="checkpoints/01"  \
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
                    --pth_pretrained='checkpoints/0046.pth'  \
                    --save_steps=10000 \
