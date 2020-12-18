export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export DEVICE_ID=5
export SLOG_PRINT_TO_STDOUT=0

python train.py --data_file="datasets/naicrs/datasetC/mindrecord"  \
                    --train_dir="checkpoints/"  \
                    --train_epochs=60  \
                    --batch_size=2  \
                    --crop_size=256  \
                    --base_lr=0.005  \
                    --lr_type=poly  \
                    --min_scale=0.7  \
                    --max_scale=1.3  \
                    --ignore_label=-1  \
                    --num_classes=14  \
                    --model=deeplabv3plus  \
                    --aux  \
                    --ckpt_pre_trained=''  \
                    --save_steps=10000 \
