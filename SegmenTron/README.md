# AI+遥感影像赛道-决赛代码

## 简介

本目录是基于Pytorch框架的代码实现。

## 环境依赖

- `python3.6`、`cuda10.1`、`pytorch1.6`、`thop`、`tqdm`、`tabulate`、`ninja`、`pyyaml`、`pillow`、`numpy`、`mmcv`

  ```shell
  conda install cudatoolkit=10.1
  conda install cudnn
  conda install pillow
  pip install thop tqdm tabulate ninja pyymal
  conda install pytorch torchvision -c pytorch
  pip install mmcv
  ```
  
- `segmentron`

  ```shell
  git clone https://github.com/czczup/NAIC_RS_NJU_FINAL.git
  cd NAIC_RS_NJU_FINAL/SegmenTron
  python setup.py develop
  ```

- apex

  ```shell
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  python setup.py install --cpp_ext --cuda_ext
  ```

## 实验设备

本项目使用8张具有32G显存的Nvidia Tesla V100显卡进行训练。

## 项目结构

```
├── configs                             # 配置文件
├── datasets                            # 数据集
|   └── naicrs                          # AI+遥感影像赛道数据集
|       └── datasetA                    # 初赛数据集
|            └── trainval
|                 └── images            # 图像
|                 └── masks             # 标签
|       └── datasetC                    # 复赛数据集
|            └── trainval
|                 └── images            # 图像
|                 └── masks             # 标签
|       └── txt                         # 训练集与验证集的划分文件
├── segmentron                          # 核心代码
├── tools                               # 训练脚本
├── pretrain                            # ImageNet预训练模型
├── test                                # 复赛本地测试的代码
├── submit                              # 复赛提交服务器测试的代码
└── runs                                # 模型文件与日志
```

## 模型训练与测试

1. 训练模型

   ```shell
   tools/trainv2.sh configs/0046_deeplabv3plus_ofav100.yaml 8
   ```

2. 复制模型至`test/model`文件夹，并重命名

   ```
   cp runs/checkpoints/0046_deeplabv3plus_ofav100/best_model.pth test/model
   cd test/model
   mv best_model.pth 0046.pth
   ```

3. 模型量化

   ```shell
   cd ../naicrs
   python quantize.py --model 0046.pth --ofa ofav100
   ```

4. 复制量化后的模型至`submit/final`文件夹

   ```shell
   cd ../model 
   cp 0046q.pth ../../submit/final
   cd ../../submit/final
   mv 0046q.pth best_model.pth
   ```

5. 执行推理脚本

   ```shell
   python main.py
   python iou.py
   ```

## 实验结果

| Model    | Branch | Epoch       | Input Size | Stride | Batch Size | FWIoU  | Time | Size   | Score  |
| -------- | ------ | ----------- | ---------- | ------ | ---------- | ------ | ---- | ------ | ------ |
| OFA-V100 | 8+14   | 120         | 256        | 32     | 16         | 0.4955 | 2887 | 2.8995 | 0.6961 |
| OFA-V100 | 8+14   | 240         | 256        | 32     | 16         | 0.5374 | 2834 | 2.7091 | 0.7185 |
| OFA-V100 | 8+14   | 240+120     | 256        | 32     | 16         | 0.5448 | 2832 | 2.5507 | 0.7229 |
| OFA-V100 | 14     | 240+120+120 | 224        | 32     | 32         | 0.5490 | 2409 | 2.1305 | 0.7326 |
| OFA-V100 | 14     | 240+120+120 | 224        | 64     | 32         | 0.5537 | 2341 | 2.1305 | 0.7358 |

