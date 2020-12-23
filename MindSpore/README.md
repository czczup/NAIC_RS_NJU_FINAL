# AI+遥感影像赛道-复赛代码

## 简介

本目录是基于华为MindSpore框架实现的推理移植。

## 环境依赖

- python3.7.5, cuda10.1
- pytorch1.6, MindSpore1.0.1 (推荐使用docker)
- opencv, pillow

## 数据下载

- 下载华为方面提供的5000张测试图像及标签，放置在`datasets/naicrs/final`文件夹中。

## 模型迁移

1. 使用隔壁`SegmenTron`项目训练分别训练`ResNeXt101_32x8d+DeepLabV3Plus`模型和`ResNeXt101_32x8d+DUNet`模型

   ```shell
   cd ../SegmenTron
   tools/trainv2.sh configs/0001_deeplabv3plus_resnext101.yaml 8
   tools/trainv2.sh configs/0002_dunet_resnext101.yaml 8
   ```

2. 复制训练完成的模型至`checkpoints`文件夹，并重命名

   ```shell
   # ResNeXt101_64x4d+DeepLabV3Plus
   cp runs/checkpoints/0001_deeplabv3plus_resnext101/best_model.pth ../MindSpore/checkpoints
   mv ../MindSpore/checkpoints/best_model.pth ../MindSpore/checkpoints/resnext101_deeplabv3plus.pth
   
   # ResNeXt101_64x4d+DUNet
   cp runs/checkpoints/0002_deeplabv3plus_dunet/best_model.pth ../MindSpore/checkpoints
   mv ../MindSpore/checkpoints/best_model.pth ../MindSpore/checkpoints/resnext101_dunet.pth
   ```

3. 运行模型迁移脚本，将`.pth`文件转换为`.ckpt`文件

   ```shell
   cd ../MindSpore
   python tools/migrate_deeplabv3plus.py
   python tools/migrate_dunet.py
   ```

4. 运行推理脚本，生成的结果保存至`datasets/naicrs/final/results`文件夹中

   ```
   python eval.pth
   ```

## 实验结果

### 初赛与复赛验证集

| 模型                           | 初赛 (8类) | 复赛 (8类) | 复赛 (14类) | 框架      |
| ------------------------------ | ---------- | ---------- | ----------- | --------- |
| ResNeXt101_32x8d+DeepLabV3Plus | 80.99      | 77.94      | 73.94       | Pytorch   |
|                                | 80.86      | 77.65      | 73.45       | MindSpore |
| ResNeXt101_32x8d+DUNet         | 81.23      | 78.04      | 73.93       | Pytorch   |
|                                | 81.10      | 77.76      | 73.67       | MindSpore |

### 决赛新测试集

| 模型                           | 8类分支 | 14类分支 | 8+14类分支 |
| ------------------------------ | ------- | -------- | ---------- |
| ResNeXt101_32x8d+DeepLabV3Plus | 69.80   | 58.60    | 58.63      |
| ResNeXt101_32x8d+DUNet         | 70.12   | 58.63    | 58.66      |
| Ensemble                       | -       | 59.76    | 59.80      |