



# 2020年全国人工智能大赛AI+遥感影像赛道代码（Amadeus团队）

## 简介

本仓库为Amadeus团队在AI+遥感影像赛道的初赛中使用的代码。

## 方案介绍

- 本项目使用Pytorch
- 语义分割方法
  - DeepLab V3+
  - Output Stride 8
  - 使用FCN-head分支辅助训练（辅助Loss的权重为0.4）
- 骨干网络
  - `ResNext50_32x4d`、`ResNext101_32x8d`、`ResNext101_64x4d`、`ResNext152_32x4d`
  - `ResNeSt200`、`ResNeSt269`
  - 使用ImageNet预训练模型
- 数据增强
  - 放大图像至640x640像素
  - 随机水平翻转、随机垂直翻转
  - 随机缩放0.7-1.3倍
  - 随机裁剪
- 训练设置
  - Loss = Cross Entropy
  - Batch Size = 8
  - Learning Rate = 0.04
  - Epoch = 120
  - Optimizer = SGD
  - Learning Scheduler = Poly
- 测试增强
  - 翻转：原图、水平翻转、垂直翻转、水平+垂直翻转（4种）
  - 多尺度：0.75、1.0、1.25（3种）
- 模型集成
  - 6个模型Ensemble
    - A榜验证集FWIoU：85.372
    - A榜测试集FWIoU：84.893
    - B榜测试集FWIoU：85.022

## 环境依赖

- `python3`、`cuda9`、`pytorch1.1`、`chop`、`tqdm`、`tabulate`、`ninja`、`pyyaml`、`pillow`、`numpy`、`mmcv`、`apex`

  - ```
    conda create -n amadeus python=3.6
    source activate amadeus
    conda install cudatoolkit=9.0
    conda install cudnn
    pip install chop tqdm tabulate ninja pyymal
    conda install pillow
    conda install pytorch torchvision -c pytorch
    pip install mmcv
    pip install apex
    ```

- `segmentron`

  - ```
    git clone https://github.com/czczup/NAIC_RS_NJU.git
    cd NAIC_RS_NJU/SegmenTron
    python setup.py develop
    ```

- 注意事项

  - 若使用项目中的`dcnv2`和`nas-fpn`，需要安装`mmcv-full`

    - ```
      git clone https://github.com/open-mmlab/mmcv.git
      cd mmcv
      MMCV_WITH_OPS=1 pip install -e .
      ```

  - 若使用混合精度训练，需要安装`apex`

    - ```
      pip install apex
      ```

  - 若需要以上两项，需使用`cuda10`和`pytorch1.6`

## 项目结构

```
├── configs                             # 配置文件
├── datasets                            # 数据集
|   └── naicrs                          # AI+遥感影像赛道数据集
|       └── datasetA                    # A榜数据集
|            └── trainval               # 训练集+验证集
|                 └── images            # 训练集+验证集图像
|                 └── masks             # 训练集+验证集标签
|            └── test                   # 测试集
|                 └── images            # 测试集图像
|                 └── results           # 测试集结果
|       └── datasetB                    # B榜数据集
|            └── test                   # 测试集
|                 └── images            # 测试集图像
|                 └── results           # 测试集结果
|       └── txt                         # A榜训练集与验证集的划分文件
├── segmentron                          # 核心代码
├── tools                               # 训练、验证、测试代码
├── pretrain                            # ImageNet预训练模型
└── runs                                # 模型文件与日志

```

## 数据准备

- A榜训练集

  - 下载并解压，放置在`datasets/naicrs/datasetA/trainval/`文件夹中，将图像文件夹重命名为`images`，将标签文件夹重命名为`masks`

    ```
    wget https://awscdn.datafountain.cn/cometition_data2/Files/PengCheng2020/RSImage/train.zip
    ```

- A榜测试集

  - 下载并解压，放置在`datasets/naicrs/datasetA/test/`文件夹中，将图像文件夹重命名为`images`，并新建一个`results`文件夹存放测试结果

    ```
    wget https://awscdn.datafountain.cn/cometition_data2/Files/PengCheng2020/RSImage/test/image_A.zip
    ```

- B榜测试集

  - 下载并解压，放置在`datasets/naicrs/datasetB/test/`文件夹中，将图像文件夹重命名为`images`，并新建一个`results`文件夹存放测试结果

    ```
    wget https://awscdn.datafountain.cn/cometition_data2/Files/PengCheng2020/RSImage/test/image_B.zip
    ```

## 模型准备

- 预训练模型（若需要训练请下载预训练模型）

  - 下载预训练模型至`pretrain/`文件夹中

  - 注：`resnext152_32x4d_batch256_20200708-aab5034c.pth`文件需重命名为`resnext152_32x4d-aab5034c.pth`

    ```shell
    wget https://download.openmmlab.com/pretrain/third_party/resnext50-32x4d-0ab1a123.pth
    wget https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    wget https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    wget https://s3.ap-northeast-2.amazonaws.com/open-mmlab/pretrain/third_party/resnext101_64x4d-ee2c6f71.pth
    wget https://download.openmmlab.com/mmclassification/v0/imagenet/resnext152_32x4d_batch256_20200708-aab5034c.pth
    wget https://s3.us-west-1.wasabisys.com/resnest/torch/resnest50-528c19ca.pth
    wget https://s3.us-west-1.wasabisys.com/resnest/torch/resnest200-75117900.pth
    wget https://s3.us-west-1.wasabisys.com/resnest/torch/resnest101-22405ba7.pth
    wget https://s3.us-west-1.wasabisys.com/resnest/torch/resnest269-0cc87c48.pth
    mv resnext152_32x4d_batch256_20200708-aab5034c.pth resnext152_32x4d-aab5034c.pth
    ```

- 已训练模型

  - 百度网盘：链接：https://pan.baidu.com/s/1FnRX41cZ9zwPBPhlB6YPpA  提取码：faky 

  - 下载模型文件并解压至`runs/checkpoints/`文件夹中

  - 解压后的目录结构为

    ```
    ├── runs
    |   └── checkpoints
    |       └── 0051_naicrs_deeplabv3_plus_resnest269
    |            └── best_model.pth
    |       └── 0052_naicrs_deeplabv3_plus_resnest200
    |            └── best_model.pth
    |       └── 0054_naicrs_deeplabv3_plus_resnext50
    |            └── best_model.pth
    |       └── 0056_naicrs_deeplabv3_plus_resnext152
    |            └── best_model.pth
    |       └── 0057_naicrs_deeplabv3_plus_resnext101
    |            └── best_model.pth
    |       └── 0058_naicrs_deeplabv3_plus_resnext101
    |            └── best_model.pth
    └── ...
    ```

## 模型测试（B榜）

- 单卡测试

  - 无测试增强

    - ```shell
      CUDA_VISIBLE_DEVICES=0 python -u tools/test.py --config-file <config-file> 
      ```

  - 有测试增强（翻转+多尺度）

    - ```shell
      CUDA_VISIBLE_DEVICES=0 python -u tools/test.py --config-file <config-file> --aug-test
      ```

- 多卡测试（推荐）

  - 无测试增强

    - ```shell
      tools/dist_test.sh <config-file> <gpu-num>
      ```

  - 有测试增强（翻转+多尺度）

    - ```shell
      tools/dist_test_aug.sh <config-file> <gpu-num>
      ```

  - 举例

    ```
    # 用8GPU测试ResNeSt269 
    tools/dist_test_aug.sh configs/0051_naicrs_deeplabv3_plus_resnest269 8
    # 用8GPU测试ResNeSt200
    tools/dist_test_aug.sh configs/0052_naicrs_deeplabv3_plus_resnest200 8
    # 用8GPU测试ResNeXt50_32x4d
    tools/dist_test_aug.sh configs/0054_naicrs_deeplabv3_plus_resnext50 8
    # 用8GPU测试ResNeXt152_32x4d
    tools/dist_test_aug.sh configs/0056_naicrs_deeplabv3_plus_resnext152 8
    # 用8GPU测试ResNeXt101_64x4d
    tools/dist_test_aug.sh configs/0057_naicrs_deeplabv3_plus_resnext101 8
    # 用8GPU测试ResNeXt101_32x8d
    tools/dist_test_aug.sh configs/0058_naicrs_deeplabv3_plus_resnext101 8
    ```

- 模型集成

  ```shell
  python tools/ensemble_test.py
  ```

- 压缩结果文件

  ```shell
  cd datasets/naicrs/datasetB/test/
  zip -r results.zip results/
  ```

## 实验结果

- 单模型实验结果

  | ID   | Method      | BackBone         | CropSize | ImageSize | MS+Flip | Offline (A) | Online (A) |
  | ---- | ----------- | ---------------- | -------- | --------- | ------- | ----------- | ---------- |
  | 0051 | DeepLab V3+ | ResNeSt269       | 480      | 512       | √       | 83.549      | 83.050     |
  | 0052 | DeepLab V3+ | ResNeSt200       | 480      | 512       | √       | 83.161      | 82.793     |
  | 0054 | DeepLab V3+ | ResNeXt50_32x4d  | 640      | 640       | √       | 83.824      | 83.330     |
  | 0056 | DeepLab V3+ | ResNeXt152_32x4d | 640      | 640       | √       | 83.929      | 83.473     |
  | 0057 | DeepLab V3+ | ResNeXt101_64x4d | 640      | 640       | √       | 84.322      | 83.842     |
  | 0058 | DeepLab V3+ | ResNeXt101_32x8d | 640      | 640       | √       | 84.415      | 83.901     |

  - 注：Offline的分数为A榜验证集的FWIoU，Online的分数为A榜测试集的FWIoU

- 模型集成实验结果

  | Model                         | Offline | Online (A) | Online (B) |
  | ----------------------------- | ------- | ---------- | ---------- |
  | 0054+0056+0057+0058           | 85.222  | 84.718     | -          |
  | 0054+0056+0057+0058+0051+0052 | 85.372  | 84.893     | 85.022     |

  