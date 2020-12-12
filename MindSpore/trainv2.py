# Copyright 2020 Huawei Technologies Co., Ltd

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
"""train deeplabv3."""

import os
import argparse
import mindspore
from mindspore import context, Tensor, Parameter
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
import mindspore.ops as P
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src.data import datasetAC as data_generator
from src.loss import loss
from src.nets import net_factory
from src.utils import learning_rates
import torch

set_seed(1)
context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                    device_target="GPU", device_id=int(os.getenv('DEVICE_ID')))


class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label8, label14):
        x1, auxout1, x2, auxout2 = self.network(input_data)
        net_loss = self.criterion(x1, auxout1, x2, auxout2, label8, label14)
        return net_loss


def parse_args():
    parser = argparse.ArgumentParser('mindspore deeplabv3plus training')
    parser.add_argument('--train_dir', type=str, default='/', help='where training log and ckpts saved')

    # dataset
    parser.add_argument('--data_file', type=str, default='', help='path and name of one mindrecord file')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[0.485, 0.456, 0.406], help='image mean')
    parser.add_argument('--image_std', type=list, default=[0.229, 0.224, 0.225], help='image std')
    parser.add_argument('--min_scale', type=float, default=0.7, help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=1.3, help='maximum scale of data argumentation')
    parser.add_argument('--ignore_label', type=int, default=-1, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes')

    # optimizer
    parser.add_argument('--train_epochs', type=int, default=60, help='epoch')
    parser.add_argument('--lr_type', type=str, default='poly', help='type of learning rate')
    parser.add_argument('--base_lr', type=float, default=0.015, help='base learning rate')
    parser.add_argument('--lr_decay_step', type=int, default=40000, help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=1.0, help='loss scale')

    # model
    parser.add_argument('--model', type=str, default='deeplabv3plus', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')
    parser.add_argument('--aux', action='store_true', help='aux')
    parser.add_argument('--ckpt_pre_trained', type=str, default='', help='pretrained model')
    parser.add_argument('--pth_pretrained', type=str, default='', help='pretrained model')

    # train
    parser.add_argument('--is_distributed', action='store_true', help='distributed training')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--save_steps', type=int, default=3000, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=None, help='max checkpoint for saving')

    args, _ = parser.parse_known_args()
    return args


def train():
    args = parse_args()

    # init multicards training
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.group_size)
    print(args)
    
    # dataset
    dataset = data_generator.DatasetAC(image_mean=args.image_mean,
                                      image_std=args.image_std,
                                      data_file=args.data_file,
                                      batch_size=args.batch_size,
                                      crop_size=args.crop_size,
                                      max_scale=args.max_scale,
                                      min_scale=args.min_scale,
                                      ignore_label=args.ignore_label,
                                      num_readers=2,
                                      num_parallel_calls=4,
                                      shard_id=args.rank,
                                      shard_num=args.group_size)
    dataset = dataset.get_dataset(repeat=1)


    network = net_factory.nets_map[args.model]('train', [8, 14], 8, args.aux)

    p2m = open("runs/checkpoints/pytorch2mindspore.csv", "r+")
    p2m = [line[:-1].split(",") for line in p2m.readlines()]
    p2m = {item[0]: item[1] for item in p2m}

    
    network.set_train(True)

    # loss
    loss_ = loss.SoftmaxCrossEntropyLossV2([8, 14], args.ignore_label, args.aux)
    loss_.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss_)

    state_dict = torch.load(args.pth_pretrained)
    param_dict = dict()
    for k, v in state_dict.items():
        print(k, k in p2m)
        if k in p2m:
            parameter = v.cpu().data.numpy()
            parameter = Parameter(Tensor(parameter), name=p2m[k])
            param_dict[p2m[k]] = parameter
    load_param_into_net(train_net, param_dict)

    # load pretrained model
    if args.ckpt_pre_trained:
        param_dict = load_checkpoint(args.ckpt_pre_trained)
        load_param_into_net(train_net, param_dict)

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * args.train_epochs
    print('total_train_steps:{}'.format(total_train_steps))
    if args.lr_type == 'cos':
        lr_iter = learning_rates.cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = learning_rates.poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    else:
        raise ValueError('unknown learning rate type')
    opt = nn.SGD(params=train_net.trainable_params(), learning_rate=lr_iter,
                 momentum=0.9, weight_decay=1e-4)
    # opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9, weight_decay=0.0001,
    #                   loss_scale=args.loss_scale)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    model = Model(train_net, optimizer=opt, amp_level="O0", loss_scale_manager=manager_loss_scale)

    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_steps,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        print('args.train_dir: {}'.format(args.train_dir))
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory=args.train_dir, config=config_ck)
        cbs.append(ckpoint_cb)

    model.train(args.train_epochs, dataset, callbacks=cbs)

if __name__ == '__main__':
    train()
