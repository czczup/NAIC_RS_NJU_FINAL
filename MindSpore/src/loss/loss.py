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

import mindspore
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P


class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=14, ignore_label=-1, aux=True, aux_weight=0.4):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.aux = aux
        self.aux_weight = aux_weight
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        
    def cross_entropy(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        logits_ = self.cast(logits_, mindspore.float32)
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss
    
    def construct(self, logits, labels):
        if self.aux:
            loss = self.cross_entropy(logits[0], labels)
            loss += self.aux_weight * self.cross_entropy(logits[1],labels)
        else:
            loss = self.cross_entropy(logits, labels)
        return loss


class SoftmaxCrossEntropyLossV2(nn.Cell):
    def __init__(self, num_class=[8, 14], ignore_label=-1, aux=True, aux_weight=0.4):
        super(SoftmaxCrossEntropyLossV2, self).__init__()
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.aux = aux
        self.aux_weight = aux_weight
        self.cast = P.Cast()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
    
    def cross_entropy(self, logits, labels, num_class):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, num_class))
        logits_ = self.cast(logits_, mindspore.float32)
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, num_class, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss
    
    def construct(self, logits, labels):
        loss_8 = self.cross_entropy(logits[0][0], labels[0], num_class=self.num_class[0])
        if self.aux:
            loss_8 += self.aux_weight * self.cross_entropy(logits[0][1], labels[0], num_class=self.num_class[0])
        
        loss_14 = self.cross_entropy(logits[1][0], labels[1], num_class=self.num_class[1])
        if self.aux:
            loss_14 += self.aux_weight * self.cross_entropy(logits[1][1], labels[1], num_class=self.num_class[1])
        
        loss = loss_8 * 0.5 + loss_14 * 0.5
        return loss
