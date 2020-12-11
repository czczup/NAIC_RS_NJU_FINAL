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

import numpy as np
import cv2
import mindspore.dataset as de
from PIL import Image

cv2.setNumThreads(0)


class DatasetAC:
    def __init__(self, image_mean, image_std, data_file='', batch_size=16, crop_size=256,
                 max_scale=1.3, min_scale=0.7, ignore_label=-1,
                 num_readers=2, num_parallel_calls=4, shard_id=None, shard_num=None):

        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label

        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        assert max_scale > min_scale
 
    def preprocess_(self, image, label, dataset):
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        image_out = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB) # to RGB image
        label_out = label_out.astype(np.int32)
        
        if dataset == 'A':
            label_out = label_out // 100 - 1
        elif dataset == 'C':
            label_out[label_out==4] = 2
            label_out[label_out>=7] -= 3
            label_out = label_out - 1

        # random scale
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])

        image_out = Image.fromarray(image_out)
        label_out = Image.fromarray(label_out)
        image_out = image_out.resize((new_w, new_h), Image.BILINEAR)
        label_out = label_out.resize((new_w, new_h), Image.NEAREST)
        image_out = np.array(image_out)
        label_out = np.array(label_out)

        # random crop
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w+self.crop_size]

        # random flip
        if np.random.uniform(0.0, 1.0) > 0.5: # horizontal flip
            image_out = cv2.flip(image_out, 1)
            label_out = cv2.flip(label_out, 1)
        if np.random.uniform(0.0, 1.0) > 0.5: # vertical flip
            image_out = cv2.flip(image_out, 0)
            label_out = cv2.flip(label_out, 0)

        # normalization
        image_out = image_out / 255.0
        image_out = (image_out - self.image_mean) / self.image_std
        image_out = image_out.transpose((2, 0, 1)) # c,h,w
        image_out = image_out.astype(np.float32)
        
        if dataset == "A":
            label_out_8 = label_out
            label_out_14 = np.zeros(label_out_8.shape) - 1  # -1不监督
        else:
            label_out_14 = label_out
            label_out_8 = label_out.copy()
            label_out_8[label_out_14==0] = 0    # 水体   -> 水体
            label_out_8[label_out_14==1] = 1    # 道路   -> 道路
            label_out_8[label_out_14==2] = 2    # 建筑物  -> 建筑物
            label_out_8[label_out_14==3] = 7    # 停车场   -> 其它
            label_out_8[label_out_14==4] = 7    # 操场    -> 其它
            label_out_8[label_out_14==5] = 3    # 普通耕地 -> 耕地
            label_out_8[label_out_14==6] = 3    # 农业大棚 -> 耕地
            label_out_8[label_out_14==7] = 4    # 自然草地 -> 草地
            label_out_8[label_out_14==8] = 4    # 绿地绿化 -> 草地
            label_out_8[label_out_14==9] = 5    # 自然林   -> 林地
            label_out_8[label_out_14==10] = 5   # 人工林   -> 林地
            label_out_8[label_out_14==11] = 6   # 自然裸土 -> 裸土
            label_out_8[label_out_14==12] = 6   # 人为裸土 -> 裸土
            label_out_8[label_out_14==13] = 7   # 其它    -> 其它
        
        image_out = image_out.copy()
        label_out_8 = label_out_8.copy()
        label_out_14 = label_out_14.copy()
        return image_out, label_out_8, label_out_14

    def get_dataset(self, repeat=1):
        data_set = de.MindDataset(dataset_file=self.data_file, columns_list=["data", "label", "dataset"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id)
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["data", "label", "dataset"],
                                output_columns=["data", "label8", "label14"],
                                num_parallel_workers=self.num_parallel_calls)
        data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set
