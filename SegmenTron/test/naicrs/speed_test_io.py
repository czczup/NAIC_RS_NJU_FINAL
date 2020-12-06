import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import cv2
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import feed_ndarray as feed_ndarray


class ExternalInputIterator(object):
    def __init__(self):
        self.image_path = None
        self.last_image_path = None
    
    def __iter__(self):
        self.i = 0
        self.n = 1
        return self
    
    def __next__(self):
        if self.i < self.n:
            image = np.fromfile(self.image_path, dtype=np.uint8)
            self.last_image_path = self.image_path
            self.i = 1
            return ([image],)
        else:
            raise StopIteration
    
    def init(self, image_path):
        self.image_path = image_path
        self.i = 0


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, eii, num_threads, device_id):
        super(ExternalSourcePipeline, self).__init__(batch_size,
                                                     num_threads,
                                                     device_id,
                                                     seed=12,
                                                     prefetch_queue_depth=1)
        self.source = ops.ExternalSource(source=eii, num_outputs=1)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
    
    def define_graph(self):
        images = self.source()
        images = self.decode(images)
        return images

device = torch.device("cuda")
transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

transform_old = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

eii = ExternalInputIterator()
pipe = ExternalSourcePipeline(batch_size=1, eii=eii, num_threads=1, device_id=0)
pipe.build()


def test_time(func):
    torch.cuda.synchronize()
    time_start = time.time()
    func()
    torch.cuda.synchronize()
    print("using time: %.2f" %(time.time() - time_start))

def load_DALI():
    for i in range(1000):
        eii.init("0.tif")
        image = pipe.run()[0][0]
        empty = torch.empty(size=image.shape(), device=device, dtype=torch.uint8)
        stream = torch.cuda.current_stream(device=device)
        feed_ndarray(image, empty, cuda_stream=stream)
        empty = empty.permute((2, 0, 1)) / 255.0
        image = transform(empty)
        image = torch.unsqueeze(image, dim=0)


def load_PIL():
    for i in range(1000):
        image = Image.open("0.tif")
        image = transform_old(image)
        image = torch.unsqueeze(image, dim=0).to(device)
    
if __name__ == '__main__':
    test_time(load_DALI)
    test_time(load_DALI)
    print("DALI:")
    test_time(load_DALI)
    
    test_time(load_PIL)
    test_time(load_PIL)
    print("PIL:")
    test_time(load_PIL)

