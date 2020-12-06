import numpy as np


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


if __name__ == '__main__':
    eii = ExternalInputIterator()
    eii.init("0.tif")
    for item in eii:
        print(item)

    eii.init("0.tif")
    for item in eii:
        print(item)
        