import numpy as np
import os
import torch
from tqdm import tqdm
from multiprocessing import Process


def statistic(start):
    filenames = train_list[start:start+data_pre_process]
    data_dict = {}
    for filename in tqdm(filenames):
        basename, _ = os.path.splitext(filename)
        npy_path = os.path.join(logits_dir, basename+".npy")
        npy = np.load(npy_path).astype(np.float32)
        tensor = torch.from_numpy(npy)
        softmax = torch.softmax(tensor, dim=0)
        mean = torch.max(softmax, dim=0).values.mean().numpy()
        data_dict[filename] = mean
        
    f = open("jupyter/statistic_%d.txt"%start, "w+")
    for key, value in data_dict.items():
        f.write("%s %f\n"%(key, value))
    f.close()

if __name__ == '__main__':
    dataset = "datasets/naicrs/datasetC"
    logits_dir = os.path.join(dataset, "trainval/logits")
    images_dir = os.path.join(dataset, "trainval/images")
    masks_dir = os.path.join(dataset, "trainval/masks")
    
    trainA = open("datasets/naicrs/txt/trainC.txt", "r+")
    train_list = [line[:-1] for line in trainA.readlines()]
    
    data_num = 90000
    process_num = 20
    data_pre_process = data_num // process_num
    pool = []
    for i in range(process_num):
        p = Process(target=statistic, args=(i * data_pre_process,))
        pool.append(p)
        p.start()

    for p in pool:  # 等待所有进程结束
        p.join()
