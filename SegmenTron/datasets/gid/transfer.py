import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def transfer_mask(mask):
    mask = np.array(mask)
    height, width, _ = mask.shape
    mask_new = np.zeros([height, width])
    mask_new[x((mask == [0, 0, 200]))] = 100      # water -> 水体
    mask_new[x((mask == [0, 150, 200]))] = 100    # lake -> 水体
    mask_new[x((mask == [0, 200, 250]))] = 100    # pond -> 水体
    
    mask_new[x((mask == [250, 150, 150]))] = 200  # traffic land -> 道路

    mask_new[x((mask == [200, 0, 0]))] = 300      # industrial land -> 建筑物
    mask_new[x((mask == [250, 0, 150]))] = 300    # urban residential -> 建筑物
    mask_new[x((mask == [200, 150, 150]))] = 300  # rural residential -> 建筑物

    mask_new[x((mask == [0, 200, 0]))] = 400      # paddy field -> 普通耕地
    mask_new[x((mask == [150, 250, 0]))] = 400    # irrigated land -> 普通耕地
    mask_new[x((mask == [150, 200, 150]))] = 400  # dry cropland -> 普通耕地
    
    mask_new[x((mask == [250, 200, 0]))] = 500  # natural grassland -> 自然草地
    mask_new[x((mask == [200, 200, 0]))] = 500  # artificial grassland -> 绿地绿化

    mask_new[x((mask == [150, 0, 250]))] = 600  # arbor woodland -> 自然林
    mask_new[x((mask == [150, 150, 250]))] = 600  # shrub land -> 自然林
    mask_new[x((mask == [200, 0, 200]))] = 600  # garden plot -> 人工林
    
    mask_new[x((mask == [0, 0, 0]))] = 800  # other -> 其它

    mask = Image.fromarray(np.uint16(mask_new))
    return mask

def make_dir():
    folder_path = ['images', 'masks']
    for folder in folder_path:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("creat a new folder named {}".format(folder))

def main(image_dir, label_dir):
    filenames = os.listdir(image_dir)
    for index_, filename in enumerate(tqdm(filenames)):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(label_dir, filename)
        # image = Image.open(image_path)
        mask = Image.open(mask_path)
        mask = transfer_mask(mask)
        mask.save('masks/{}'.format(filename.replace("tif", "png")))


if __name__ == '__main__':
    x = lambda a: a[:, :, 0] & a[:, :, 1] & a[:, :, 2]
    make_dir()
    main(image_dir='images', label_dir='labels')


