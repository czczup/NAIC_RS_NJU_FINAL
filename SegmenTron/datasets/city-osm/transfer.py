import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def transfer_mask(mask):
    mask = np.array(mask)
    height, width, _ = mask.shape
    mask_new = np.zeros([height, width])
    index = x((mask == [0, 0, 255]))
    mask_new[index] = 200  # 交通
    index = x((mask == [255, 0, 0]))
    mask_new[index] = 300                                 # 建筑
    # index = x((mask==[255, 255, 255]))
    # mask_new[index] = 0                                   # 背景
    mask = Image.fromarray(np.uint16(mask_new))
    return mask

def make_dir():
    folder_path = ['images', 'masks']
    for folder in folder_path:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print("creat a new folder named {}".format(folder))

def crop_data(name):
    image_dir = name
    
    filenames = os.listdir(image_dir)
    images_filename = [item for item in filenames if item.endswith("image.png")]
    for index_, filename in enumerate(tqdm(images_filename)):
        image_path = os.path.join(image_dir, filename)
        mask_path = image_path.replace("image.png", "labels.png")
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        mask = transfer_mask(mask)
        image.save('images/{}'.format(filename.split("_")[0]+".tif"))
        mask.save('masks/{}'.format(filename.split("_")[0]+".png"))
    print(len(os.listdir("images")), len(os.listdir("masks")))


if __name__ == '__main__':
    x = lambda a: a[:, :, 0] & a[:, :, 1] & a[:, :, 2]
    make_dir()
    crop_data(name='berlin')
    crop_data(name='zurich')
    crop_data(name='chicago')


