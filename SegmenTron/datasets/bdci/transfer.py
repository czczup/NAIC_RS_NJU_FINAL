import numpy as np
import os
from PIL import Image
from tqdm import tqdm


folder_path = ['images', 'masks']
for folder in folder_path:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("creat a new folder named {}".format(folder))

image_path = 'img_train'
label_path = 'lab_train'

filenames = ["T%06d.jpg"%i for i in range(90000,100000)]

for filename in tqdm(filenames):
    image = Image.open(os.path.join(image_path, filename))
    image.save(os.path.join(folder_path[0], filename))
    
    mask = Image.open(os.path.join(label_path, filename.replace("jpg", "png")))
    mask = np.array(mask)
    mask_new = np.zeros([256, 256])

    mask_new[mask==3] = 100                                 # 水体
    mask_new[mask==4] = 200                                 # 交通
    mask_new[mask==0] = 300                                 # 建筑
    mask_new[mask==1] = 400                                 # 耕地
    mask_new[mask==5] = 500                                 # 草地
    mask_new[mask==2] = 600                                 # 林地
    mask_new[mask==6] = 800                                 # 其他
    mask_new[mask==255] = 800                               # 其他

    label = Image.fromarray(np.uint16(mask_new))
    label.save(os.path.join(folder_path[1], filename.replace("jpg", "png")))
    
print(len(os.listdir("images")), len(os.listdir("masks")))