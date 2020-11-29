import numpy as np
import os
from PIL import Image
from tqdm import tqdm


folder_path = ['images', 'masks']
for folder in folder_path:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("creat a new folder named {}".format(folder))

image_path = 'images'
label_path = 'labels'

img_size = 2448     # 原图片2448*2448

x = lambda a : a[:,:,0] & a[:,:,1] & a[:,:,2]        # 截取第一维和第二维
for index_, file in enumerate(tqdm(os.listdir(label_path))):
    mask = Image.open(label_path +  '/' + file)
    mask = np.array(mask)
    mask_new = np.zeros([2448, 2448])

    index = x((mask==[0,0,255]))
    mask_new[index] = 100                                 # 水体
    index = x((mask == [0,255,255]))
    mask_new[index] = 300                                 # 建筑
    index = x((mask==[255,255,0]))
    mask_new[index] = 400                                 # 耕地
    index = x((mask==[255,0,255]))
    mask_new[index] = 500                                 # 草地
    index = x((mask==[0,255,0]))
    mask_new[index] = 600                                 # 林地
    index = x((mask==[255,255,255]))
    mask_new[index] = 700                                 # 裸土
    index = x((mask==[0,0,0]))
    mask_new[index] = 0                                   # 无意义
    label = Image.fromarray(np.uint16(mask_new))
    label.save('masks/{}'.format(file.replace("tif", "png")))
    
print(len(os.listdir("images")), len(os.listdir("masks")))