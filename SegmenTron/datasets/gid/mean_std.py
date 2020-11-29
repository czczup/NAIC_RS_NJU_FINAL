import numpy as np
from PIL import Image
from tqdm import tqdm
gid_mean = np.array([0.35592559, 0.3763364, 0.34348521])
gid_std = np.array([0.23231347, 0.22758195, 0.22442952])

competition_mean = np.array([[0.41726797, 0.4177039, 0.40032812]])
competition_std = np.array([0.18120442, 0.16814594, 0.17162799])

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

for i in tqdm(range(10)):
    image = Image.open("images/%d.tif"%(i+1))
    image = np.array(image) / 255.0
    image = (image - gid_mean) / gid_std
    image = image * competition_std + competition_mean
    image = np.clip(image*255.0, 0, 255)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save("new/%d.jpg"%(i+1))