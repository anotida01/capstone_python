

from sys import argv
import os
from PIL import Image

DATA_DIR = './data/'
LR_DIR = DATA_DIR + 'LR/'
SR_DIR = DATA_DIR + 'SR_bicubic/'

def iterate_dir():

    images = list()

    legal_types = ['.png', '.jpg', '.tif']
    for f in sorted(os.listdir(LR_DIR)):
        if os.path.splitext(f)[1] in legal_types:
            images.append(
                (os.path.join(LR_DIR, f),os.path.join(SR_DIR, f))
            )

    return images

if __name__ == '__main__':

    images = iterate_dir()

    if (not os.path.exists(SR_DIR)): os.makedirs(SR_DIR)

    for i, (img_lr_path, img_sr_path) in enumerate(images):
        
        img_lr = Image.open(img_lr_path).convert('RGB')

        img_sr = img_lr.resize((img_lr.width*2, img_lr.height*2), Image.Resampling.BICUBIC)

        img_sr.save(img_sr_path)

        print("Saved a new file: {}".format(img_sr_path))

