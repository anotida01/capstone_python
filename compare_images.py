
import os
from PIL import Image
from utils.psnr import PSNR
import numpy as np

DATA_DIR = './data/'
SR_DIR = DATA_DIR + 'bicubic/'
HR_DIR = DATA_DIR + 'HR/'

def iterate_dir():

    images = list()

    legal_types = ['.png', '.jpg', '.tif']
    for f in sorted(os.listdir(SR_DIR)):
        if os.path.splitext(f)[1] in legal_types:
            images.append(
                (os.path.join(SR_DIR, f),os.path.join(HR_DIR, f))
            )

    return images


if __name__ == '__main__':

    images = iterate_dir()

    for i, (img_sr_path, img_model_path) in enumerate(images):

        img_model = Image.open(img_model_path).convert('RGB')
        img_sr = Image.open(img_sr_path).convert('RGB')

        img_model = np.uint8(img_model)
        img_sr = np.uint8(img_sr)

        psnr = PSNR(img_model, img_sr)

        img_name = os.path.basename(img_sr_path)

        print("PSNR of {} is {}dB".format(img_name, psnr))
