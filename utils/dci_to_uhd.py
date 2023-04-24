

from sys import argv
import os
from PIL import Image

DATA_DIR = './data/'
LR_DIR = DATA_DIR + 'LR/'
HR_DIR = DATA_DIR + 'HR/'
SR_DIR = DATA_DIR + 'SR/'

def iterate_dir():

    images = list()

    legal_types = ['.png', '.jpg', '.tif']
    for f in sorted(os.listdir(HR_DIR)):
        if os.path.splitext(f)[1] in legal_types:
            images.append(
                # (os.path.join(path, f),os.path.join(hr_path, f))
                (os.path.join(LR_DIR, f),os.path.join(HR_DIR, f))
            )

    return images

if __name__ == '__main__':

    # if (len(argv) != 2):
        # print("Syntax:{} [video_file]".format(argv[0]))
        # exit(1)

    images = iterate_dir()

    for i, (img_lr_path, img_hr_path) in enumerate(images):
        
        img_hr = Image.open(img_hr_path).convert('RGB')

        if (img_hr.width <= 3840):
            print("Width of {} is not 4096. Skipping".format(img_hr_path))
            continue;

        img_hr = img_hr.crop(box=(128, 0, 3968, 2160))

        img_lr = img_hr.resize((img_hr.width//2, img_hr.height//2), Image.Resampling.BICUBIC)

        # img_lr_path = open(img_lr_path, 'w+')
        img_lr.save(img_lr_path)
        # img_lr_path.close()

        # img_hr_path = open(img_hr_path, 'w+')
        img_hr.save(img_hr_path)
        # img_hr_path.close()

