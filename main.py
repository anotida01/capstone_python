import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sys import argv
import os
from utils.psnr import PSNR
import datetime as dt

DATA_DIR = './data/'
LR_DIR = DATA_DIR + 'LR/'
HR_DIR = DATA_DIR + 'HR/'
SR_DIR = DATA_DIR + 'SR/'

INPUT_SHAPE = (1, 3, 1080, 1920)
OUTPUT_SHAPE = (1, 3, 2160, 3840)

def iterate_dir():

    images = list()

    legal_types = ['.png', '.jpg', '.tif']
    for f in sorted(os.listdir(LR_DIR)):
        if os.path.splitext(f)[1] in legal_types:
            images.append(
                # (os.path.join(path, f),os.path.join(hr_path, f))
                (os.path.join(LR_DIR, f),os.path.join(HR_DIR, f))
            )

    return images

if __name__ == '__main__':

    lib = ctypes.CDLL("./fmen1080_new.so") # shared library

    # return and arg types
    lib.py_entry.argtypes = [
        ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=INPUT_SHAPE),
        ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=OUTPUT_SHAPE)
    ]

    # output log file
    out_file = open('out.csv', 'w+')
    out_file.write('image, psnr (dB), runtime (seconds)\n')

    # read images
    if (len(argv) != 1):
        print('Attempting to read image: ', argv[1])
        images = [(argv[1], None)]
    else:
        print('No arguments provided. Searching {} for images'.format(DATA_DIR))
        images = iterate_dir()

    for i, (img_lr, img_hr) in enumerate(images):

        print ('Found: ', img_lr, img_hr)
        in_img = Image.open(img_lr).convert('RGB')
        py_input_tensor = (np.float32(in_img))
        py_input_tensor = np.expand_dims(py_input_tensor, 0)
        py_input_tensor = np.transpose(py_input_tensor, [0, 3, 1, 2])

        # convert to continuous numpy array
        py_input_tensor = np.ascontiguousarray(py_input_tensor)

        py_output_tensor = (np.float32(np.zeros(shape=OUTPUT_SHAPE)))

        # call C function and start timers
        old_time = dt.datetime.now()
        lib.py_entry(py_input_tensor, py_output_tensor)
        new_time = dt.datetime.now()
        runtime =  dt.timedelta.total_seconds(new_time - old_time)

        out_img = np.squeeze(py_output_tensor)
        out_img = np.transpose(out_img, (1, 2, 0))
        out_img_clipped = np.clip(out_img, a_min=0, a_max=255)
        out_img_clipped = np.uint8(out_img_clipped.round())

        try:
            hr = Image.open(img_hr).convert('RGB')
            psnr = PSNR(hr, out_img_clipped)
            print("PSNR is {}dB".format(psnr))
        except Exception as e:
            print(e)
            psnr = 0
        
        # save results to log file
        img_lr_basename = os.path.basename(img_lr)
        line = "{}, {}, {}\n".format(img_lr_basename, psnr, runtime)
        out_file.write(line)

        # save image
        out_img_path = os.path.join(SR_DIR, img_lr_basename)

        try:
            print('Saving SR image as {}'.format(out_img_path))
            plt.imsave(out_img_path, out_img_clipped)
        except Exception as e:
            print(e)
    
    out_file.close()
    exit(0)






    # plt.imshow(in_img, interpolation='nearest')
    # plt.show()




    # call c function
    # print("py: {}".format(hex(id(py_input_tensor))))

    # print ("py: py_input_tensor[0][0][0][0]", py_input_tensor[0][0][0][0])
    # print ("py: py_input_tensor[0][1][567][976]", py_input_tensor[0][1][567][976])
    # print ("py: py_input_tensor[0][2][1234][399]", py_input_tensor[0][2][1234][399])


    # # pixel shuffle
    # x = py_output_tensor
    # b, c, h, w = x.shape
    # blocksize=2
    # tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
    # tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
    # y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])



    plt.imshow(out_img_clipped, interpolation='nearest')
    plt.show()

    # save image
    try:
        plt.imsave('out_img_plt-imsave.png', out_img_clipped)
    except Exception as e:
        print(e)


    exit(0)
