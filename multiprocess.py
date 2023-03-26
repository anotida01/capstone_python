import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sys import argv
import os
from utils.psnr import PSNR
import time
import multiprocessing as mp

DATA_DIR = './data/'
LR_DIR = DATA_DIR + 'LR/'
HR_DIR = DATA_DIR + 'HR/'
SR_DIR = DATA_DIR + 'SR/'

INPUT_SHAPE = (1, 3, 1080, 1920)
OUTPUT_SHAPE = (1, 3, 2160, 3840)

NUM_WORKERS = os.cpu_count() - 1

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


def mp_print(string):
    PID = os.getpid()
    print("[{}] - {}".format(PID, string))


def run_image(img_lr, img_hr):
    
    lib = ctypes.CDLL("./fmen1080_new.so") # shared library

    # return and arg types
    lib.py_entry.argtypes = [
        ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=INPUT_SHAPE),
        ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=OUTPUT_SHAPE)
    ]

    mp_print('Found: ' + img_lr + " " + img_hr)
    in_img = Image.open(img_lr).convert('RGB')
    py_input_tensor = (np.float32(in_img))
    py_input_tensor = np.expand_dims(py_input_tensor, 0)
    py_input_tensor = np.transpose(py_input_tensor, [0, 3, 1, 2])

    # convert to continuous numpy array
    py_input_tensor = np.ascontiguousarray(py_input_tensor)

    py_output_tensor = (np.float32(np.zeros(shape=OUTPUT_SHAPE)))

    # call C function and start timers
    old_time = time.process_time()
    lib.py_entry(py_input_tensor, py_output_tensor)
    new_time = time.process_time()
    runtime =  new_time - old_time
    mp_print("C Function Exited. CPU_TIME: {}s".format(runtime))

    out_img = np.squeeze(py_output_tensor)
    out_img = np.transpose(out_img, (1, 2, 0))
    out_img_clipped = np.clip(out_img, a_min=0, a_max=255)
    out_img_clipped = np.uint8(out_img_clipped.round())

    # save image
    img_hr_basename = os.path.basename(img_hr)
    img_lr_basename = os.path.basename(img_lr)
    out_img_path = os.path.join(SR_DIR, img_lr_basename)

    if (not os.path.exists(SR_DIR)): os.makedirs(SR_DIR)

    try:
        mp_print('Saving SR image as {}'.format(out_img_path))
        plt.imsave(out_img_path, out_img_clipped)
    except Exception as e:
        print(e)

    try:
        hr = Image.open(img_hr).convert('RGB')
        psnr = PSNR(hr, out_img_clipped)
        mp_print("PSNR of {} is {}dB".format(img_hr_basename, psnr))
    except Exception as e:
        mp_print(e)
        psnr = 0


def check_workers(workers, finish):
    # check if any other workers are still active
    while (True):
        alive = 0
        for i, p in enumerate(workers):
            if p.is_alive():
                alive += 1
            else:
                p.close()
                del(workers[i])
            
        if (finish): # wait for all workers to finish
            if (alive == 0): break
            else: continue
        elif (alive != NUM_WORKERS):
            break

        time.sleep(1)


if __name__ == '__main__':

    # read images
    images = iterate_dir()

    workers = list()

    for i, (img_lr, img_hr) in enumerate(images):

        # dispatch a worker for each image
        p = mp.Process(target=run_image, args=((img_lr, img_hr)))
        workers.append(p)
        p.start()

        # max assigned workers hit 
        if (len(workers) >= NUM_WORKERS):
            check_workers(workers, 0) # wait for at least one to finish

    # images are all assigned to workers. wait for all to finish
    check_workers(workers, 1)
