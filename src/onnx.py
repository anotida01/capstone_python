import onnx
import onnxruntime as ort
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from utils.psnr import PSNR
import time

ONNX_PATH = './model/onnx/fmen_x2_div2k_35dB_FULL.onnx'

DATA_DIR = './data/'
LR_DIR = DATA_DIR + 'LR/'
HR_DIR = DATA_DIR + 'HR/'
SR_DIR = DATA_DIR + 'SR_ONNX/'

INPUT_SHAPE = (1, 3, 1920, 1080)
OUTPUT_SHAPE = (1, 3, 3840, 2160)

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

    # output log file
    out_file = open('onnx.csv', 'w+')
    out_file.write('image, psnr (dB), runtime (seconds)\n')

    images = iterate_dir()

    for i, (img_lr, img_hr) in enumerate(images):

        print ('Found: ', img_lr, img_hr)
        in_img = Image.open(img_lr).convert('RGB')
        py_input_tensor = (np.float32(in_img))
        py_input_tensor = np.expand_dims(py_input_tensor, 0)
        py_input_tensor = np.transpose(py_input_tensor, [0, 3, 2, 1])

        # convert to continuous numpy array
        # py_input_tensor = np.ascontiguousarray(py_input_tensor)

        # py_output_tensor = (np.float32(np.zeros(shape=OUTPUT_SHAPE)))

        # call ONNX function and start timers
        old_time = time.process_time()
        ort_sess = ort.InferenceSession(ONNX_PATH)
        py_output_tensor = ort_sess.run(None, {'input': py_input_tensor})
        new_time = time.process_time()
        runtime =  new_time - old_time

        out_img = np.squeeze(py_output_tensor)
        out_img = np.transpose(out_img, (2, 1, 0))
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
        out_file.flush()

        # save image
        out_img_path = os.path.join(SR_DIR, img_lr_basename)

        if (not os.path.exists(SR_DIR)): os.makedirs(SR_DIR)

        try:
            print('Saving SR image as {}'.format(out_img_path))
            plt.imsave(out_img_path, out_img_clipped)
        except Exception as e:
            print(e)
    
    out_file.close()
    exit(0)
