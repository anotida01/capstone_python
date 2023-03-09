
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

from PIL import Image
import pickle
import utils.utils_image as util

lib = ctypes.CDLL("./fmen1080_new.so") # shared library

INPUT_SHAPE = (1, 3, 1080, 1920)
OUTPUT_SHAPE = (1, 50, 1080, 1920)
OUTPUT_FILE_NAME = "node__head_Conv"

lib.py_entry.argtypes = [
    ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=INPUT_SHAPE),
    ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=OUTPUT_SHAPE)
]

in_img = Image.open('in.png').convert('RGB')

img_lr = util.imread_uint('in.png', n_channels=3)
img_lr = util.uint2tensor4(img_lr, 255)
img_lr = np.float32(img_lr)
py_input_tensor = img_lr
py_input_tensor = np.ascontiguousarray(py_input_tensor)

# convert to continuous numpy array
# py_input_tensor = (np.float32(in_img))
# py_input_tensor = np.transpose(py_input_tensor, [2, 1, 0])

# normalize to 0 to 1
# py_input_tensor = py_input_tensor / 255

# py_input_tensor = np.ascontiguousarray(np.expand_dims(py_input_tensor, 0))

py_output_tensor = (np.float32(np.zeros(shape=OUTPUT_SHAPE)))

# grab tensor from C
lib.py_entry(py_input_tensor, py_output_tensor)

# write tensor to outfile for comparison
outfile = open("node__head_Conv.pickle", "wb+")
pickle.dump(py_output_tensor, outfile)
outfile.close()
