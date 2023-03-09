import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np

from matplotlib import pyplot as plt

from PIL import Image

# np.set_printoptions(threshold=np.inf, suppress=True)

lib = ctypes.CDLL("./fmen1080.so") # shared library

# return and arg types

lib.py_entry.argtypes = [
    ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=(1, 3, 1920, 1080)),
    ndpointer(dtype=ctypes.c_float, flags="C_CONTIGUOUS", shape=(1, 3, 3840, 2160))
]

in_img = Image.open('in.png').convert('RGB')

# plt.imshow(in_img, interpolation='nearest')
# plt.show()

# convert to continuous numpy array
py_input_tensor = (np.float32(in_img))
py_input_tensor = np.transpose(py_input_tensor, [2, 1, 0])
py_input_tensor = np.ascontiguousarray(np.expand_dims(py_input_tensor, 0))

py_output_tensor = (np.float32(np.zeros(shape=[1, 3, 3840, 2160])))

# call c function
print("py: {}".format(hex(id(py_input_tensor))))

print ("py: py_input_tensor[0][0][0][0]", py_input_tensor[0][0][0][0])
print ("py: py_input_tensor[0][1][567][976]", py_input_tensor[0][1][567][976])
print ("py: py_input_tensor[0][2][1234][399]", py_input_tensor[0][2][1234][399])

lib.py_entry(py_input_tensor, py_output_tensor)

# # pixel shuffle
# x = final_conv_layer_out
# b, c, h, w = x.shape
# blocksize=4
# tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
# tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
# y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

y = np.squeeze(py_output_tensor)
out_img = np.transpose(y, [2, 1, 0])
out_img_clipped = np.clip(out_img, a_min=0, a_max=255)
out_img_clipped = np.uint8(out_img_clipped)

plt.imshow(out_img_clipped, interpolation='nearest')
plt.show()

# save image
try:
    plt.imsave('out_img_plt-imsave.png', out_img_clipped)
except Exception as e:
    print(e)


exit(0)
