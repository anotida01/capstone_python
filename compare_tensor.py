import pickle
import torch
import numpy as np


# grab pytorch tensor
infile = open('head_Conv_py.pickle', 'rb')
torch_device = torch.device('cpu')
py = torch.load(infile).to(torch_device)

py_tensor = torch.Tensor.numpy(py)


# grab ONNX tensor
infile.close()
infile = open('node__head_Conv.pickle', 'rb')

onnx_tensor = pickle.load(infile)

# onnx_tensor = np.transpose(onnx_tensor, [0, 1, 3, 2])

eq = np.isclose(py_tensor, onnx_tensor)

num_true = np.count_nonzero(eq)
num_false = eq.size - num_true

print("matches: ", num_true)
print("mismatch: ", num_false)
print("mismatch pct: ", (num_false/eq.size) * 100)

