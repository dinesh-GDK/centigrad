import numpy as np

from .tensor import Tensor        

def mse(output, label):
    out = Tensor(np.sum(label.data - output.data), (output, label))

    def _backward():
        output.grad += -2*np.sum(label.data - output.data)
    out._backward = _backward

    return out

def cross_entropy(output, label):

    out = Tensor(-np.sum(label.data*np.log(output.data)), (output, label))

    def _backward():
        output.grad += -label.data/output.data
    out._backward = _backward

    return out