import numpy as np

from .tensor import Tensor

def relu(output):
    out = np.copy(output.data)
    out[out < 0] = 0
    out = Tensor(out, (output,))

    def _backward():
        output.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

def tanh(output):
    t = (np.exp(2*output.data) - 1) / (np.exp(2*output.data) + 1)
    out = Tensor(t, (output,))

    def _backward():
        output.grad += (1 - out.data**2) * out.grad
    out._backward = _backward

    return out

def softmax(output):
    exps = np.exp(output.data - np.max(output.data))
    out = Tensor(exps / exps.sum(), (output,))

    def _backward():
        s = out.data.reshape(-1, 1)
        grad_matrix = np.diagflat(s) - np.dot(s, s.T)
        output.grad += np.dot(out.grad, grad_matrix)

    out._backward = _backward

    return out