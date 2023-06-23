import numpy as np

from .tensor import Tensor        

def mse(output: Tensor, label: Tensor):
    """
    Mean squared error loss

    Args:
        output (Tensor): output tensor
        label (Tensor): label tensor

    Returns:
        Tensor: loss tensor
    """
    out = Tensor(np.sum(label.data - output.data), (output, label))

    def _backward():
        output.grad += -2*np.sum(label.data - output.data)
    out._backward = _backward

    return out

def cross_entropy(output: Tensor, label: Tensor):
    """
    Cross entropy loss

    Args:
        output (Tensor): output tensor
        label (Tensor): label tensor

    Returns:
        Tensor: loss tensor
    """
    eps = 1e-6
    out = Tensor(-np.sum(label.data*np.log(output.data + eps), axis=-1), (output, label))

    def _backward():
        output.grad += -label.data/(output.data + eps)
    out._backward = _backward

    return out