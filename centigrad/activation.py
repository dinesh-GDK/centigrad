import numpy as np

from .tensor import Tensor

def relu(tensor: Tensor) -> Tensor:
    """
    ReLU activation function

    Args:
        tensor (Tensor): tensor to apply ReLU

    Returns:
        Tensor: output of the ReLU layer
    """
    out = np.copy(tensor.data)
    out[out < 0] = 0
    out = Tensor(out, (tensor,))

    def _backward():
        tensor.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out

def tanh(tensor: Tensor) -> Tensor:
    """
    Tanh activation function

    Args:
        tensor (Tensor): tensor to apply Tanh

    Returns:
        Tensor: tensor of the Tanh layer
    """
    t = (np.exp(2*tensor.data) - 1) / (np.exp(2*tensor.data) + 1)
    out = Tensor(t, (tensor,))

    def _backward():
        tensor.grad += (1 - out.data**2) * out.grad
    out._backward = _backward

    return out

def softmax(tensor: Tensor) -> Tensor:
    """
    Softmax activation function

    Args:
        tensor (Tensor): tensor to apply Softmax

    Returns:
        Tensor: tensor of the Softmax layer
    """
    exps = np.exp(tensor.data - np.max(tensor.data))
    out = Tensor(exps / exps.sum(), (tensor,))

    def _backward():
        s = out.data.reshape(-1, 1)
        grad_matrix = np.diagflat(s) - np.dot(s, s.T)
        tensor.grad += np.dot(out.grad, grad_matrix)

    out._backward = _backward

    return out