from __future__ import annotations
from typing import Union

import numpy as np
from scipy.signal import convolve

class Tensor:
    """
    Tensor class
    """
    def __init__(self, data: Union[int, float, np.ndarray], prev: tuple[Tensor]=()) -> None:
        """
        Initialize the tensor

        Args:
            data (int, float, np.ndarray): data
            prev (tuple[Tensor]): previous tensors

        Returns:
            None

        Note:
            - The gradient of the tensor is initialized to zero
            - Each tensor has a backward function (_backward) that computes the
              gradient of the tensor
        """
        if isinstance(data, int) or isinstance(data, float):
            self.data = np.array([[data]])
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(prev)
        self.shape = self.data.shape

    def __add__(self, other: Union[int, float, np.ndarray]) -> Tensor:

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other: Union[int, float, np.ndarray]) -> Tensor:

        if isinstance(other, int) or isinstance(other, float):
            other = np.ones(self.data.shape) * other

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def dot(self, other: Union[int, float, np.ndarray]):

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other))

        def _backward():
            self.grad += np.dot(other.data, out.grad.T).T
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out

    def __pow__(self, other: Union[int, float]) -> Tensor:

        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data**other, (self,))

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def flatten(self) -> Tensor:
        """
        Flatten the tensor
        """        
        initial_shape = self.data.shape
        out = Tensor(np.reshape(self.data, (1, -1)), (self,))

        def _backward():
            self.grad += np.reshape(out.grad, initial_shape)
        out._backward = _backward

        return out
    
    def conv2d(self, other: Union[int, float, np.ndarray]) -> Tensor:
        """
        Convolve the tensor with a kernel
        
        Note:
        always has stride 1 and padding 1 for now
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        assert self.shape[0] == other.shape[1], "number of channels must match"
        assert other.shape[-1] % 2 == 1 and other.shape[-2] % 2 == 1, "only supporting odd kernel sizes"

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.zeros((other.shape[0], self.shape[1] - 2*(other.shape[-2]//2), self.shape[2] - 2*(other.shape[-1]//2))), (self, other))
        # TODO: remove for loop
        for i in range(other.shape[0]):
            out.data[i] = convolve(self.data, other.data[i], mode="valid")

        def _backward():
            # TODO: remove for loop
            for i in range(other.shape[0]):
                other.grad[i] += convolve(other.data[i][:, :, ::-1][:, ::-1, :], out.grad, mode="same")
            self.grad += convolve(self.data, out.grad, mode="same")
        out._backward = _backward

        return out
    
    def maxpool2d(self, ksize: tuple[int, int]) -> Tensor:
        """
        Maxpool the tensor
        """
        slid = np.lib.stride_tricks.sliding_window_view(self.data, ksize, axis=(-2, -1))[:, ::ksize[0], ::ksize[1]]
        out = Tensor(np.amax(slid, axis=(-1, -2)), (self,))

        def _backward():
            idx_max = slid.reshape(-1, ksize[0]*ksize[1]).argmax(axis=1)
            m, n, _ = self.data.shape
            f = idx_max.shape[0]
            for k in range(len(idx_max)):
                x = k//(f//m)
                y = ksize[0]*((k%(f//m))//(n//ksize[0])) + idx_max[k]//ksize[0]
                z = ksize[1]*(k%(n//ksize[0])) + idx_max[k]%ksize[0]
                self.grad[x, y, z] += out.grad[x, (k%(f//m))//(n//ksize[0]), k%(n//ksize[0])]
        out._backward = _backward

        return out
    
    def dropout(self, p: float, is_train: bool=True) -> Tensor:
        """
        Dropout the tensor
        """
        if not is_train:
            return self

        mask = np.random.binomial(1, p, size=self.data.shape)
        out = Tensor(self.data * mask, (self,))

        def _backward():
            self.grad += mask * out.grad
        out._backward = _backward

        return out

    def backward(self) -> None:
        """
        Backpropagate the gradients
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.array([1])
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        """
        String representation of the tensor
        """
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def parameters(self) -> Tensor:
        """
        Returns the parameters of the tensor
        """
        return self

    def item(self) -> Union[float, np.ndarray]:
        """
        Returns the data of the tensor as a numpy array
        """
        return self.data.squeeze()
    
    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1