import numpy as np
from scipy.signal import convolve2d

class Tensor:
    def __init__(self, data, prev=()):

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

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):

        if isinstance(other, int) or isinstance(other, float):
            other = np.ones(self.data.shape) * other

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def dot(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other))

        def _backward():
            self.grad += np.dot(other.data, out.grad.T).T
            other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward

        return out

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data**other, (self,))

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def conv2d(self, other):
        # self -> input, other -> filter
        # right always has stride one and padding 1
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(convolve2d(self.data, other.data, mode="same"), (self, other))

        def _backward():
            self.grad += convolve2d(self.data, out.grad, mode="same")
            other.grad += convolve2d(np.flipud(np.fliplr(other.data)), out.grad, mode="valid")
        out._backward = _backward

        return out


    def backward(self):

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

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def parameters(self):
        return self
    
    def item(self):
        return self.data.squeeze()