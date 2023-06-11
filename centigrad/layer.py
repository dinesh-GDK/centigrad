import numpy as np

from .tensor import Tensor

class Flatten:
    def __init__(self):
        pass

    def __call__(self, x):
        return Tensor(np.reshape(x.data, (1, x.data.shape[0]*x.data.shape[1])))

class FullyConnected:
    def __init__(self, dim_in, dim_out):
        self._weight = Tensor(np.random.uniform(low=-1., high=1., size=(dim_in, dim_out))/np.sqrt(dim_in*dim_out))
        self._bias = Tensor(np.random.uniform(low=-1., high=1., size=(1, dim_out))/np.sqrt(dim_out))

    def __call__(self, x):
        return Tensor.dot(x, self._weight) + self._bias

    def shape(self):
        print(f"Weight: {self._weight.shape} | Bias: {self._bias.shape}")

    def parameters(self):
        return [self._weight.parameters(), self._bias.parameters()]
    
class Conv2d:
    def __init__(self, ksize=(3, 3)):
        self._filter = Tensor(np.random.uniform(low=-1., high=1., size=ksize)/np.sqrt(ksize[0]*ksize[1]))

    def __call__(self, x):
        return Tensor.conv2d(x, self._filter)
    
    def shape(self):
        print(f"Kernel; {self._filter.shape}")

    def parameters(self):
        return [self._filter.parameters()]

