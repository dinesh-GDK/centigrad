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
