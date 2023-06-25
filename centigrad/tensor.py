from __future__ import annotations
from typing import Union

import numpy as np
from scipy.signal import convolve


class Tensor:
    """
    Tensor class
    """

    def __init__(
        self,
        data: Union[int, float, np.ndarray],
        prev: tuple[Tensor] = (),
        requires_grad: bool = True,
    ) -> None:
        """
        Initialize the tensor

        Args:
            data (int, float, np.ndarray): data
            prev (tuple[Tensor]): previous tensors
            requires_grad (bool): True if the gradient of the tensor is required,

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

        self.requires_grad = requires_grad
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
            # Can't use lambda function here because of the limitation of pickle
            self._backward = self._default_backward
            self._prev = set(prev)
        self.shape = self.data.shape

    def _default_backward(self) -> None:
        return None

    def __add__(self, other: Union[int, float, np.ndarray]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += np.sum(out.grad, axis=0)

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

    def __pow__(self, other: Union[int, float]) -> Tensor:
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Tensor(self.data**other, (self,))

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def dot(self, other: Union[int, float, np.ndarray]):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other))

        def _backward():
            self.grad += np.dot(other.data, out.grad.squeeze().T).T
            other.grad += np.dot(self.data.T, out.grad)

        out._backward = _backward

        return out

    def flatten(self) -> Tensor:
        """
        Flatten the tensor
        """
        initial_shape = self.data.shape
        out = Tensor(np.reshape(self.data, (initial_shape[0], -1)), (self,))

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

        assert self.shape[1] == other.shape[1], "number of channels must match"
        assert (
            other.shape[-1] % 2 == 1 and other.shape[-2] % 2 == 1
        ), "only supporting odd kernel sizes"

        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            np.zeros(
                (
                    self.shape[0],
                    other.shape[0],
                    self.shape[-2] - 2 * (other.shape[-2] // 2),
                    self.shape[-1] - 2 * (other.shape[-1] // 2),
                )
            ),
            (self, other),
        )
        # TODO: remove for loop
        for i in range(self.shape[0]):
            for j in range(other.shape[0]):
                out.data[i][j] = convolve(self.data[i], other.data[j], mode="valid")

        def _backward():
            # TODO: remove for loop
            for i in range(self.shape[0]):
                for j in range(other.shape[0]):
                    other.grad[j] += convolve(
                        other.data[j][:, :, ::-1][:, ::-1, :],
                        np.expand_dims(out.grad[i][j], axis=0),
                        mode="same",
                    )
                self.grad[i] += convolve(self.data[i], out.grad[i], mode="same")

        out._backward = _backward

        return out

    def maxpool2d(self, ksize: tuple[int, int]) -> Tensor:
        """
        Maxpool the tensor
        """
        out = Tensor(
            np.zeros(
                (
                    self.shape[0],
                    self.shape[1],
                    self.shape[2] // ksize[0],
                    self.shape[3] // ksize[1],
                )
            ),
            (self,),
        )
        idx_max = np.zeros(
            (
                self.shape[0],
                self.shape[-3]
                * (self.shape[-2] // ksize[-2])
                * (self.shape[-1] // ksize[-1]),
            )
        )
        for i in range(self.shape[0]):
            slid = np.lib.stride_tricks.sliding_window_view(
                self.data[i], ksize, axis=(-2, -1)
            )[:, :: ksize[0], :: ksize[1]]
            out.data[i] = np.amax(slid, axis=(-1, -2))
            idx_max[i] = slid.reshape(-1, ksize[0] * ksize[1]).argmax(axis=1)

        def _backward():
            _, m, n, _ = self.data.shape
            f = idx_max.shape[1]
            for i in range(self.shape[0]):
                for k in range(len(idx_max[i])):
                    x = k // (f // m)
                    y = int(
                        ksize[0] * ((k % (f // m)) // (n // ksize[0]))
                        + idx_max[i][k] // ksize[0]
                    )
                    z = int(ksize[1] * (k % (n // ksize[0])) + idx_max[i][k] % ksize[0])
                    self.grad[i, x, y, z] += out.grad[
                        i, x, (k % (f // m)) // (n // ksize[0]), k % (n // ksize[0])
                    ]

        out._backward = _backward

        return out

    def dropout2d(self, p: float, is_train: bool = True) -> Tensor:
        """
        Dropout the tensor
        """
        if not is_train:
            return self

        mask = np.zeros(self.data.shape)
        out = Tensor(np.zeros(self.data.shape), (self,))

        for i in range(self.data.shape[0]):
            mask[i] = np.random.binomial(1, 1 - p, size=self.data.shape[1:])
            out.data[i] = self.data[i] * mask[i]

        def _backward():
            for i in range(self.data.shape[0]):
                self.grad[i] += mask[i] * out.grad[i]

        out._backward = _backward

        return out

    def batchnorm2d(
        self,
        gamma: Union[Tensor, float],
        beta: Union[Tensor, float],
        is_train: bool,
        run: int,
        run_mean: np.ndarray,
        run_var: np.ndarray,
    ) -> Tensor:
        """
        Batch normalize the tensor
        """
        eps = 1e-5

        if is_train:
            mean = self.data.mean(axis=(0, 2, 3), keepdims=True)
            var = self.data.var(axis=(0, 2, 3), keepdims=True)

            run += 1
            temp = run_mean + (mean - run_mean) / run
            run_var = run_var + (mean - run_mean) * (mean - temp)
            run_mean = temp

            norm = (self.data - mean) / np.sqrt(var + eps)
        else:
            norm = (self.data - run_mean) / np.sqrt(run_var + eps)

        out = Tensor(gamma.data * norm + beta.data, (self, gamma, beta))

        def _backward():
            self.grad += out.grad * gamma.data
            gamma.grad += np.sum(
                np.mean(out.grad * out.data, axis=(2, 3), keepdims=True), axis=0
            )
            beta.grad += np.sum(
                np.mean(out.grad, axis=(2, 3), keepdims=True), axis=0, keepdims=True
            )

        out._backward = _backward

        return out, run, run_mean, run_var

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
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

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
