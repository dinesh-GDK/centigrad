import numpy as np

from .tensor import Tensor


class Layer:
    """
    Base class for all layers
    """

    def __init__(self) -> None:
        """
        Initialize the layer

        Args:
            None

        Returns:
            None
        """
        self.is_train = True

    def set_mode(self, is_train: bool) -> None:
        """
        Set the mode of the layer

        Args:
            is_train (bool): True if the layer is in training mode, False otherwise

        Returns:
            None
        """
        self.is_train = is_train

    def __call__(self, x: Tensor):
        """
        Apply the layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """
        Get the parameters of the layer

        Args:
            None

        Returns:
            list[Tensor]: list of parameters
        """
        raise NotImplementedError


class Flatten(Layer):
    """
    Flatten layer
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the flatten layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return Tensor.flatten(x)

    def parameters(self) -> list[None]:
        """
        Get the parameters of the layer

        Returns:
            list[None]: empty list
        """
        return []


class FullyConnected(Layer):
    """
    Fully connected layer
    """

    def __init__(self, dim_in: int, dim_out: int):
        """
        Initialize the fully connected layer

        Args:
            dim_in (int): input dimension
            dim_out (int): output dimension

        Returns:
            None
        """
        super().__init__()
        self._weight = Tensor(
            np.random.normal(0, np.sqrt(2.0 / dim_in), size=(dim_in, dim_out))
        )
        self._bias = Tensor(
            np.random.normal(0, np.sqrt(2.0 / dim_in), size=(1, dim_out))
        )

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the fully connected layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return Tensor.dot(x, self._weight) + self._bias

    def shape(self) -> None:
        """
        Print the shape of the layer

        Args:
            None

        Returns:
            None
        """
        print(f"Weight: {self._weight.shape} | Bias: {self._bias.shape}")

    def parameters(self) -> list[Tensor]:
        """
        Get the parameters of the layer

        Args:
            None

        Returns:
            list[Tensor]: list of parameters
        """
        return [self._weight.parameters(), self._bias.parameters()]


class Conv2d(Layer):
    def __init__(self, channel_in: int, channel_out: int, ksize: tuple = (3, 3)):
        """
        Initialize the convolution layer

        Args:
            channel_in (int): number of input channels
            channel_out (int): number of output channels
            ksize (tuple): kernel size

        Returns:
            None
        """
        super().__init__()
        kernel = np.random.randn(channel_out, channel_in, ksize[0], ksize[1])
        self._filter = Tensor(kernel / np.sum(kernel))

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the convolution layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return Tensor.conv2d(x, self._filter)

    def shape(self) -> None:
        """
        Print the shape of the layer

        Args:
            None

        Returns:
            None
        """
        print(f"Kernel: {self._filter.shape}")

    def parameters(self) -> list[Tensor]:
        """
        Get the parameters of the layer

        Args:
            None

        Returns:
            list[Tensor]: list of parameters
        """
        return [self._filter.parameters()]


class MaxPool2d(Layer):
    """
    Max pooling layer
    """

    def __init__(self, ksize: tuple = (2, 2)):
        """
        Initialize the max pooling layer

        Args:
            ksize (tuple): kernel size

        Returns:
            None
        """
        super().__init__()
        self._ksize = ksize

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the max pooling layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return Tensor.maxpool2d(x, self._ksize)

    def parameters(self) -> list[None]:
        """
        Get the parameters of the layer

        Args:
            None

        Returns:
            list[None]: empty list
        """
        return []


class Dropout2d(Layer):
    def __init__(self, p: float = 0.2):
        """
        Initialize the dropout layer

        Args:
            p (float): dropout probability

        Returns:
            None
        """
        super().__init__()
        self._p = p

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the dropout layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return Tensor.dropout2d(x, self._p, self.is_train)

    def parameters(self) -> list[None]:
        """
        Get the parameters of the layer

        Args:
            None

        Returns:
            list[None]: empty list
        """
        return []


class BatchNorm2d(Layer):
    """
    Batch normalization layer
    """

    def __init__(self, channels: int):
        """
        Initialize the batch normalization layer

        Args:
            channels (int): number of channels

        Returns:
            None
        """
        super().__init__()
        # gamma -> scale factor; beta -> shift factor
        self._gamma = Tensor(np.ones((1, channels, 1, 1)))
        self._beta = Tensor(np.zeros((1, channels, 1, 1)))

        # parameters used during inference (running metrics)
        # number of batches seen so far
        self.run = 0
        # running mean and variance of the batches seen so far
        self.run_mean = np.zeros((1, channels, 1, 1))
        self.run_var = np.zeros((1, channels, 1, 1))

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the batch normalization layer on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        norm, self.run, self.run_mean, self.run_var = Tensor.batchnorm2d(
            x,
            self._gamma,
            self._beta,
            self.is_train,
            self.run,
            self.run_mean,
            self.run_var,
        )
        return norm

    def parameters(self) -> list[Tensor]:
        """
        Get the parameters of the layer

        Args:
            None

        Returns:
            list[Tensor]: list of parameters
        """
        return [self._gamma, self._beta]
