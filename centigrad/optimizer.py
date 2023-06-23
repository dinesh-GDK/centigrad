from .tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers
    """

    def __init__(self, params: list[Tensor], batch_size: int, lr: float):
        """
        Initialize the optimizer

        Args:
            params (list[Tensor]): list of parameters
            batch_size (int): batch size
            lr (float): learning rate

        Returns:
            None
        """
        self.params = params
        self.batch_size = batch_size
        self.lr = lr

    def step(self) -> None:
        """
        Update the parameters

        Args:
            None

        Returns:
            None
        """
        raise NotImplementedError

    def zero_grad(self) -> None:
        """
        Zero out the gradients of the parameters

        Args:
            None

        Returns:
            None
        """
        for param in self.params:
            if isinstance(param, Tensor):
                if hasattr(param, "grad"):
                    param.grad.fill(0)
                else:
                    raise ValueError(f"Tensor {param} does not have attribute grad")
            else:
                raise ValueError(f"Unsupported param type: {type(param)}")


class GradientDescent(Optimizer):
    """
    Gradient descent optimizer
    """

    def __init__(self, params: list[Tensor], batch_size: int, lr: float = 0.01) -> None:
        """
        Initialize the optimizer

        Args:
            params (list[Tensor]): list of parameters
            batch_size (int): batch size
            lr (float): learning rate

        Returns:
            None
        """
        super().__init__(params, batch_size, lr)

    def step(self) -> None:
        """
        Update the parameters

        Args:
            None

        Returns:
            None
        """
        for param in self.params:
            param.data += -(self.lr / self.batch_size) * param.grad
