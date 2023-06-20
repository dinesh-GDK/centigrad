from .tensor import Tensor

class Optimizer:
    """
    Base class for all optimizers
    """
    def __init__(self, params: list[Tensor], lr: float):
        """
        Initialize the optimizer

        Args:
            params (list[Tensor]): list of parameters
            lr (float): learning rate

        Returns:
            None
        """
        self.params = params
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
            param.grad.fill(0)

class GradientDescent(Optimizer):
    """
    Gradient descent optimizer
    """
    def __init__(self, params: list[Tensor], lr: float=0.01) -> None:
        """
        Initialize the optimizer

        Args:
            params (list[Tensor]): list of parameters
            lr (float): learning rate

        Returns:
            None
        """
        super().__init__(params, lr)

    def step(self, batch_size: int=1) -> None:
        """
        Update the parameters

        Args:
            batch_size (int): batch size

        Returns:
            None
        """
        for param in self.params:
            param.data += -self.lr * param.grad/batch_size