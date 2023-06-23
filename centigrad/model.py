from .layer import Layer
from .tensor import Tensor


class Model:
    """
    Base class for all models
    """

    def __init__(self):
        self.train()

    def __call__(self, x: Tensor) -> Tensor:
        """
        Apply the model on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return self.forward(x)

    def train(self) -> None:
        """
        Set the model to train mode

        Args:
            None

        Returns:
            None
        """
        self.set_mode(True)

    def inference(self) -> None:
        """
        Set the model to inference mode

        Args:
            None

        Returns:
            None
        """
        self.set_mode(False)

    def set_mode(self, is_train: bool) -> None:
        """
        Set the mode of the model

        Args:
            is_train (bool): True if the model is in training mode, False otherwise

        Returns:
            None
        """
        for layer in self.__dict__.values():
            if isinstance(layer, Layer):
                layer.set_mode(is_train)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply the model on the input tensor

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: output tensor
        """
        return x

    def parameters(self) -> list[Tensor]:
        """
        Get the parameters of the model

        Args:
            None

        Returns:
            list[Tensor]: list of parameters
        """
        params = list()
        for layer in self.__dict__.values():
            if isinstance(layer, Layer):
                params.extend(layer.parameters())
        return params
