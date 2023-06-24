# &#x1F52E; centigrad

Autograd engine and neural network library based on numpy.

Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

Unlike [micrograd](https://github.com/karpathy/micrograd) which uses scalars, **centigrad** uses vectors.

## Installation
```
git clone https://github.com/dinesh-GDK/centigrad.git
pip install -e centigrad
```

## Features

***centigrad*** has the basic building blocks to construct ***Fully Connected Neural Networks*** and 2D ***Convolution Neural Networks***.

### Layers

- Flatten
- Fully Connected
- 2D Convolution
- 2D Max Pooling
- 2D Dropout
- 2D Batch Normalization

### Activations

- ReLu
- Tanh
- Softmax

### Losses

- Mean Square Error
- Cross Entropy

### Optimizers

- Gradient Descent

## Example

Here is an example how a model is defined in ***centigrad***

```python
class MnistNet(Model):
    def __init__(self):
        super().__init__()
        self.layerc1 = Conv2d(1, 2)
        self.maxpool = MaxPool2d()
        self.dropout = Dropout2d()
        self.batchnorm = BatchNorm2d(2)
        self.flatten = Flatten()
        self.layer1 = FullyConnected(338, 10)

    def forward(self, x):
        x = relu(self.layerc1(x))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.batchnorm(x)
        x = self.flatten(x)
        x = softmax(self.layer1(x))
        return x
```

See [demo noteb](https://github.com/dinesh-GDK/centigrad/blob/main/demo.ipynb)

 for more details

## References

- [micrograd](https://github.com/karpathy/micrograd)
