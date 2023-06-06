class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.grad.fill(0)

class GradientDescent(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

    def step(self, batch_size):
        for param in self.params:
            param.data += -self.lr * param.grad/batch_size