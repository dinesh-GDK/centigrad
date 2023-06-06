class Model:
    def __init__(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return x
    
    def parameters(self):
        params = list()
        for name, layer in self.__dict__.items():
            if name == "flatten":
                continue
            params.extend(layer.parameters())
        return params
