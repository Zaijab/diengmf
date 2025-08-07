import equinox as eqx


class AbstractInvertibleNeuralNetwork(eqx.Module, strict=True):
    def forward():
        raise NotImplementedError

    def __call__():
        raise NotImplementedError
