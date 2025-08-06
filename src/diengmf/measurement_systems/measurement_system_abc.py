import abc

import equinox as eqx
from jaxtyping import Array, Key


class AbstractMeasurementSystem(eqx.Module, strict=True):
    covariance: eqx.AbstractVar[Array]

    def __check_init__(self):
        # check Shape is square
        # check Positive Definite
        pass

    @abc.abstractmethod
    def __call__(self, state, key: Key[Array, "..."] | None = None):
        """
        Generate a measurement from a state, potentially with noise.

        Args:
        - state: The state of the system to measure
        - key: The optional random key to generate noise
        """
        raise NotImplementedError
