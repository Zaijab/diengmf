import abc

import equinox as eqx
from jaxtyping import Array, Float, Key
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.dynamical_systems import AbstractDynamicalSystem
import distrax


class AbstractFilter(eqx.Module, strict=True):
    """
    This is an abstract base class representing the functionality of a filter.
    Stochastic filters assume three parts:

    - Initialization: What is the initial guess of my state?
    - Prediction: Where do I expect the state to go next?
    - Update: 
    """

    # measurement_system: AbstractMeasurementSystem
    # dynamical_system: AbstractDynamicalSystem
    
    # @abc.abstractmethod
    # def initialize(
    #     self,
    #     key: Key[Array, "..."],
    #     initial_belief: distrax.Distribution
    # ) -> distrax.Distribution:
    #     raise NotImplementedError
    
    
    # @abc.abstractmethod
    # def predict(
    #     self,
    #     key: Key[Array, "..."],
    #     posterior_distribution: distrax.Distribution,
    #     measurement: Float[Array, "*num_measurements measurement_dim"],
    # ) -> distrax.Distribution:
    #     raise NotImplementedError
    

    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: distrax.Distribution,
        measurement: Float[Array, "measurement_dim"],
    ) -> distrax.Distribution:
        """
        Given some noisy measurement and my current understanding of the state, how should I update my degrees of beliefs?
        """
        raise NotImplementedError
