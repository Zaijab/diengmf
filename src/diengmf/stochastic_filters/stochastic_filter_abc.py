import abc

import equinox as eqx
from jaxtyping import Array, Float, Key
from diengmf.measurement_systems import AbstractMeasurementSystem
from diengmf.dynamical_systems import AbstractDynamicalSystem
from distreqx.distributions import AbstractDistribution


class AbstractFilter(eqx.Module, strict=True):
    """
    This is an abstract base class representing the functionality of a filter.
    Stochastic filters assume three parts:

    - Initialization: What is the initial guess of my state?
    - Prediction: Where do I expect the state to go next?
    - Update: Given an observation, how should I update my beliefs?
    """

    # State-Space Model
    measurement_system: AbstractMeasurementSystem
    dynamical_system: AbstractDynamicalSystem

    # Probability Discretization
    @abc.abstractmethod
    def initialize(
        self,
        key: Key[Array, "..."],
        belief: AbstractDistribution
    ) -> AbstractDistribution:
        raise NotImplementedError
    
    
    @abc.abstractmethod
    def predict(
        self,
        # key: Key[Array, "..."],
        posterior_distribution: AbstractDistribution,
        start_time, final_time,
    ) -> AbstractDistribution:
        raise NotImplementedError
    

    @abc.abstractmethod
    def update(
        self,
        key: Key[Array, "..."],
        prior_ensemble: Float[Array, "batch_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
    ) -> Float[Array, "batch_dim state_dim"]:
        """
        Given some noisy measurement and my current understanding of the state, how should I update my degrees of beliefs?
        """
        raise NotImplementedError
