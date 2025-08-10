"""
This file contains the definition for the composite invertible neural network, the normalizing flow.
This builds on the other invertible layers by chaining together compositions and summing the log-determinant-jacobians.
"""
import equinox as eqx


class NormalizingFlow(eqx.Module):
    pass

