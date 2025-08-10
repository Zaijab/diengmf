"""
This file contains the definition for the composite invertible neural network, the normalizing flow.
This builds on the other invertible layers by chaining together compositions and summing the log-determinant-jacobians.
"""

"""


class InvertibleNN(eqx.Module):
    coupling_layers: List[CouplingLayer]
    plu_layers: List[PLULinear]
    input_dim: int

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        input_dim: int = 2,
        num_coupling_layers: int = 6,
        hidden_dim: int = 64,
        num_hidden_layers: int = 4,
        *,
        key: Key[Array, "..."],
    ) -> None:
        self.input_dim = input_dim

        # Split keys for coupling layers and PLU layers
        coupling_keys = jax.random.split(key, num_coupling_layers)
        plu_keys = jax.random.split(key, num_coupling_layers - 1)

        # Initialize coupling layers
        self.coupling_layers = []
        for i in range(num_coupling_layers):
            swap = i % 2 == 1
            layer = CouplingLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                swap=swap,
                key=coupling_keys[i],
            )
            self.coupling_layers.append(layer)

        # Initialize PLU layers between coupling layers
        self.plu_layers = []
        for i in range(num_coupling_layers - 1):
            plu_layer = PLULinear(n=input_dim, key=plu_keys[i])
            self.plu_layers.append(plu_layer)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Array, "... d_in"]
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Forward transformation: map from data space to latent space.

        Args:
            x: Input tensor in data space

        Returns:
            Tuple of (transformed tensor in latent space, log determinant of Jacobian)
        """
        log_det_jacobian = 0

        # Apply first coupling layer
        x, ldj = self.coupling_layers[0].forward(x)
        log_det_jacobian += ldj

        # Apply alternating PLU and coupling layers
        for i in range(len(self.plu_layers)):
            # Apply PLU layer
            x, ldj = self.plu_layers[i].forward(x)
            log_det_jacobian += ldj

            # Apply next coupling layer
            x, ldj = self.coupling_layers[i + 1].forward(x)
            log_det_jacobian += ldj

        return x, log_det_jacobian

    @jaxtyped(typechecker=typechecker)
    def inverse(
        self, z: Float[Array, "... d_in"]
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Inverse transformation: map from latent space to data space.

        Args:
            z: Input tensor in latent space

        Returns:
            Tuple of (transformed tensor in data space, log determinant of Jacobian)
        """
        log_det_jacobian = 0

        # Apply layers in reverse order
        # Start with the last coupling layer
        z, ldj = self.coupling_layers[-1].inverse(z)
        log_det_jacobian += ldj

        # Apply alternating PLU and coupling layers in reverse
        for i in range(len(self.plu_layers) - 1, -1, -1):
            # Apply PLU layer in reverse
            z, ldj = self.plu_layers[i].inverse(z)
            log_det_jacobian += ldj

            # Apply coupling layer in reverse
            z, ldj = self.coupling_layers[i].inverse(z)
            log_det_jacobian += ldj

        return z, log_det_jacobian

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, x: Float[Array, "... d_in"], inverse: bool = False
    ) -> Tuple[Float[Array, "... d_in"], Float[Array, "..."]]:
        """
        Apply forward or inverse transformation.

        Args:
            x: Input tensor
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        return self.inverse(x) if inverse else self.forward(x)
"""


import equinox as eqx


class NormalizingFlow(eqx.Module):
    pass

