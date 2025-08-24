from typing import List, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped


class PLULinear(eqx.Module):
    """
    Invertible linear layer using PLU decomposition.
    Represents a linear transformation y = Ax + b where A = PLU.
    """

    input_dim: int
    P: jax.Array  # Permutation indices
    L_params: jax.Array  # Strictly lower triangular elements
    U_diag: jax.Array  # Diagonal elements of U
    U_upper: jax.Array  # Strictly upper triangular elements
    bias: Optional[jax.Array]
    use_bias: bool
    initialization_scale: float

    @jaxtyped(typechecker=typechecker)
    def __init__(
        self,
        input_dim: int,
        use_bias: bool = True,
        initialization_scale: float = 0.2,
        permutation_type="random",
        *,
        key: Key[Array, "..."],
    ) -> None:
        self.initialization_scale = initialization_scale
        p_key, l_key, u_diag_key, u_upper_key, bias_key = jax.random.split(key, 5)

        if permutation_type == "identity":
            self.P = jnp.arange(input_dim)
        elif permutation_type == "random":
            self.P = jax.random.permutation(p_key, input_dim)
        elif permutation_type == "reverse":
            self.P = jnp.arange(input_dim)[::-1]
        else:
            raise ValueError(f"Unknown permutation_type: {permutation_type}")

        # L parameters: Glorot uniform initialization for lower triangular part
        l_init = jax.nn.initializers.glorot_uniform()
        l_full = l_init(l_key, (input_dim, input_dim))
        # Extract only the strictly lower triangular part
        self.L_params = l_full[jnp.tril_indices(input_dim, -1)]

        # U diagonal: initialize for softplus(U_diag) ≈ 1.0
        target_value = jnp.log(jnp.exp(1.0) - 1.0)  # ≈ 0.313
        u_diag_base_init = jax.nn.initializers.constant(target_value)
        u_diag_noise_init = jax.nn.initializers.normal(stddev=initialization_scale)
        self.U_diag = u_diag_base_init(u_diag_key, (input_dim,)) + u_diag_noise_init(
            u_diag_key, (input_dim,)
        )

        # U upper: Glorot uniform initialization for upper triangular part
        u_upper_init = jax.nn.initializers.glorot_uniform()
        u_full = u_upper_init(u_upper_key, (input_dim, input_dim))
        # Extract only the strictly upper triangular part
        self.U_upper = u_full[jnp.triu_indices(input_dim, 1)]

        # Bias: standard initialization
        if use_bias:
            bias_init = jax.nn.initializers.normal(stddev=initialization_scale)
            self.bias = bias_init(bias_key, (input_dim,))
        else:
            self.bias = None

        self.use_bias = use_bias
        self.input_dim = input_dim

    def _construct_L(self) -> jax.Array:
        """Construct lower triangular matrix L with ones on diagonal."""
        L = jnp.eye(self.input_dim)
        indices = jnp.tril_indices(
            self.input_dim, -1
        )  # Strictly lower triangular indices
        L = L.at[indices].set(self.L_params)
        return L

    def _construct_U(self) -> jax.Array:
        """Construct upper triangular matrix U."""
        # Apply softplus to ensure diagonal elements are positive
        u_diag = jax.nn.softplus(self.U_diag)  # + 1e-5

        # Create diagonal matrix
        U = jnp.diag(u_diag)

        # Fill in strictly upper triangular part
        indices = jnp.triu_indices(
            self.input_dim, 1
        )  # Strictly upper triangular indices
        U = U.at[indices].set(self.U_upper)

        return U

    def _construct_P_matrix(self) -> jax.Array:
        """Construct permutation matrix from permutation indices."""
        P = jnp.zeros((self.input_dim, self.input_dim))
        P = P.at[jnp.arange(self.input_dim), self.P].set(1.0)
        return P

    def _log_det(self) -> Float[Array, "..."]:
        """Compute log determinant of the transformation matrix."""
        # Log det of a permutation matrix is 0 if even permutation, log(-1) if odd
        # But we'll ignore this for now since we're not updating P during training
        # log_det_P = jnp.log(jnp.linalg.det(self._construct_P_matrix()))

        # Log det of L is 0 since diagonal elements are 1
        # log_det_L = 0.0

        # Log det of U is sum of log of diagonal elements
        u_diag = jax.nn.softplus(self.U_diag)  # + 1e-5
        log_det_U = jnp.sum(jnp.log(u_diag))

        # Total log det: P contributes sign only
        # parity = jnp.array(0.0, jnp.float32)  # Even permutation initially

        return log_det_U

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, x: Float[Array, "... n"]
    ) -> tuple[Float[Array, "... n"], Float[Array, "..."]]:
        """
        Forward transformation: x → PLUx + b

        Args:
            x: Input tensor with last dimension n

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        # Construct matrices
        L = self._construct_L()
        U = self._construct_U()
        P_matrix = self._construct_P_matrix()

        # Apply PLU transformation
        y = x @ (P_matrix @ L @ U).T

        # Add bias if specified
        if self.use_bias and self.bias is not None:
            y = y + self.bias

        # Compute log determinant of Jacobian
        log_det = self._log_det()

        return y, log_det

    def inverse(
        self, y: Float[Array, "... n"]
    ) -> tuple[Float[Array, "... n"], Float[Array, "..."]]:
        """
        Inverse transformation: y → U⁻¹L⁻¹P⁻¹(y - b)
        """
        # Remove bias if specified
        if self.use_bias and self.bias is not None:
            y = y - self.bias

        # Construct matrices
        L = self._construct_L()
        U = self._construct_U()
        P_matrix = self._construct_P_matrix()

        # Step 1: Apply P inverse (P^T)^(-1) = P
        y_p = y @ P_matrix

        # Step 2: Solve y_p = w @ L^T for w
        # Since L is lower triangular, L^T is upper triangular
        # We solve L @ w^T = y_p^T with lower=True
        w_t = jax.scipy.linalg.solve_triangular(L, y_p.T, lower=True)

        # Step 3: Solve w = x @ U^T for x
        # Since U is upper triangular, U^T is lower triangular
        # We solve U @ x^T = w^T with lower=False
        x_t = jax.scipy.linalg.solve_triangular(U, w_t, lower=False)

        # Transpose to get x
        x = x_t.T

        # Compute negative log determinant
        log_det = -self._log_det()

        return x, log_det

    def __call__(self, x, inverse=False):
        """
        Apply forward or inverse transformation.

        Args:
            x: Input tensor
            inverse: Whether to apply inverse transformation

        Returns:
            Tuple of (transformed tensor, log determinant of Jacobian)
        """
        return self.inverse(x) if inverse else self.forward(x)
