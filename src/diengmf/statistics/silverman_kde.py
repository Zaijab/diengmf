import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped

# CR3BP Earth-Moon mass parameter
CR3BP_MU = 0.012150584269940

def stable_log_mills_factor(t):
    def large_t_case():
        return jnp.log(jnp.abs(t)) + t * (jnp.sign(t) * jnp.abs(t) - 1.0)
    
    def normal_case():  
        phi_t = jax.scipy.stats.norm.pdf(t)
        big_phi_t = jax.scipy.stats.norm.cdf(t)
        ratio = big_phi_t / phi_t
        return jnp.log(ratio + t * (1.0 + t * ratio))
    
    return jax.lax.cond(jnp.abs(t) > 8.0, large_t_case, normal_case)


@jaxtyped(typechecker=typechecker)
class GMM(eqx.Module):
    means: Float[Array, "num_components state_dim"]
    covs: Float[Array, "num_components state_dim state_dim"]
    weights: Float[Array, "num_components"]
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, means, covs, weights, max_components=1):
        max_components = max(means.shape[0], max_components)
        pad_width = max_components - means.shape[0]
        self.means = jnp.pad(means, ((0, pad_width), (0, 0)))
        self.covs = jnp.pad(covs, ((0, pad_width), (0, 0), (0, 0)))
        self.weights = jnp.pad(weights, (0, pad_width))
    
    @jaxtyped(typechecker=typechecker)
    def pdf(self, x: Float[Array, "state_dim"]) -> Float[Array, ""]:
        """Compute probability density at point x."""
        
        def component_pdf(mean, cov, weight):
            L = jnp.linalg.cholesky(cov)
            
            diff = x - mean
            y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
            
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quad_form = jnp.sum(y**2)
            
            k = mean.shape[0]
            log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + log_det + quad_form)
            
            return weight * jnp.exp(log_prob)
        
        component_probs = eqx.filter_vmap(component_pdf)(
            self.means, self.covs, self.weights
        )
        
        return jnp.sum(component_probs)
    
    @jaxtyped(typechecker=typechecker) 
    def log_pdf(self, x: Float[Array, "state_dim"]) -> Float[Array, ""]:
        """Compute log probability density."""
        
        def component_log_pdf(mean, cov, log_weight):
            L = jnp.linalg.cholesky(cov)
            
            diff = x - mean
            y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
            
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quad_form = jnp.sum(y**2)
            
            k = mean.shape[0]
            log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + log_det + quad_form)
            
            return log_weight + log_prob
        
        log_weights = jnp.log(self.weights)
        log_component_probs = eqx.filter_vmap(component_log_pdf)(
            self.means, self.covs, log_weights
        )
        
        return jax.scipy.special.logsumexp(log_component_probs)

    @jaxtyped(typechecker=typechecker)
    def marginalize_to_position(self) -> "GMM":
        """Marginalize 6D GMM (x,y,z,vx,vy,vz) to 3D position (x,y,z)."""
        position_means = self.means[:, :3]
        position_covs = self.covs[:, :3, :3]
        return GMM(position_means, position_covs, self.weights)

    @jaxtyped(typechecker=typechecker)
    def spherical_angles_to_unit_vector(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, "3"]:
        """Convert (azimuth, inclination) to unit vector v ∈ S²."""
        azimuth, inclination = angles[0], angles[1]

        unit_vector = jnp.array([
            jnp.cos(azimuth) * jnp.sin(inclination),  # x = cos(θ₁)sin(θ₂)
            jnp.sin(azimuth) * jnp.sin(inclination),  # y = sin(θ₁)sin(θ₂) 
            jnp.cos(inclination)                      # z = cos(θ₂)
        ])
        # assert jnp.allclose(jnp.linalg.norm(unit_vector), 1.0)
        return unit_vector

    @jaxtyped(typechecker=typechecker)
    def positional_logpdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal logpdf at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_logpdf(idx):
            return self.positional_component_logpdf(idx, angles)

        component_logpdfs = eqx.filter_vmap(single_component_logpdf)(component_indices)
        return jax.scipy.special.logsumexp(component_logpdfs)


    @jaxtyped(typechecker=typechecker)
    def positional_component_pdf(
        self, component_idx: int | Int[Array, ""], angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Single component positional normal PDF per Wikipedia formula."""
        angles = jnp.deg2rad(angles)
        # Convert spherical angles (azimuth, inclination) to unit vector u ∈ S²
        unit_vector = self.spherical_angles_to_unit_vector(angles)

        # Extract parameters for this mixture component
        mean = self.means[component_idx, :3]           # μ ∈ ℝ³ (mean vector)
        cov = self.covs[component_idx, :3, :3]         # Σ ∈ ℝ³ˣ³ (covariance matrix)  
        weight = self.weights[component_idx]           # mixture weight
        mean = mean + jnp.array([0.012150584269940, 0.0, 0.0]) # Shifting means to be Earth centered
        
        # Efficient computation via Cholesky decomposition: Σ = LLᵀ
        L = jnp.linalg.cholesky(cov)                   # L: lower triangular Cholesky factor
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)        # Σ⁻¹μ
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)  # Σ⁻¹u

        # Compute t-statistic: t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)
        numerator = jnp.dot(mean, sigma_inv_v)         # μᵀΣ⁻¹u
        denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))  # √(uᵀΣ⁻¹u)
        t_statistic = numerator / denominator          # t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)

        # Standard normal PDF and CDF evaluated at t
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)  # φ(t) = (1/√2π)exp(-t²/2)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)  # Φ(t) = ∫_{-∞}^t φ(s)ds

        # Compute normalization constant: 1/((2π)^(3/2)|Σ|^(1/2))
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ| = 2∑log(Lᵢᵢ)
        normalization = jnp.exp(-0.5 * (3.0 * jnp.log(2.0 * jnp.pi) + log_det_sigma))

        # Wikipedia quadratic form: exp(-½μᵀΣ⁻¹μ)
        quadratic_form = jnp.dot(mean, sigma_inv_mu)   # μᵀΣ⁻¹μ
        mean_correction = jnp.exp(-0.5 * quadratic_form)

        # Wikipedia projected normal factor: φ(t)[Φ(t) + tφ(t)]
        projected_factor = phi_t * (big_phi_t + t_statistic * phi_t)

        # Complete Wikipedia formula: weight × normalization × mean_correction × projected_factor
        result = weight * normalization * mean_correction * projected_factor

        return result

    @jaxtyped(typechecker=typechecker)
    def positional_pdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal PDF at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_pdf(idx):
            return self.positional_component_pdf(idx, angles)

        component_pdfs = eqx.filter_vmap(single_component_pdf)(component_indices)
        return jnp.sum(component_pdfs)

@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) #* (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    return GMM(means, covs, weights)

# # Usage:
# # my_dist is a GMM with 10 components over 6D space.
# # We can then evaluate this GMM over a single point in 6D.
# my_dist = silverman_kde_estimate(jax.random.normal(jax.random.key(0), (10,6)))
# # my_dist.pdf(jnp.arange(6).astype(float))
# my_dist.positional_component_pdf(0, jnp.array([jnp.deg2rad(5.0), jnp.deg2rad(5.0)]))
# my_dist.positional_pdf(jnp.array([jnp.deg2rad(5.0), jnp.deg2rad(50.0)]))

###

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped


def stable_log_mills_ratio(t: Float[Array, ""]) -> Float[Array, ""]:
    # Computes log(Φ(t)/φ(t)) in a numerically stable way
    log_phi = -0.5 * t**2 - 0.5 * jnp.log(2 * jnp.pi)
    log_big_phi = jax.scipy.special.log_ndtr(t)
    return log_big_phi - log_phi

@jaxtyped(typechecker=typechecker)
class GMM(eqx.Module):
    means: Float[Array, "num_components state_dim"]
    covs: Float[Array, "num_components state_dim state_dim"]
    weights: Float[Array, "num_components"]
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, means, covs, weights, max_components=1):
        max_components = max(means.shape[0], max_components)
        pad_width = max_components - means.shape[0]
        self.means = jnp.pad(means, ((0, pad_width), (0, 0)))
        self.covs = jnp.pad(covs, ((0, pad_width), (0, 0), (0, 0)))
        self.weights = jnp.pad(weights, (0, pad_width))
    
    @jaxtyped(typechecker=typechecker)
    def pdf(self, x: Float[Array, "state_dim"]) -> Float[Array, ""]:
        """Compute probability density at point x."""
        
        def component_pdf(mean, cov, weight):
            L = jnp.linalg.cholesky(cov)
            
            diff = x - mean
            y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
            
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quad_form = jnp.sum(y**2)
            
            k = mean.shape[0]
            log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + log_det + quad_form)
            
            return weight * jnp.exp(log_prob)
        
        component_probs = eqx.filter_vmap(component_pdf)(
            self.means, self.covs, self.weights
        )
        
        return jnp.sum(component_probs)
    
    @jaxtyped(typechecker=typechecker) 
    def log_pdf(self, x: Float[Array, "state_dim"]) -> Float[Array, ""]:
        """Compute log probability density."""
        
        def component_log_pdf(mean, cov, log_weight):
            L = jnp.linalg.cholesky(cov)
            
            diff = x - mean
            y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
            
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quad_form = jnp.sum(y**2)
            
            k = mean.shape[0]
            log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + log_det + quad_form)
            
            return log_weight + log_prob
        
        log_weights = jnp.log(self.weights)
        log_component_probs = eqx.filter_vmap(component_log_pdf)(
            self.means, self.covs, log_weights
        )
        
        return jax.scipy.special.logsumexp(log_component_probs)

    @jaxtyped(typechecker=typechecker)
    def marginalize_to_position(self) -> "GMM":
        """Marginalize 6D GMM (x,y,z,vx,vy,vz) to 3D position (x,y,z)."""
        position_means = self.means[:, :3]
        position_covs = self.covs[:, :3, :3]
        return GMM(position_means, position_covs, self.weights)

    @jaxtyped(typechecker=typechecker)
    def spherical_angles_to_unit_vector(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, "3"]:
        """Convert (azimuth, inclination) to unit vector v ∈ S²."""
        azimuth, inclination = angles[0], angles[1]

        unit_vector = jnp.array([
            jnp.cos(azimuth) * jnp.sin(inclination),  # x = cos(θ₁)sin(θ₂)
            jnp.sin(azimuth) * jnp.sin(inclination),  # y = sin(θ₁)sin(θ₂) 
            jnp.cos(inclination)                      # z = cos(θ₂)
        ])
        # assert jnp.allclose(jnp.linalg.norm(unit_vector), 1.0)
        return unit_vector

    ###
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def positional_component_logpdf(
        self, component_idx,
        point: Float[Array, "2"],
    ) -> Float[Array, ""]:
        """CORRECTED implementation matching Wikipedia formula exactly."""
        point = jnp.deg2rad(point)
        mean = self.means[component_idx, :3]
        # mean = mean + jnp.array([0.012150584269940, 0.0, 0.0]) # Shifting means to be Earth centered
        cov = self.covs[component_idx, :3, :3]
        ###
        eigval, eigvec = jnp.linalg.eigh(cov)
        min_eigval = 1e-4
        eigval = jnp.clip(eigval, min_eigval, None)  # e.g., min_eigval = 1e-5
        cov = (eigvec * eigval) @ eigvec.T
        ###
        
        unit_vector = self.spherical_angles_to_unit_vector(point)

        L = jnp.linalg.cholesky(cov)
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)

        numerator = jnp.dot(mean, sigma_inv_v)
        denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))
        t_statistic = numerator / denominator

        # Wikipedia factor: Φ(T)/φ(T) + T(1 + TΦ(T)/φ(T))
        # NOT [Φ(T)/φ(T) + T][1 + TΦ(T)/φ(T)]
        
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)
        ratio_term = big_phi_t / phi_t  # Φ(T)/φ(T)
        wikipedia_factor = ratio_term + t_statistic * (1.0 + t_statistic * ratio_term)
        log_wikipedia_factor = jnp.log(wikipedia_factor)

        ###
        # log_ratio_term = stable_log_mills_ratio(t_statistic)
        # log_wikipedia_factor = jnp.log1p(t_statistic * (1.0 + t_statistic * jnp.exp(log_ratio_term))) + log_ratio_term
        ###


        # CORRECTED normalization: Wikipedia formula exactly
        # p = (e^(-½μᵀΣ⁻¹μ)) / (√|Σ| × (2πγᵀΣ⁻¹γ)^(3/2)) × wikipedia_factor

        quadratic_form = jnp.dot(mean, sigma_inv_mu)  # μᵀΣ⁻¹μ
        # quadratic_form = jnp.sum(jax.scipy.linalg.solve_triangular(L, mean, lower=True)**2)

        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ|
        gamma_term = jnp.dot(unit_vector, sigma_inv_v)  # γᵀΣ⁻¹γ

        # Log normalization: -½μᵀΣ⁻¹μ - ½log|Σ| - (3/2)log(2πγᵀΣ⁻¹γ)
        log_normalization = (
            -0.5 * quadratic_form
            - 0.5 * log_det_sigma  
            - 1.5 * jnp.log(2.0 * jnp.pi * gamma_term)
        )

        return log_normalization + log_wikipedia_factor


    ###

    @jaxtyped(typechecker=typechecker)
    def positional_logpdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal logpdf at (azimuth, inclination) angles. IN DEGREES"""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_logpdf(idx):
            return self.positional_component_logpdf(idx, angles)

        component_logpdfs = eqx.filter_vmap(single_component_logpdf)(component_indices)
        return jax.scipy.special.logsumexp(component_logpdfs)


        # @jaxtyped(typechecker=typechecker)
    # @eqx.filter_jit
    # def positional_component_logpdf(
    #     self,
    #     component_idx,
    #     point: Float[Array, "2"],
    #     # means: Float[Array, "num_components 3"],
    #     # covs: Float[Array, "num_components 3 3"], 
    #     # weights: Float[Array, "num_components"],
    #     mu: float = CR3BP_MU
    # ) -> Float[Array, ""]:
    #     """
    #     CR3BP-aware projected normal logpdf.

    #     Args:
    #         component_idx: Which mixture component
    #         point: [azimuth, elevation] in radians, measured from Earth
    #         means: Satellite positions in barycentric coordinates  
    #         covs: Covariance matrices in barycentric coordinates
    #         weights: Mixture weights
    #         mu: CR3BP mass parameter (Earth-Moon system)
    #     """

    #     # Extract component parameters (barycentric coordinates)
    #     mean_barycentric = self.means[component_idx, :3]
    #     cov_barycentric = self.covs[component_idx, :3, :3]  
    #     weight = self.weights[component_idx]

    #     # Transform to Earth-centered coordinates
    #     # Earth is at (-μ, 0, 0) in barycentric frame
    #     earth_position = jnp.array([-mu, 0.0, 0.0])
    #     mean_earth_centered = mean_barycentric - earth_position
    #     # Covariance matrix doesn't change under translation
    #     cov_earth_centered = cov_barycentric

    #     # Convert spherical angles to unit vector (from Earth perspective)
    #     def spherical_to_cartesian_from_earth(angles):
    #         azimuth, elevation = angles[0], angles[1]
    #         return jnp.array([
    #             jnp.cos(elevation) * jnp.cos(azimuth),
    #             jnp.cos(elevation) * jnp.sin(azimuth), 
    #             jnp.sin(elevation)
    #         ])

    #     unit_vector = spherical_to_cartesian_from_earth(point)

    #     # Now compute projected normal with Earth-centered parameters
    #     L = jnp.linalg.cholesky(cov_earth_centered)
    #     sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean_earth_centered)
    #     sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)

    #     numerator = jnp.dot(mean_earth_centered, sigma_inv_v)
    #     denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))
    #     t_statistic = numerator / denominator

    #     # Wikipedia factor with numerical stability
    #     phi_t = jax.scipy.stats.norm.pdf(t_statistic)
    #     big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)

    #     # Safe ratio computation for extreme values
    #     def safe_ratio(t, phi, big_phi):
    #         normal_ratio = jnp.where(phi > 1e-50, big_phi / phi, 0.0)
    #         mills_approx = t + 1.0 / jnp.maximum(jnp.abs(t), 1.0)

    #         return jnp.where(
    #             t > 8.0, mills_approx,
    #             jnp.where(t < -8.0, 0.0, normal_ratio)
    #         )

    #     ratio_term = safe_ratio(t_statistic, phi_t, big_phi_t)
    #     wikipedia_factor = ratio_term + t_statistic * (1.0 + t_statistic * ratio_term)
    #     wikipedia_factor = jnp.maximum(wikipedia_factor, 1e-50)  # Ensure positive

    #     # Normalization components
    #     quadratic_form = jnp.dot(mean_earth_centered, sigma_inv_mu)
    #     log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    #     gamma_term = jnp.dot(unit_vector, sigma_inv_v)

    #     # Log normalization: -½μᵀΣ⁻¹μ - ½log|Σ| - (3/2)log(2πγᵀΣ⁻¹γ)
    #     log_normalization = (
    #         -0.5 * quadratic_form
    #         - 0.5 * log_det_sigma  
    #         - 1.5 * jnp.log(2.0 * jnp.pi * gamma_term)
    #     )

    #     log_wikipedia_factor = jnp.log(wikipedia_factor)
    #     log_weight = jnp.log(jnp.maximum(weight, 1e-50))

    #     return log_weight + log_normalization + log_wikipedia_factor

    ###
    
    @jaxtyped(typechecker=typechecker)
    def positional_component_pdf(
        self, component_idx: int | Int[Array, ""], angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Single component positional normal PDF per Wikipedia formula."""
        angles = jnp.deg2rad(angles)
        # Convert spherical angles (azimuth, inclination) to unit vector u ∈ S²
        unit_vector = self.spherical_angles_to_unit_vector(angles)

        # Extract parameters for this mixture component
        mean = self.means[component_idx, :3]           # μ ∈ ℝ³ (mean vector)
        cov = self.covs[component_idx, :3, :3]         # Σ ∈ ℝ³ˣ³ (covariance matrix)  
        weight = self.weights[component_idx]           # mixture weight
        mean = mean + jnp.array([0.012150584269940, 0.0, 0.0]) # Shifting means to be Earth centered
        
        # Efficient computation via Cholesky decomposition: Σ = LLᵀ
        L = jnp.linalg.cholesky(cov)                   # L: lower triangular Cholesky factor
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)        # Σ⁻¹μ
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)  # Σ⁻¹u

        # Compute t-statistic: t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)
        numerator = jnp.dot(mean, sigma_inv_v)         # μᵀΣ⁻¹u
        denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))  # √(uᵀΣ⁻¹u)
        t_statistic = numerator / denominator          # t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)

        # Standard normal PDF and CDF evaluated at t
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)  # φ(t) = (1/√2π)exp(-t²/2)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)  # Φ(t) = ∫_{-∞}^t φ(s)ds

        # Compute normalization constant: 1/((2π)^(3/2)|Σ|^(1/2))
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ| = 2∑log(Lᵢᵢ)
        normalization = jnp.exp(-0.5 * (3.0 * jnp.log(2.0 * jnp.pi) + log_det_sigma))

        # Wikipedia quadratic form: exp(-½μᵀΣ⁻¹μ)
        quadratic_form = jnp.dot(mean, sigma_inv_mu)   # μᵀΣ⁻¹μ
        mean_correction = jnp.exp(-0.5 * quadratic_form)

        # Wikipedia projected normal factor: φ(t)[Φ(t) + tφ(t)]
        projected_factor = phi_t * (big_phi_t + t_statistic * phi_t)

        # Complete Wikipedia formula: weight × normalization × mean_correction × projected_factor
        result = weight * normalization * mean_correction * projected_factor

        return result

    @jaxtyped(typechecker=typechecker)
    def positional_pdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal PDF at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_pdf(idx):
            return self.positional_component_pdf(idx, angles)

        component_pdfs = eqx.filter_vmap(single_component_pdf)(component_indices)
        return jnp.sum(component_pdfs)

###

@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) * (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    return GMM(means, covs, weights)

###

from diengmf.dynamical_systems import CR3BP
from diengmf.measurement_systems import AnglesOnly
dynamical_system = CR3BP()
angle_measurement = AnglesOnly()
key = jax.random.key(10)
posterior_ensemble = dynamical_system.generate(key)
true_state = dynamical_system.initial_state()
gmm = silverman_kde_estimate(posterior_ensemble)
true_angles = jnp.rad2deg(angle_measurement(true_state))
true_angles_approx = jnp.rad2deg(angle_measurement(true_state, key))

gmm.positional_logpdf(true_angles), gmm.positional_logpdf(true_angles_approx)
