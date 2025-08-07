"""
Normalizing Flow Hyperparameter Study Framework
================================================

Organized by importance and computational efficiency for Optuna optimization.
"""

from dataclasses import dataclass
from typing import Literal, List, Optional
import jax.numpy as jnp


@dataclass
class FlowHyperparameters:
    """Complete hyperparameter configuration for normalizing flows."""
    
    # ========== TIER 1: Critical Architecture Choices ==========
    # These have the largest impact on model expressivity and performance
    
    # Core Architecture
    num_flow_layers: int = 6  # [2, 4, 6, 8, 12, 16, 24]
    # Deeper flows can model more complex distributions but harder to train
    
    transform_type: Literal["rqs", "affine", "mixed"] = "rqs"  
    # RQS is most expressive, affine is faster, mixed alternates
    
    layer_pattern: Literal["coupling_only", "sandwich", "interleaved", "multiscale"] = "sandwich"
    # coupling_only: Only coupling layers
    # sandwich: PLU -> Couplings -> PLU  
    # interleaved: Coupling -> PLU -> Coupling -> PLU
    # multiscale: Hierarchical with squeeze operations (for images)
    
    # ========== TIER 2: Transform-Specific Parameters ==========
    # Critical for RQS performance
    
    # RQS Parameters
    num_bins: int = 8  # [4, 8, 16, 32]
    # More bins = more flexible, but diminishing returns after 8-16
    
    spline_range: tuple[float, float] = (-5.0, 5.0)  # [(-3,3), (-5,5), (-10,10)]
    # Should cover expected data range
    
    # Conditioner Network
    conditioner_hidden_dim: int = 128  # [32, 64, 128, 256]
    conditioner_depth: int = 3  # [2, 3, 4, 5]
    conditioner_activation: Literal["relu", "gelu", "swish", "elu"] = "relu"
    
    # ========== TIER 3: Masking and Permutation Strategy ==========
    # Controls information flow between dimensions
    
    mask_type_sequence: List[Literal["half", "alternating", "random", "learned"]] = None
    # Can specify custom sequence like ["half", "alternating", "half", "alternating"]
    
    permutation_type: Literal["none", "reverse", "random", "learned", "lu"] = "reverse"
    # How to permute between coupling layers
    
    use_actnorm: bool = True  # Activation normalization for stable training
    
    # ========== TIER 4: Linear Transform Parameters ==========
    
    num_plu_layers: int = 2  # [0, 1, 2, 3]
    plu_share_weights: bool = False  # Share PLU weights across layers
    
    # ========== TIER 5: Regularization and Training Stability ==========
    
    dropout_rate: float = 0.0  # [0.0, 0.1, 0.2]
    weight_decay: float = 1e-5  # [0, 1e-6, 1e-5, 1e-4]
    
    gradient_clip_norm: Optional[float] = 1.0  # [None, 0.5, 1.0, 5.0]
    
    # Initialization
    weight_init_scale: float = 0.01  # [0.001, 0.01, 0.1]
    zero_init_last_layer: bool = True  # Initialize last layer to zero for stability
    
    # ========== TIER 6: Training Dynamics ==========
    
    learning_rate: float = 1e-3  # [1e-4, 3e-4, 1e-3, 3e-3]
    lr_schedule: Literal["constant", "cosine", "exponential", "warmup_cosine"] = "cosine"
    warmup_steps: int = 1000  # For warmup schedules
    
    batch_size: int = 256  # [32, 64, 128, 256, 512]
    
    # ========== TIER 7: Advanced Techniques ==========
    
    use_residual_blocks: bool = False  # ResNet-style conditioners
    use_self_attention: bool = False  # For high-dim data
    
    variational_dequantization: bool = False  # For discrete data
    
    # Flow-specific tricks
    safe_transforms: bool = True  # Clamp values to prevent numerical issues
    sum_log_det: bool = True  # Sum vs mean for log determinant
    
    beta_schedule: Optional[str] = None  # For annealed importance sampling


def get_search_space_for_optuna():
    """Returns search space configuration for Optuna optimization."""
    
    return {
        # Most important (always search)
        "num_flow_layers": {
            "type": "int",
            "low": 2,
            "high": 16,
            "step": 2,
            "importance": "critical"
        },
        "transform_type": {
            "type": "categorical",
            "choices": ["rqs", "mixed"],  # Skip pure affine
            "importance": "critical"
        },
        "num_bins": {
            "type": "categorical", 
            "choices": [4, 8, 16],
            "importance": "high",
            "condition": "transform_type in ['rqs', 'mixed']"
        },
        
        # Important architecture choices
        "conditioner_hidden_dim": {
            "type": "categorical",
            "choices": [64, 128, 256],
            "importance": "high"
        },
        "conditioner_depth": {
            "type": "int",
            "low": 2,
            "high": 4,
            "importance": "medium"
        },
        "layer_pattern": {
            "type": "categorical",
            "choices": ["coupling_only", "sandwich", "interleaved"],
            "importance": "high"
        },
        
        # Permutation strategy
        "permutation_type": {
            "type": "categorical",
            "choices": ["reverse", "random", "lu"],
            "importance": "medium"
        },
        
        # Regularization
        "dropout_rate": {
            "type": "float",
            "low": 0.0,
            "high": 0.3,
            "importance": "low"
        },
        "weight_decay": {
            "type": "loguniform",
            "low": 1e-6,
            "high": 1e-3,
            "importance": "medium"
        },
        
        # Training
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-4,
            "high": 1e-2,
            "importance": "high"
        },
        "batch_size": {
            "type": "categorical",
            "choices": [64, 128, 256],
            "importance": "medium"
        }
    }


def get_progressive_search_strategy():
    """Returns a progressive search strategy for efficient exploration."""
    
    return [
        # Stage 1: Find good base architecture (fast)
        {
            "name": "architecture_search",
            "budget": 50,  # trials
            "params": ["num_flow_layers", "transform_type", "layer_pattern"],
            "fixed": {"num_bins": 8, "conditioner_hidden_dim": 128}
        },
        
        # Stage 2: Optimize transform parameters
        {
            "name": "transform_tuning", 
            "budget": 100,
            "params": ["num_bins", "conditioner_hidden_dim", "conditioner_depth"],
            "use_best_from": "architecture_search"
        },
        
        # Stage 3: Fine-tune training dynamics
        {
            "name": "training_optimization",
            "budget": 50,
            "params": ["learning_rate", "batch_size", "weight_decay"],
            "use_best_from": "transform_tuning"
        },
        
        # Stage 4: Final comprehensive search
        {
            "name": "final_search",
            "budget": 200,
            "params": "all",
            "init_with_best": True  # Start from best configs found
        }
    ]


def get_recommended_configs_by_problem():
    """Recommended starting configurations for different problem types."""
    
    return {
        "2d_toy": {
            # For 2D visualization (Ikeda, moons, spirals)
            "num_flow_layers": 4,
            "transform_type": "rqs",
            "num_bins": 8,
            "conditioner_hidden_dim": 64,
            "conditioner_depth": 2,
            "layer_pattern": "coupling_only"
        },
        
        "low_dim_dynamics": {
            # For Lorenz63 (3D), Lorenz96 (40D)
            "num_flow_layers": 6,
            "transform_type": "rqs",
            "num_bins": 8,
            "conditioner_hidden_dim": 128,
            "conditioner_depth": 3,
            "layer_pattern": "sandwich",
            "num_plu_layers": 2
        },
        
        "high_dim_tabular": {
            # For tabular data with 100+ dimensions
            "num_flow_layers": 10,
            "transform_type": "mixed",  # Alternate RQS and affine for efficiency
            "num_bins": 16,
            "conditioner_hidden_dim": 256,
            "conditioner_depth": 3,
            "layer_pattern": "interleaved",
            "use_residual_blocks": True
        },
        
        "images": {
            # For image modeling
            "num_flow_layers": 24,
            "transform_type": "mixed",
            "num_bins": 8,
            "conditioner_hidden_dim": 512,
            "conditioner_depth": 4,
            "layer_pattern": "multiscale",
            "use_actnorm": True,
            "variational_dequantization": True
        }
    }


def estimate_model_complexity(config: FlowHyperparameters, input_dim: int):
    """Estimates parameter count and FLOPs for a configuration."""
    
    # Approximate parameter count
    params_per_coupling = 0
    
    if config.transform_type in ["rqs", "mixed"]:
        # Conditioner network parameters
        conditioner_params = (
            input_dim // 2 * config.conditioner_hidden_dim +  # First layer
            config.conditioner_hidden_dim ** 2 * (config.conditioner_depth - 2) +  # Hidden layers
            config.conditioner_hidden_dim * (input_dim // 2) * (3 * config.num_bins + 1)  # Output layer
        )
        params_per_coupling += conditioner_params
    
    if config.transform_type == "affine":
        # Simpler affine coupling
        conditioner_params = (
            input_dim // 2 * config.conditioner_hidden_dim +
            config.conditioner_hidden_dim ** 2 * (config.conditioner_depth - 2) +
            config.conditioner_hidden_dim * input_dim  # Just shift and scale
        )
        params_per_coupling += conditioner_params
    
    # PLU parameters
    params_per_plu = input_dim ** 2  # LU decomposition
    
    # Total parameters
    total_params = (
        config.num_flow_layers * params_per_coupling +
        config.num_plu_layers * params_per_plu
    )
    
    # Approximate FLOPs per forward pass
    flops_per_sample = total_params * 2  # Very rough estimate
    
    return {
        "total_parameters": int(total_params),
        "parameters_per_layer": int(params_per_coupling),
        "flops_per_sample": int(flops_per_sample),
        "memory_mb": total_params * 4 / 1024 / 1024  # float32
    }


# ========== Experimental Design for Your Study ==========

def get_ablation_studies():
    """Key ablation studies to understand component importance."""
    
    return {
        "transform_comparison": {
            "description": "Compare RQS vs Affine coupling",
            "vary": ["transform_type"],
            "fixed": {"num_flow_layers": 6, "conditioner_hidden_dim": 128},
            "metrics": ["final_loss", "training_speed", "expressivity"]
        },
        
        "depth_study": {
            "description": "Effect of model depth",
            "vary": ["num_flow_layers"],
            "values": [2, 4, 6, 8, 12, 16],
            "metrics": ["convergence_speed", "final_performance", "gradient_stability"]
        },
        
        "spline_resolution": {
            "description": "Number of bins in RQS",
            "vary": ["num_bins"],
            "values": [4, 8, 16, 32],
            "condition": "transform_type == 'rqs'",
            "metrics": ["expressivity", "computational_cost"]
        },
        
        "permutation_importance": {
            "description": "Effect of permutation strategies",
            "vary": ["permutation_type"],
            "values": ["none", "reverse", "random", "lu"],
            "metrics": ["mixing_efficiency", "final_loss"]
        },
        
        "conditioner_capacity": {
            "description": "Conditioner network size impact",
            "vary": ["conditioner_hidden_dim", "conditioner_depth"],
            "grid": [(64, 2), (128, 2), (128, 3), (256, 3), (256, 4)],
            "metrics": ["expressivity", "overfitting", "training_time"]
        }
    }


if __name__ == "__main__":
    # Example: Print recommended config for your use case
    config = get_recommended_configs_by_problem()["low_dim_dynamics"]
    print("Recommended config for Lorenz systems:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Estimate complexity
    from types import SimpleNamespace
    config_obj = SimpleNamespace(**config)
    config_obj.transform_type = "rqs"
    config_obj.num_plu_layers = 2
    
    complexity = estimate_model_complexity(config_obj, input_dim=40)
    print(f"\nModel complexity for 40D input:")
    print(f"  Total parameters: {complexity['total_parameters']:,}")
    print(f"  Memory: {complexity['memory_mb']:.2f} MB")
