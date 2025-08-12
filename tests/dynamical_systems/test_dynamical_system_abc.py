import jax
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96

def test_dynamical_system_abc():
    dynamical_systems = [Ikeda(), Lorenz63(), Lorenz96()]
    
    for dynamical_system in dynamical_systems:
        initial_state = dynamical_system.initial_state()
        initial_state = dynamical_system.flow(0.0, 10.0, initial_state)
        batch = dynamical_system.generate(jax.random.key(0))

    assert True


import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import jax
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96

@pytest.mark.parametrize(
    "dynamical_system,filename",
    [
        (Ikeda(), "ikeda_attractor.png"),
        (Lorenz63(), "lorenz63_attractor.png"),
        (Lorenz96(), "lorenz96_attractor.png"),
    ],
)
def test_dynamical_system_pretty_plotting(dynamical_system, filename, ROOT_DIR):
    batch = dynamical_system.generate(jax.random.key(0), batch_size=5000)
    
    if dynamical_system.dimension == 2:
        fig, ax = plt.subplots()
        ax.scatter(batch[:, 0], batch[:, 1], s=0.1, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    elif dynamical_system.dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2], s=0.1, alpha=0.7)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots()
        for dim in range(min(4, dynamical_system.dimension)):
            ax.plot(batch[:, dim], alpha=0.7, label=f"dim_{dim}")
        ax.legend()
    
    os.makedirs(ROOT_DIR / "figures" / "dynamical_system_attractors", exist_ok=True)
    save_path = os.path.join(ROOT_DIR / "figures" / "dynamical_system_attractors", filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    assert os.path.exists(save_path)

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import jax
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96

@pytest.mark.parametrize(
    "dynamical_system,filename",
    [
        (Ikeda(), "ikeda_attractor.png"),
        (Lorenz63(), "lorenz63_attractor.png"),
        (Lorenz96(), "lorenz96_attractor.png"),
    ],
)
def test_dynamical_system_pretty_plotting(dynamical_system, filename, ROOT_DIR):
    batch = dynamical_system.generate(jax.random.key(0), batch_size=5000)
    
    if dynamical_system.dimension == 2:
        fig, ax = plt.subplots()
        ax.scatter(batch[:, 0], batch[:, 1], s=0.1, alpha=0.7)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    elif dynamical_system.dimension == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2], s=0.1, alpha=0.7)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        n_dims_to_plot = min(4, dynamical_system.dimension)
        for i in range(n_dims_to_plot):
            axes[i].plot(batch[:, i], alpha=0.7)
            axes[i].set_title(f"x{i+1}")
            axes[i].set_xlabel("sample")
            axes[i].set_ylabel("value")
        for i in range(n_dims_to_plot, 4):
            axes[i].axis('off')
    
    os.makedirs(ROOT_DIR / "figures" / "dynamical_system_attractors", exist_ok=True)
    save_path = os.path.join(ROOT_DIR / "figures" / "dynamical_system_attractors", filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    assert os.path.exists(save_path)

###

import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
import jax
from diengmf.dynamical_systems import Ikeda, Lorenz63, Lorenz96

@pytest.mark.parametrize(
    "dynamical_system,filename",
    [
        (Ikeda(), "ikeda_attractor.png"),
        (Lorenz63(), "lorenz63_attractor.png"),
        (Lorenz96(), "lorenz96_attractor.png"),
    ],
)
def test_dynamical_system_pretty_plotting(dynamical_system, filename, ROOT_DIR):
    batch = dynamical_system.generate(jax.random.key(0), batch_size=5000)
    
    if dynamical_system.dimension == 2:
        fig, ax = plt.subplots()
        ax.scatter(batch[:, 0], batch[:, 1], s=0.1, alpha=0.7)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(batch[:, 0], batch[:, 1], batch[:, 2], s=0.1, alpha=0.7)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")

    ax.set_xlim(dynamical_system.plot_limits[0])
    ax.set_ylim(dynamical_system.plot_limits[1])
    if hasattr(ax, 'set_zlim'): ax.set_zlim(dynamical_system.plot_limits[2])
    
    # print(f"{dynamical_system.__class__.__name__}: xlim={ax.get_xlim()}, ylim={ax.get_ylim()}")
    # if hasattr(ax, 'get_zlim'): print(f"zlim={ax.get_zlim()}")
    os.makedirs(ROOT_DIR / "figures" / "dynamical_system_attractors", exist_ok=True)
    save_path = os.path.join(ROOT_DIR / "figures" / "dynamical_system_attractors", filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    assert os.path.exists(save_path)
