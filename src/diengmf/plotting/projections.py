
def plot_3d_state_projections(true_state, posterior_ensemble, prior_ensemble=None, 
                             figsize=(15, 10), title_prefix="State"):
    """
    Plot 3D position state in all 2D projections plus 3D view.
    
    Parameters:
    -----------
    true_state : Array, shape (6,)
        True state vector [x, y, z, vx, vy, vz]
    posterior_ensemble : Array, shape (N, 6) 
        Posterior ensemble of states
    prior_ensemble : Array, shape (N, 6), optional
        Prior ensemble of states
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for plot titles
    """
    
    # Extract position components (first 3 dimensions)
    true_pos = true_state[:3]
    post_pos = posterior_ensemble[:, :3]
    
    if prior_ensemble is not None:
        prior_pos = prior_ensemble[:, :3]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Define projection pairs and labels
    projections = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
    
    # Plot 2D projections
    for i, (dim1, dim2, label1, label2) in enumerate(projections):
        ax = fig.add_subplot(2, 2, i+1)
        
        # Plot prior if available
        if prior_ensemble is not None:
            ax.scatter(prior_pos[:, dim1], prior_pos[:, dim2], 
                      c='red', alpha=0.6, s=20, label='Prior')
        
        # Plot posterior
        ax.scatter(post_pos[:, dim1], post_pos[:, dim2], 
                  c='blue', alpha=0.7, s=20, label='Posterior')
        
        # Plot true state
        ax.scatter(true_pos[dim1], true_pos[dim2], 
                  c='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2, label='True State')
        
        ax.set_xlabel(f'{label1} Position')
        ax.set_ylabel(f'{label2} Position')
        ax.set_title(f'{title_prefix}: {label1}-{label2} Projection')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3D view
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot prior if available
    if prior_ensemble is not None:
        ax3d.scatter(prior_pos[:, 0], prior_pos[:, 1], prior_pos[:, 2],
                    c='red', alpha=0.6, s=20, label='Prior')
    
    # Plot posterior
    ax3d.scatter(post_pos[:, 0], post_pos[:, 1], post_pos[:, 2],
                c='blue', alpha=0.7, s=20, label='Posterior')
    
    # Plot true state
    ax3d.scatter(true_pos[0], true_pos[1], true_pos[2],
                c='green', s=100, marker='o',
                edgecolors='black', linewidth=2, label='True State')
    
    ax3d.set_xlabel('X Position')
    ax3d.set_ylabel('Y Position')
    ax3d.set_zlabel('Z Position')
    ax3d.set_title(f'{title_prefix}: 3D View')
    ax3d.legend()
    
    plt.tight_layout()
    return fig
