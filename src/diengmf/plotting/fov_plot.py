def plot_tracking_fov(predicted_state, true_state, step, elevation_fov=(5 * jnp.pi / 180), azimuth_fov=(5 * jnp.pi / 180), range_fov=1.0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Predicted state spherical coordinates
    rho_pred = np.linalg.norm(predicted_state[:3])
    theta_pred = np.arcsin(predicted_state[2] / rho_pred)
    phi_pred = np.arctan2(predicted_state[1], predicted_state[0])
    
    # Box corners in spherical coordinates
    theta_range = [theta_pred - elevation_fov/2, theta_pred + elevation_fov/2]
    phi_range = [phi_pred - azimuth_fov/2, phi_pred + azimuth_fov/2]
    rho_range = [rho_pred - range_fov/2, rho_pred + range_fov/2]
    
    # Generate 8 corners of the box
    corners = []
    for theta in theta_range:
        for phi in phi_range:
            for rho in rho_range:
                x = rho * np.cos(theta) * np.cos(phi)
                y = rho * np.cos(theta) * np.sin(phi)
                z = rho * np.sin(theta)
                corners.append([x, y, z])
    corners = np.array(corners)
    
    # Draw wireframe edges
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for edge in edges:
        points = corners[list(edge)]
        ax.plot(points[:,0], points[:,1], points[:,2], 'b-', alpha=0.6)
    
    # Check if true state is measurable
    is_measurable = tracking_measurability(true_state, predicted_state, elevation_fov, azimuth_fov, range_fov)
    true_color = 'green' if is_measurable else 'red'
    
    ax.scatter(*predicted_state[:3], c='blue', s=100, label='Predicted State')
    ax.scatter(*true_state[:3], c=true_color, s=100, label=f'True State ({"Measurable" if is_measurable else "Not Measurable"})')
    
    # Dynamic axis limits to show both states and wireframe
    all_points = np.vstack([corners, predicted_state[:3].reshape(1,-1), true_state[:3].reshape(1,-1)])
    margin = 0.02
    ax.set_xlim(all_points[:,0].min() - margin, all_points[:,0].max() + margin)
    ax.set_ylim(all_points[:,1].min() - margin, all_points[:,1].max() + margin)
    ax.set_zlim(all_points[:,2].min() - margin, all_points[:,2].max() + margin)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.title(f'Optical Sensor FOV - Step {step}')
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/fov_step_{step:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()
