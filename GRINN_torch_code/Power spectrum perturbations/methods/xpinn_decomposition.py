"""
XPINN Domain Decomposition Utilities

Provides functions for subdomain boundary calculation, interface identification,
and collocation point generation for XPINN-style domain decomposition.
"""

import numpy as np
import torch


def get_num_subdomains(nx_sub, ny_sub):
    """
    Calculate total number of subdomains.
    
    Args:
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        Total number of subdomains
    """
    return nx_sub * ny_sub


def subdomain_idx_to_grid(idx, nx_sub, ny_sub):
    """
    Convert linear subdomain index to (i, j) grid position.
    
    Args:
        idx: Linear subdomain index (0 to nx_sub*ny_sub-1)
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        Tuple (i, j) representing grid position where:
        - i is x-direction index (0 to nx_sub-1)
        - j is y-direction index (0 to ny_sub-1)
    """
    i = idx // ny_sub
    j = idx % ny_sub
    return (i, j)


def get_subdomain_bounds(subdomain_idx, xmin, xmax, ymin, ymax, nx_sub, ny_sub):
    """
    Compute spatial boundaries for a given subdomain.
    
    Args:
        subdomain_idx: Linear subdomain index
        xmin, xmax: Global domain x-bounds
        ymin, ymax: Global domain y-bounds
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        Tuple (x_min, x_max, y_min, y_max) for the subdomain
    """
    i, j = subdomain_idx_to_grid(subdomain_idx, nx_sub, ny_sub)
    
    # Calculate subdomain width and height
    dx = (xmax - xmin) / nx_sub
    dy = (ymax - ymin) / ny_sub
    
    # Calculate subdomain boundaries (non-overlapping)
    x_min = xmin + i * dx
    x_max = xmin + (i + 1) * dx
    y_min = ymin + j * dy
    y_max = ymin + (j + 1) * dy
    
    return (x_min, x_max, y_min, y_max)


def get_interfaces(nx_sub, ny_sub):
    """
    Identify all interfaces between adjacent subdomains.
    
    Args:
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        List of tuples (subdomain_i, subdomain_j, interface_type, position_idx)
        where:
        - subdomain_i, subdomain_j: indices of adjacent subdomains
        - interface_type: 'vertical', 'horizontal', 'periodic_vertical', or 'periodic_horizontal'
        - position_idx: which vertical/horizontal line (for position calculation)
    """
    interfaces = []
    
    # Vertical interfaces (constant x, between subdomains in x-direction)
    for i in range(nx_sub - 1):  # Between x-slices i and i+1
        for j in range(ny_sub):  # For each y-slice
            subdomain_left = i * ny_sub + j
            subdomain_right = (i + 1) * ny_sub + j
            interfaces.append((subdomain_left, subdomain_right, 'vertical', i + 1))
    
    # Horizontal interfaces (constant y, between subdomains in y-direction)
    for i in range(nx_sub):  # For each x-slice
        for j in range(ny_sub - 1):  # Between y-slices j and j+1
            subdomain_bottom = i * ny_sub + j
            subdomain_top = i * ny_sub + (j + 1)
            interfaces.append((subdomain_bottom, subdomain_top, 'horizontal', j + 1))
    
    # Periodic wrap-around interfaces (enforce periodic BC across domain boundaries)
    # Vertical wrap-around: right edge ↔ left edge (at x=xmax ↔ x=xmin)
    for j in range(ny_sub):  # For each y-slice
        subdomain_left = 0 * ny_sub + j              # Leftmost column (i=0)
        subdomain_right = (nx_sub - 1) * ny_sub + j  # Rightmost column (i=nx_sub-1)
        interfaces.append((subdomain_right, subdomain_left, 'periodic_vertical', nx_sub))
    
    # Horizontal wrap-around: top edge ↔ bottom edge (at y=ymax ↔ y=ymin)
    for i in range(nx_sub):  # For each x-slice
        subdomain_bottom = i * ny_sub + 0              # Bottom row (j=0)
        subdomain_top = i * ny_sub + (ny_sub - 1)      # Top row (j=ny_sub-1)
        interfaces.append((subdomain_top, subdomain_bottom, 'periodic_horizontal', ny_sub))
    
    return interfaces


def generate_interface_points(interface_info, xmin, xmax, ymin, ymax, 
                               tmin, tmax, n_points, device='cpu'):
    """
    Generate collocation points along an interface.
    
    Args:
        interface_info: Tuple (subdomain_i, subdomain_j, interface_type, position_idx)
        xmin, xmax: Global domain x-bounds
        ymin, ymax: Global domain y-bounds
        tmin, tmax: Time bounds
        n_points: Number of collocation points to generate
        device: PyTorch device
    
    Returns:
        List [x, y, t] of torch tensors with gradients enabled
    """
    subdomain_i, subdomain_j, interface_type, position_idx = interface_info
    
    # Generate points uniformly distributed along the interface and in time
    # Split n_points between spatial and temporal sampling
    n_spatial = max(int(np.sqrt(n_points)), 1)
    n_temporal = max(n_points // n_spatial, 1)
    
    if interface_type in ('vertical', 'periodic_vertical'):
        # Constant x interface
        from config import NUM_SUBDOMAINS_X
        dx = (xmax - xmin) / NUM_SUBDOMAINS_X
        
        if interface_type == 'periodic_vertical':
            # Periodic wrap-around: sample at x=xmax (which is equivalent to x=xmin due to periodic BC)
            x_interface = xmax
        else:
            # Interior vertical interface
            x_interface = xmin + position_idx * dx
        
        # Sample along y and t
        y_vals = torch.empty(n_spatial, 1, device=device, dtype=torch.float32).uniform_(ymin, ymax).requires_grad_()
        t_vals = torch.empty(n_temporal, 1, device=device, dtype=torch.float32).uniform_(tmin, tmax).requires_grad_()
        
        # Create meshgrid-like structure
        y_grid = y_vals.repeat(n_temporal, 1)
        t_grid = t_vals.repeat_interleave(n_spatial, dim=0)
        x_grid = torch.full_like(y_grid, x_interface).requires_grad_()
        
    else:  # 'horizontal' or 'periodic_horizontal'
        # Constant y interface
        from config import NUM_SUBDOMAINS_Y
        dy = (ymax - ymin) / NUM_SUBDOMAINS_Y
        
        if interface_type == 'periodic_horizontal':
            # Periodic wrap-around: sample at y=ymax (which is equivalent to y=ymin due to periodic BC)
            y_interface = ymax
        else:
            # Interior horizontal interface
            y_interface = ymin + position_idx * dy
        
        # Sample along x and t
        x_vals = torch.empty(n_spatial, 1, device=device, dtype=torch.float32).uniform_(xmin, xmax).requires_grad_()
        t_vals = torch.empty(n_temporal, 1, device=device, dtype=torch.float32).uniform_(tmin, tmax).requires_grad_()
        
        # Create meshgrid-like structure
        x_grid = x_vals.repeat(n_temporal, 1)
        t_grid = t_vals.repeat_interleave(n_spatial, dim=0)
        y_grid = torch.full_like(x_grid, y_interface).requires_grad_()
    
    # Ensure all have gradients enabled
    if not x_grid.requires_grad:
        x_grid = x_grid.requires_grad_()
    if not y_grid.requires_grad:
        y_grid = y_grid.requires_grad_()
    if not t_grid.requires_grad:
        t_grid = t_grid.requires_grad_()
    
    return [x_grid, y_grid, t_grid]


def generate_subdomain_collocation(subdomain_bounds, n_residual, n_ic, 
                                    tmin, tmax, startup_dt, device='cpu'):
    """
    Generate residual and IC collocation points within a subdomain.
    
    Args:
        subdomain_bounds: Tuple (x_min, x_max, y_min, y_max)
        n_residual: Number of residual collocation points
        n_ic: Number of initial condition points
        tmin, tmax: Time bounds
        startup_dt: Time offset for PDE enforcement
        device: PyTorch device
    
    Returns:
        Tuple (colloc_domain, colloc_ic) where each is a list [x, y, t]
    """
    x_min, x_max, y_min, y_max = subdomain_bounds
    
    # Generate residual/domain collocation points
    x_domain = torch.empty(n_residual, 1, device=device, dtype=torch.float32).uniform_(x_min, x_max).requires_grad_()
    y_domain = torch.empty(n_residual, 1, device=device, dtype=torch.float32).uniform_(y_min, y_max).requires_grad_()
    # Shift PDE enforcement to start at t = startup_dt (like original implementation)
    t_domain = torch.empty(n_residual, 1, device=device, dtype=torch.float32).uniform_(max(tmin, startup_dt), tmax).requires_grad_()
    colloc_domain = [x_domain, y_domain, t_domain]
    
    # Generate initial condition collocation points (at t=0)
    x_ic = torch.empty(n_ic, 1, device=device, dtype=torch.float32).uniform_(x_min, x_max).requires_grad_()
    y_ic = torch.empty(n_ic, 1, device=device, dtype=torch.float32).uniform_(y_min, y_max).requires_grad_()
    t_ic = torch.empty(n_ic, 1, device=device, dtype=torch.float32).fill_(tmin).requires_grad_()
    colloc_ic = [x_ic, y_ic, t_ic]
    
    return (colloc_domain, colloc_ic)


def point_in_subdomain(x, y, subdomain_bounds):
    """
    Check if point(s) belong to a subdomain.
    
    Args:
        x, y: Coordinates (can be scalars, arrays, or tensors)
        subdomain_bounds: Tuple (x_min, x_max, y_min, y_max)
    
    Returns:
        Boolean or boolean array indicating if points are in subdomain
    """
    x_min, x_max, y_min, y_max = subdomain_bounds
    
    # Handle both numpy arrays and torch tensors
    if isinstance(x, torch.Tensor):
        in_x = (x >= x_min) & (x <= x_max)
        in_y = (y >= y_min) & (y <= y_max)
    else:
        in_x = (x >= x_min) & (x <= x_max)
        in_y = (y >= y_min) & (y <= y_max)
    
    return in_x & in_y


def get_exterior_boundary_info(subdomain_idx, nx_sub, ny_sub, xmin, xmax, ymin, ymax):
    """
    Determine which boundaries of a subdomain are exterior boundaries.
    
    Args:
        subdomain_idx: Linear subdomain index
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
        xmin, xmax: Global domain x-bounds
        ymin, ymax: Global domain y-bounds
    
    Returns:
        Dict with keys 'left', 'right', 'bottom', 'top' indicating if boundary is exterior
        and corresponding coordinate values
    """
    i, j = subdomain_idx_to_grid(subdomain_idx, nx_sub, ny_sub)
    x_min, x_max, y_min, y_max = get_subdomain_bounds(subdomain_idx, xmin, xmax, ymin, ymax, nx_sub, ny_sub)
    
    boundary_info = {
        'left': (i == 0, x_min),           # Left edge of domain
        'right': (i == nx_sub - 1, x_max), # Right edge of domain
        'bottom': (j == 0, y_min),         # Bottom edge of domain
        'top': (j == ny_sub - 1, y_max)    # Top edge of domain
    }
    
    return boundary_info


# ==================== XPINN Setup and Initialization ====================

def setup_xpinn_devices(num_subdomains, device, use_multi_gpu=False):
    """
    Setup device assignment for XPINN subdomains.
    
    Args:
        num_subdomains: Total number of subdomains
        device: Default device
        use_multi_gpu: Whether to use multi-GPU setup
    
    Returns:
        Tuple (subdomain_devices, num_gpus)
    """
    if use_multi_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Multi-GPU enabled: {num_gpus} GPUs available")
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        subdomain_devices = [devices[i % len(devices)] for i in range(num_subdomains)]
    else:
        num_gpus = 1
        subdomain_devices = [device] * num_subdomains
    
    return subdomain_devices, num_gpus


def setup_xpinn_networks(num_subdomains, subdomain_devices, xmin, xmax, ymin, ymax, 
                         dimension, num_neurons, num_layers, harmonics, 
                         default_activation, nx_sub, ny_sub):
    """
    Initialize XPINN subdomain networks.
    
    Args:
        num_subdomains: Total number of subdomains
        subdomain_devices: List of device assignments per subdomain
        xmin, xmax, ymin, ymax: Global domain bounds
        dimension: Spatial dimension
        num_neurons: Default number of neurons per layer
        num_layers: Default number of hidden layers
        harmonics: Default number of Fourier harmonics
        default_activation: Default activation function type
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        List of initialized neural networks
    """
    from config import SUBDOMAIN_CONFIGS
    from core.model_architecture import PINN
    
    # Validate subdomain configs
    subdomain_configs = None
    if SUBDOMAIN_CONFIGS and len(SUBDOMAIN_CONFIGS) != num_subdomains:
        print(f"WARNING: SUBDOMAIN_CONFIGS has {len(SUBDOMAIN_CONFIGS)} entries but {num_subdomains} subdomains expected.")
        print(f"         Using global defaults for all subdomains.")
    elif SUBDOMAIN_CONFIGS:
        subdomain_configs = SUBDOMAIN_CONFIGS
    
    nets = []
    for i in range(num_subdomains):
        # Get subdomain-specific configuration or use defaults
        if subdomain_configs and i < len(subdomain_configs):
            config = subdomain_configs[i]
            sub_neurons = config.get('num_neurons', num_neurons)
            sub_layers = config.get('num_layers', num_layers)
            sub_harmonics = config.get('n_harmonics', harmonics)
            sub_activation = config.get('activation', default_activation)
        else:
            sub_neurons = num_neurons
            sub_layers = num_layers
            sub_harmonics = harmonics
            sub_activation = default_activation
        
        # Create network
        net = PINN(num_neurons=sub_neurons, num_layers=sub_layers, 
                  n_harmonics=sub_harmonics, activation_type=sub_activation)
        
        # Use GLOBAL domain for periodic embeddings
        net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=dimension)
        net = net.to(subdomain_devices[i])
        nets.append(net)
        
        # Print configuration
        subdomain_bounds = get_subdomain_bounds(i, xmin, xmax, ymin, ymax, nx_sub, ny_sub)
        print(f"  Subdomain {i}: bounds={subdomain_bounds}")
        print(f"    Architecture: neurons={sub_neurons}, layers={sub_layers}, harmonics={sub_harmonics}, activation={sub_activation}, device={subdomain_devices[i]}")
    
    return nets


def setup_xpinn_collocation(num_subdomains, subdomain_devices, xmin, xmax, ymin, ymax, 
                            tmin, tmax, n_r, n_0, startup_dt, nx_sub, ny_sub):
    """
    Generate collocation points for XPINN subdomains.
    
    Args:
        num_subdomains: Total number of subdomains
        subdomain_devices: Device assignment per subdomain
        xmin, xmax, ymin, ymax: Domain bounds
        tmin, tmax: Time bounds
        n_r: Total residual collocation points
        n_0: Total IC collocation points
        startup_dt: Time offset for PDE enforcement
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        Tuple (subdomain_collocs, subdomain_ic_collocs)
    """
    from core.data_generator import distribute_collocation_points
    
    # Distribute points
    n_r_per_subdomain = distribute_collocation_points(n_r, num_subdomains)
    n_0_per_subdomain = distribute_collocation_points(n_0, num_subdomains)
    
    subdomain_collocs = []
    subdomain_ic_collocs = []
    
    for i in range(num_subdomains):
        subdomain_bounds = get_subdomain_bounds(i, xmin, xmax, ymin, ymax, nx_sub, ny_sub)
        colloc_domain, colloc_ic = generate_subdomain_collocation(
            subdomain_bounds, n_r_per_subdomain[i], n_0_per_subdomain[i],
            tmin, tmax, startup_dt, device=subdomain_devices[i]
        )
        subdomain_collocs.append(colloc_domain)
        subdomain_ic_collocs.append(colloc_ic)
    
    return subdomain_collocs, subdomain_ic_collocs


def setup_xpinn_interfaces(subdomain_devices, xmin, xmax, ymin, ymax, tmin, tmax, 
                           n_interface, nx_sub, ny_sub):
    """
    Generate interface collocation points for XPINN.
    
    Args:
        subdomain_devices: Device assignment per subdomain
        xmin, xmax, ymin, ymax: Domain bounds
        tmin, tmax: Time bounds
        n_interface: Number of interface collocation points
        nx_sub: Number of subdomain splits in x-direction
        ny_sub: Number of subdomain splits in y-direction
    
    Returns:
        Tuple (interfaces, interface_collocs)
    """
    interfaces = get_interfaces(nx_sub, ny_sub)
    
    interface_collocs = {}
    for interface in interfaces:
        subdomain_i, subdomain_j, _, _ = interface
        interface_device = subdomain_devices[subdomain_i]
        interface_points = generate_interface_points(
            interface, xmin, xmax, ymin, ymax, tmin, tmax, n_interface, device=interface_device
        )
        interface_collocs[(subdomain_i, subdomain_j)] = interface_points
    
    return interfaces, interface_collocs


def cache_xpinn_initial_conditions(num_subdomains, subdomain_ic_collocs, ic_functions):
    """
    Pre-compute and cache initial condition values for all subdomains.
    
    Args:
        num_subdomains: Total number of subdomains
        subdomain_ic_collocs: IC collocation points per subdomain
        ic_functions: Dictionary of IC functions
    
    Returns:
        List of cached IC dictionaries per subdomain
    """
    print("Caching IC values for all subdomains...")
    cached_ic_values = []
    
    for i in range(num_subdomains):
        colloc_ic = subdomain_ic_collocs[i]
        ic_cache = {
            'rho': ic_functions['rho'](colloc_ic),
            'vx': ic_functions['vx'](colloc_ic),
            'vy': ic_functions['vy'](colloc_ic),
            'phi': ic_functions['phi'](colloc_ic)
        }
        cached_ic_values.append(ic_cache)
    
    print("IC values cached successfully!")
    return cached_ic_values
