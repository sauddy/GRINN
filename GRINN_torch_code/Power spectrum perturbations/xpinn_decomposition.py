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
        - interface_type: 'vertical' (constant x) or 'horizontal' (constant y)
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
    
    if interface_type == 'vertical':
        # Constant x interface
        dx = (xmax - xmin) / (position_idx + 1 - 1)  # This will be recalculated properly
        # Get x position from interface position
        # Vertical interface position_idx means between x-slice (position_idx-1) and position_idx
        from config import NUM_SUBDOMAINS_X
        dx = (xmax - xmin) / NUM_SUBDOMAINS_X
        x_interface = xmin + position_idx * dx
        
        # Sample along y and t
        y_vals = torch.empty(n_spatial, 1, device=device, dtype=torch.float32).uniform_(ymin, ymax).requires_grad_()
        t_vals = torch.empty(n_temporal, 1, device=device, dtype=torch.float32).uniform_(tmin, tmax).requires_grad_()
        
        # Create meshgrid-like structure
        y_grid = y_vals.repeat(n_temporal, 1)
        t_grid = t_vals.repeat_interleave(n_spatial, dim=0)
        x_grid = torch.full_like(y_grid, x_interface).requires_grad_()
        
    else:  # 'horizontal'
        # Constant y interface
        from config import NUM_SUBDOMAINS_Y
        dy = (ymax - ymin) / NUM_SUBDOMAINS_Y
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

