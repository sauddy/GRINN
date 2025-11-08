"""
Initial Conditions Module

This module contains all initial condition functions and power spectrum generation
for PINNs training. Extracted from solver.py for better code organization.

Functions:
- initialize_shared_velocity_fields: Setup shared velocity fields for PINN/FD consistency
- generate_power_spectrum_field: Generate vx component using power spectrum
- generate_power_spectrum_field_vy: Generate vy component using power spectrum
- fun_rho_0: Initial density condition
- fun_vx_0: Initial x-velocity condition
- fun_vy_0: Initial y-velocity condition
- func: Placeholder function for phi initial condition
"""

import numpy as np
import torch
from config import (cs, rho_o, N_GRID, POWER_EXPONENT, FILTER_SCALE, 
                    PERTURBATION_TYPE, KX, KY, RANDOM_SEED)

# Global shared velocity fields for consistent initial conditions
_shared_vx_interp = None
_shared_vy_interp = None


def initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=None):
    """
    Initialize shared velocity fields for consistent PINN/FD initial conditions.
    This should be called once at the beginning of training.
    
    IMPORTANT: The parameters used here (POWER_EXPONENT, v_1=a*cs, seed) MUST match
    the parameters used in FD plotting functions to ensure identical initial conditions.
    All FD visualization functions should use the same defaults.
    
    Args:
        lam: Wavelength
        num_of_waves: Number of waves in domain
        v_1: Velocity amplitude
        seed: Random seed for reproducibility
    
    Returns:
        Tuple (vx_np, vy_np): Velocity field arrays
    """
    global _shared_vx_interp, _shared_vy_interp
    
    if seed is None:
        seed = RANDOM_SEED
    
    # Import LAX_2D functions
    from numerical_solvers.LAX_2D import generate_shared_velocity_field
    
    # Calculate domain size to match FD solver
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    
    # Generate shared velocity fields
    vx_np, vy_np, vx_interp, vy_interp = generate_shared_velocity_field(
        N_GRID, N_GRID, Lx, Ly, 
        power_index=POWER_EXPONENT, 
        amplitude=v_1, 
        random_seed=seed
    )
    
    # Store interpolation functions globally
    _shared_vx_interp = vx_interp
    _shared_vy_interp = vy_interp
    
    return vx_np, vy_np


def _interpolate_shared_field(x, field_interp):
    """
    Helper function to interpolate shared velocity field to collocation points.
    
    Args:
        x: Collocation coordinates [x, y, ...]
        field_interp: Interpolation function from shared fields
    
    Returns:
        Interpolated field values as torch tensor
    """
    # Convert tensor coordinates to numpy for interpolation
    x_np = x[0].detach().cpu().numpy()
    y_np = x[1].detach().cpu().numpy()
    
    # Create coordinate pairs for interpolation
    coords = np.stack([x_np.flatten(), y_np.flatten()], axis=1)
    
    # Interpolate shared velocity field
    field_interp_values = field_interp(coords)
    
    # Convert back to tensor and reshape
    field_tensor = torch.from_numpy(field_interp_values).float().to(x[0].device)
    
    # Ensure correct shape [N, 1]
    if field_tensor.dim() == 1:
        return field_tensor.unsqueeze(-1)
    else:
        return field_tensor


def _generate_power_spectrum_fallback(lam, v_1, x, seed=None):
    """
    Fallback power spectrum generation when shared fields are not available.
    
    Args:
        lam: Wavelength
        v_1: Velocity amplitude
        x: Collocation coordinates
        seed: Random seed
    
    Returns:
        Generated power spectrum field
    """
    if seed is None:
        seed = RANDOM_SEED
    
    Lx = lam * 2  # Domain size
    dx = Lx / N_GRID
    
    # Calculate wave numbers
    kx = 2 * np.pi * torch.fft.fftfreq(N_GRID, dx, device=x[0].device)
    ky = 2 * np.pi * torch.fft.fftfreq(N_GRID, dx, device=x[0].device)
    KX_grid, KY_grid = torch.meshgrid(kx, ky, indexing='ij')
    
    # Calculate magnitude of wave number
    K = torch.sqrt(KX_grid**2 + KY_grid**2)
    
    # Power spectrum: P(k) ~ k^expon * exp((-k*Rf)^2)
    K_safe = torch.where(K == 0, torch.tensor(1e-10, device=x[0].device), K)
    power_spectrum = K_safe**POWER_EXPONENT * torch.exp(-(K_safe * FILTER_SCALE)**2)
    
    # Remove DC (uniform) mode to avoid bulk drift
    power_spectrum[K == 0] = 0.0
    
    # Safety check: limit extreme values
    power_spectrum = torch.clamp(power_spectrum, 0, 1e6)
    
    # Generate random phases
    torch.manual_seed(seed)
    random_phases = torch.randn(N_GRID, N_GRID, device=x[0].device) + 1j * torch.randn(N_GRID, N_GRID, device=x[0].device)
    
    # Create complex field in Fourier space and transform to real space
    field_fourier = torch.sqrt(power_spectrum) * random_phases
    field_real = torch.real(torch.fft.ifft2(field_fourier))
    
    # Remove any residual mean (bulk flow) and normalize rms to v_1
    field_real = field_real - torch.mean(field_real)
    field_real = field_real / torch.std(field_real) * v_1
    
    # Interpolate to the actual collocation points
    x_norm = torch.clamp((x[0] / Lx) * (N_GRID - 1), 0, N_GRID - 1)
    y_norm = torch.clamp((x[1] / Lx) * (N_GRID - 1), 0, N_GRID - 1)
    
    x_idx = torch.round(x_norm).long()
    y_idx = torch.round(y_norm).long()
    
    # Ensure correct tensor shape [N, 1]
    result = field_real[x_idx, y_idx]
    if result.dim() == 1:
        return result.unsqueeze(-1)
    else:
        return result


def generate_power_spectrum_field(lam, v_1, x, seed=None):
    """
    Generate 2D Gaussian random field with power spectrum using shared fields if available.
    
    Args:
        lam: Wavelength
        v_1: Velocity amplitude
        x: Collocation coordinates [x, y, ...]
        seed: Random seed for reproducibility
    
    Returns:
        vx component of velocity field
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Use shared velocity fields if available
    if _shared_vx_interp is not None:
        return _interpolate_shared_field(x, _shared_vx_interp)
    
    # Fallback to original method if shared fields not available
    return _generate_power_spectrum_fallback(lam, v_1, x, seed)


def generate_power_spectrum_field_vy(lam, v_1, x, seed=None):
    """
    Generate vy component using shared fields if available.
    
    Args:
        lam: Wavelength
        v_1: Velocity amplitude
        x: Collocation coordinates [x, y, ...]
        seed: Random seed for reproducibility
    
    Returns:
        vy component of velocity field
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Use shared velocity fields if available
    if _shared_vy_interp is not None:
        return _interpolate_shared_field(x, _shared_vy_interp)
    
    # Fallback to original method if shared fields not available
    return _generate_power_spectrum_fallback(lam, v_1, x, seed)


def _sinusoidal_component(coord, lam, jeans, v_1, k_component=None):
    """
    Helper function to generate sinusoidal velocity component.
    
    Args:
        coord: Coordinate tensor [N,1] or [N]
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        k_component: Wave vector component (for 2D case)
    
    Returns:
        Sinusoidal velocity component
    """
    u = coord if coord.dim() > 1 else coord.unsqueeze(-1)
    
    if k_component is not None:
        # Use specific k component for 2D case
        wave_phase = k_component * u
    else:
        # Fallback to original behavior for 1D case
        wave_phase = 2*np.pi*u/lam
    
    if lam > jeans:
        return -v_1 * torch.sin(wave_phase)
    else:
        return v_1 * torch.cos(wave_phase)


def _coupled_2d_velocity_components(x, lam, jeans, v_1):
    """
    Generate coupled 2D velocity components from the same wave pattern.
    
    Args:
        x: Coordinates [x, y, ...]
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
    
    Returns:
        Tuple (vx, vy) of velocity components
    """
    if len(x) < 2:
        # Fallback to 1D case
        return _sinusoidal_component(x[0], lam, jeans, v_1), torch.zeros_like(x[0])
    
    x_coord = x[0] if x[0].dim() > 1 else x[0].unsqueeze(-1)
    y_coord = x[1] if x[1].dim() > 1 else x[1].unsqueeze(-1)
    
    # Calculate wave vector magnitude (convert to tensor)
    k_magnitude = torch.sqrt(torch.tensor(KX**2 + KY**2, device=x_coord.device, dtype=x_coord.dtype))
    
    # Generate the coupled wave pattern
    wave_phase = KX * x_coord + KY * y_coord
    
    if lam > jeans:
        # Gravitational instability case
        wave_field = -v_1 * torch.sin(wave_phase)
    else:
        # Oscillatory case
        wave_field = v_1 * torch.cos(wave_phase)
    
    # Coupled velocity components
    if k_magnitude > 0:
        vx = wave_field * (KX / k_magnitude)
        vy = wave_field * (KY / k_magnitude)
    else:
        vx = wave_field
        vy = torch.zeros_like(wave_field)
    
    return vx, vy


def fun_rho_0(rho_1, lam, x):
    """
    Define initial condition for density.
    
    Args:
        rho_1: Perturbation amplitude
        lam: Wavelength
        x: Spatial coordinates [x, y, t] or [x, t]
    
    Returns:
        rho_0: Initial density field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use separate kx and ky components for 2D wave vector
        if len(x) >= 2:  # 2D case
            x_coord = x[0] if x[0].dim() > 1 else x[0].unsqueeze(-1)
            y_coord = x[1] if x[1].dim() > 1 else x[1].unsqueeze(-1)
            rho_0 = rho_o + rho_1 * torch.cos(KX * x_coord + KY * y_coord)
        else:  # 1D case - fallback to original behavior
            coord = x[0]
            u = coord if coord.dim() > 1 else coord.unsqueeze(-1)
            rho_0 = rho_o + rho_1 * torch.cos(2*np.pi*u/lam)
    else:
        # Power spectrum: uniform initial density
        rho_0 = torch.full_like(x[0], rho_o)
        # Ensure correct shape [N, 1]
        if rho_0.dim() == 1:
            rho_0 = rho_0.unsqueeze(-1)
    
    return rho_0


def fun_vx_0(lam, jeans, v_1, x):
    """
    Initial condition for x-velocity.
    
    Args:
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        x: Spatial coordinates
    
    Returns:
        vx_0: Initial x-velocity field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use coupled 2D velocity components for proper 2D wave physics
        if len(x) >= 2:  # 2D case
            vx, _ = _coupled_2d_velocity_components(x, lam, jeans, v_1)
            return vx
        else:  # 1D case
            return _sinusoidal_component(x[0], lam, jeans, v_1)
    else:
        # Power spectrum case
        return generate_power_spectrum_field(lam, v_1, x, seed=RANDOM_SEED)


def fun_vy_0(lam, jeans, v_1, x):
    """
    Initial condition for y-velocity.
    
    Args:
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        x: Spatial coordinates
    
    Returns:
        vy_0: Initial y-velocity field
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use coupled 2D velocity components for proper 2D wave physics
        if len(x) >= 2:  # 2D case
            _, vy = _coupled_2d_velocity_components(x, lam, jeans, v_1)
            return vy
        else:
            # Fallback to x if y is unavailable (1D)
            return _sinusoidal_component(x[0], lam, jeans, v_1)
    else:
        # Power spectrum case
        return generate_power_spectrum_field_vy(lam, v_1, x, seed=RANDOM_SEED)


def func(x):
    """
    Placeholder function for phi initial condition (zero potential).
    
    Args:
        x: Spatial coordinates
    
    Returns:
        Zero tensor matching the shape of x[0]
    """
    return x[0] * 0

