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
from config import (cs, rho_o, N_GRID, POWER_EXPONENT, 
                    PERTURBATION_TYPE, KX, KY, KZ, RANDOM_SEED)
from numerical_solvers.LAX import generate_shared_velocity_field

# Global shared velocity fields for consistent initial conditions
_shared_vx_interp = None
_shared_vy_interp = None

def _ensure_column_tensor(tensor):
    return tensor if tensor.dim() > 1 else tensor.unsqueeze(-1)

def _extract_spatial_coords(coords):
    return coords[:-1] if len(coords) > 1 else coords


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
        lam: Wavelength (unused for domain sizing in fallback)
        v_1: Velocity amplitude
        x: Collocation coordinates
        seed: Random seed
    
    Returns:
        Generated power spectrum field
    """
    if seed is None:
        seed = RANDOM_SEED
    
    # Infer domain extents directly from the collocation coordinates to support arbitrary num_of_waves
    # Use conservative defaults if tensors are degenerate (e.g., single point during a unit test)
    x_coords = x[0].detach()
    y_coords = x[1].detach() if len(x) > 1 else x[0].detach()

    xmin_val = torch.min(x_coords).item() if x_coords.numel() > 0 else 0.0
    xmax_val = torch.max(x_coords).item() if x_coords.numel() > 0 else float(lam * 2.0)
    ymin_val = torch.min(y_coords).item() if y_coords.numel() > 0 else 0.0
    ymax_val = torch.max(y_coords).item() if y_coords.numel() > 0 else float(lam * 2.0)

    # Ensure positive lengths; fall back to 2*lam if bounds collapse
    Lx = float(max(xmax_val - xmin_val, 1e-6))
    Ly = float(max(ymax_val - ymin_val, 1e-6))
    if not torch.isfinite(torch.tensor(Lx)) or Lx < 1e-6:
        Lx = float(lam * 2.0)
    if not torch.isfinite(torch.tensor(Ly)) or Ly < 1e-6:
        Ly = float(lam * 2.0)

    dx = Lx / N_GRID
    dy = Ly / N_GRID
    
    # Calculate wave numbers
    kx = 2 * np.pi * torch.fft.fftfreq(N_GRID, dx, device=x[0].device)
    ky = 2 * np.pi * torch.fft.fftfreq(N_GRID, dy, device=x[0].device)
    KX_grid, KY_grid = torch.meshgrid(kx, ky, indexing='ij')
    
    # Calculate magnitude of wave number
    K = torch.sqrt(KX_grid**2 + KY_grid**2)
    
    # Power spectrum: P(k) ~ k^expon
    K_safe = torch.where(K == 0, torch.tensor(1e-10, device=x[0].device), K)
    power_spectrum = K_safe**POWER_EXPONENT
    
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
    x_norm = torch.clamp(((x[0] - xmin_val) / Lx) * (N_GRID - 1), 0, N_GRID - 1)
    if len(x) > 1:
        y_norm = torch.clamp(((x[1] - ymin_val) / Ly) * (N_GRID - 1), 0, N_GRID - 1)
    else:
        # 1D fallback: mirror x for y to preserve shape
        y_norm = x_norm.clone()
    
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


def _compute_wave_phase(spatial_coords, lam):
    """
    Compute wave phase and wave-vector components for sinusoidal perturbations.
    """
    if not spatial_coords:
        raise ValueError("Spatial coordinates are required to compute wave phase.")
    
    x_coord = _ensure_column_tensor(spatial_coords[0])
    zeros = torch.zeros_like(x_coord)
    y_coord = _ensure_column_tensor(spatial_coords[1]) if len(spatial_coords) >= 2 else zeros
    z_coord = _ensure_column_tensor(spatial_coords[2]) if len(spatial_coords) >= 3 else zeros
    
    device = x_coord.device
    dtype = x_coord.dtype
    kx = torch.as_tensor(float(KX), device=device, dtype=dtype)
    ky = torch.as_tensor(float(KY), device=device, dtype=dtype)
    kz = torch.as_tensor(float(KZ), device=device, dtype=dtype)
    
    phase = kx * x_coord + ky * y_coord + kz * z_coord
    
    # Fallback to fundamental wavelength if wave-vector is zero (e.g., user-specified)
    if torch.allclose(kx.abs() + ky.abs() + kz.abs(), torch.tensor(0.0, device=device, dtype=dtype)):
        fundamental = torch.as_tensor(2 * np.pi / lam, device=device, dtype=dtype)
        phase = fundamental * x_coord
        kx, ky, kz = fundamental, torch.zeros_like(fundamental), torch.zeros_like(fundamental)
    
    return phase, kx, ky, kz, x_coord, y_coord, z_coord


def _coupled_velocity_components(coords, lam, jeans, v_1):
    """
    Generate coupled velocity components from the same wave pattern (supports 1D/2D/3D).
    """
    spatial_coords = _extract_spatial_coords(coords)
    phase, kx, ky, kz, _, _, _ = _compute_wave_phase(spatial_coords, lam)
    dtype = phase.dtype
    device = phase.device
    v_scale = torch.as_tensor(float(v_1), device=device, dtype=dtype)
    
    if lam > jeans:
        wave_field = -v_scale * torch.sin(phase)
    else:
        wave_field = v_scale * torch.cos(phase)
    
    k_mag = torch.sqrt(kx**2 + ky**2 + kz**2)
    if k_mag <= torch.tensor(1e-12, device=device, dtype=dtype):
        vx = wave_field
        vy = torch.zeros_like(wave_field)
        vz = torch.zeros_like(wave_field)
    else:
        inv_mag = 1.0 / k_mag
        vx = wave_field * (kx * inv_mag)
        vy = wave_field * (ky * inv_mag)
        vz = wave_field * (kz * inv_mag)
    
    return vx, vy, vz


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
        spatial_coords = _extract_spatial_coords(x)
        phase, *_ = _compute_wave_phase(spatial_coords, lam)
        rho_0 = rho_o + rho_1 * torch.cos(phase)
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
        vx, _, _ = _coupled_velocity_components(x, lam, jeans, v_1)
        return vx
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
        _, vy, _ = _coupled_velocity_components(x, lam, jeans, v_1)
        return vy
    else:
        # Power spectrum case
        return generate_power_spectrum_field_vy(lam, v_1, x, seed=RANDOM_SEED)


def fun_vz_0(lam, jeans, v_1, x):
    """
    Initial condition for z-velocity (used in 3D sinusoidal runs).
    """
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        _, _, vz = _coupled_velocity_components(x, lam, jeans, v_1)
        return vz
    else:
        # Power spectrum setup is currently 2D; default to zero
        return func(x)


def func(x):
    """
    Placeholder function for phi initial condition (zero potential).
    
    Args:
        x: Spatial coordinates
    
    Returns:
        Zero tensor matching the shape of x[0]
    """
    return x[0] * 0