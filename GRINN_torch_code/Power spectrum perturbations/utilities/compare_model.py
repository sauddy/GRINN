"""
Script to generate comparison plots (PINN vs FD) for a saved model at custom time points.

Usage:
    python compare_model.py [model_path] <time_points>
    
    If model_path is omitted, defaults to SNAPSHOT_DIR/GRINN/model.pth
    
Example:
    # Spatial comparison plots (default)
    python compare_model.py /path/to/model.pth 0.0,1.0,2.0,3.0
    python compare_model.py 0.0,1.0,2.0,3.0  # Uses default model path
    python compare_model.py /path/to/model.pth 0.5,1.5,2.5 --which velocity
    
    # Density PDF plots
    python compare_model.py /path/to/model.pth 0.0,1.0,2.0 --plot-type pdf
    python compare_model.py 1.5 --plot-type pdf --no-fit  # Uses default model path
    
    # Use GPU-accelerated FD solver (faster for large grids)
    python compare_model.py model.pth 1.5,2.0,3.0 --plot-type pdf --fd-backend gpu
    python compare_model.py model.pth 0.0,1.0,2.0 --fd-backend torch  # Same as gpu
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import sys
import argparse
from scipy.optimize import curve_fit

# Add parent directory to path for imports when running from utilities directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from config import (
    xmin, ymin, tmin, tmax, wave, a, cs, rho_o, harmonics,
    PERTURBATION_TYPE, RANDOM_SEED, N_GRID, POWER_EXPONENT, DIMENSION, SNAPSHOT_DIR
)
# Import num_of_waves with different name to avoid scoping conflict in main()
from config import num_of_waves as num_of_waves_config
from core.model_architecture import PINN
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import initialize_shared_velocity_fields
from visualization.Plotting_2D import set_shared_velocity_fields
import visualization.Plotting_2D as plotting_module
from numerical_solvers.LAX_2D import lax_solution, lax_solution_with_shared_velocity
from numerical_solvers.LAX_2D_torch import lax_solution_torch
from scipy.interpolate import RegularGridInterpolator

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

if device.startswith('cuda'):
    torch.cuda.empty_cache()


def load_model(model_path, xmax, ymax):
    """
    Load a saved single PINN model from disk.
    
    Args:
        model_path: Path to model file
        xmax: Maximum x coordinate
        ymax: Maximum y coordinate
    
    Returns:
        net: Loaded neural network
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    net = PINN(n_harmonics=harmonics)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=DIMENSION)
    net = net.to(device)
    net.eval()
    print(f"Loaded PINN model from {model_path}")
    return net


def create_comparison_plots(net, initial_params, time_points, which="density", N=None, nu=0.5, save_plots=True, fd_backend="cpu"):
    """
    Create comparison plots showing PINN, FD, and epsilon metric at custom time points.
    Based on create_5x3_comparison_table but accepts custom time points.
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: Array of time points to plot
        which: "density" or "velocity"
        N: Grid resolution for LAX solver (defaults to N_GRID to match training plots)
        nu: Courant number for LAX solver
        save_plots: Whether to save the plots to disk (default: True)
        fd_backend: FD solver backend - "cpu" (default) or "gpu"/"torch" for GPU-accelerated solver
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Use N_GRID by default to match training comparison plots
    if N is None:
        N = N_GRID
    
    num_times = len(time_points)
    print(f"Creating {num_times}x3 comparison table for {which}...")
    print(f"Time points: {time_points}")
    
    # Create subplot grid
    fig, axes = plt.subplots(num_times, 3, figsize=(15, 4*num_times))
    if num_times == 1:
        axes = axes.reshape(1, -1)
    
    # Store data
    pinn_data = []
    fd_data = []
    pinn_velocity_data = []
    fd_velocity_data = []
    
    # First pass: collect data
    for i, t in enumerate(time_points):
        print(f"Collecting data for t = {t:.4f}")
        
        # Get PINN data - use N_GRID for consistency with FD solver
        Q = N_GRID
        xs = np.linspace(xmin, xmax, Q, endpoint=False)
        ys = np.linspace(ymin, ymax, Q, endpoint=False)
        tau, phi = np.meshgrid(xs, ys)
        Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
        t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
        
        pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
        
        output_00 = net([pt_x_collocation, pt_y_collocation, pt_t_collocation])
        
        if which == "density":
            # PINN already outputs actual density (not log density) due to _apply_density_constraint
            pinn_field = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
            U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
            V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)
            pinn_vx = U
            pinn_vy = V
        else:  # velocity magnitude
            U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
            V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)
            pinn_field = np.sqrt(U**2 + V**2)
            pinn_vx = U
            pinn_vy = V
        
        # Get FD data - use same parameters as PINN for power spectrum
        num_of_waves = (xmax - xmin) / lam
        
        if (fd_backend.lower() == "gpu" or fd_backend.lower() == "torch") and str(PERTURBATION_TYPE).lower() == "power_spectrum":
            # Use GPU-accelerated torch solver (only for power_spectrum perturbations)
            x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_torch(
                time_val=t, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                gravity=True, use_velocity_ps=True, ps_index=POWER_EXPONENT, 
                vel_rms=a*cs, random_seed=RANDOM_SEED
            )
            # Note: torch solver returns None for phi, but we don't use it in comparison plots
        else:
            # Use CPU solver (default or when GPU not supported for perturbation type)
            if fd_backend.lower() in ["gpu", "torch"] and str(PERTURBATION_TYPE).lower() != "power_spectrum":
                print(f"Warning: GPU solver only supports power_spectrum perturbations. Using CPU solver.")
            if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                # For power spectrum, use shared velocity fields if available
                shared_vx = getattr(plotting_module, '_shared_vx_np', None)
                shared_vy = getattr(plotting_module, '_shared_vy_np', None)
                if shared_vx is not None and shared_vy is not None:
                    x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_with_shared_velocity(
                        t, N, nu, lam, num_of_waves, rho_1, shared_vx, shared_vy,
                        gravity=True, isplot=False, comparison=False, animation=True
                    )
                else:
                    # Fallback to original method
                    x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                        t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                        use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                    )
            else:
                # For sinusoidal, use original parameters
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                    t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                    use_velocity_ps=False, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                )
        
        # Build y-array consistent with solver setup
        # Calculate domain boundaries from config
        xmax_calc = xmin + lam * num_of_waves
        ymax_calc = ymin + lam * num_of_waves
        Lx = xmax_calc - xmin
        Nx = x_fd.shape[0]
        Ny = rho_fd.shape[1]
        y_fd = np.linspace(ymin, ymax_calc, Ny, endpoint=False)
        
        # Create meshgrid for FD data
        X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
        
        if which == "density":
            fd_field = rho_fd
        else:  # velocity magnitude
            fd_field = np.sqrt(vx_fd**2 + vy_fd**2)
        
        # Interpolate FD data to PINN grid
        points_pinn = np.column_stack([tau.ravel(), phi.ravel()])
        
        interpolator = RegularGridInterpolator(
            (x_fd, y_fd), fd_field,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        fd_field_interp = interpolator(points_pinn).reshape(Q, Q)
        
        # Interpolate FD velocity components
        vx_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), vx_fd,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        fd_vx_interp = vx_interpolator(points_pinn).reshape(Q, Q)
        
        vy_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), vy_fd,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        fd_vy_interp = vy_interpolator(points_pinn).reshape(Q, Q)
        
        pinn_data.append(pinn_field)
        fd_data.append(fd_field_interp)
        pinn_velocity_data.append((pinn_vx, pinn_vy))
        fd_velocity_data.append((fd_vx_interp, fd_vy_interp))
    
    # Second pass: create plots
    for i, t in enumerate(time_points):
        pinn_field = pinn_data[i]
        fd_field = fd_data[i]
        
        # Extract velocity components
        pinn_vx, pinn_vy = pinn_velocity_data[i]
        fd_vx, fd_vy = fd_velocity_data[i]
        
        # Calculate epsilon metric: ε = 2 * |PINN - FD| / (PINN + FD) * 100
        eps = 1e-6
        epsilon_metric = 200.0 * np.abs(pinn_field - fd_field) / (pinn_field + fd_field + eps)
        
        # Column 1: PINN
        ax_pinn = axes[i, 0]
        if which == "density":
            pc_pinn = ax_pinn.pcolormesh(tau, phi, pinn_field, shading='auto', cmap='YlOrBr',
                                       vmin=np.min(pinn_field), vmax=np.max(pinn_field))
        else:
            pc_pinn = ax_pinn.pcolormesh(tau, phi, pinn_field, shading='auto', cmap='viridis',
                                       vmin=np.min(pinn_field), vmax=np.max(pinn_field))
        
        # Add velocity vectors
        if pinn_vx is not None and pinn_vy is not None:
            skip_x = max(1, Q // 20)
            skip_y = max(1, Q // 20)
            skip = (slice(None, None, skip_x), slice(None, None, skip_y))
            ax_pinn.quiver(tau[skip], phi[skip], pinn_vx[skip], pinn_vy[skip],
                          color='k', headwidth=3.0, width=0.003, alpha=0.7)
        
        ax_pinn.set_title(f"PINN {which.title()}, t={t:.4f}")
        ax_pinn.set_xlim(xmin, xmax)
        ax_pinn.set_ylim(ymin, ymax)
        cbar_pinn = plt.colorbar(pc_pinn, ax=ax_pinn, shrink=0.6)
        cbar_pinn.ax.set_title(r"$\rho$" if which == "density" else r"$|v|$", fontsize=14)
        
        # Column 2: FD
        ax_fd = axes[i, 1]
        if which == "density":
            pc_fd = ax_fd.pcolormesh(tau, phi, fd_field, shading='auto', cmap='YlOrBr',
                                    vmin=np.min(fd_field), vmax=np.max(fd_field))
        else:
            pc_fd = ax_fd.pcolormesh(tau, phi, fd_field, shading='auto', cmap='viridis',
                                    vmin=np.min(fd_field), vmax=np.max(fd_field))
        
        # Add velocity vectors
        if fd_vx is not None and fd_vy is not None:
            skip_x = max(1, Q // 20)
            skip_y = max(1, Q // 20)
            skip = (slice(None, None, skip_x), slice(None, None, skip_y))
            ax_fd.quiver(tau[skip], phi[skip], fd_vx[skip], fd_vy[skip],
                        color='k', headwidth=3.0, width=0.003, alpha=0.7)
        
        ax_fd.set_title(f"FD {which.title()}, t={t:.4f}")
        ax_fd.set_xlim(xmin, xmax)
        ax_fd.set_ylim(ymin, ymax)
        cbar_fd = plt.colorbar(pc_fd, ax=ax_fd, shrink=0.6)
        cbar_fd.ax.set_title(r"$\rho$" if which == "density" else r"$|v|$", fontsize=14)
        
        # Column 3: Epsilon Metric
        ax_diff = axes[i, 2]
        pc_diff = ax_diff.pcolormesh(tau, phi, epsilon_metric, shading='auto', cmap='coolwarm')
        ax_diff.set_title(f"ε (%), t={t:.4f}")
        ax_diff.set_xlim(xmin, xmax)
        ax_diff.set_ylim(ymin, ymax)
        cbar_diff = plt.colorbar(pc_diff, ax=ax_diff, shrink=0.6)
        cbar_diff.ax.set_title("ε (%)", fontsize=14)
        
        # Add axis labels
        if i == num_times - 1:
            ax_pinn.set_xlabel("x")
            ax_fd.set_xlabel("x")
            ax_diff.set_xlabel("x")
        
        ax_pinn.set_ylabel("y")
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_plots:
        desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
        output_dir = os.path.join(desktop_path, "model test plots")
        os.makedirs(output_dir, exist_ok=True)
        time_str = "_".join([f"{t:.4f}" for t in time_points])
        save_path = os.path.join(output_dir, f"{which}_comparison_{num_times}x3_t{time_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()
    return fig, axes


def compute_density_pdf(rho, rho_ref=None, bin_width=0.01, log_min=None, log_max=None):
    """
    Compute density PDF from 2D density field.
    
    Args:
        rho: 2D array of density values
        rho_ref: Reference density for normalization (default: mean of rho)
        bin_width: Bin width in log space (default: 0.01)
        log_min: Minimum log density (default: auto)
        log_max: Maximum log density (default: auto)
    
    Returns:
        bin_centers: Log density bin centers
        pdf_values: PDF values (normalized)
        rho_tilde_centers: Normalized density values at bin centers
        y_values: log(ρ̃ × f(ρ̃)) for plotting
    """
    # Flatten density array
    rho_flat = rho.flatten()
    
    # Remove any invalid values
    rho_flat = rho_flat[np.isfinite(rho_flat)]
    rho_flat = rho_flat[rho_flat > 0]  # Only positive densities
    
    if len(rho_flat) == 0:
        raise ValueError("No valid density values found")
    
    # Normalize by reference density
    if rho_ref is None:
        rho_ref = np.mean(rho_flat)
    
    rho_tilde = rho_flat / rho_ref
    
    # Take logarithm
    log_rho_tilde = np.log10(rho_tilde)
    
    # Determine bin range
    if log_min is None:
        log_min = np.min(log_rho_tilde) - bin_width
    if log_max is None:
        log_max = np.max(log_rho_tilde) + bin_width
    
    # Create bins
    bins = np.arange(log_min, log_max + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Compute histogram
    counts, _ = np.histogram(log_rho_tilde, bins=bins)
    
    # Normalize to PDF: f(log ρ) = dN / (N_total * d(log ρ))
    total_points = len(rho_flat)
    pdf = counts / (total_points * bin_width)
    
    # Convert bin centers back to linear scale for y-axis
    rho_tilde_centers = 10**bin_centers
    
    # Compute y = log(ρ̃ × f(ρ̃)) = log(ρ̃) + log(f(ρ̃))
    # But we need to be careful: f(ρ̃) is the PDF in log space, so we need f(ρ̃) in linear space
    # f_linear(ρ̃) = f_log(log ρ̃) / (ρ̃ * ln(10))
    # Actually, for the plot we want: log(ρ̃ × f(ρ̃)) where f(ρ̃) is PDF in linear space
    # The relationship is: f_linear(ρ̃) = f_log(log ρ̃) / (ρ̃ * ln(10))
    # So: log(ρ̃ × f_linear(ρ̃)) = log(ρ̃) + log(f_log(log ρ̃)) - log(ρ̃) - log(ln(10))
    #     = log(f_log(log ρ̃)) - log(ln(10))
    # But the standard astrophysical convention is simpler:
    # We plot log(ρ̃ × f(ρ̃)) where f is the PDF in log space, which is what we computed
    # So: y = log(ρ̃) + log(f) = bin_centers + log10(pdf)
    # But we need to handle zeros in pdf
    pdf_safe = np.maximum(pdf, 1e-10)  # Avoid log(0)
    y_values = bin_centers + np.log10(pdf_safe)
    
    # Filter out invalid y values
    valid_mask = np.isfinite(y_values) & (pdf > 0)
    bin_centers = bin_centers[valid_mask]
    pdf_values = pdf[valid_mask]
    rho_tilde_centers = rho_tilde_centers[valid_mask]
    y_values = y_values[valid_mask]
    
    return bin_centers, pdf_values, rho_tilde_centers, y_values


def fit_lognormal(bin_centers, pdf_values):
    """
    Fit log-normal distribution to PDF.
    
    Returns:
        mu: Mean of log-normal in log space
        sigma: Standard deviation in log space
        fit_pdf: Fitted PDF values
    """
    # Convert to linear space for fitting
    rho_tilde = 10**bin_centers
    pdf_linear = pdf_values / (rho_tilde * np.log(10))  # Convert from log-space PDF to linear-space PDF
    
    # Fit log-normal: f(ρ̃) = 1/(ρ̃*σ*√(2π)) * exp(-(ln(ρ̃) - μ)²/(2σ²))
    # In log space: f(log ρ̃) = 1/(σ*√(2π)) * exp(-(log ρ̃ - μ_log)²/(2σ²)) where μ_log = μ/ln(10)
    try:
        # Use weighted least squares on log-space PDF
        valid = pdf_values > 0
        if np.sum(valid) < 3:
            return None, None, None
        
        # Estimate parameters from data
        log_rho = bin_centers[valid]
        pdf_fit = pdf_values[valid]
        
        # Method: fit to log-normal in log space
        # f(log ρ) = A * exp(-(log ρ - μ)²/(2σ²))
        # log(f) = log(A) - (log ρ - μ)²/(2σ²)
        log_pdf = np.log10(pdf_fit + 1e-10)
        
        # Simple polynomial fit to estimate parameters
        p = np.polyfit(log_rho, log_pdf, 2)
        # p[0]*x² + p[1]*x + p[2] = -1/(2σ²)*x² + μ/σ²*x - μ²/(2σ²) + log(A)
        # Check if p[0] is negative (required for log-normal fit)
        if p[0] >= 0:
            # If not negative, use default estimates
            sigma_est = 0.1
            mu_est = np.mean(log_rho)
        else:
            sigma_est = np.sqrt(-1.0 / (2 * p[0]))
            mu_est = -p[1] * sigma_est**2
        # Ensure sigma_est is positive and reasonable
        sigma_est = max(sigma_est, 1e-6)
        
        # Refine with curve_fit
        def lognormal_logspace(x, mu, sigma, A):
            # Ensure A is positive to avoid log10 of negative/zero
            # Handle both scalar and array inputs
            A_safe = np.maximum(A, 1e-10)
            log_term = np.log10(A_safe)
            quad_term = (x - mu)**2 / (2 * sigma**2)
            result = log_term - quad_term
            # Ensure result is finite (handle cases where quad_term > log_term)
            # Replace any invalid values with a very negative number
            result = np.where(np.isfinite(result), result, -10.0)
            return result
        
        # Set bounds to ensure A > 0 and sigma > 0
        bounds = ([np.min(log_rho) - 1, 1e-6, 1e-10], [np.max(log_rho) + 1, np.inf, np.inf])
        try:
            # Suppress warnings during curve fitting
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                popt, _ = curve_fit(lognormal_logspace, log_rho, log_pdf, 
                                   p0=[mu_est, sigma_est, max(10**(p[2]), 1e-10)], 
                                   bounds=bounds,
                                   maxfev=1000)
            mu, sigma, A = popt
            # Ensure A is positive
            A = max(A, 1e-10)
        except (RuntimeError, ValueError) as e:
            # If curve fitting fails, return None
            return None, None, None
        
        # Generate fitted PDF
        fit_pdf = A * np.exp(-(bin_centers - mu)**2 / (2 * sigma**2))
        
        return mu, sigma, fit_pdf
    except:
        return None, None, None


def fit_powerlaw(bin_centers, pdf_values, threshold=None):
    """
    Fit power-law distribution to PDF tail.
    
    Args:
        bin_centers: Log density bin centers
        pdf_values: PDF values
        threshold: Minimum log density for power-law fit (default: 0.8, matching example)
    
    Returns:
        alpha: Power-law index
        fit_pdf: Fitted PDF values
        threshold_used: Actual threshold used
    """
    if threshold is None:
        threshold = 0.8  # Default from example plot
    
    # Find tail region
    tail_mask = bin_centers >= threshold
    if np.sum(tail_mask) < 3:
        return None, None, None
    
    log_rho_tail = bin_centers[tail_mask]
    pdf_tail = pdf_values[tail_mask]
    
    # Power-law in log space: log(f) = -α * log(ρ̃) + const
    # Fit: log(pdf) = -α * log(ρ̃) + C
    try:
        log_pdf_tail = np.log10(pdf_tail + 1e-10)
        p = np.polyfit(log_rho_tail, log_pdf_tail, 1)
        alpha = -p[0]  # Power-law index
        C = p[1]
        
        # Generate fitted PDF for tail region only
        fit_pdf = np.zeros_like(pdf_values)
        fit_pdf[tail_mask] = 10**(C - alpha * log_rho_tail)
        
        return alpha, fit_pdf, threshold
    except:
        return None, None, None


def create_density_pdf_plot(net, initial_params, time_points, N=None, nu=0.5, save_plots=True, 
                            fit_lognorm=True, fit_powerlaw_tail=True, powerlaw_threshold=0.8, fd_backend="cpu"):
    """
    Create density PDF plots comparing PINN and FD solutions.
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: Array of time points to plot
        N: Grid resolution for LAX solver (defaults to N_GRID)
        nu: Courant number for LAX solver
        save_plots: Whether to save plots to disk
        fit_lognorm: Whether to fit log-normal distribution
        fit_powerlaw_tail: Whether to fit power-law tail
        powerlaw_threshold: Minimum log density for power-law fit
        fd_backend: FD solver backend - "cpu" (default) or "gpu"/"torch" for GPU-accelerated solver
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    if N is None:
        N = N_GRID
    
    num_times = len(time_points)
    print(f"Creating density PDF plots for {num_times} time points...")
    print(f"Time points: {time_points}")
    
    # Create subplot grid (one row per time point)
    fig, axes = plt.subplots(num_times, 1, figsize=(8, 5*num_times))
    if num_times == 1:
        axes = [axes]
    
    num_of_waves = (xmax - xmin) / lam
    Q = N_GRID
    
    for i, t in enumerate(time_points):
        print(f"Processing density PDF for t = {t:.4f}")
        ax = axes[i]
        
        # Get PINN density field
        xs = np.linspace(xmin, xmax, Q, endpoint=False)
        ys = np.linspace(ymin, ymax, Q, endpoint=False)
        tau, phi = np.meshgrid(xs, ys)
        Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
        t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
        
        pt_x = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
        pt_t = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
        
        output_00 = net([pt_x, pt_y, pt_t])
        # PINN already outputs actual density (not log density) due to _apply_density_constraint
        rho_pinn = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
        
        # Get FD density field
        if (fd_backend.lower() == "gpu" or fd_backend.lower() == "torch") and str(PERTURBATION_TYPE).lower() == "power_spectrum":
            # Use GPU-accelerated torch solver (only for power_spectrum perturbations)
            x_fd, rho_fd, _, _, _, _, _ = lax_solution_torch(
                time_val=t, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                gravity=True, use_velocity_ps=True, ps_index=POWER_EXPONENT, 
                vel_rms=a*cs, random_seed=RANDOM_SEED
            )
        else:
            # Use CPU solver (default or when GPU not supported for perturbation type)
            if fd_backend.lower() in ["gpu", "torch"] and str(PERTURBATION_TYPE).lower() != "power_spectrum":
                print(f"Warning: GPU solver only supports power_spectrum perturbations. Using CPU solver.")
            if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                shared_vx = getattr(plotting_module, '_shared_vx_np', None)
                shared_vy = getattr(plotting_module, '_shared_vy_np', None)
                if shared_vx is not None and shared_vy is not None:
                    x_fd, rho_fd, _, _, _, _, _ = lax_solution_with_shared_velocity(
                        t, N, nu, lam, num_of_waves, rho_1, shared_vx, shared_vy,
                        gravity=True, isplot=False, comparison=False, animation=True
                    )
                else:
                    x_fd, rho_fd, _, _, _, _, _ = lax_solution(
                        t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                        use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                    )
            else:
                x_fd, rho_fd, _, _, _, _, _ = lax_solution(
                    t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                    use_velocity_ps=False
                )
        
        # Interpolate FD to PINN grid for consistent comparison
        xmax_calc = xmin + lam * num_of_waves
        ymax_calc = ymin + lam * num_of_waves
        Nx = x_fd.shape[0]
        Ny = rho_fd.shape[1]
        y_fd = np.linspace(ymin, ymax_calc, Ny, endpoint=False)
        X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
        
        points_pinn = np.column_stack([tau.ravel(), phi.ravel()])
        rho_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), rho_fd, method='linear', bounds_error=False, fill_value=None
        )
        rho_fd_interp = rho_interpolator(points_pinn).reshape(Q, Q)
        
        # Compute PDFs
        try:
            bin_centers_pinn, pdf_pinn, rho_tilde_pinn, y_pinn = compute_density_pdf(rho_pinn, rho_ref=rho_o)
            bin_centers_fd, pdf_fd, rho_tilde_fd, y_fd = compute_density_pdf(rho_fd_interp, rho_ref=rho_o)
        except Exception as e:
            print(f"Error computing PDF for t={t:.4f}: {e}")
            continue
        
        # Plot data
        ax.plot(bin_centers_pinn, y_pinn, 'k-', linewidth=2, label='PINN', alpha=0.8)
        ax.plot(bin_centers_fd, y_fd, 'r--', linewidth=2, label='FD', alpha=0.8)
        
        # Fit and plot log-normal (optional)
        if fit_lognorm:
            mu_pinn, sigma_pinn, fit_pdf_pinn = fit_lognormal(bin_centers_pinn, pdf_pinn)
            if mu_pinn is not None:
                y_fit_pinn = bin_centers_pinn + np.log10(np.maximum(fit_pdf_pinn, 1e-10))
                ax.plot(bin_centers_pinn, y_fit_pinn, 'k:', linewidth=1.5, 
                       label=f'LN fit (μ={mu_pinn:.2f}, σ={sigma_pinn:.2f})', alpha=0.6)
        
        # Fit and plot power-law tail (optional)
        if fit_powerlaw_tail:
            alpha_pinn, fit_pdf_tail_pinn, threshold_used = fit_powerlaw(
                bin_centers_pinn, pdf_pinn, threshold=powerlaw_threshold
            )
            if alpha_pinn is not None:
                tail_mask = bin_centers_pinn >= threshold_used
                y_fit_tail_pinn = bin_centers_pinn[tail_mask] + np.log10(np.maximum(fit_pdf_tail_pinn[tail_mask], 1e-10))
                ax.plot(bin_centers_pinn[tail_mask], y_fit_tail_pinn, 'b--', linewidth=1.5,
                       label=f'PL fit (α={alpha_pinn:.2f})', alpha=0.6)
                # Add vertical line at threshold
                ax.axvline(x=threshold_used, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        
        # Formatting
        ax.set_xlabel(r'log $\tilde{\rho}$', fontsize=12)
        ax.set_ylabel(r'log $\tilde{\rho}$ $f(\tilde{\rho})$', fontsize=12)
        ax.set_title(f'Density PDF, t = {t:.4f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        ax.set_xlim(bin_centers_pinn[0], bin_centers_pinn[-1])
        
        # Add resolution info
        ax.text(0.02, 0.98, f'resolution ({Q}, {Q})', transform=ax.transAxes,
               fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if requested
    if save_plots:
        desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
        output_dir = os.path.join(desktop_path, "model test plots")
        os.makedirs(output_dir, exist_ok=True)
        time_str = "_".join([f"{t:.4f}" for t in time_points])
        save_path = os.path.join(output_dir, f"density_pdf_t{time_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved density PDF plot to {save_path}")
    
    plt.show()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots for a saved model')
    parser.add_argument('model_path', type=str, nargs='?', default=None,
                       help='Path to model file (optional: defaults to SNAPSHOT_DIR/GRINN/model.pth)')
    parser.add_argument('time_points', type=str, help='Comma-separated time points (e.g., "0.0,1.0,2.0,3.0")')
    parser.add_argument('--plot-type', type=str, default='spatial', choices=['spatial', 'pdf'],
                       help='Type of plot: spatial (PINN/FD/epsilon comparison) or pdf (density PDF) (default: spatial)')
    parser.add_argument('--which', type=str, default='both', choices=['density', 'velocity', 'both'],
                       help='Which field to plot for spatial plots: density, velocity, or both (default: both)')
    parser.add_argument('--N', type=int, default=None, help='Grid resolution for FD solver (default: N_GRID from config)')
    parser.add_argument('--nu', type=float, default=0.5, help='Courant number for FD solver (default: 0.5)')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots to disk (only display them)')
    parser.add_argument('--no-fit', action='store_true', help='Do not fit distributions to PDF plots')
    parser.add_argument('--powerlaw-threshold', type=float, default=0.8, 
                       help='Minimum log density for power-law fit (default: 0.8)')
    parser.add_argument('--fd-backend', type=str, default='cpu', choices=['cpu', 'gpu', 'torch'],
                       help='FD solver backend: cpu (default) or gpu/torch for GPU-accelerated solver')
    
    args = parser.parse_args()
    
    # Parse time points
    try:
        time_points = np.array([float(t.strip()) for t in args.time_points.split(',')])
    except ValueError:
        print("Error: time_points must be comma-separated numbers")
        sys.exit(1)
    
    # Set up initial parameters (same as train.py) - need to calculate xmax/ymax before loading model
    # Use num_of_waves_config (imported at top level) to avoid scoping issues
    lam, rho_1, num_of_waves, tmax_calc, _, _, _ = input_taker(wave, a, num_of_waves_config, tmax, 0, 0, 0)
    jeans, alpha = req_consts_calc(lam, rho_1)
    
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    
    # Determine model path (use default if not provided)
    if args.model_path is None:
        # Use default path: SNAPSHOT_DIR/GRINN/model.pth
        model_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
        model_path = os.path.join(model_dir, "model.pth")
        print(f"No model path provided, using default: {model_path}")
    else:
        model_path = args.model_path
    
    # Load model
    try:
        net = load_model(model_path, xmax, ymax)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize shared velocity fields for consistent PINN/FD initial conditions
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        v_1 = a * cs
        vx_np, vy_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
        set_shared_velocity_fields(vx_np, vy_np)
    
    initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)
    
    # Generate plots
    save_plots = not args.no_save
    
    # Check if GPU is available when GPU backend is requested
    if args.fd_backend.lower() in ['gpu', 'torch']:
        if not torch.cuda.is_available():
            print(f"Warning: GPU backend requested but CUDA not available. Falling back to CPU.")
            args.fd_backend = 'cpu'
        else:
            print(f"Using GPU-accelerated FD solver (CUDA available)")
    
    if args.plot_type == 'pdf':
        # Density PDF plots
        print(f"\nGenerating density PDF plots...")
        create_density_pdf_plot(
            net, initial_params, time_points, N=args.N, nu=args.nu, save_plots=save_plots,
            fit_lognorm=not args.no_fit, fit_powerlaw_tail=not args.no_fit,
            powerlaw_threshold=args.powerlaw_threshold, fd_backend=args.fd_backend
        )
    else:
        # Spatial comparison plots
        if args.which == 'both':
            print(f"\nGenerating density comparison plots...")
            create_comparison_plots(net, initial_params, time_points, which='density', N=args.N, nu=args.nu, save_plots=save_plots, fd_backend=args.fd_backend)
            print(f"\nGenerating velocity comparison plots...")
            create_comparison_plots(net, initial_params, time_points, which='velocity', N=args.N, nu=args.nu, save_plots=save_plots, fd_backend=args.fd_backend)
        else:
            print(f"\nGenerating {args.which} comparison plots...")
            create_comparison_plots(net, initial_params, time_points, which=args.which, N=args.N, nu=args.nu, save_plots=save_plots, fd_backend=args.fd_backend)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
