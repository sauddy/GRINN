"""
Script to generate comparison plots (PINN vs FD) for a saved model at custom time points.

Usage:
    python compare_model.py <model_path> <time_points>
    
Example:
    python compare_model.py /path/to/model.pth 0.0,1.0,2.0,3.0
    python compare_model.py /path/to/model.pth 0.5,1.5,2.5 --which velocity
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import sys
import argparse

# Import necessary modules
from config import (
    xmin, ymin, tmin, tmax, wave, a, cs, rho_o, harmonics,
    PERTURBATION_TYPE, RANDOM_SEED, N_GRID, POWER_EXPONENT, DIMENSION
)
from model_architecture import PINN
from solver import input_taker, req_consts_calc, initialize_shared_velocity_fields
from Plotting_2D import (
    convert_log_density_to_density, set_shared_velocity_fields
)
import Plotting_2D as plotting_module
from LAX_2D import lax_solution, lax_solution_with_shared_velocity
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


def create_comparison_plots(net, initial_params, time_points, which="density", N=None, nu=0.5, save_plots=True):
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
        print(f"Collecting data for t = {t:.2f}")
        
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
            pinn_field = convert_log_density_to_density(output_00[:, 0].data.cpu().numpy().reshape(Q, Q))
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
        Lx = lam * num_of_waves
        Nx = x_fd.shape[0]
        Ny = rho_fd.shape[1]
        y_fd = np.linspace(0.0, Lx, Ny, endpoint=False)
        
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
        
        ax_pinn.set_title(f"PINN {which.title()}, t={t:.2f}")
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
        
        ax_fd.set_title(f"FD {which.title()}, t={t:.2f}")
        ax_fd.set_xlim(xmin, xmax)
        ax_fd.set_ylim(ymin, ymax)
        cbar_fd = plt.colorbar(pc_fd, ax=ax_fd, shrink=0.6)
        cbar_fd.ax.set_title(r"$\rho$" if which == "density" else r"$|v|$", fontsize=14)
        
        # Column 3: Epsilon Metric
        ax_diff = axes[i, 2]
        pc_diff = ax_diff.pcolormesh(tau, phi, epsilon_metric, shading='auto', cmap='coolwarm')
        ax_diff.set_title(f"ε (%), t={t:.2f}")
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
        time_str = "_".join([f"{t:.2f}" for t in time_points])
        save_path = os.path.join(output_dir, f"{which}_comparison_{num_times}x3_t{time_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description='Generate comparison plots for a saved model')
    parser.add_argument('model_path', type=str, help='Path to model file')
    parser.add_argument('time_points', type=str, help='Comma-separated time points (e.g., "0.0,1.0,2.0,3.0")')
    parser.add_argument('--which', type=str, default='both', choices=['density', 'velocity', 'both'],
                       help='Which field to plot: density, velocity, or both (default: both)')
    parser.add_argument('--N', type=int, default=None, help='Grid resolution for FD solver (default: N_GRID from config)')
    parser.add_argument('--nu', type=float, default=0.5, help='Courant number for FD solver (default: 0.5)')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots to disk (only display them)')
    
    args = parser.parse_args()
    
    # Parse time points
    try:
        time_points = np.array([float(t.strip()) for t in args.time_points.split(',')])
    except ValueError:
        print("Error: time_points must be comma-separated numbers")
        sys.exit(1)
    
    # Set up initial parameters (same as train.py) - need to calculate xmax/ymax before loading model
    lam, rho_1, num_of_waves, tmax_calc, _, _, _ = input_taker(wave, a, 2, tmax, 0, 0, 0)
    jeans, alpha = req_consts_calc(lam, rho_1)
    
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    
    # Load model
    try:
        net = load_model(args.model_path, xmax, ymax)
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
    if args.which == 'both':
        print(f"\nGenerating density comparison plots...")
        create_comparison_plots(net, initial_params, time_points, which='density', N=args.N, nu=args.nu, save_plots=save_plots)
        print(f"\nGenerating velocity comparison plots...")
        create_comparison_plots(net, initial_params, time_points, which='velocity', N=args.N, nu=args.nu, save_plots=save_plots)
    else:
        print(f"\nGenerating {args.which} comparison plots...")
        create_comparison_plots(net, initial_params, time_points, which=args.which, N=args.N, nu=args.nu, save_plots=save_plots)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
