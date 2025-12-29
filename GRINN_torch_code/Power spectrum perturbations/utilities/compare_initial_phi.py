r"""
Script to compare initial phi (potential) fields between PINN and FD solver.
Shows PINN phi, FD phi, and epsilon metric at t=0.

Usage:
    python compare_initial_phi.py <model_path>
    
Example:
    python compare_initial_phi.py "C:\Users\tirth\Downloads\exponential.pth"
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    xmin, ymin, tmin, tmax, wave, a, cs, rho_o, const, G,
    PERTURBATION_TYPE, RANDOM_SEED, N_GRID, POWER_EXPONENT, DIMENSION, harmonics
)
from config import num_of_waves as num_of_waves_config
from core.model_architecture import PINN
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import initialize_shared_velocity_fields, fun_rho_0
from numerical_solvers.LAX import lax_solution, lax_solution_with_shared_velocity, fft_solver
import visualization.Plotting_2D as plotting_module

# Device setup
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"
if device.startswith('cuda'):
    torch.cuda.empty_cache()

print(f"Using device: {device}")


def load_model(model_path, xmax, ymax):
    """Load a saved PINN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    net = PINN(n_harmonics=harmonics)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=DIMENSION)
    net = net.to(device)
    net.eval()
    print(f"Loaded PINN model from {model_path}")
    return net


def compute_phi_from_rho(rho, Lx, Ly, Nx, Ny):
    """
    Compute potential phi from density using FFT-based Poisson solver.
    phi_xx + phi_yy = const * (rho - rho_o)
    """
    dx = Lx / Nx
    dy = Ly / Ny
    rho_perturbation = rho - rho_o
    phi = fft_solver(rho_perturbation, Lx, Nx, Ly, Ny, dim=2)
    return phi


def create_initial_phi_comparison(net, model_path, N=400, nu=0.5):
    """
    Create comparison plots of PINN phi vs FD phi at t=0.
    
    Args:
        net: Trained PINN model
        model_path: Path to model (for reference in plots)
        N: Grid resolution for spatial plots
        nu: Courant number for FD solver
    """
    # Setup parameters
    lam, rho_1, num_of_waves, tmax_calc, _, _, _ = input_taker(
        wave, a, num_of_waves_config, tmax, 0, 0, 0
    )
    jeans, alpha = req_consts_calc(lam, rho_1)
    
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    
    print(f"\nSetup parameters:")
    print(f"  Domain: x in [{xmin}, {xmax}], y in [{ymin}, {ymax}]")
    print(f"  Lambda: {lam:.4f}, num_of_waves: {num_of_waves:.2f}")
    print(f"  Rho_1: {rho_1:.6f}, Jeans length: {jeans:.4f}")
    print(f"  Grid resolution: {N}x{N}")
    
    # Initialize shared velocity fields if power spectrum
    shared_vx = None
    shared_vy = None
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        v_1 = a * cs
        shared_vx, shared_vy = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
        plotting_module.set_shared_velocity_fields(shared_vx, shared_vy)
        print(f"  Initialized shared velocity fields for power spectrum")
    
    # Get FD solution at t=0
    print(f"\nComputing FD solution at t=0...")
    if str(PERTURBATION_TYPE).lower() == "power_spectrum" and shared_vx is not None:
        x_fd, rho_fd, vx_fd, vy_fd, phi_fd, _n, _rho_max = lax_solution_with_shared_velocity(
            0.0, N, nu, lam, num_of_waves, rho_1, shared_vx, shared_vy,
            gravity=True, isplot=False, comparison=False, animation=False
        )
    else:
        x_fd, rho_fd, vx_fd, vy_fd, phi_fd, _n, _rho_max = lax_solution(
            0.0, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False,
            comparison=False, animation=False, use_velocity_ps=True,
            ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
        )
    
    x_coords = np.asarray(x_fd) + xmin
    y_extent = lam * num_of_waves
    y_coords = np.linspace(ymin, ymin + y_extent, np.asarray(rho_fd).shape[1], endpoint=False)
    rho_fd = np.asarray(rho_fd)
    phi_fd = np.asarray(phi_fd)
    
    print(f"  FD rho shape: {rho_fd.shape}, min={np.min(rho_fd):.6f}, max={np.max(rho_fd):.6f}")
    print(f"  FD phi shape: {phi_fd.shape}, min={np.min(phi_fd):.6f}, max={np.max(phi_fd):.6f}")
    
    # Get PINN predictions at t=0
    print(f"\nComputing PINN predictions at t=0...")
    Q = N
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.vstack([XX.flatten(), YY.flatten()]).T
    t_array = np.zeros((Q**2, 1))
    
    pt_x = Variable(torch.from_numpy(grid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(grid[:, 1:2]).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_array).float(), requires_grad=True).to(device)
    
    with torch.no_grad():
        net_output = net([pt_x, pt_y, pt_t])
    
    rho_pinn = net_output[:, 0].data.cpu().numpy().reshape(Q, Q)
    vx_pinn = net_output[:, 1].data.cpu().numpy().reshape(Q, Q)
    vy_pinn = net_output[:, 2].data.cpu().numpy().reshape(Q, Q)
    phi_pinn = net_output[:, 3].data.cpu().numpy().reshape(Q, Q)
    
    print(f"  PINN rho shape: {rho_pinn.shape}, min={np.min(rho_pinn):.6f}, max={np.max(rho_pinn):.6f}")
    print(f"  PINN phi shape: {phi_pinn.shape}, min={np.min(phi_pinn):.6f}, max={np.max(phi_pinn):.6f}")
    
    # Interpolate FD data to PINN grid
    from scipy.interpolate import RegularGridInterpolator
    
    points_pinn = np.column_stack([XX.ravel(), YY.ravel()])
    
    phi_interpolator = RegularGridInterpolator(
        (x_coords, y_coords), phi_fd,
        method='linear', bounds_error=False, fill_value=None
    )
    phi_fd_interp = phi_interpolator(points_pinn).reshape(Q, Q)
    
    rho_interpolator = RegularGridInterpolator(
        (x_coords, y_coords), rho_fd,
        method='linear', bounds_error=False, fill_value=None
    )
    rho_fd_interp = rho_interpolator(points_pinn).reshape(Q, Q)
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(f"Initial Potential φ Comparison (t=0)\nModel: {os.path.basename(model_path)}", 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Density
    # Column 1: PINN density
    ax = axes[0, 0]
    im = ax.pcolormesh(XX, YY, rho_pinn, shading='auto', cmap='YlOrBr')
    ax.set_title("PINN ρ(t=0)", fontsize=12)
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="ρ")
    
    # Column 2: FD density
    ax = axes[0, 1]
    im = ax.pcolormesh(XX, YY, rho_fd_interp, shading='auto', cmap='YlOrBr')
    ax.set_title("FD ρ(t=0)", fontsize=12)
    plt.colorbar(im, ax=ax, label="ρ")
    
    # Column 3: Density epsilon
    ax = axes[0, 2]
    eps_rho = 200.0 * np.abs(rho_pinn - rho_fd_interp) / (rho_pinn + rho_fd_interp + 1e-6)
    im = ax.pcolormesh(XX, YY, eps_rho, shading='auto', cmap='coolwarm', vmin=0, vmax=50)
    ax.set_title("ε(ρ) (%)", fontsize=12)
    plt.colorbar(im, ax=ax, label="ε (%)")
    ax.text(0.02, 0.98, f"mean={np.mean(eps_rho):.2f}%\nmax={np.max(eps_rho):.2f}%", 
            transform=ax.transAxes, verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Row 2: Potential φ
    # Column 1: PINN phi
    ax = axes[1, 0]
    im = ax.pcolormesh(XX, YY, phi_pinn, shading='auto', cmap='RdBu_r')
    ax.set_title("PINN φ(t=0)", fontsize=12)
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="φ")
    
    # Column 2: FD phi
    ax = axes[1, 1]
    im = ax.pcolormesh(XX, YY, phi_fd_interp, shading='auto', cmap='RdBu_r')
    ax.set_title("FD φ(t=0)", fontsize=12)
    plt.colorbar(im, ax=ax, label="φ")
    
    # Column 3: Phi epsilon
    ax = axes[1, 2]
    # Avoid division by zero in epsilon metric
    phi_max = np.maximum(np.abs(phi_pinn), np.abs(phi_fd_interp)).max()
    if phi_max > 1e-10:
        eps_phi = 200.0 * np.abs(phi_pinn - phi_fd_interp) / (np.abs(phi_pinn) + np.abs(phi_fd_interp) + 1e-6)
    else:
        eps_phi = np.zeros_like(phi_pinn)
    
    im = ax.pcolormesh(XX, YY, eps_phi, shading='auto', cmap='coolwarm', vmin=0, vmax=100)
    ax.set_title("ε(φ) (%)", fontsize=12)
    plt.colorbar(im, ax=ax, label="ε (%)")
    valid_mask = phi_max > 1e-10
    if valid_mask:
        mean_eps = np.nanmean(eps_phi[eps_phi < np.inf])
        max_eps = np.nanmax(eps_phi[eps_phi < np.inf])
        ax.text(0.02, 0.98, f"mean={mean_eps:.2f}%\nmax={max_eps:.2f}%", 
                transform=ax.transAxes, verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Row 3: Difference φ_PINN - φ_FD
    # Column 1: Absolute difference
    ax = axes[2, 0]
    diff_phi = phi_pinn - phi_fd_interp
    im = ax.pcolormesh(XX, YY, diff_phi, shading='auto', cmap='seismic')
    ax.set_title("φ_PINN - φ_FD", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, label="Δφ")
    
    # Column 2: Relative difference
    ax = axes[2, 1]
    rel_diff = np.abs(diff_phi) / (np.abs(phi_fd_interp) + 1e-6)
    im = ax.pcolormesh(XX, YY, rel_diff, shading='auto', cmap='Purples')
    ax.set_title("|φ_PINN - φ_FD| / |φ_FD|", fontsize=12)
    ax.set_xlabel("x")
    plt.colorbar(im, ax=ax, label="rel diff")
    
    # Column 3: Statistics
    ax = axes[2, 2]
    ax.axis('off')
    
    stats_text = f"""
Initial Condition Statistics (t=0):

PINN ρ:
  min={np.min(rho_pinn):.6f}, max={np.max(rho_pinn):.6f}
  mean={np.mean(rho_pinn):.6f}
  
FD ρ:
  min={np.min(rho_fd_interp):.6f}, max={np.max(rho_fd_interp):.6f}
  mean={np.mean(rho_fd_interp):.6f}
  
PINN φ:
  min={np.min(phi_pinn):.6e}, max={np.max(phi_pinn):.6e}
  mean={np.mean(phi_pinn):.6e}
  std={np.std(phi_pinn):.6e}
  
FD φ:
  min={np.min(phi_fd_interp):.6e}, max={np.max(phi_fd_interp):.6e}
  mean={np.mean(phi_fd_interp):.6e}
  std={np.std(phi_fd_interp):.6e}
  
φ Difference:
  |Δφ| mean={np.mean(np.abs(diff_phi)):.6e}
  |Δφ| max={np.max(np.abs(diff_phi)):.6e}
  
Poisson Consistency:
  ∇²φ should satisfy: ∇²φ = const*(ρ-ρ₀)
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontfamily='monospace', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save the figure
    desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
    output_dir = os.path.join(desktop_path, "model test plots")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "initial_phi_comparison_t0.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] Saved initial phi comparison plot to {save_path}")
    
    plt.show()
    
    return fig, axes


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare initial phi fields between PINN and FD')
    parser.add_argument('model_path', type=str, help='Path to trained PINN model')
    parser.add_argument('--N', type=int, default=400, help='Grid resolution for comparison (default: 400)')
    parser.add_argument('--nu', type=float, default=0.5, help='Courant number for FD solver (default: 0.5)')
    
    args = parser.parse_args()
    
    # Setup parameters to get domain size
    lam, rho_1, num_of_waves, tmax_calc, _, _, _ = input_taker(
        wave, a, num_of_waves_config, tmax, 0, 0, 0
    )
    jeans, alpha = req_consts_calc(lam, rho_1)
    
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    
    # Load model
    try:
        net = load_model(args.model_path, xmax, ymax)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Create comparison plots
    try:
        fig, axes = create_initial_phi_comparison(net, args.model_path, N=args.N, nu=args.nu)
        print("\n[*] Comparison complete!")
    except Exception as e:
        print(f"\n[ERROR] Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

