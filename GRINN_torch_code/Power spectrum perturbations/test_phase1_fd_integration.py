"""
Warm-start vs Full FD Comparison Script

This script:
1. Loads a trained PINN model
2. Evaluates the PINN at the warm-start time to extract fields
3. Runs the FD solver forward from that state (warm-start)
4. Runs a full FD simulation from t=0 for reference
5. Generates diagnostic plots comparing the two

Usage:
    # Default: model at default path, warm-start from t=3.0
    python test_phase1_fd_integration.py
    
    # Custom model and warm-start time
    python test_phase1_fd_integration.py --model path/to/model.pth --t_restart 2.0
    
    # See all options
    python test_phase1_fd_integration.py --help
"""

import os
import sys
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import config and modules
from config import (
    xmin, ymin, tmin, tmax, wave, a, cs, rho_o, harmonics, const, G,
    PERTURBATION_TYPE, RANDOM_SEED, N_GRID, POWER_EXPONENT, DIMENSION,
    num_of_waves, FD_N_2D
)
from core.model_architecture import PINN
from numerical_solvers.LAX_2D import lax_solution, fft_solver, lax_solution_warm_start

# Try to import torch version for GPU acceleration
try:
    from numerical_solvers.LAX_2D_torch import lax_solution_warm_start_torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: Torch solver not available, using CPU version")
from core.initial_conditions import initialize_shared_velocity_fields

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

if device.startswith('cuda'):
    torch.cuda.empty_cache()

# Default configuration (can be overridden by command-line arguments)
DEFAULT_MODEL_PATH = r"C:\Users\tirth\OneDrive\Desktop\models\power_spectrum_strong_collapse.pth"
DEFAULT_T_RESTART = 3.0  # Time at which to extract PINN state
T_END = 4.0      # Final time for FD run
FD_N = 400       # FD grid resolution (can be lower than N_GRID for speed)
FD_COURANT = 0.25  # Courant number (nu) for FD solver stability
POINTS_PER_TIME = 1000  # Number of anchor points per time slice


def load_pinn_model(model_path):
    """
    Load trained PINN model.
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        net: Loaded PINN model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Calculate domain extents
    lam = wave
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    
    # Create and load model
    net = PINN(n_harmonics=harmonics, num_neurons=64, num_layers=5)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=DIMENSION)
    net = net.to(device)
    net.eval()
    
    print(f"[OK] Loaded PINN model from {model_path}")
    print(f"  Domain: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
    print(f"  Model parameters: {sum(p.numel() for p in net.parameters())} total")
    
    return net


def evaluate_pinn_at_time(net, t_eval, N_grid=None):
    """
    Evaluate PINN at a specific time on a grid.
    
    Args:
        net: PINN model
        t_eval: Time to evaluate at
        N_grid: Grid resolution (defaults to N_GRID from config)
        
    Returns:
        x, y, rho, vx, vy, phi: Grid coordinates and fields
    """
    if N_grid is None:
        N_grid = N_GRID
    
    # Calculate domain extents
    lam = wave
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    
    # Create evaluation grid
    x = np.linspace(xmin, xmax, N_grid, endpoint=False)
    y = np.linspace(ymin, ymax, N_grid, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Flatten for network evaluation
    x_flat = X.flatten()
    y_flat = Y.flatten()
    t_flat = np.full_like(x_flat, t_eval)
    
    # Convert to tensors
    x_tensor = Variable(torch.from_numpy(x_flat).float(), requires_grad=False).to(device).unsqueeze(-1)
    y_tensor = Variable(torch.from_numpy(y_flat).float(), requires_grad=False).to(device).unsqueeze(-1)
    t_tensor = Variable(torch.from_numpy(t_flat).float(), requires_grad=False).to(device).unsqueeze(-1)
    
    # Evaluate network
    with torch.no_grad():
        output = net([x_tensor, y_tensor, t_tensor])
        
        rho = output[:, 0].cpu().numpy().reshape(N_grid, N_grid)
        vx = output[:, 1].cpu().numpy().reshape(N_grid, N_grid)
        vy = output[:, 2].cpu().numpy().reshape(N_grid, N_grid)
        phi = output[:, 3].cpu().numpy().reshape(N_grid, N_grid)
    
    print(f"[OK] Evaluated PINN at t={t_eval:.3f}")
    print(f"  Density range: [{np.min(rho):.4f}, {np.max(rho):.4f}]")
    print(f"  Velocity magnitude range: [{np.min(np.sqrt(vx**2 + vy**2)):.4f}, {np.max(np.sqrt(vx**2 + vy**2)):.4f}]")
    
    return x, y, rho, vx, vy, phi


def run_fd_from_custom_ic(rho_ic, vx_ic, vy_ic, x_grid, y_grid, t_start, t_end, nu=FD_COURANT, save_times=None):
    """
    Run FD solver from custom initial conditions using warm-start function.
    
    Args:
        rho_ic: Initial density field (Nx, Ny)
        vx_ic: Initial x-velocity field (Nx, Ny)
        vy_ic: Initial y-velocity field (Nx, Ny)
        x_grid: x coordinates (Nx,)
        y_grid: y coordinates (Ny,)
        t_start: Starting time
        t_end: Ending time
        nu: Courant number
        save_times: List of times to save snapshots
        
    Returns:
        Dictionary with time series data: {time: (rho, vx, vy, phi, x, y)}
    """
    if save_times is None:
        save_times = [t_end]
    
    print(f"\n   Running FD warm-start from t={t_start} to t={t_end}")
    print(f"   Grid: {len(x_grid)} x {len(y_grid)}")
    print(f"   Saving snapshots at: {save_times}")
    
    # Use torch version if available and GPU is present, otherwise use CPU version
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print(f"   Using GPU-accelerated torch solver")
        snapshots = lax_solution_warm_start_torch(
            rho_ic, vx_ic, vy_ic, x_grid, y_grid,
            t_start=t_start, t_end=t_end, nu=nu,
            save_times=save_times, gravity=True
        )
    else:
        print(f"   Using CPU numpy solver")
        snapshots = lax_solution_warm_start(
            rho_ic, vx_ic, vy_ic, x_grid, y_grid,
            t_start=t_start, t_end=t_end, nu=nu,
            save_times=save_times, gravity=True
        )
    
    print(f"   [OK] FD run completed. Saved {len(snapshots)} snapshots")
    
    return snapshots


def create_comparison_plots(rho_warm, vx_warm, vy_warm, x_warm, y_warm,
                            rho_full, vx_full, vy_full, x_full, y_full,
                            t_eval, output_dir):
    """
    Create comparison plots between warm-start FD and full FD solutions.
    
    Args:
        rho_warm, vx_warm, vy_warm: Warm-start FD fields
        x_warm, y_warm: Warm-start coordinates
        rho_full, vx_full, vy_full: Full FD fields
        x_full, y_full: Full FD coordinates
        t_eval: Time of comparison
        output_dir: Directory to save plots
    """
    # Calculate epsilon metric (same as Plotting_2D.py)
    # For density: ε = 200 * |warm - full| / (warm + full + eps)
    eps = 1e-6
    rho_epsilon = 200.0 * np.abs(rho_warm - rho_full) / (rho_warm + rho_full + eps)
    
    # For velocity magnitude: ε = 200 * |warm - full| / (warm + full + 2.0)
    v_mag_warm = np.sqrt(vx_warm**2 + vy_warm**2)
    v_mag_full = np.sqrt(vx_full**2 + vy_full**2)
    v_mag_epsilon = 200.0 * np.abs(v_mag_warm - v_mag_full) / (v_mag_warm + v_mag_full + 2.0)
    
    # Create meshgrids
    X_warm, Y_warm = np.meshgrid(x_warm, y_warm, indexing='ij')
    X_full, Y_full = np.meshgrid(x_full, y_full, indexing='ij')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Density comparison
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.contourf(X_warm, Y_warm, rho_warm, levels=20, cmap='viridis')
    ax1.set_title(f'Warm-start FD: Density at t={t_eval:.1f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.contourf(X_full, Y_full, rho_full, levels=20, cmap='viridis')
    ax2.set_title(f'Full FD: Density at t={t_eval:.1f}')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(3, 3, 3)
    # Interpolate full FD to warm-start grid for epsilon calculation
    from scipy.interpolate import RegularGridInterpolator
    rho_full_interp = RegularGridInterpolator((x_full, y_full), rho_full, method='linear')
    points_warm = np.stack([X_warm.flatten(), Y_warm.flatten()], axis=1)
    rho_full_on_warm = rho_full_interp(points_warm).reshape(rho_warm.shape)
    
    # Calculate epsilon metric (same as Plotting_2D.py)
    eps = 1e-6
    rho_epsilon_aligned = 200.0 * np.abs(rho_warm - rho_full_on_warm) / (rho_warm + rho_full_on_warm + eps)
    
    im3 = ax3.contourf(X_warm, Y_warm, rho_epsilon_aligned, levels=20, cmap='coolwarm')
    ax3.set_title(f'ε (%), t={t_eval:.1f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    cbar3 = plt.colorbar(im3, ax=ax3)
    cbar3.ax.set_title("ε (%)", fontsize=12)
    
    # Row 2: Velocity magnitude comparison
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.contourf(X_warm, Y_warm, v_mag_warm, levels=20, cmap='plasma')
    ax4.set_title(f'Warm-start FD: |v| at t={t_eval:.1f}')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4)
    
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.contourf(X_full, Y_full, v_mag_full, levels=20, cmap='plasma')
    ax5.set_title(f'Full FD: |v| at t={t_eval:.1f}')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5)
    
    v_mag_full_interp = RegularGridInterpolator((x_full, y_full), v_mag_full, method='linear')
    v_mag_full_on_warm = v_mag_full_interp(points_warm).reshape(v_mag_warm.shape)
    
    # Calculate epsilon metric for velocity (same as Plotting_2D.py)
    v_mag_epsilon_aligned = 200.0 * np.abs(v_mag_warm - v_mag_full_on_warm) / (v_mag_warm + v_mag_full_on_warm + 2.0)
    
    ax6 = plt.subplot(3, 3, 6)
    im6 = ax6.contourf(X_warm, Y_warm, v_mag_epsilon_aligned, levels=20, cmap='coolwarm')
    ax6.set_title(f'ε (%), t={t_eval:.1f}')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    cbar6 = plt.colorbar(im6, ax=ax6)
    cbar6.ax.set_title("ε (%)", fontsize=12)
    
    # Row 3: 1D slices through collapse region
    # Find collapse location (max density)
    i_max_warm, j_max_warm = np.unravel_index(np.argmax(rho_warm), rho_warm.shape)
    i_max_full, j_max_full = np.unravel_index(np.argmax(rho_full), rho_full.shape)
    
    # X-slice at y where max occurs
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(x_warm, rho_warm[:, j_max_warm], 'b-', label='Warm-start', linewidth=2)
    # Interpolate full FD to warm-start x coordinates
    rho_full_slice = np.array([rho_full_interp([xi, y_warm[j_max_warm]])[0] for xi in x_warm])
    ax7.plot(x_warm, rho_full_slice, 'r--', label='Full FD', linewidth=2)
    ax7.set_xlabel('x')
    ax7.set_ylabel('Density')
    ax7.set_title(f'X-slice at y={y_warm[j_max_warm]:.2f}')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Y-slice at x where max occurs
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(y_warm, rho_warm[i_max_warm, :], 'b-', label='Warm-start', linewidth=2)
    rho_full_slice_y = np.array([rho_full_interp([x_warm[i_max_warm], yi])[0] for yi in y_warm])
    ax8.plot(y_warm, rho_full_slice_y, 'r--', label='Full FD', linewidth=2)
    ax8.set_xlabel('y')
    ax8.set_ylabel('Density')
    ax8.set_title(f'Y-slice at x={x_warm[i_max_warm]:.2f}')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Max density evolution (if we had time series, but for now just show single point)
    ax9 = plt.subplot(3, 3, 9)
    ax9.bar(['Warm-start', 'Full FD'], 
            [np.max(rho_warm), np.max(rho_full)],
            color=['blue', 'red'], alpha=0.7)
    ax9.set_ylabel('Max Density')
    ax9.set_title(f'Max Density at t={t_eval:.1f}')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    comparison_file = os.path.join(output_dir, f"fd_comparison_t{t_eval:.1f}.png")
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   [OK] Comparison plot saved to {comparison_file}")


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Phase 1: Generate FD Data from PINN State',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default model and t=3.0
  python test_phase1_fd_integration.py
  
  # Specify custom model path
  python test_phase1_fd_integration.py --model path/to/model.pth
  
  # Specify custom warm-start time
  python test_phase1_fd_integration.py --t_restart 2.0
  
  # Both custom
  python test_phase1_fd_integration.py --model model_t2.pth --t_restart 2.0
        """
    )
    
    parser.add_argument(
        '--model', '--model_path', '-m',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f'Path to trained PINN model (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--t_restart', '--t_start', '-t',
        type=float,
        default=DEFAULT_T_RESTART,
        help=f'Time at which to extract PINN state for warm-start (default: {DEFAULT_T_RESTART})'
    )
    
    parser.add_argument(
        '--t_end',
        type=float,
        default=T_END,
        help=f'Final time for FD run (default: {T_END})'
    )
    
    parser.add_argument(
        '--fd_n',
        type=int,
        default=FD_N,
        help=f'FD grid resolution (default: {FD_N})'
    )
    
    parser.add_argument(
        '--courant', '--nu',
        type=float,
        default=FD_COURANT,
        help=f'Courant number for FD solver stability (default: {FD_COURANT})'
    )
    
    return parser.parse_args()


def main():
    """
    Main test function for Phase 1.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set configuration from arguments
    MODEL_PATH = args.model
    T_RESTART = args.t_restart
    T_END = args.t_end
    FD_N = args.fd_n
    COURANT = args.courant

    # Snapshots to store from warm-start FD run (only final state needed for comparison)
    SAVE_TIMES = [float(T_END)]

    print("=" * 70)
    print("Warm-Start vs Full FD Comparison")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Warm-start time: t={T_RESTART}")
    print(f"  Final time: t={T_END}")
    print(f"  FD grid resolution: {FD_N}x{FD_N}")
    print(f"  Courant number: {COURANT}")
    print(f"  Snapshot times: {[f'{t:.2f}' for t in SAVE_TIMES]}")
    print("=" * 70)
    
    # Step 1: Load PINN model
    print("\n[Step 1] Loading PINN model...")
    try:
        net = load_pinn_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return
    
    # Step 2: Evaluate PINN at t=3
    print("\n[Step 2] Evaluating PINN at t=3...")
    try:
        x_grid, y_grid, rho_t3, vx_t3, vy_t3, phi_t3 = evaluate_pinn_at_time(net, T_RESTART)
    except Exception as e:
        print(f"[ERROR] Error evaluating PINN: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Prepare for FD warm-start
    print("\n[Step 3] Preparing FD warm-start...")
    print(f"   PINN state extracted at t={T_RESTART}")
    print(f"   Will run FD from t={T_RESTART} to t={T_END}")
    print(f"   Grid resolution: {len(x_grid)} x {len(y_grid)}")
    
    # Interpolate PINN state to FD grid if needed
    if len(x_grid) != FD_N:
        print(f"   Interpolating PINN state from {len(x_grid)}x{len(y_grid)} to {FD_N}x{FD_N}...")
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolators
        rho_interp = RegularGridInterpolator((x_grid, y_grid), rho_t3, method='linear')
        vx_interp = RegularGridInterpolator((x_grid, y_grid), vx_t3, method='linear')
        vy_interp = RegularGridInterpolator((x_grid, y_grid), vy_t3, method='linear')
        
        # Create FD grid
        lam = wave
        xmax = xmin + lam * num_of_waves
        ymax = ymin + lam * num_of_waves
        x_fd = np.linspace(xmin, xmax, FD_N, endpoint=False)
        y_fd = np.linspace(ymin, ymax, FD_N, endpoint=False)
        X_fd, Y_fd = np.meshgrid(x_fd, y_fd, indexing='ij')
        
        # Interpolate
        points = np.stack([X_fd.flatten(), Y_fd.flatten()], axis=1)
        rho_ic = rho_interp(points).reshape(FD_N, FD_N)
        vx_ic = vx_interp(points).reshape(FD_N, FD_N)
        vy_ic = vy_interp(points).reshape(FD_N, FD_N)
        
        x_grid = x_fd
        y_grid = y_fd
    else:
        rho_ic = rho_t3
        vx_ic = vx_t3
        vy_ic = vy_t3
    
    print(f"   [OK] Initial conditions prepared on {FD_N}x{FD_N} grid")
    
    # Step 4: Run FD from custom IC
    print("\n[Step 4] Running FD solver from custom IC...")
    fd_snapshots = run_fd_from_custom_ic(
        rho_ic, vx_ic, vy_ic, x_grid, y_grid, T_RESTART, T_END,
        nu=COURANT,
        save_times=SAVE_TIMES
    )
    
    # Step 5: Compare with full FD run
    print("\n[Step 5] Comparing warm-start FD with full FD run...")
    output_dir = "warm_start_vs_full"
    os.makedirs(output_dir, exist_ok=True)

    print(f"   Running full FD from t=0 to t={T_END} for comparison...")
    
    # Run full FD from t=0 to t=4
    lam = wave
    num_of_waves_val = num_of_waves
    rho_1_val = a * rho_o  # Density perturbation amplitude
    
    # Use the same parameters as PINN training
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        # Generate shared velocity fields for consistency
        from numerical_solvers.LAX_2D import generate_shared_velocity_field
        Lx = lam * num_of_waves_val
        Ly = lam * num_of_waves_val
        vx0_shared, vy0_shared, _, _ = generate_shared_velocity_field(
            FD_N, FD_N, Lx, Ly,
            power_index=POWER_EXPONENT,
            amplitude=a*cs,
            random_seed=RANDOM_SEED
        )
        
        x_full, rho_full, vx_full, vy_full, phi_full, n_full, rho_max_full = lax_solution(
            time=T_END, N=FD_N, nu=COURANT, lam=lam, num_of_waves=num_of_waves_val,
            rho_1=rho_1_val, gravity=True, use_velocity_ps=True,
            ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED,
            vx0_shared=vx0_shared, vy0_shared=vy0_shared
        )
    else:
        x_full, rho_full, vx_full, vy_full, phi_full, n_full, rho_max_full = lax_solution(
            time=T_END, N=FD_N, nu=COURANT, lam=lam, num_of_waves=num_of_waves_val,
            rho_1=rho_1_val, gravity=True, use_velocity_ps=False
        )
    
    # Build y grid for full FD
    Lx = lam * num_of_waves_val
    y_full = np.linspace(0, Lx, FD_N, endpoint=False)
    
    print(f"   [OK] Full FD run completed (t=0 to t={T_END})")
    print(f"   Full FD max density at t={T_END}: {rho_max_full:.4f}")
    
    # Compare at t=4.0
    comparison_results = None
    if T_END in fd_snapshots:
        rho_warm, vx_warm, vy_warm, phi_warm, x_warm, y_warm = fd_snapshots[T_END]
        
        # Calculate comparison metrics using epsilon (same as Plotting_2D.py)
        # Interpolate full FD to warm-start grid for proper comparison
        from scipy.interpolate import RegularGridInterpolator
        rho_full_interp = RegularGridInterpolator((x_full, y_full), rho_full, method='linear')
        X_warm, Y_warm = np.meshgrid(x_warm, y_warm, indexing='ij')
        points_warm = np.stack([X_warm.flatten(), Y_warm.flatten()], axis=1)
        rho_full_on_warm = rho_full_interp(points_warm).reshape(rho_warm.shape)
        
        # Epsilon metric for density: ε = 200 * |warm - full| / (warm + full + eps)
        eps = 1e-6
        rho_epsilon = 200.0 * np.abs(rho_warm - rho_full_on_warm) / (rho_warm + rho_full_on_warm + eps)
        
        rho_max_warm = np.max(rho_warm)
        rho_max_full_val = np.max(rho_full)
        
        # Statistics
        rho_epsilon_mean = np.mean(rho_epsilon)
        rho_epsilon_max = np.max(rho_epsilon)
        rho_max_error = abs(rho_max_warm - rho_max_full_val) / rho_max_full_val * 100
        
        # Store results for summary
        comparison_results = {
            'rho_max_warm': rho_max_warm,
            'rho_max_full': rho_max_full_val,
            'rho_max_error': rho_max_error,
            'rho_epsilon_mean': rho_epsilon_mean,
            'rho_epsilon_max': rho_epsilon_max
        }
        
        print(f"\n   Comparison at t={T_END}:")
        print(f"   Warm-start max density: {rho_max_warm:.4f}")
        print(f"   Full FD max density:    {rho_max_full_val:.4f}")
        print(f"   Max density error:      {rho_max_error:.2f}%")
        print(f"   Mean epsilon (ε):       {rho_epsilon_mean:.2f}%")
        print(f"   Max epsilon (ε):        {rho_epsilon_max:.2f}%")
        
        # Create comparison plots
        print(f"\n   Creating comparison plots...")
        create_comparison_plots(
            rho_warm, vx_warm, vy_warm, x_warm, y_warm,
            rho_full, vx_full, vy_full, x_full, y_full,
            T_END, output_dir
        )
        print(f"   [OK] Comparison plots saved to {output_dir}/")
    
    # Summary
    print("\n" + "=" * 70)
    print("Warm-Start vs Full FD Summary")
    print("=" * 70)
    print(f"[OK] PINN model loaded and evaluated at t={T_RESTART}")
    print(f"[OK] Initial conditions prepared for FD warm-start")
    print(f"[OK] FD warm-start solver executed from t={T_RESTART} to t={T_END}")
    print(f"[OK] Full FD reference solution generated")

    # Add comparison summary if validation was done
    if comparison_results is not None:
        print(f"\nValidation Results:")
        print(f"  Warm-start max density at t={T_END}: {comparison_results['rho_max_warm']:.4f}")
        print(f"  Full FD max density at t={T_END}:    {comparison_results['rho_max_full']:.4f}")
        print(f"  Max density error:                    {comparison_results['rho_max_error']:.2f}%")
        print(f"  Mean epsilon (ε):                     {comparison_results['rho_epsilon_mean']:.2f}%")
        print(f"  Max epsilon (ε):                      {comparison_results['rho_epsilon_max']:.2f}%")

    print(f"\nNext steps:")
    print(f"  Review comparison plot: {output_dir}/fd_comparison_t{T_END:.1f}.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

