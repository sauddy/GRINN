"""
Script to diagnose FD solution quality for PINN-FD comparisons.
Helps identify problematic regions and provides recommendations.

Usage:
    python diagnose_comparison.py [model_path] [time] [--N-fd N] [--nu NU] [--fd-backend BACKEND]

Example:
    python diagnose_comparison.py model.pth 3.0 --N-fd 800 --nu 0.25 --fd-backend gpu
"""

import numpy as np
import torch
import sys
import os
import argparse
from torch.autograd import Variable

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    xmin, ymin, tmin, tmax, wave, a, cs, rho_o, harmonics, const, G,
    PERTURBATION_TYPE, RANDOM_SEED, N_GRID, POWER_EXPONENT, DIMENSION
)
from config import num_of_waves as num_of_waves_config
from core.model_architecture import PINN
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import initialize_shared_velocity_fields
from utilities.compare_model import FDSolutionManager, load_model
from utilities.fd_diagnostics import diagnose_fd_solution, plot_diagnostic_masks
from visualization.Plotting_2D import set_shared_velocity_fields

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description='Diagnose FD solution quality')
    parser.add_argument('model_path', type=str, help='Path to model file')
    parser.add_argument('time', type=float, help='Time point to analyze')
    parser.add_argument('--N-fd', type=int, default=None, help='Grid resolution for FD solver')
    parser.add_argument('--nu', type=float, default=0.5, help='Courant number')
    parser.add_argument('--fd-backend', type=str, default='cpu', choices=['cpu', 'gpu', 'torch'],
                       help='FD solver backend')
    parser.add_argument('--save-plot', action='store_true', help='Save diagnostic plot')
    
    args = parser.parse_args()
    
    # Set up parameters
    lam, rho_1, num_of_waves, tmax_calc, _, _, _ = input_taker(wave, a, num_of_waves_config, tmax, 0, 0, 0)
    jeans, alpha = req_consts_calc(lam, rho_1)
    
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    zmax = None
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    net = load_model(args.model_path, xmax, ymax, zmax=zmax)
    
    # Initialize shared velocity fields
    shared_vx_np = None
    shared_vy_np = None
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        v_1 = a * cs
        shared_vx_np, shared_vy_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
        set_shared_velocity_fields(shared_vx_np, shared_vy_np)
    
    initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)
    fd_cache = FDSolutionManager(initial_params, lam, num_of_waves, rho_1, PERTURBATION_TYPE, 
                                 shared_vx_np, shared_vy_np)
    
    # Determine N
    N_fd = args.N_fd if args.N_fd is not None else N_GRID
    
    print(f"\nGenerating FD solution at t={args.time} with N={N_fd}, nu={args.nu}, backend={args.fd_backend}")
    
    # Get FD solution
    fd_solution = fd_cache.get_solution(args.time, N=N_fd, nu=args.nu, backend=args.fd_backend)
    
    x_fd = fd_solution["x"]
    y_fd = fd_solution["y"]
    rho_fd = fd_solution["rho"]
    vx_fd = fd_solution["vx"]
    vy_fd = fd_solution["vy"]
    
    print(f"FD solution shape: {rho_fd.shape}")
    
    # Get PINN solution on same grid
    print("Evaluating PINN on FD grid...")
    Q = rho_fd.shape[0]
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    tau, phi = np.meshgrid(xs, ys)
    Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
    t_00 = args.time * np.ones(Q**2).reshape(Q**2, 1)
    
    pt_x = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
    pt_t = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    
    output_00 = net([pt_x, pt_y, pt_t])
    rho_pinn = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
    
    # Interpolate FD to PINN grid if needed
    if rho_fd.shape != rho_pinn.shape:
        from scipy.interpolate import RegularGridInterpolator
        points_pinn = np.column_stack([tau.ravel(), phi.ravel()])
        interpolator = RegularGridInterpolator(
            (x_fd, y_fd), rho_fd,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        rho_fd_interp = interpolator(points_pinn).reshape(Q, Q)
        
        vx_interpolator = RegularGridInterpolator((x_fd, y_fd), vx_fd, method='linear',
                                                  bounds_error=False, fill_value=None)
        vy_interpolator = RegularGridInterpolator((x_fd, y_fd), vy_fd, method='linear',
                                                  bounds_error=False, fill_value=None)
        vx_fd_interp = vx_interpolator(points_pinn).reshape(Q, Q)
        vy_fd_interp = vy_interpolator(points_pinn).reshape(Q, Q)
    else:
        rho_fd_interp = rho_fd
        vx_fd_interp = vx_fd
        vy_fd_interp = vy_fd
    
    # Run diagnostics
    diagnostics = diagnose_fd_solution(
        rho_fd_interp, vx_fd_interp, vy_fd_interp, rho_pinn, 
        rho_o=rho_o, verbose=True
    )
    
    # Compute error metric
    epsilon = 200.0 * np.abs(rho_pinn - rho_fd_interp) / (rho_pinn + rho_fd_interp + 1e-6)
    
    # Create diagnostic plot
    save_path = None
    if args.save_plot:
        desktop_path = os.path.expanduser("~/Desktop")
        if not os.path.exists(desktop_path):
            desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
        output_dir = os.path.join(desktop_path, "model test plots")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"fd_diagnostics_t{args.time:.4f}.png")
    
    plot_diagnostic_masks(xs, ys, diagnostics, rho_fd_interp, epsilon, 
                         vx_fd=vx_fd_interp, vy_fd=vy_fd_interp, save_path=save_path)
    
    # Print recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    quality = diagnostics['quality_score']
    
    if quality >= 0.98:
        print("[OK] FD solution is high quality - comparisons are reliable")
        print(f"  Mean error in valid regions: {diagnostics['mean_epsilon_valid']:.2f}%")
    elif quality >= 0.90:
        print("[WARNING] FD solution has some quality issues")
        print("  Comparisons are mostly reliable but be cautious in flagged regions")
        print(f"  Mean error (all): {diagnostics['mean_epsilon']:.2f}%")
        print(f"  Mean error (valid only): {diagnostics['mean_epsilon_valid']:.2f}%")
    else:
        print("[ERROR] FD solution has significant quality problems")
        print("  PINN-FD comparisons may not be reliable!")
        print()
        print("Try these solutions:")
        print(f"  1. Reduce Courant number: --nu 0.1 (currently {args.nu})")
        print(f"  2. Increase resolution: --N-fd {N_fd * 2} (currently {N_fd})")
        print(f"  3. Compare at earlier time (currently t={args.time})")
        if str(PERTURBATION_TYPE).lower() == "power_spectrum":
            print(f"  4. Try different random seed (currently {RANDOM_SEED})")
    
    print("="*60)


if __name__ == "__main__":
    main()

