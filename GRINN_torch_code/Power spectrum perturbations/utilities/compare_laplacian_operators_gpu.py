"""
GPU-Accelerated Comparison: Discrete vs Continuous Laplacian Operators in FFT Poisson Solver

This script compares the effect of using discrete vs continuous Laplacian operators
in the FFT-based Poisson solver using PyTorch on GPU for acceleration.

The discrete Laplacian accounts for the finite difference discretization:
    Discrete: 2(cos(kx·dx) - 1)/dx² + 2(cos(ky·dy) - 1)/dy²
    Continuous: -(kx² + ky²)

Usage:
    python utilities/compare_laplacian_operators_gpu.py
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numerical_solvers.LAX_torch import lax_solution_torch, device, dtype
from config import (
    RANDOM_SEED, POWER_EXPONENT, num_of_waves as NUM_OF_WAVES_CONFIG,
    xmin, ymin, cs, rho_o, const, G, a, wave, PERTURBATION_TYPE, KX, KY
)

# Configuration
N = 400                    # Grid resolution (Nx = Ny)
nu = 0.25                   # Courant number for stability
lam = wave                 # Wavelength (from config.py)
num_of_waves = NUM_OF_WAVES_CONFIG  # Number of wavelengths in domain
time_points = [1.0, 2.0, 3.0]  # Times to compare

perturbation_type = PERTURBATION_TYPE
power_index = POWER_EXPONENT
vel_rms = a * cs
random_seed = RANDOM_SEED
kx = KX
ky = KY

gravity = True            # Must be True to use FFT solver
output_dir = "laplacian_comparison_gpu_output"  # Directory to save plots
save_plots = True         # Save plots to files
show_plots = True         # Display plots on screen

print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def fft_solver_torch_continuous(rho, Lx, nx, Ly, ny):
    """
    PyTorch FFT solver using continuous Laplacian operator: -(kx² + ky²)
    This is the standard implementation.
    """
    dx = Lx / nx
    dy = Ly / ny
    
    # Calculate the Fourier modes of the gas density
    rhohat = torch.fft.fft2(rho)
    
    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
    ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
    
    # Construct the Laplacian operator in Fourier space (continuous)
    # Using 'xy' indexing to match NumPy behavior, then transpose
    kx2, ky2 = torch.meshgrid(kx**2, ky**2, indexing='xy')
    laplace = -(kx2.T + ky2.T)
    
    # Handle zero mode (k=0)
    laplace = torch.where(laplace == 0, torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    
    # Solve for the potential in Fourier space
    phihat = rhohat / laplace
    
    # Transform back to real space
    phi = torch.real(torch.fft.ifft2(phihat))
    
    return phi

def fft_solver_torch_discrete(rho, Lx, nx, Ly, ny):
    """
    PyTorch FFT solver using discrete Laplacian operator.
    
    The discrete Laplacian for finite differences is:
        ∇²_discrete = 2(cos(kx·dx) - 1)/dx² + 2(cos(ky·dy) - 1)/dy²
    
    This matches the actual discretization used in finite difference methods.
    For small k·dx, this approaches the continuous form -(kx² + ky²).
    """
    dx = Lx / nx
    dy = Ly / ny
    
    # Calculate the Fourier modes of the gas density
    rhohat = torch.fft.fft2(rho)
    
    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
    ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
    
    # Construct the discrete Laplacian operator in Fourier space
    # Need to create 2D grids of kx and ky values
    # Using 'ij' indexing: kx varies along axis 0, ky varies along axis 1
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
    
    # Discrete Laplacian: 2(cos(k·dx) - 1)/dx² for each direction
    laplace_x = 2 * (torch.cos(kx_grid * dx) - 1) / (dx**2)
    laplace_y = 2 * (torch.cos(ky_grid * dy) - 1) / (dy**2)
    laplace = laplace_x + laplace_y
    
    # Handle zero mode (k=0)
    laplace = torch.where(torch.abs(laplace) < 1e-12, 
                         torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    
    # Solve for the potential in Fourier space
    phihat = rhohat / laplace
    
    # Transform back to real space
    phi = torch.real(torch.fft.ifft2(phihat))
    
    return phi

def run_lax_with_custom_fft_torch(time, N, nu, lam, num_of_waves, a, gravity,
                                   power_index, vel_rms, random_seed, fft_solver_func):
    """
    Run LAX solver with a custom FFT solver function (PyTorch version).
    
    This patches the fft_solver_torch in the LAX_torch module temporarily.
    """
    # Import the LAX_torch module to patch it
    import numerical_solvers.LAX_torch as lax_torch_module
    
    # Save original fft_solver
    original_fft_solver = lax_torch_module.fft_solver_torch
    
    # Replace with custom function
    lax_torch_module.fft_solver_torch = fft_solver_func
    
    try:
        # Determine perturbation type
        use_velocity_ps = (str(perturbation_type).lower() == "power_spectrum")
        
        # Use unique seed based on power_index
        if use_velocity_ps and abs(power_index - POWER_EXPONENT) < 1e-6:
            unique_seed = random_seed
        elif use_velocity_ps:
            unique_seed = random_seed + int(abs(power_index) * 1000)
        else:
            unique_seed = random_seed
        
        # Calculate domain boundaries
        xmax = xmin + lam * num_of_waves
        ymax = ymin + lam * num_of_waves
        
        # Run LAX solver (PyTorch version)
        result = lax_solution_torch(
            time_val=time,
            N=N,
            nu=nu,
            lam=lam,
            num_of_waves=num_of_waves,
            rho_1=a,
            gravity=gravity,
            use_velocity_ps=use_velocity_ps,
            ps_index=power_index,
            vel_rms=vel_rms,
            random_seed=unique_seed
        )
        
        x, rho, vx, vy, phi, n, rho_max = result
        
        # Convert to numpy for plotting (handle both torch tensors and numpy arrays)
        if torch.is_tensor(x):
            x = x.cpu().numpy() + xmin
        else:
            x = np.asarray(x) + xmin
        
        if torch.is_tensor(rho):
            rho = rho.cpu().numpy()
        else:
            rho = np.asarray(rho)
        
        if torch.is_tensor(vx):
            vx = vx.cpu().numpy()
        else:
            vx = np.asarray(vx)
        
        if torch.is_tensor(vy):
            vy = vy.cpu().numpy()
        else:
            vy = np.asarray(vy)
        
        if phi is not None:
            if torch.is_tensor(phi):
                phi = phi.cpu().numpy()
            else:
                phi = np.asarray(phi)
        
        # Create y-coordinates
        y = np.linspace(ymin, ymax, rho.shape[1])
        
        return x, y, rho, vx, vy, phi, rho_max
        
    finally:
        # Restore original fft_solver
        lax_torch_module.fft_solver_torch = original_fft_solver

def create_comparison_plot(x, y, field_continuous, field_discrete, diff, title, 
                          time, cmap='viridis', diff_cmap='RdBu_r'):
    """
    Create a comparison plot showing continuous, discrete, and difference.
    """
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Continuous Laplacian result
    pc1 = axes[0].pcolormesh(X, Y, field_continuous, shading='auto', cmap=cmap)
    axes[0].set_title(f'Continuous Laplacian\n-(kx² + ky²)', fontsize=12)
    axes[0].set_xlabel('x', fontsize=10)
    axes[0].set_ylabel('y', fontsize=10)
    plt.colorbar(pc1, ax=axes[0], shrink=0.8)
    
    # Discrete Laplacian result
    pc2 = axes[1].pcolormesh(X, Y, field_discrete, shading='auto', cmap=cmap)
    axes[1].set_title(f'Discrete Laplacian\n2(cos(k·dx)-1)/dx² + 2(cos(k·dy)-1)/dy²', fontsize=11)
    axes[1].set_xlabel('x', fontsize=10)
    axes[1].set_ylabel('y', fontsize=10)
    plt.colorbar(pc2, ax=axes[1], shrink=0.8)
    
    # Difference
    vmax_diff = np.max(np.abs(diff))
    pc3 = axes[2].pcolormesh(X, Y, diff, shading='auto', cmap=diff_cmap, 
                             vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title(f'Difference\n(Discrete - Continuous)', fontsize=12)
    axes[2].set_xlabel('x', fontsize=10)
    axes[2].set_ylabel('y', fontsize=10)
    cbar3 = plt.colorbar(pc3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Difference', fontsize=10)
    
    # Add statistics text
    stats_text = (
        f'Max |diff|: {np.max(np.abs(diff)):.2e}\n'
        f'Mean |diff|: {np.mean(np.abs(diff)):.2e}\n'
        f'RMS diff: {np.sqrt(np.mean(diff**2)):.2e}\n'
        f'Rel. RMS: {np.sqrt(np.mean(diff**2))/np.std(field_continuous):.2e}'
    )
    axes[2].text(0.02, 0.98, stats_text,
                transform=axes[2].transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3',
                fc='white', ec='gray', alpha=0.8))
    
    fig.suptitle(f'{title} at t={time:.2f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_statistics_plot(times, stats_continuous, stats_discrete, stats_diff):
    """
    Create a plot showing how statistics evolve over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    times = np.array(times)
    
    # Max density comparison
    axes[0, 0].plot(times, stats_continuous['rho_max'], 'o-', 
                   label='Continuous', linewidth=2, markersize=6)
    axes[0, 0].plot(times, stats_discrete['rho_max'], 's-', 
                   label='Discrete', linewidth=2, markersize=6)
    axes[0, 0].set_title('Maximum Density', fontsize=12)
    axes[0, 0].set_xlabel('Time', fontsize=10)
    axes[0, 0].set_ylabel('Density', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Max potential comparison
    axes[0, 1].plot(times, stats_continuous['phi_max'], 'o-', 
                   label='Continuous', linewidth=2, markersize=6)
    axes[0, 1].plot(times, stats_discrete['phi_max'], 's-', 
                   label='Discrete', linewidth=2, markersize=6)
    axes[0, 1].set_title('Maximum Potential', fontsize=12)
    axes[0, 1].set_xlabel('Time', fontsize=10)
    axes[0, 1].set_ylabel('Potential', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Difference in max density
    rho_max_diff = np.array(stats_discrete['rho_max']) - np.array(stats_continuous['rho_max'])
    axes[1, 0].plot(times, rho_max_diff, 'o-', color='red', linewidth=2, markersize=6)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Difference in Max Density (Discrete - Continuous)', fontsize=12)
    axes[1, 0].set_xlabel('Time', fontsize=10)
    axes[1, 0].set_ylabel('Density Difference', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMS difference in potential
    axes[1, 1].plot(times, stats_diff['phi_rms'], 'o-', color='purple', 
                   linewidth=2, markersize=6)
    axes[1, 1].set_title('RMS Difference in Potential', fontsize=12)
    axes[1, 1].set_xlabel('Time', fontsize=10)
    axes[1, 1].set_ylabel('RMS Difference', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    return fig

def analyze_laplacian_operators():
    """
    Analyze the difference between continuous and discrete Laplacian operators
    in Fourier space directly.
    """
    print("\n" + "=" * 60)
    print("Analyzing Laplacian Operators in Fourier Space")
    print("=" * 60)
    
    # Setup grid
    Lx = Ly = lam * num_of_waves
    Nx = Ny = N
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Calculate wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)
    
    # Create 2D grids
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # Continuous Laplacian
    laplace_continuous = -(kx_grid**2 + ky_grid**2)
    
    # Discrete Laplacian
    laplace_discrete = (2 * (np.cos(kx_grid * dx) - 1) / (dx**2) + 
                       2 * (np.cos(ky_grid * dy) - 1) / (dy**2))
    
    # Avoid division by zero
    mask = k_magnitude > 1e-10
    
    # Relative difference
    rel_diff = np.zeros_like(k_magnitude)
    rel_diff[mask] = np.abs(laplace_discrete[mask] - laplace_continuous[mask]) / np.abs(laplace_continuous[mask])
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1D slice along kx (ky=0)
    ky_zero_idx = Ny // 2
    axes[0, 0].plot(kx[:Nx//2], -kx[:Nx//2]**2, 'o-', label='Continuous: -kx²', markersize=3)
    axes[0, 0].plot(kx[:Nx//2], 2*(np.cos(kx[:Nx//2]*dx)-1)/(dx**2), 's-', 
                   label='Discrete: 2(cos(kx·dx)-1)/dx²', markersize=3)
    axes[0, 0].set_xlabel('kx', fontsize=10)
    axes[0, 0].set_ylabel('Laplacian operator', fontsize=10)
    axes[0, 0].set_title('1D Slice: Laplacian along kx (ky=0)', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Relative difference along kx
    rel_diff_1d = np.abs(2*(np.cos(kx[:Nx//2]*dx)-1)/(dx**2) + kx[:Nx//2]**2) / (kx[:Nx//2]**2 + 1e-10)
    axes[0, 1].plot(kx[:Nx//2], rel_diff_1d, 'o-', color='red', markersize=3)
    axes[0, 1].set_xlabel('kx', fontsize=10)
    axes[0, 1].set_ylabel('Relative Difference', fontsize=10)
    axes[0, 1].set_title('Relative Difference along kx', fontsize=11)
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2D relative difference
    pc = axes[1, 0].pcolormesh(kx_grid[:Nx//2, :Ny//2], ky_grid[:Nx//2, :Ny//2], 
                               rel_diff[:Nx//2, :Ny//2], shading='auto', cmap='hot')
    axes[1, 0].set_xlabel('kx', fontsize=10)
    axes[1, 0].set_ylabel('ky', fontsize=10)
    axes[1, 0].set_title('2D Relative Difference |Discrete - Continuous|/|Continuous|', fontsize=10)
    plt.colorbar(pc, ax=axes[1, 0])
    
    # Relative difference vs k magnitude
    k_flat = k_magnitude[mask].flatten()
    rel_diff_flat = rel_diff[mask].flatten()
    
    # Sort by k magnitude for cleaner plot
    sort_idx = np.argsort(k_flat)
    k_sorted = k_flat[sort_idx][::100]  # Subsample for clarity
    rel_diff_sorted = rel_diff_flat[sort_idx][::100]
    
    axes[1, 1].scatter(k_sorted, rel_diff_sorted, alpha=0.5, s=1)
    axes[1, 1].set_xlabel('|k| (wave number magnitude)', fontsize=10)
    axes[1, 1].set_ylabel('Relative Difference', fontsize=10)
    axes[1, 1].set_title('Relative Difference vs Wave Number Magnitude', fontsize=11)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add text with key information
    k_nyquist = np.pi / dx
    text = (f'Grid: {Nx}×{Ny}\n'
            f'dx = dy = {dx:.4f}\n'
            f'k_Nyquist = π/dx = {k_nyquist:.2f}\n'
            f'Max rel. diff: {np.max(rel_diff):.2e}')
    axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
                   fc='white', ec='gray', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    
    if save_plots:
        filepath = os.path.join(output_dir, "laplacian_operator_analysis.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"Maximum relative difference: {np.max(rel_diff):.6e}")
    print(f"Mean relative difference: {np.mean(rel_diff[mask]):.6e}")
    print("=" * 60)

def main():
    """Main function to run the comparison."""
    print("=" * 60)
    print("GPU-Accelerated Laplacian Operator Comparison")
    print("Discrete vs Continuous")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Grid resolution: {N}×{N}")
    print(f"Perturbation type: {perturbation_type.upper()}")
    if str(perturbation_type).lower() == "power_spectrum":
        print(f"Power spectrum index: {power_index}")
    else:
        print(f"Wave vector: KX={kx:.3f}, KY={ky:.3f}")
    print(f"Time points: {time_points}")
    print("=" * 60)
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # First, analyze the Laplacian operators themselves
    analyze_laplacian_operators()
    
    # Storage for statistics
    stats_continuous = {
        'times': [],
        'rho_max': [],
        'rho_min': [],
        'phi_max': [],
        'phi_min': []
    }
    stats_discrete = {
        'times': [],
        'rho_max': [],
        'rho_min': [],
        'phi_max': [],
        'phi_min': []
    }
    stats_diff = {
        'times': [],
        'rho_rms': [],
        'phi_rms': [],
        'rho_max_diff': [],
        'phi_max_diff': []
    }
    
    # Run comparison for each time point
    print(f"\nRunning LAX solver comparison for {len(time_points)} time points...")
    for time in tqdm(time_points, desc="Comparing", unit="time"):
        print(f"\n--- Time t = {time:.2f} ---")
        
        # Run with continuous Laplacian
        print("  Running with continuous Laplacian...")
        x_cont, y_cont, rho_cont, vx_cont, vy_cont, phi_cont, rho_max_cont = \
            run_lax_with_custom_fft_torch(
                time, N, nu, lam, num_of_waves, a, gravity,
                power_index, vel_rms, random_seed, fft_solver_torch_continuous
            )
        
        # Run with discrete Laplacian
        print("  Running with discrete Laplacian...")
        x_disc, y_disc, rho_disc, vx_disc, vy_disc, phi_disc, rho_max_disc = \
            run_lax_with_custom_fft_torch(
                time, N, nu, lam, num_of_waves, a, gravity,
                power_index, vel_rms, random_seed, fft_solver_torch_discrete
            )
        
        # Store statistics
        stats_continuous['times'].append(time)
        stats_continuous['rho_max'].append(np.max(rho_cont))
        stats_continuous['rho_min'].append(np.min(rho_cont))
        if phi_cont is not None:
            stats_continuous['phi_max'].append(np.max(phi_cont))
            stats_continuous['phi_min'].append(np.min(phi_cont))
        
        stats_discrete['times'].append(time)
        stats_discrete['rho_max'].append(np.max(rho_disc))
        stats_discrete['rho_min'].append(np.min(rho_disc))
        if phi_disc is not None:
            stats_discrete['phi_max'].append(np.max(phi_disc))
            stats_discrete['phi_min'].append(np.min(phi_disc))
        
        # Calculate differences
        rho_diff = rho_disc - rho_cont
        if phi_cont is not None and phi_disc is not None:
            phi_diff = phi_disc - phi_cont
        else:
            phi_diff = None
        
        stats_diff['times'].append(time)
        stats_diff['rho_rms'].append(np.sqrt(np.mean(rho_diff**2)))
        stats_diff['rho_max_diff'].append(np.max(rho_disc) - np.max(rho_cont))
        if phi_diff is not None:
            stats_diff['phi_rms'].append(np.sqrt(np.mean(phi_diff**2)))
            stats_diff['phi_max_diff'].append(np.max(phi_disc) - np.max(phi_cont))
        
        # Print summary statistics
        print(f"  Max density - Continuous: {np.max(rho_cont):.6f}, Discrete: {np.max(rho_disc):.6f}")
        print(f"  Max density difference: {np.max(rho_disc) - np.max(rho_cont):.6e}")
        print(f"  RMS density difference: {np.sqrt(np.mean(rho_diff**2)):.6e}")
        if phi_diff is not None:
            print(f"  RMS potential difference: {np.sqrt(np.mean(phi_diff**2)):.6e}")
            print(f"  Max potential difference: {np.max(np.abs(phi_diff)):.6e}")
        
        # Create comparison plots
        # Density comparison
        fig_rho = create_comparison_plot(
            x_cont, y_cont, rho_cont, rho_disc, rho_diff,
            'Density', time, cmap='YlOrBr'
        )
        if save_plots:
            filename = f"density_comparison_t_{time:.2f}.png"
            filepath = os.path.join(output_dir, filename)
            fig_rho.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_rho)
        
        # Potential comparison (if available)
        if phi_cont is not None and phi_disc is not None:
            fig_phi = create_comparison_plot(
                x_cont, y_cont, phi_cont, phi_disc, phi_diff,
                'Potential', time, cmap='viridis'
            )
            if save_plots:
                filename = f"potential_comparison_t_{time:.2f}.png"
                filepath = os.path.join(output_dir, filename)
                fig_phi.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"  Saved: {filepath}")
            if show_plots:
                plt.show()
            else:
                plt.close(fig_phi)
    
    # Create summary statistics plot
    if len(time_points) > 1:
        print("\nGenerating summary statistics plot...")
        fig_stats = create_statistics_plot(
            stats_continuous['times'], stats_continuous, stats_discrete, stats_diff
        )
        if save_plots:
            filepath = os.path.join(output_dir, "statistics_comparison.png")
            fig_stats.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        if show_plots:
            plt.show()
        else:
            plt.close(fig_stats)
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    if save_plots:
        print(f"Results saved to: {output_dir}/")
    print("=" * 60)
    
    # Summary of findings
    print("\nKey Findings:")
    print(f"  - The discrete Laplacian accounts for finite grid spacing")
    print(f"  - Differences are most significant at high wave numbers (near Nyquist)")
    print(f"  - For this grid (dx={lam*num_of_waves/N:.4f}), the operators differ by:")
    print(f"    * Max density difference: {np.max(np.abs(stats_diff['rho_max_diff'])):.6e}")
    print(f"    * Max RMS potential diff: {np.max(stats_diff['phi_rms']):.6e}")
    print("=" * 60)

if __name__ == "__main__":
    main()

