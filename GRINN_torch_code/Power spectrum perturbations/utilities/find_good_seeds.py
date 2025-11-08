"""
Script to find good random seeds for visualization

This script scans through RANDOM_SEED to RANDOM_SEED + 100 and identifies seeds
where collapse occurs away from domain boundaries, making them good for visualization.

Usage:
    python find_good_seeds.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numerical_solvers.LAX_2D import lax_solution as lax_solution_cpu
from numerical_solvers.LAX_2D_torch import lax_solution_torch
from config import RANDOM_SEED, POWER_EXPONENT, cs, rho_o

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Grid & Domain Parameters (should match analyze_lax.py)
N = 300                    # Grid resolution (Nx = Ny)
nu = 0.5                   # Courant number for stability
lam = 7.0                  # Wavelength
num_of_waves = 2.0         # Number of wavelengths in domain
time = 4.0                 # Time to evaluate (should match time_points[0] from analyze_lax.py)

# Physical Constants
a = 0.1                   # Amplitude parameter (same as in config.py)
power_index = POWER_EXPONENT  # Power spectrum exponent (matches config.py)
vel_rms = a * cs          # RMS velocity amplitude
gravity = True            # Whether to include self-gravity

# Boundary threshold (fraction of domain size from edges)
# If collapse is within this fraction from any boundary, consider it "close to boundary"
BOUNDARY_THRESHOLD = 0.25  # 15% of domain size from edges

# Seed search range
NUM_SEEDS_TO_TEST = 10  # Number of seeds to test (from RANDOM_SEED to RANDOM_SEED + NUM_SEEDS_TO_TEST)

# Solver backend selection: "cpu" uses numerical_solvers.LAX_2D (reference implementation),
# "torch" uses numerical_solvers.LAX_2D_torch (experimental GPU version)
SOLVER_BACKEND = "torch"

# Plot mode selection
MAKE_SUMMARY_PLOT = True  # If True: create max density vs seed plot for t=3.0 and t=4.0 (disables t=4.0 spatial plots)
                           # If False: create spatial plots for t=4.0 (current behavior)

# Output Settings
output_dir = "seed_search_output"  # Directory to save plots
save_plots = True          # Save plots to files

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

def find_max_density_location(rho, x, y):
    """
    Find the spatial coordinates of the point with maximum density.
    
    Args:
        rho: 2D density array
        x, y: 1D coordinate arrays
    
    Returns:
        x_max, y_max: Coordinates of maximum density
        max_density: Maximum density value
    """
    # Find index of maximum density
    max_idx = np.unravel_index(np.argmax(rho), rho.shape)
    i_max, j_max = max_idx
    
    # Get coordinates
    x_max = x[i_max]
    y_max = y[j_max]
    max_density = rho[i_max, j_max]
    
    return x_max, y_max, max_density

def is_far_from_boundary(x_max, y_max, Lx, Ly, threshold):
    """
    Check if point is far from boundaries.
    
    Args:
        x_max, y_max: Coordinates of point
        Lx, Ly: Domain size
        threshold: Fraction of domain size from edges
    
    Returns:
        True if far from boundaries, False otherwise
    """
    x_threshold = Lx * threshold
    y_threshold = Ly * threshold
    
    # Check if point is away from all boundaries
    far_from_left = x_max > x_threshold
    far_from_right = x_max < (Lx - x_threshold)
    far_from_bottom = y_max > y_threshold
    far_from_top = y_max < (Ly - y_threshold)
    
    return far_from_left and far_from_right and far_from_bottom and far_from_top


def run_lax_solver(time_val, unique_seed, Lx, Ly):
    """
    Dispatch helper that runs the chosen LAX solver backend and
    returns a unified tuple.
    """
    backend = SOLVER_BACKEND.lower().strip()
    if backend not in {"cpu", "torch"}:
        raise ValueError(f"Unsupported SOLVER_BACKEND='{SOLVER_BACKEND}'. Use 'cpu' or 'torch'.")

    if backend == "torch":
        result = lax_solution_torch(
            time_val=time_val,
            N=N,
            nu=nu,
            lam=lam,
            num_of_waves=num_of_waves,
            rho_1=a,
            gravity=gravity,  # Now supports gravity
            use_velocity_ps=True,
            ps_index=power_index,
            vel_rms=vel_rms,
            random_seed=unique_seed
        )

        x, rho, vx, vy, phi, n, rho_max = result

    else:  # CPU reference solver
        result = lax_solution_cpu(
            time=time_val,
            N=N,
            nu=nu,
            lam=lam,
            num_of_waves=num_of_waves,
            rho_1=a,
            gravity=gravity,
            isplot=False,
            comparison=False,
            animation=True,
            use_velocity_ps=True,
            ps_index=power_index,
            vel_rms=vel_rms,
            random_seed=unique_seed
        )

        if gravity:
            x, rho, vx, vy, phi, n, rho_max = result
        else:
            rho, vx, rho_max = result
            x = np.linspace(0, Lx, rho.shape[0], endpoint=False)
            vy = np.zeros_like(rho)
            phi = None
            n = None

    return x, rho, vx, vy, phi, n, rho_max

def create_summary_plot(seeds, max_densities_t3, max_densities_t4, output_dir):
    """
    Create and save a plot showing max density vs seed number for two times.
    
    Args:
        seeds: List of seed numbers
        max_densities_t3: List of max densities at t=3.0
        max_densities_t4: List of max densities at t=4.0
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot both time series
    ax.plot(seeds, max_densities_t3, 'o-', label='t = 3.0', linewidth=2, markersize=6)
    ax.plot(seeds, max_densities_t4, 's-', label='t = 4.0', linewidth=2, markersize=6)
    
    # Formatting
    ax.set_xlabel('Seed Number', fontsize=12)
    ax.set_ylabel('Max Density', fontsize=12)
    ax.set_title('Maximum Density vs Seed Number', fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        filename = "max_density_vs_seed.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"\nSummary plot saved to: {filepath}")
    
    plt.close(fig)

def create_plot_with_seed(x, y, rho, seed, x_max, y_max, max_density, output_dir, time_val):
    """
    Create and save a density plot with seed information.
    
    Args:
        x, y: Coordinate arrays
        rho: Density field
        seed: Random seed used
        x_max, y_max: Location of maximum density
        max_density: Maximum density value
        output_dir: Directory to save plot
        time_val: Time value for the plot
    """
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the density field
    pc = ax.pcolormesh(X, Y, rho, shading='auto', cmap='YlOrBr')
    
    # Mark the maximum density location
    ax.plot(x_max, y_max, 'r*', markersize=15, label=f'Max density: {max_density:.2f}')
    
    # Add domain boundaries for reference
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    threshold_x = Lx * BOUNDARY_THRESHOLD
    threshold_y = Ly * BOUNDARY_THRESHOLD
    
    # Draw boundary threshold lines
    ax.axvline(x=threshold_x, color='gray', linestyle='--', alpha=0.5, label='Boundary threshold')
    ax.axvline(x=Lx - threshold_x, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=threshold_y, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=Ly - threshold_y, color='gray', linestyle='--', alpha=0.5)
    
    # Formatting
    ax.set_title(f'Density at t={time_val:.2f}, Seed={seed}', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(loc='upper right')
    
    # Add text box with max density location
    ax.text(0.02, 0.98, f'Max density location:\n({x_max:.2f}, {y_max:.2f})\nMax density: {max_density:.2f}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5',
            fc='white', ec='gray', alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(pc, shrink=0.6, location='right')
    cbar.ax.set_title(r'$\rho$', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    if save_plots:
        filename = f"density_seed_{seed:04d}_t_{time_val:.2f}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    plt.close(fig)

def scan_seeds():
    """
    Scan through seeds from RANDOM_SEED to RANDOM_SEED + NUM_SEEDS_TO_TEST and find good ones.
    """
    print("=" * 60)
    if MAKE_SUMMARY_PLOT:
        print("Creating summary plot: max density vs seed")
    else:
        print("Scanning for good random seeds")
    print("=" * 60)
    print(f"Testing seeds from {RANDOM_SEED} to {RANDOM_SEED + NUM_SEEDS_TO_TEST}")
    print(f"Solver backend: {SOLVER_BACKEND.upper()}")
    
    if MAKE_SUMMARY_PLOT:
        print(f"Mode: Summary plot (t=3.0 and t=4.0)")
        print("Spatial plots will be disabled - only summary plot will be created")
    else:
        print(f"Mode: Spatial plots at t = {time:.2f}")
        print(f"Boundary threshold: {BOUNDARY_THRESHOLD * 100:.0f}% of domain size")
    print("=" * 60)
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}")
    
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    
    # For summary plot mode, store data for both times
    if MAKE_SUMMARY_PLOT:
        seeds_list = []
        max_densities_t3 = []
        max_densities_t4 = []
        times_to_compute = [3.0, 4.0]
        good_seeds = []  # Not used in summary mode, but keep for compatibility
    else:
        good_seeds = []
        times_to_compute = [time]
    
    # Create progress bar with custom format similar to analyze_lax.py
    seed_range = range(RANDOM_SEED, RANDOM_SEED + NUM_SEEDS_TO_TEST + 1)
    pbar = tqdm(seed_range, desc="Scanning seeds", unit="seed",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                disable=False, leave=True)
    
    for seed in pbar:
        try:
            # Run LAX solver with this seed
            # Use RANDOM_SEED directly when power_index matches POWER_EXPONENT from config
            # This ensures consistency with PINN training and comparison code.
            if abs(power_index - POWER_EXPONENT) < 1e-6:
                unique_seed = seed  # Match PINN/comparison code exactly
            else:
                unique_seed = seed + int(abs(power_index) * 1000)  # Different pattern for exploration
            
            # Save current random state and set unique seed
            original_state = np.random.get_state()
            np.random.seed(unique_seed)
            
            # Process each time point
            current_max_rho = None
            for time_val in times_to_compute:
                # Run chosen LAX solver backend
                x, rho, vx, vy, phi, n, rho_max = run_lax_solver(
                    time_val=time_val,
                    unique_seed=unique_seed,
                    Lx=Lx,
                    Ly=Ly
                )
                
                # Create y-coordinates (LAX solver uses square domain)
                y = np.linspace(0, Lx, rho.shape[1])
                
                # Find maximum density (just the value, not location for summary plot)
                max_density = np.max(rho)
                
                if MAKE_SUMMARY_PLOT:
                    # Store data for summary plot (no spatial plots, no boundary checks)
                    if time_val == 3.0:
                        max_densities_t3.append(max_density)
                    elif time_val == 4.0:
                        max_densities_t4.append(max_density)
                        current_max_rho = max_density  # Use t=4.0 for progress bar
                else:
                    # Original behavior: create spatial plot and check boundaries
                    x_max, y_max, _ = find_max_density_location(rho, x, y)
                    is_good = is_far_from_boundary(x_max, y_max, Lx, Ly, BOUNDARY_THRESHOLD)
                    create_plot_with_seed(x, y, rho, seed, x_max, y_max, max_density, output_dir, time_val)
                    current_max_rho = max_density  # Use current max density for progress bar
                    
                    # If good seed, record it
                    if is_good:
                        good_seeds.append(seed)
            
            # Store seed for summary plot
            if MAKE_SUMMARY_PLOT:
                seeds_list.append(seed)
            
            # Update progress bar with current info
            if current_max_rho is None:
                current_max_rho = 0.0  # Fallback if somehow not set
            if MAKE_SUMMARY_PLOT:
                pbar.set_postfix({
                    'seed': seed,
                    'max_rho': f'{current_max_rho:.2f}'
                })
            else:
                pbar.set_postfix({
                    'seed': seed,
                    'good': len(good_seeds),
                    'max_rho': f'{current_max_rho:.2f}'
                })
            pbar.update(1)
            
            # Restore original random state
            np.random.set_state(original_state)
            
        except Exception as e:
            print(f"\n✗ Error with seed {seed}: {e}")
            pbar.set_postfix({
                'seed': seed,
                'error': True
            })
            pbar.update(1)
            continue
    
    pbar.close()
    
    # Create summary plot if requested
    if MAKE_SUMMARY_PLOT and seeds_list:
        create_summary_plot(seeds_list, max_densities_t3, max_densities_t4, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total seeds tested: {len(seed_range)}")
    
    if MAKE_SUMMARY_PLOT:
        print(f"Summary plot created with max density data for t=3.0 and t=4.0")
        if seeds_list:
            print(f"Seeds processed: {len(seeds_list)}")
    else:
        print(f"Good seeds found: {len(good_seeds)}")
        print(f"\nBoundary threshold: {BOUNDARY_THRESHOLD * 100:.0f}% of domain size")
        Lx = lam * num_of_waves
        threshold_value = Lx * BOUNDARY_THRESHOLD
        print(f"Domain: [0, {Lx:.1f}] for both x and y")
        print(f"Boundary regions (avoid): [0, {threshold_value:.1f}] and [{Lx - threshold_value:.1f}, {Lx:.1f}]")
        print(f"Good region (central): [{threshold_value:.1f}, {Lx - threshold_value:.1f}]")
        
        if good_seeds:
            print("\nGood seeds (collapse away from boundaries):")
            for seed in good_seeds:
                print(f"  Seed: {seed}")
            print(f"\nTo use a good seed, set RANDOM_SEED = {good_seeds[0]} in config.py")
        else:
            print("\nNo good seeds found in this range.")
            print("Try increasing the seed range or adjusting the boundary threshold.")
    
    print("=" * 60)
    
    return good_seeds

if __name__ == "__main__":
    good_seeds = scan_seeds()

