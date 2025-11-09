"""
LAX Analysis Script for Power Spectrum Perturbations

This script generates 2D surface plots using the LAX finite difference solver
with power spectrum velocity initial conditions. It provides easy configuration
of all relevant parameters for analysis purposes.

Usage:
    python analyze_numerical.py

Modify the configuration section below to explore different parameter values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Add parent directory to path for imports when running from utilities directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from numerical_solvers.LAX_2D import lax_solution as lax_solution_cpu, generate_velocity_field_power_spectrum
from numerical_solvers.LAX_2D_torch import lax_solution_torch
from config import (
    RANDOM_SEED, POWER_EXPONENT, num_of_waves as NUM_OF_WAVES_CONFIG,
    xmin, ymin, cs, rho_o, const, G, a, wave
)

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS FOR ANALYSIS
# =============================================================================

# Grid & Domain Parameters
N = 400                    # Grid resolution (Nx = Ny)
nu = 0.25                   # Courant number for stability (typically 0.1-0.9)
lam = wave                  # Wavelength (from config.py)
num_of_waves = NUM_OF_WAVES_CONFIG  # Number of wavelengths in domain (from config.py)
time_points = [3.5, 4.0, 4.5]  # Times to plot

# Physical Constants (all from config.py)
# cs, rho_o, const, G, a, xmin, ymin are imported from config.py above

# Power Spectrum Parameters
power_index = POWER_EXPONENT  # Power spectrum exponent (matches config.py for consistency)
vel_rms = a * cs          # RMS velocity amplitude (consistent with train.py)
random_seed = RANDOM_SEED         # Seed for reproducibility (from config.py)

# Solver backend selection: "cpu" uses numerical_solvers.LAX_2D (reference implementation),
# "torch" uses numerical_solvers.LAX_2D_torch (experimental GPU version)
SOLVER_BACKEND = "torch"      # Options: "cpu" or "torch"

# Output Settings
output_dir = "lax_analysis_output"  # Directory to save plots
gravity = True              # Whether to include self-gravity

# Plot Settings
plot_density = True        # Generate density plots
plot_velocity = False       # Generate velocity magnitude plots
show_vectors = True         # Show velocity vectors on plots
save_plots = True          # Save plots to files
show_plots = False          # Display plots on screen

# Collapse Time Settings
find_collapse_time = False   # Whether to find collapse time (set to False for faster execution)
collapse_method = "full_lax"    # Method: "fast" (integrated solver) or "full_lax" (parallel LAX calls)
target_density_ratio = 10.0  # Target density ratio (times initial density)
max_search_time = 10.0     # Maximum time to search for collapse
collapse_dt = 0.1          # Time step for collapse search (used by full_lax method)

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

def run_lax_solver(time, N, nu, lam, num_of_waves, a, gravity, 
                   power_index, vel_rms, random_seed):
    """
    Dispatch helper that runs the chosen LAX solver backend and
    returns a unified tuple.
    
    Returns:
        x, y, rho, vx, vy, phi, rho_max
    """
    backend = SOLVER_BACKEND.lower().strip()
    if backend not in {"cpu", "torch"}:
        raise ValueError(f"Unsupported SOLVER_BACKEND='{SOLVER_BACKEND}'. Use 'cpu' or 'torch'.")
    
    # Use RANDOM_SEED directly when power_index matches POWER_EXPONENT from config
    if abs(power_index - POWER_EXPONENT) < 1e-6:
        unique_seed = random_seed
    else:
        unique_seed = random_seed + int(abs(power_index) * 1000)
    
    # Save current random state and set unique seed
    original_state = np.random.get_state()
    np.random.seed(unique_seed)
    
    # Calculate domain boundaries from config
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    Lx = xmax - xmin
    Ly = ymax - ymin
    
    if backend == "torch":
        # Run PyTorch (GPU) solver
        result = lax_solution_torch(
            time_val=time,
            N=N,
            nu=nu,
            lam=lam,
            num_of_waves=num_of_waves,
            rho_1=a,
            gravity=gravity,
            use_velocity_ps=True,
            ps_index=power_index,
            vel_rms=vel_rms,
            random_seed=unique_seed
        )
        
        x, rho, vx, vy, phi, n, rho_max = result
        # LAX solver returns x starting from 0, so offset by xmin
        x = x + xmin
        
    else:  # CPU reference solver
        # Run CPU solver
        result = lax_solution_cpu(
            time=time,
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
            # LAX solver returns x starting from 0, so offset by xmin
            x = x + xmin
        else:
            rho, vx, rho_max = result
            x = np.linspace(xmin, xmax, rho.shape[0], endpoint=False)
            vy = np.zeros_like(rho)
            phi = None
            n = None
    
    # Create y-coordinates (LAX solver uses square domain)
    # LAX solver creates y starting from 0, so we create it with ymin offset
    y = np.linspace(ymin, ymax, rho.shape[1])
    
    # Restore original random state
    np.random.set_state(original_state)
    
    return x, y, rho, vx, vy, phi, rho_max

def create_2d_surface_plot(x, y, field, title, cmap='viridis', 
                          vx=None, vy=None, show_vectors=False):
    """
    Create a 2D surface plot with optional velocity vectors.
    """
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the field
    pc = ax.pcolormesh(X, Y, field, shading='auto', cmap=cmap)
    
    # Add velocity vectors if requested
    if show_vectors and vx is not None and vy is not None:
        skip_x = max(1, len(x) // 20)
        skip_y = max(1, len(y) // 20)
        skip = (slice(None, None, skip_x), slice(None, None, skip_y))
        ax.quiver(X[skip], Y[skip], vx[skip], vy[skip], 
                 color='k', headwidth=3.0, width=0.003, alpha=0.7)
    
    # Formatting
    ax.set_title(title, fontsize=14)
    ax.text(0.02, 0.96, f"a={a}, power_index={power_index}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2',
            fc='white', ec='gray', alpha=0.6))
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(pc, shrink=0.6, location='right')
    cbar.ax.set_title(title.split()[0], fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def find_collapse_time_search():
    """
    Search for the time when density reaches target_density_ratio * rho_o.
    
    Returns:
        collapse_time: Time when target density is reached, or None if not found
    """
    target_density = target_density_ratio * rho_o
    print(f"\nSearching for collapse time (target density: {target_density:.2f})...")
    print(f"Method: {collapse_method}")
    print(f"Searching from t={collapse_dt} to t={max_search_time} with dt={collapse_dt}")
    
    if collapse_method == "full_lax":
        # Search by running full LAX solver at each time step
        # Start from a small time to avoid checking initial conditions
        current_time = collapse_dt
        last_rho_max = rho_o
        num_checks = 0
        
        with tqdm(total=int(max_search_time / collapse_dt), desc="Searching for collapse", unit="step") as pbar:
            while current_time <= max_search_time:
                # Run solver at current time
                x, y, rho, vx, vy, phi, rho_max = run_lax_solver(
                    current_time, N, nu, lam, num_of_waves, a, gravity,
                    power_index, vel_rms, random_seed
                )
                
                max_density = np.max(rho)
                num_checks += 1
                
                # Check if target is reached
                if max_density >= target_density:
                    print(f"\n✓ Collapse found at t = {current_time:.3f}")
                    print(f"  Max density: {max_density:.2f} (target: {target_density:.2f})")
                    print(f"  Searched {num_checks} time steps")
                    return current_time
                
                # Check if density is decreasing (might indicate numerical issues or wrong direction)
                if max_density < last_rho_max and current_time > 0.5:
                    print(f"\n  Warning: Density decreased from {last_rho_max:.2f} to {max_density:.2f} at t={current_time:.3f}")
                
                last_rho_max = max_density
                
                # Update progress bar
                pbar.set_postfix({
                    't': f'{current_time:.2f}',
                    'max_rho': f'{max_density:.2f}',
                    'target': f'{target_density:.2f}'
                })
                pbar.update(1)
                
                current_time += collapse_dt
        
        print(f"\n✗ Collapse not found within t={max_search_time}")
        print(f"  Final max density: {last_rho_max:.2f} (target: {target_density:.2f})")
        return None
    
    else:
        raise ValueError(f"Unknown collapse_method: {collapse_method}. Use 'full_lax'.")

def generate_analysis_plots():
    """Generate all analysis plots for the configured parameters."""
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find collapse time if requested
    collapse_time = None
    if find_collapse_time:
        collapse_time = find_collapse_time_search()
        if collapse_time is not None:
            print(f"\nCollapse time: {collapse_time:.3f}")
            # Optionally add collapse time to time_points for plotting
            if collapse_time not in time_points:
                print(f"Note: Collapse time {collapse_time:.3f} is not in time_points list.")
        else:
            print(f"\nNo collapse found within max_search_time={max_search_time}")
    
    # Storage for statistics
    stats = {
        'times': [],
        'rho_max': [],
        'rho_min': []
    }
    
    # Generate plots for each time point
    print(f"\nGenerating plots for {len(time_points)} time points...")
    for time in tqdm(time_points, desc="Plotting", unit="plot"):
        # Run LAX solver
        x, y, rho, vx, vy, phi, rho_max = run_lax_solver(
            time, N, nu, lam, num_of_waves, a, gravity,
            power_index, vel_rms, random_seed
        )
        
        # Store statistics
        stats['times'].append(time)
        stats['rho_max'].append(np.max(rho))
        stats['rho_min'].append(np.min(rho))
        
        # Generate density plot
        if plot_density:
            fig_dens, ax_dens = create_2d_surface_plot(
                x, y, rho, f"Density at t={time:.2f}", 
                cmap='YlOrBr', vx=vx, vy=vy, show_vectors=show_vectors
            )
            
            if save_plots:
                filename = f"density_t_{time:.2f}.png"
                filepath = os.path.join(output_dir, filename)
                fig_dens.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig_dens)
        
        # Generate velocity magnitude plot
        if plot_velocity:
            v_mag = np.sqrt(vx**2 + vy**2)
            fig_vel, ax_vel = create_2d_surface_plot(
                x, y, v_mag, f"Velocity Magnitude at t={time:.2f}", 
                cmap='viridis', vx=vx, vy=vy, show_vectors=show_vectors
            )
            
            if save_plots:
                filename = f"velocity_t_{time:.2f}.png"
                filepath = os.path.join(output_dir, filename)
                fig_vel.savefig(filepath, dpi=300, bbox_inches='tight')
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig_vel)
    
    # Generate summary statistics plot
    if len(time_points) > 1:
        create_summary_plot(stats)

def create_summary_plot(stats):
    """Create a summary plot showing evolution of key density quantities over time."""
    times = np.array(stats['times'])
    rho_max = np.array(stats['rho_max'])
    rho_min = np.array(stats['rho_min'])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"Summary (a={a}, power_index={power_index})", fontsize=12)

    # Density evolution (max/min)
    axes[0].plot(times, rho_max, 'o-', label='Max', linewidth=2)
    axes[0].plot(times, rho_min, 's-', label='Min', linewidth=2)
    axes[0].set_title('Density Evolution')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)

    # Density range
    rho_range = rho_max - rho_min
    axes[1].plot(times, rho_range, 'o-', linewidth=2, color='green')
    axes[1].set_title('Density Range (Max - Min)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Density Range')
    axes[1].grid(True)

    if save_plots:
        filepath = os.path.join(output_dir, "summary_statistics.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def main():
    """Main function to run the LAX analysis."""
    print("LAX Power Spectrum Analysis Script")
    print("=" * 40)
    print("Modify the configuration section at the top of this script")
    print("to explore different parameter values.")
    print(f"Solver backend: {SOLVER_BACKEND.upper()}")
    print("=" * 40)
    
    try:
        generate_analysis_plots()
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
