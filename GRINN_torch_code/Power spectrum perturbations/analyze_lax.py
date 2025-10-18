"""
LAX Analysis Script for Power Spectrum Perturbations

This script generates 2D surface plots using only the LAX finite difference solver
with power spectrum velocity initial conditions. It provides easy configuration
of all relevant parameters for analysis purposes.

Usage:
    python analyze_lax.py

Modify the configuration section below to explore different parameter values.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from functools import partial
import time
from tqdm import tqdm
from LAX_2D import lax_solution, generate_velocity_field_power_spectrum

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS FOR ANALYSIS
# =============================================================================

# Grid & Domain Parameters
N = 300                    # Grid resolution (Nx = Ny)
nu = 0.5                   # Courant number for stability (typically 0.1-0.9)
lam = 7.0                  # Wavelength
num_of_waves = 2.0         # Number of wavelengths in domain
time_points = [1.5]  # Times to plot

# Physical Constants
cs = 1.0                   # Sound speed
rho_o = 1.0                # Background density
G = 1.0                    # Gravitational constant
const = 1.0                # Constant multiplier
a = 0.01                   # Amplitude parameter (same as in config.py)

# Power Spectrum Parameters
power_index = 0         # Power spectrum exponent (e.g., -3.0, -4.0)
vel_rms = a * cs          # RMS velocity amplitude (consistent with train.py)
random_seed = 1234         # Seed for reproducibility

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
find_collapse_time = True   # Whether to find collapse time (set to False for faster execution)
collapse_method = "full_lax"    # Method: "fast" (integrated solver) or "full_lax" (parallel LAX calls)
target_density_ratio = 100.0  # Target density ratio (times initial density)
max_search_time = 20.0     # Maximum time to search for collapse
collapse_dt = 0.1          # Time step for collapse search (used by full_lax method)

# Note: Validation functions removed for cleaner code - fast algorithm works well for core formation detection

# =============================================================================
# END CONFIGURATION SECTION
# =============================================================================

def run_lax_solver(time, N, nu, lam, num_of_waves, a, gravity, 
                   power_index, vel_rms, random_seed):
    """
    Run LAX solver with power spectrum velocity initial conditions.
    
    Returns:
        x, y, rho, vx, vy, phi, rho_max
    """
    # Create a unique seed that combines random_seed and power_index
    # This ensures different power_index values create different spatial patterns
    unique_seed = random_seed + int(abs(power_index) * 1000)
    
    # Save current random state and set unique seed
    original_state = np.random.get_state()
    np.random.seed(unique_seed)
    
    # Run LAX solver with power spectrum velocity field
    result = lax_solution(
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
    
    # Extract results
    x, rho, vx, vy, phi, n, rho_max = result
    
    # Create y-coordinates (LAX solver uses square domain)
    Lx = lam * num_of_waves
    y = np.linspace(0, Lx, rho.shape[1])
    
    # Restore original random state
    np.random.set_state(original_state)
    
    return x, y, rho, vx, vy, phi, rho_max

def run_lax_solver_with_velocity_field(time, N, nu, lam, num_of_waves, a, gravity, 
                                      power_index, vel_rms, velocity_field):
    """
    Run LAX solver with pre-generated velocity field for consistent results.
    
    Returns:
        x, y, rho, vx, vy, phi, rho_max
    """
    # Create a unique seed that combines random_seed and power_index
    # This ensures different power_index values create different spatial patterns
    unique_seed = random_seed + int(abs(power_index) * 1000)
    np.random.seed(unique_seed)
    
    # Run LAX solver with shared velocity field
    result = lax_solution(
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
        random_seed=unique_seed,
        vx0_shared=velocity_field[0],
        vy0_shared=velocity_field[1]
    )
    
    # Extract results
    x, rho, vx, vy, phi, n, rho_max = result
    
    # Create y-coordinates (LAX solver uses square domain)
    Lx = lam * num_of_waves
    y = np.linspace(0, Lx, rho.shape[1])
    
    return x, y, rho, vx, vy, phi, rho_max

def create_2d_surface_plot(x, y, field, title, cmap='viridis', 
                          vx=None, vy=None, show_vectors=False):
    """
    Create a 2D surface plot with optional velocity vectors.
    
    Args:
        x, y: Coordinate arrays
        field: 2D field to plot
        title: Plot title
        cmap: Colormap
        vx, vy: Velocity components for vectors
        show_vectors: Whether to show velocity vectors
    """
    # Create meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the field
    pc = ax.pcolormesh(X, Y, field, shading='auto', cmap=cmap)
    
    # Add velocity vectors if requested
    if show_vectors and vx is not None and vy is not None:
        # Subsample vectors for clarity
        skip_x = max(1, len(x) // 20)
        skip_y = max(1, len(y) // 20)
        skip = (slice(None, None, skip_x), slice(None, None, skip_y))
        
        ax.quiver(X[skip], Y[skip], vx[skip], vy[skip], 
                 color='k', headwidth=3.0, width=0.003, alpha=0.7)
    
    # Formatting
    ax.set_title(title, fontsize=14)
    # Annotate with parameter values
    try:
        ax.text(0.02, 0.96, f"a={a}, power_index={power_index}",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.2',
                fc='white', ec='gray', alpha=0.6))
    except Exception:
        pass
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Add colorbar
    cbar = plt.colorbar(pc, shrink=0.6, location='right')
    cbar.ax.set_title(title.split()[0], fontsize=12)
    
    plt.tight_layout()
    return fig, ax

def find_collapse_time_fast(target_ratio=100.0, max_time=10.0, random_seed=1234):
    """
    Fast algorithm to find collapse time by integrating directly in LAX solver.
    
    Args:
        target_ratio: Target density ratio (default: 100.0)
        max_time: Maximum time to search (default: 10.0)
        random_seed: Random seed for reproducibility
    
    Returns:
        collapse_time: Time when target ratio is reached, or None if not reached
        velocity_field: Tuple of (vx0, vy0) initial velocity field for reuse
        collapse_state: Tuple of (x, y, rho, vx, vy, phi) at collapse time
    """
    print(f"Fast search for collapse time (density ratio = {target_ratio}x)...")
    print(f"Searching from t=0 to t={max_time}")
    
    target_density = target_ratio * rho_o
    
    # Import LAX solver components
    from LAX_2D import generate_velocity_field_power_spectrum, fft_solver
    from numpy.fft import fft2, ifft2
    
    # Create a unique seed that combines random_seed and power_index
    # This ensures different power_index values create different spatial patterns
    unique_seed = random_seed + int(abs(power_index) * 1000)
    
    # Save current random state and set unique seed
    original_state = np.random.get_state()
    np.random.seed(unique_seed)
    
    # Set up domain
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    Nx = N
    Ny = N
    dx = float(Lx/Nx)
    dy = float(Ly/Ny)
    dt = nu*dx/cs
    mux = dt/(2*dx)
    muy = dt/(2*dy)
    n_max = int(max_time/dt)
    
    # Initialize arrays
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Initial conditions (use same seed as plotting)
    rho0 = rho_o * np.ones((Nx, Ny))
    vx0, vy0 = generate_velocity_field_power_spectrum(Nx, Ny, Lx, Ly, 
                                                     power_index=power_index, 
                                                     amplitude=vel_rms, 
                                                     random_seed=unique_seed)
    
    rho1 = rho0.copy()
    vx1 = vx0.copy()
    vy1 = vy0.copy()
    Px0 = rho0*vx0
    Py0 = rho0*vy0
    Px1 = Px0.copy()
    Py1 = Py0.copy()
    
    # Initial potential
    phi0 = fft_solver(const*(rho0-rho_o), Lx, Nx, Ly, Ny, dim=2)
    phi1 = phi0.copy()
    
    # Time integration with collapse detection
    current_time = 0.0
    current_max_density = rho_o  # Initialize
    
    # Create progress bar for time integration
    pbar = tqdm(total=n_max, desc="Fast Algorithm", unit="steps", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                disable=False, leave=True)
    
    for k in range(1, n_max):
        current_time += dt
        
        # Update progress bar with current info
        pbar.set_postfix({
            't': f'{current_time:.2f}',
            'max_rho': f'{current_max_density:.2f}',
            'target': f'{target_density:.0f}'
        })
        pbar.update(1)
        
        # LAX update (same as in LAX_2D.py)
        rho1 = (1/4)*(np.roll(rho0, -1, axis=0) + np.roll(rho0, 1, axis=0) +
                      np.roll(rho0, -1, axis=1) + np.roll(rho0, 1, axis=1)) - \
               (mux*(np.roll(rho0,-1,axis=0)*np.roll(vx0,-1,axis=0) - 
                     np.roll(rho0,1, axis=0)*np.roll(vx0,1,axis=0))) - \
               (muy*(np.roll(rho0,-1,axis=1)*np.roll(vy0,-1,axis=1) - 
                     np.roll(rho0,1, axis=1)*np.roll(vy0,1,axis=1)))
        
        if gravity:
            Px1 = 0.25*(np.roll(Px0,-1,axis=0) + np.roll(Px0,1,axis=0) + 
                        np.roll(Px0,-1,axis=1) + np.roll(Px0,1,axis=1)) - \
                  (mux*(np.roll(Px0,-1,axis=0)*np.roll(vx0,-1,axis=0) - 
                        np.roll(Px0,1,axis=0)*np.roll(vx0,1,axis=0))) - \
                  (muy*(np.roll(Px0,-1,axis=1)*np.roll(vy0,-1,axis=1) - 
                        np.roll(Px0,1,axis=1)*np.roll(vy0,1,axis=1))) - \
                  ((cs**2)*mux*(np.roll(rho0,-1,axis=0) - np.roll(rho0,1,axis=0))) - \
                  (mux*rho0*(np.roll(phi0,-1,axis=0) - np.roll(phi0,1,axis=0)))
            
            Py1 = 0.25*(np.roll(Py0,-1,axis=0) + np.roll(Py0,1,axis=0) + 
                        np.roll(Py0,-1,axis=1) + np.roll(Py0,1,axis=1)) - \
                  (muy*(np.roll(Py0,-1,axis=1)*np.roll(vy0,-1,axis=1) - 
                        np.roll(Py0,1,axis=1)*np.roll(vy0,1,axis=1))) - \
                  (mux*(np.roll(Py0,-1,axis=0)*np.roll(vx0,-1,axis=0) - 
                        np.roll(Py0,1,axis=0)*np.roll(vx0,1,axis=0))) - \
                  ((cs**2)*muy*(np.roll(rho0,-1,axis=1) - np.roll(rho0,1,axis=1))) - \
                  (muy*rho0*(np.roll(phi0,-1,axis=1) - np.roll(phi0,1,axis=1)))
            
            phi1 = fft_solver(const*(rho1-rho_o), Lx, Nx, Ly, Ny, dim=2)
        
        vx1 = Px1/rho1
        vy1 = Py1/rho1
        
        # Check for collapse
        current_max_density = np.max(rho1)
        if current_max_density >= target_density:
            pbar.close()  # Close progress bar
            print(f"\nCollapse reached at t = {current_time:.3f}")
            print(f"  Maximum density: {current_max_density:.4f}")
            print(f"  Target density: {target_density:.4f}")
            print(f"  Ratio achieved: {current_max_density/rho_o:.1f}x")
            
            # Save the exact collapse state
            y = np.linspace(0, Lx, rho1.shape[1])
            collapse_state = (x.copy(), y.copy(), rho1.copy(), vx1.copy(), vy1.copy(), phi1.copy())
            
            # Restore original random state
            np.random.set_state(original_state)
            
            return current_time, (vx0.copy(), vy0.copy()), collapse_state
        
        # Update for next iteration
        rho0, vx0, vy0, Px0, Py0, phi0 = rho1, vx1, vy1, Px1, Py1, phi1
        
        # Adaptive time step
        dt1 = nu*dx/np.max([abs(vx1), abs(vy1)])
        dt2 = nu*dx/cs
        dt = np.min([dt1, dt2])
        mux = dt/(2*dx)
        muy = dt/(2*dy)
    
    pbar.close()  # Close progress bar
    print(f"\nCollapse not reached within t={max_time}")
    print(f"  Final maximum density: {current_max_density:.4f}")
    print(f"  Target density: {target_density:.4f}")
    print(f"  Ratio achieved: {current_max_density/rho_o:.1f}x")
    
    # Restore original random state
    np.random.set_state(original_state)
    
    return None, None, None

def run_lax_for_time_parallel(args):
    """
    Function to run LAX solver for a single time point (for multiprocessing).
    """
    time_point, unique_seed = args
    try:
        # Set unique seed for this process
        np.random.seed(unique_seed)
        
        result = lax_solution(
            time=time_point,
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
        
        x, rho, vx, vy, phi, n, rho_max = result
        return time_point, rho_max, (x, rho, vx, vy, phi)
        
    except Exception as e:
        print(f"Error at t={time_point}: {e}")
        return time_point, 0.0, None

def find_collapse_time_full_lax(target_ratio=100.0, max_time=10.0, random_seed=1234):
    """
    Find collapse time using full LAX solver with parallel processing.
    
    This method runs the complete LAX solver for each time point in parallel,
    providing the most accurate results but taking longer than the fast algorithm.
    
    Args:
        target_ratio: Target density ratio (times initial density)
        max_time: Maximum time to search
        random_seed: Random seed for reproducibility
    
    Returns:
        collapse_time, initial_velocity_field, collapse_state, results
        where results is a list of (time_point, max_density, state) tuples
    """
    print(f"Full LAX search for collapse time (density ratio = {target_ratio}x)...")
    print(f"Searching from t=0 to t={max_time}")
    
    # Create a unique seed that combines random_seed and power_index
    unique_seed = random_seed + int(abs(power_index) * 1000)
    
    # Save current random state and set unique seed
    original_state = np.random.get_state()
    np.random.seed(unique_seed)
    
    # Set up domain
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    Nx = N
    Ny = N
    dx = float(Lx/Nx)
    dy = float(Ly/Ny)
    
    # Generate initial velocity field
    vx0, vy0 = generate_velocity_field_power_spectrum(
        Nx, Ny, Lx, Ly, power_index, vel_rms, unique_seed
    )
    
    # Restore original random state
    np.random.set_state(original_state)
    
    # Create time points to test
    time_points = np.arange(0, max_time + collapse_dt, collapse_dt)
    
    # Prepare arguments for parallel processing
    args_list = [(time_point, unique_seed) for time_point in time_points]
    
    # Use all available CPU cores
    num_cores = mp.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing...")
    
    # Run LAX solver sequentially until collapse is found
    print(f"Running LAX simulations sequentially until collapse is found...")
    print(f"Searching from t=0 to t={max_time} with dt={collapse_dt}")
    
    target_density = rho_o * target_ratio
    collapse_time = None
    collapse_state = None
    results = []
    
    # Create progress bar for sequential search
    pbar = tqdm(total=len(time_points), desc="Full LAX", unit="sim")
    
    for i, time_point in enumerate(time_points):
        # Run single LAX simulation
        result = run_lax_for_time_parallel((time_point, unique_seed))
        time_point, max_density, state = result
        results.append(result)
        
        # Update progress bar
        pbar.set_postfix({
            't': f'{time_point:.2f}',
            'max_rho': f'{max_density:.2f}',
            'target': f'{target_density:.0f}'
        })
        pbar.update(1)
        
        # Check for collapse
        if max_density >= target_density and state is not None and collapse_time is None:
            collapse_time = time_point
            pbar.close()  # Close progress bar
            print(f"\nCollapse reached at t = {time_point:.3f}")
            print(f"  Maximum density: {max_density:.4f}")
            print(f"  Target density: {target_density:.4f}")
            print(f"  Ratio achieved: {max_density/rho_o:.1f}x")
            
            # Create collapse state
            x, rho, vx, vy, phi = state
            y = np.linspace(0, Lx, rho.shape[1])
            collapse_state = (x.copy(), y.copy(), rho.copy(), vx.copy(), vy.copy(), phi.copy())
            
            # Stop searching - collapse found!
            break
    
    if collapse_time is None:
        pbar.close()  # Close progress bar if no collapse found
    
    if collapse_time is None:
        print(f"Collapse not reached within t={max_time}")
        if results:
            final_max_density = max(rho_max for _, rho_max, _ in results)
            print(f"  Final maximum density: {final_max_density:.4f}")
            print(f"  Target density: {target_density:.4f}")
            print(f"  Ratio achieved: {final_max_density/rho_o:.1f}x")
    
    # Return all results for summary statistics
    return collapse_time, (vx0.copy(), vy0.copy()) if collapse_time else None, collapse_state, results

def generate_analysis_plots(full_lax_results=None):
    """
    Generate all analysis plots for the configured parameters.
    
    Args:
        full_lax_results: Optional results from full LAX method (list of tuples)
    """
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Storage for statistics
    stats = {
        'times': [],
        'rho_max': [],
        'rho_min': [],
        'v_max': [],
        'v_min': []
    }
    
    # Generate plots for each time point (regular density/velocity plots)
    print(f"\nGenerating plots for {len(time_points)} time points...")
    for i, time in enumerate(tqdm(time_points, desc="Plotting", unit="plot")):
        # Run LAX solver
        x, y, rho, vx, vy, phi, rho_max = run_lax_solver(
            time, N, nu, lam, num_of_waves, a, gravity,
            power_index, vel_rms, random_seed
        )
        
        # Calculate velocity magnitude
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Store statistics (only if not using full_lax_results)
        if full_lax_results is None:
            stats['times'].append(time)
            stats['rho_max'].append(np.max(rho))
            stats['rho_min'].append(np.min(rho))
            stats['v_max'].append(np.max(v_mag))
            stats['v_min'].append(np.min(v_mag))
        
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
    # Use full_lax_results if provided, otherwise use stats from time_points
    if full_lax_results is not None:
        # Extract statistics from full LAX results
        print(f"\nExtracting statistics from {len(full_lax_results)} full LAX results...")
        for i, (time_point, max_density, state) in enumerate(tqdm(full_lax_results, desc="Statistics", unit="point")):
            if state is not None:
                x, rho, vx, vy, phi = state
                v_mag = np.sqrt(vx**2 + vy**2)
                
                stats['times'].append(time_point)
                stats['rho_max'].append(np.max(rho))
                stats['rho_min'].append(np.min(rho))
                stats['v_max'].append(np.max(v_mag))
                stats['v_min'].append(np.min(v_mag))
        
        if len(stats['times']) > 1:
            create_summary_plot(stats)
    elif len(time_points) > 1:
        create_summary_plot(stats)

def create_summary_plot(stats):
    """
    Create a summary plot showing evolution of key density quantities over time.
    Plots only density-based metrics (no velocities).
    """
    # Sort by time to ensure monotonic x-axis
    times_np = np.array(stats['times'])
    order = np.argsort(times_np)
    times = times_np[order]
    rho_max_sorted = np.array(stats['rho_max'])[order]
    rho_min_sorted = np.array(stats['rho_min'])[order]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    # Global annotation with parameters
    try:
        fig.suptitle(f"Summary (a={a}, power_index={power_index})", fontsize=12)
    except Exception:
        pass

    # Density evolution (max/min)
    axes[0].plot(times, rho_max_sorted, 'o-', label='Max', linewidth=2)
    axes[0].plot(times, rho_min_sorted, 's-', label='Min', linewidth=2)
    axes[0].set_title('Density Evolution')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(True)

    # Density range
    rho_range = rho_max_sorted - rho_min_sorted
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
    """
    Main function to run the LAX analysis.
    """
    print("LAX Power Spectrum Analysis Script")
    print("=" * 40)
    print("Modify the configuration section at the top of this script")
    print("to explore different parameter values.")
    print("=" * 40)
    
    try:
        # First, find collapse time if enabled
        collapse_time = None
        velocity_field = None
        collapse_state = None
        full_lax_results = None
        
        if find_collapse_time:
            if collapse_method == "fast":
                print(f"Using fast algorithm for collapse detection...")
                collapse_time, velocity_field, collapse_state = find_collapse_time_fast(
                    target_ratio=target_density_ratio, 
                    max_time=max_search_time,
                    random_seed=random_seed
                )
            elif collapse_method == "full_lax":
                print(f"Using full LAX method for collapse detection...")
                collapse_time, velocity_field, collapse_state, full_lax_results = find_collapse_time_full_lax(
                    target_ratio=target_density_ratio, 
                    max_time=max_search_time,
                    random_seed=random_seed
                )
            else:
                print(f"Unknown collapse method: {collapse_method}")
                print("Available methods: 'fast', 'full_lax'")
                return
        
        # Then generate analysis plots
        # Pass full_lax_results if using full_lax method for summary statistics
        generate_analysis_plots(full_lax_results=full_lax_results)
        
        # Plot at collapse time if found
        if collapse_time is not None and collapse_state is not None:
            x, y, rho, vx, vy, phi = collapse_state
            
            if plot_density:
                fig_collapse, ax_collapse = create_2d_surface_plot(
                    x, y, rho, f"Density at Collapse Time t={collapse_time:.3f}", 
                    cmap='YlOrBr', vx=vx, vy=vy, show_vectors=show_vectors
                )
                
                if save_plots:
                    filename = f"density_collapse_t_{collapse_time:.3f}.png"
                    filepath = os.path.join(output_dir, filename)
                    fig_collapse.savefig(filepath, dpi=300, bbox_inches='tight')
                
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig_collapse)
        
        # Print collapse time summary
        if collapse_time is not None:
            print(f"\n" + "="*60)
            print("COLLAPSE TIME SUMMARY")
            print("="*60)
            print(f"Maximum density reached {target_density_ratio}x initial density at t = {collapse_time:.3f}")
            print(f"Initial density: {rho_o}")
            print(f"Target density: {target_density_ratio * rho_o}")
            print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
