#!/usr/bin/env python3
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
from LAX_2D import lax_solution

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS FOR ANALYSIS
# =============================================================================

# Grid & Domain Parameters
N = 300                    # Grid resolution (Nx = Ny)
nu = 0.5                   # Courant number for stability (typically 0.1-0.9)
lam = 7.0                  # Wavelength
num_of_waves = 2.0         # Number of wavelengths in domain
time_points = [1.0]  # Times to plot

# Physical Constants
cs = 1.0                   # Sound speed
rho_o = 1.0                # Background density
G = 1.0                    # Gravitational constant
const = 1.0                # Constant multiplier
a = 0.1                   # Amplitude parameter (same as in config.py)

# Power Spectrum Parameters
power_index = -4         # Power spectrum exponent (e.g., -3.0, -4.0)
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
target_density_ratio = 100.0  # Target density ratio (times initial density)
max_search_time = 10.0     # Maximum time to search for collapse
collapse_dt = 0.1          # Time step for collapse search (not used in fast algorithm)

# Validation Settings
run_validation = True      # Whether to run validation test (set to False for faster execution)
validation_times = [1.0, 2.0, 3.0, 4.0, 5.0]  # Times to test for validation

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
    # Set random seed to ensure consistent initial conditions
    np.random.seed(random_seed)
    
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
        random_seed=random_seed
    )
    
    # Extract results
    x, rho, vx, vy, phi, n, rho_max = result
    
    # Create y-coordinates (LAX solver uses square domain)
    Lx = lam * num_of_waves
    y = np.linspace(0, Lx, rho.shape[1])
    
    return x, y, rho, vx, vy, phi, rho_max

def run_lax_solver_with_velocity_field(time, N, nu, lam, num_of_waves, a, gravity, 
                                      power_index, vel_rms, velocity_field):
    """
    Run LAX solver with pre-generated velocity field for consistent results.
    
    Returns:
        x, y, rho, vx, vy, phi, rho_max
    """
    # Set random seed to ensure consistent initial conditions
    np.random.seed(random_seed)
    
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
        random_seed=random_seed,
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
    print(f"\nFast search for collapse time (density ratio = {target_ratio}x)...")
    print(f"Searching from t=0 to t={max_time}")
    
    target_density = target_ratio * rho_o
    
    # Import LAX solver components
    from LAX_2D import generate_velocity_field_power_spectrum, fft_solver
    from numpy.fft import fft2, ifft2
    
    # Set random seed to ensure consistent initial conditions
    np.random.seed(random_seed)
    
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
                                                     random_seed=random_seed)
    
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
    for k in range(1, n_max):
        current_time += dt
        
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
            print(f"Collapse reached at t = {current_time:.3f}")
            print(f"  Maximum density: {current_max_density:.4f}")
            print(f"  Target density: {target_density:.4f}")
            print(f"  Ratio achieved: {current_max_density/rho_o:.1f}x")
            
            # Save the exact collapse state
            y = np.linspace(0, Lx, rho1.shape[1])
            collapse_state = (x.copy(), y.copy(), rho1.copy(), vx1.copy(), vy1.copy(), phi1.copy())
            return current_time, (vx0.copy(), vy0.copy()), collapse_state
        
        # Update for next iteration
        rho0, vx0, vy0, Px0, Py0, phi0 = rho1, vx1, vy1, Px1, Py1, phi1
        
        # Adaptive time step
        dt1 = nu*dx/np.max([abs(vx1), abs(vy1)])
        dt2 = nu*dx/cs
        dt = np.min([dt1, dt2])
        mux = dt/(2*dx)
        muy = dt/(2*dy)
        
        # Progress indicator (reduced frequency)
        if k % 500 == 0:
            print(f"  t = {current_time:.2f}, max density = {current_max_density:.4f}")
    
    print(f"Collapse not reached within t={max_time}")
    print(f"  Final maximum density: {current_max_density:.4f}")
    print(f"  Target density: {target_density:.4f}")
    print(f"  Ratio achieved: {current_max_density/rho_o:.1f}x")
    return None, None, None

def generate_analysis_plots():
    """
    Generate all analysis plots for the configured parameters.
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
    
    # Generate plots for each time point
    for i, time in enumerate(time_points):
        # Run LAX solver
        x, y, rho, vx, vy, phi, rho_max = run_lax_solver(
            time, N, nu, lam, num_of_waves, a, gravity,
            power_index, vel_rms, random_seed
        )
        
        # Calculate velocity magnitude
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Store statistics
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
    if len(time_points) > 1:
        create_summary_plot(stats)
    
    # Print final statistics
    print_summary_statistics(stats)

def create_summary_plot(stats):
    """
    Create a summary plot showing evolution of key quantities over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    times = np.array(stats['times'])
    
    # Density evolution
    axes[0, 0].plot(times, stats['rho_max'], 'o-', label='Max', linewidth=2)
    axes[0, 0].plot(times, stats['rho_min'], 's-', label='Min', linewidth=2)
    axes[0, 0].set_title('Density Evolution')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Velocity evolution
    axes[0, 1].plot(times, stats['v_max'], 'o-', label='Max', linewidth=2)
    axes[0, 1].plot(times, stats['v_min'], 's-', label='Min', linewidth=2)
    axes[0, 1].set_title('Velocity Magnitude Evolution')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Density range
    rho_range = np.array(stats['rho_max']) - np.array(stats['rho_min'])
    axes[1, 0].plot(times, rho_range, 'o-', linewidth=2, color='green')
    axes[1, 0].set_title('Density Range (Max - Min)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Density Range')
    axes[1, 0].grid(True)
    
    # Velocity range
    v_range = np.array(stats['v_max']) - np.array(stats['v_min'])
    axes[1, 1].plot(times, v_range, 'o-', linewidth=2, color='purple')
    axes[1, 1].set_title('Velocity Range (Max - Min)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Velocity Range')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        filepath = os.path.join(output_dir, "summary_statistics.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)

def print_summary_statistics(stats):
    """
    Print summary statistics to console.
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"{'Time':<8} {'Rho Max':<10} {'Rho Min':<10} {'V Max':<10} {'V Min':<10}")
    print("-" * 60)
    
    for i, time in enumerate(stats['times']):
        print(f"{time:<8.2f} {stats['rho_max'][i]:<10.4f} {stats['rho_min'][i]:<10.4f} "
              f"{stats['v_max'][i]:<10.4f} {stats['v_min'][i]:<10.4f}")

def validate_fast_algorithm(target_ratio=100.0, test_times=[1.0, 2.0, 3.0, 4.0, 5.0]):
    """
    Validate the fast algorithm against the original LAX solver.
    
    Args:
        target_ratio: Target density ratio for collapse detection
        test_times: List of times to compare results
    
    Returns:
        validation_results: Dictionary with comparison data
    """
    print("\n" + "="*60)
    print("VALIDATION TEST: Fast Algorithm vs Original LAX")
    print("="*60)
    
    # Set random seed for consistency
    np.random.seed(random_seed)
    
    validation_results = {
        'times': [],
        'fast_density': [],
        'original_density': [],
        'density_error': [],
        'density_error_percent': []
    }
    
    print(f"Testing at times: {test_times}")
    print(f"Using parameters: a={a}, power_index={power_index}, vel_rms={vel_rms}")
    print("-" * 60)
    
    for time in test_times:
        # Run fast algorithm (integrate to this time)
        fast_density = get_density_at_time_fast(time, random_seed)
        
        # Run original LAX solver
        x, y, rho, vx, vy, phi, rho_max = run_lax_solver(
            time, N, nu, lam, num_of_waves, a, gravity,
            power_index, vel_rms, random_seed
        )
        original_density = np.max(rho)
        
        # Calculate errors
        density_error = abs(fast_density - original_density)
        density_error_percent = (density_error / original_density) * 100
        
        # Store results
        validation_results['times'].append(time)
        validation_results['fast_density'].append(fast_density)
        validation_results['original_density'].append(original_density)
        validation_results['density_error'].append(density_error)
        validation_results['density_error_percent'].append(density_error_percent)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"{'Time':<8} {'Fast':<12} {'Original':<12} {'Error':<10} {'Error %':<10}")
    print("-" * 60)
    
    for i, time in enumerate(validation_results['times']):
        print(f"{time:<8.1f} {validation_results['fast_density'][i]:<12.4f} "
              f"{validation_results['original_density'][i]:<12.4f} "
              f"{validation_results['density_error'][i]:<10.4f} "
              f"{validation_results['density_error_percent'][i]:<10.2f}")
    
    # Calculate overall statistics
    avg_error_percent = np.mean(validation_results['density_error_percent'])
    max_error_percent = np.max(validation_results['density_error_percent'])
    
    print("-" * 60)
    print(f"Average relative error: {avg_error_percent:.2f}%")
    print(f"Maximum relative error: {max_error_percent:.2f}%")
    
    if avg_error_percent < 5.0:
        print("VALIDATION PASSED: Fast algorithm is highly accurate")
    elif avg_error_percent < 15.0:
        print("VALIDATION ACCEPTABLE: Fast algorithm has moderate accuracy")
    else:
        print("VALIDATION FAILED: Fast algorithm has poor accuracy")
    
    print("="*60)
    
    return validation_results

def get_density_at_time_fast(target_time, random_seed):
    """
    Get maximum density at a specific time using the fast algorithm.
    """
    # Import LAX solver components
    from LAX_2D import generate_velocity_field_power_spectrum, fft_solver
    from numpy.fft import fft2, ifft2
    
    # Set random seed
    np.random.seed(random_seed)
    
    # Set up domain (same as fast algorithm)
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    Nx = N
    Ny = N
    dx = float(Lx/Nx)
    dy = float(Ly/Ny)
    dt = nu*dx/cs
    mux = dt/(2*dx)
    muy = dt/(2*dy)
    
    # Initialize arrays
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Initial conditions
    rho0 = rho_o * np.ones((Nx, Ny))
    vx0, vy0 = generate_velocity_field_power_spectrum(Nx, Ny, Lx, Ly, 
                                                     power_index=power_index, 
                                                     amplitude=vel_rms, 
                                                     random_seed=random_seed)
    
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
    
    # Time integration
    current_time = 0.0
    while current_time < target_time:
        # LAX update (same as in fast algorithm)
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
        
        # Update for next iteration
        rho0, vx0, vy0, Px0, Py0, phi0 = rho1, vx1, vy1, Px1, Py1, phi1
        
        # Adaptive time step
        dt1 = nu*dx/np.max([abs(vx1), abs(vy1)])
        dt2 = nu*dx/cs
        dt = np.min([dt1, dt2])
        mux = dt/(2*dx)
        muy = dt/(2*dy)
        
        current_time += dt
        
        # Check if we've reached the target time
        if current_time >= target_time:
            break
    
    return np.max(rho1)

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
        # Run validation test if enabled
        if run_validation:
            validation_results = validate_fast_algorithm(
                target_ratio=target_density_ratio,
                test_times=validation_times
            )
        
        # First, find collapse time if enabled
        collapse_time = None
        velocity_field = None
        collapse_state = None
        if find_collapse_time:
            collapse_time, velocity_field, collapse_state = find_collapse_time_fast(
                target_ratio=target_density_ratio, 
                max_time=max_search_time,
                random_seed=random_seed
            )
        
        # Then generate analysis plots
        generate_analysis_plots()
        
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
