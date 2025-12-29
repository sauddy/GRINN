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
    
    # 1D cross-section plots (creates both spatial 2D plots AND 1D cross-section plots)
    python compare_model.py /path/to/model.pth 3.0,6.0,8.0 --plot-type 1d
    python compare_model.py 3.0,4.0,5.0 --plot-type cross-section --y-fixed 0.5
    python compare_model.py model.pth 3.0,6.0 --plot-type 1d --N-fd 2000 --nu-fd 0.25
    
    # Density growth plot (power spectrum only)
    python compare_model.py model.pth 0.0 --plot-growth
    python compare_model.py model.pth 0.0 --plot-growth --growth-tmax 5.0 --growth-dt 0.1
    
    # Use GPU-accelerated FD solver (faster for large grids, not used for 1D plots)
    python compare_model.py model.pth 1.5,2.0,3.0 --plot-type pdf --fd-backend gpu
    python compare_model.py model.pth 0.0,1.0,2.0 --fd-backend torch  # Same as gpu
    
    # 3D sinusoidal block plots (density + epsilon)
    python compare_model.py model.pth 2.0 --plot-type 3d --N-fd 64
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import os
import sys
import argparse
from scipy.optimize import curve_fit
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LightSource

# Add parent directory to path for imports when running from utilities directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from config import (
    xmin, ymin, zmin, tmin, tmax, wave, a, cs, rho_o, harmonics, const, G,
    PERTURBATION_TYPE, RANDOM_SEED, N_GRID, POWER_EXPONENT, DIMENSION, SNAPSHOT_DIR, FD_N_1D,
    FD_N_3D, SLICE_Z, GROWTH_PLOT_TMAX, GROWTH_PLOT_DT
)
# Import num_of_waves with different name to avoid scoping conflict in main()
from config import num_of_waves as num_of_waves_config
from core.model_architecture import PINN
from core.data_generator import input_taker, req_consts_calc
from core.initial_conditions import initialize_shared_velocity_fields
from visualization.Plotting_2D import set_shared_velocity_fields, create_density_growth_plot
import visualization.Plotting_2D as plotting_module
from numerical_solvers.LAX import (
    lax_solution,
    lax_solution_with_shared_velocity,
    lax_solution_3d_sinusoidal,
    lax_solution1D_sinusoidal,
)
from numerical_solvers.LAX_torch import (
    lax_solution_torch,
    lax_solution_3d_sinusoidal_torch,
)
from scipy.interpolate import RegularGridInterpolator, interp1d
from config import KX, KY, SHOW_LINEAR_THEORY

# Device setup
has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

if device.startswith('cuda'):
    torch.cuda.empty_cache()


class FDSolutionManager:
    """
    Cache wrapper to avoid redundant FD solver executions across plot types.
    Stores FD outputs keyed by (time, N, nu, backend, mode) so repeated requests
    reuse existing solutions.
    """

    def __init__(self, initial_params, lam, num_of_waves, rho_1, perturbation_type, shared_vx=None, shared_vy=None):
        xmin, xmax, ymin, ymax, *_ = initial_params
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.lam = lam
        self.num_of_waves = num_of_waves
        self.rho_1 = rho_1
        self.perturbation_type = str(perturbation_type).lower()
        self.dimension = DIMENSION
        self.shared_vx = shared_vx
        self.shared_vy = shared_vy
        self.cache = {}

    def _normalize_backend(self, backend):
        backend_lower = backend.lower()
        if backend_lower in ("gpu", "torch") and torch.cuda.is_available():
            return "gpu"
        return "cpu"

    def get_solution(self, time_value, N=None, nu=0.5, backend="cpu"):
        """
        Retrieve FD solution for given parameters, computing it only once.
        """
        backend_key = self._normalize_backend(backend)
        if N is None:
            default_n = FD_N_3D if (self.dimension >= 3 and self.perturbation_type == "sinusoidal") else N_GRID
            N_eff = default_n
        else:
            N_eff = int(N)

        use_velocity_ps = (self.perturbation_type == "power_spectrum")
        use_3d = self.dimension >= 3 and self.perturbation_type == "sinusoidal"
        cache_key = (
            round(float(time_value), 8),
            int(N_eff),
            round(float(nu), 6),
            backend_key,
            "3d" if use_3d else "2d",
            use_velocity_ps
        )

        if cache_key in self.cache:
            return self.cache[cache_key]

        if use_3d:
            solution = self._run_solver_3d(time_value, N_eff, nu, backend_key)
        else:
            solution = self._run_solver_2d(time_value, N_eff, nu, backend_key, use_velocity_ps)

        self.cache[cache_key] = solution
        return solution

    def _run_solver_2d(self, time_value, N, nu, backend_key, use_velocity_ps):
        """
        Execute 2D FD solver (power spectrum or sinusoidal) and package results.
        """
        xmin, ymin = self.xmin, self.ymin
        lam = self.lam
        num_of_waves = self.num_of_waves
        rho_1 = self.rho_1

        if backend_key == "gpu":
            # Use shared velocity fields if available for power spectrum perturbations
            if use_velocity_ps and self.shared_vx is not None and self.shared_vy is not None:
                n_shared = int(self.shared_vx.shape[0])
                if n_shared != N:
                    print(f"Shared velocity fields available at resolution {n_shared}. Using that instead of N={N}.")
                    N = n_shared
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_torch(
                    time_val=time_value, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                    gravity=True, use_velocity_ps=use_velocity_ps, ps_index=POWER_EXPONENT,
                    vel_rms=a*cs, random_seed=RANDOM_SEED,
                    vx0_shared=self.shared_vx, vy0_shared=self.shared_vy
                )
            else:
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_torch(
                    time_val=time_value, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                    gravity=True, use_velocity_ps=use_velocity_ps, ps_index=POWER_EXPONENT,
                    vel_rms=a*cs, random_seed=RANDOM_SEED
                )
            x_fd = np.asarray(x_fd)
        else:
            if use_velocity_ps and self.shared_vx is not None and self.shared_vy is not None:
                n_shared = int(self.shared_vx.shape[0])
                if n_shared != N:
                    print(f"Shared velocity fields available at resolution {n_shared}. Using that instead of N={N}.")
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_with_shared_velocity(
                    time_value, n_shared, nu, lam, num_of_waves, rho_1, self.shared_vx, self.shared_vy,
                    gravity=True, isplot=False, comparison=False, animation=True
                )
                N = n_shared
            else:
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                    time_value, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False,
                    comparison=False, animation=True, use_velocity_ps=use_velocity_ps,
                    ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                )

        x_coords = np.asarray(x_fd) + xmin
        y_extent = lam * num_of_waves
        y_coords = np.linspace(ymin, ymin + y_extent, np.asarray(rho_fd).shape[1], endpoint=False)

        return {
            "dimension": 2,
            "x": x_coords,
            "y": y_coords,
            "rho": np.asarray(rho_fd),
            "vx": np.asarray(vx_fd),
            "vy": np.asarray(vy_fd),
            "backend": backend_key,
            "N": N
        }

    def _run_solver_3d(self, time_value, N, nu, backend_key):
        """
        Execute 3D sinusoidal FD solver (torch or numpy version) and package results.
        """
        xmin, ymin, zmin_local = self.xmin, self.ymin, self.zmin
        lam = self.lam
        num_of_waves = self.num_of_waves
        rho_1 = self.rho_1

        if backend_key == "gpu":
            fd_result = lax_solution_3d_sinusoidal_torch(
                time_val=time_value, N=N, nu=nu, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1, gravity=True
            )
        else:
            fd_result = lax_solution_3d_sinusoidal(
                time_value, N, nu, lam, num_of_waves, rho_1, gravity=True
            )

        x_fd = np.asarray(fd_result[0]) + xmin
        y_fd = np.asarray(fd_result[1]) + ymin
        z_fd = np.asarray(fd_result[2]) + zmin_local
        rho_fd = np.asarray(fd_result[3])
        vx_fd = np.asarray(fd_result[4])
        vy_fd = np.asarray(fd_result[5])
        vz_fd = np.asarray(fd_result[6])

        return {
            "dimension": 3,
            "x": x_fd,
            "y": y_fd,
            "z": z_fd,
            "rho": rho_fd,
            "vx": vx_fd,
            "vy": vy_fd,
            "vz": vz_fd,
            "backend": backend_key,
            "N": N
        }


def load_model(model_path, xmax, ymax, zmax=None):
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
    rmin = [xmin]
    rmax = [xmax]
    if DIMENSION >= 2:
        rmin.append(ymin)
        rmax.append(ymax)
    if DIMENSION >= 3:
        if zmax is None:
            raise ValueError("zmax must be provided when DIMENSION >= 3")
        rmin.append(zmin)
        rmax.append(zmax)
    net.set_domain(rmin=rmin, rmax=rmax, dimension=DIMENSION)
    net = net.to(device)
    net.eval()
    print(f"Loaded PINN model from {model_path}")
    return net


def create_comparison_plots(net, initial_params, time_points, which="density", N=None, nu=0.5,
                            save_plots=True, fd_backend="cpu", show_plot=True, fd_cache=None):
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
        show_plot: Whether to show the plot immediately (default: True). Set to False to show later.
        fd_cache: Instance of FDSolutionManager for reusing FD solutions.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    num_of_waves = (xmax - xmin) / lam
    z_slice = SLICE_Z if DIMENSION >= 3 else None
    if fd_cache is None:
        raise ValueError("fd_cache is required for create_comparison_plots to reuse FD data.")
    
    # Use N_GRID by default to match training comparison plots
    if N is None:
        if DIMENSION >= 3 and str(PERTURBATION_TYPE).lower() == "sinusoidal":
            N = FD_N_3D
            print(f"3D spatial plot: N not provided, using default FD_N_3D={FD_N_3D} from config")
        else:
            N = N_GRID
            print(f"2D spatial plot: N not provided, using default N_GRID={N_GRID} from config")
    else:
        print(f"Spatial plot: Using N={N} from command line argument")
    
    num_times = len(time_points)
    print(f"Creating {num_times}x3 comparison table for {which}...")
    print(f"Time points: {time_points}")
    
    # Create subplot grid
    fig, axes = plt.subplots(num_times, 3, figsize=(15, 4*num_times), constrained_layout=True)
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
        if DIMENSION >= 3:
            z_values = np.full((Q**2, 1), z_slice, dtype=np.float32)
            pt_z_collocation = Variable(torch.from_numpy(z_values).float(), requires_grad=True).to(device)
        else:
            pt_z_collocation = None
        
        net_inputs = [pt_x_collocation]
        if DIMENSION >= 2:
            net_inputs.append(pt_y_collocation)
        if DIMENSION >= 3 and pt_z_collocation is not None:
            net_inputs.append(pt_z_collocation)
        net_inputs.append(pt_t_collocation)
        
        output_00 = net(net_inputs)
        
        # Extract components
        rho_pinn = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
        pinn_vx = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
        pinn_vy = output_00[:, 2].data.cpu().numpy().reshape(Q, Q) if DIMENSION >= 2 else None
        pinn_vz = output_00[:, 3].data.cpu().numpy().reshape(Q, Q) if DIMENSION >= 3 else None
        
        if which == "density":
            pinn_field = rho_pinn
        else:  # velocity magnitude
            vel_components = [comp for comp in (pinn_vx, pinn_vy, pinn_vz) if comp is not None]
            if vel_components:
                pinn_field = np.sqrt(np.sum([comp**2 for comp in vel_components], axis=0))
            else:
                pinn_field = np.zeros_like(pinn_vx)
        
        fd_solution = fd_cache.get_solution(t, N=N, nu=nu, backend=fd_backend)
        if fd_solution["dimension"] == 3:
            x_fd = fd_solution["x"]
            y_fd = fd_solution["y"]
            z_fd = fd_solution["z"]
            z_idx = np.argmin(np.abs(z_fd - z_slice))
            if np.abs(z_fd[z_idx] - z_slice) > 1e-6:
                print(f"Using nearest z-slice at {z_fd[z_idx]:.4f} for requested z={z_slice:.4f}")
            rho_fd = fd_solution["rho"][:, :, z_idx]
            vx_fd = fd_solution["vx"][:, :, z_idx]
            vy_fd = fd_solution["vy"][:, :, z_idx]
            fd_slice_vz = fd_solution["vz"][:, :, z_idx]
        else:
            x_fd = fd_solution["x"]
            y_fd = fd_solution["y"]
            rho_fd = fd_solution["rho"]
            vx_fd = fd_solution["vx"]
            vy_fd = fd_solution["vy"]
            fd_slice_vz = None
        
        if which == "density":
            fd_field_native = rho_fd
        else:
            if fd_slice_vz is not None:
                fd_field_native = np.sqrt(vx_fd**2 + vy_fd**2 + fd_slice_vz**2)
            else:
                fd_field_native = np.sqrt(vx_fd**2 + vy_fd**2)
        
        # Interpolate FD data to PINN grid
        points_pinn = np.column_stack([tau.ravel(), phi.ravel()])
        
        interpolator = RegularGridInterpolator(
            (x_fd, y_fd), fd_field_native,
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
        
        if fd_slice_vz is not None:
            vz_interpolator = RegularGridInterpolator(
                (x_fd, y_fd), fd_slice_vz,
                method='linear',
                bounds_error=False,
                fill_value=None
            )
            fd_vz_interp = vz_interpolator(points_pinn).reshape(Q, Q)
        else:
            fd_vz_interp = None
        
        pinn_data.append(pinn_field)
        fd_data.append(fd_field_interp)
        pinn_velocity_data.append((pinn_vx, pinn_vy, pinn_vz))
        fd_velocity_data.append((fd_vx_interp, fd_vy_interp, fd_vz_interp))
    
    # Second pass: create plots
    for i, t in enumerate(time_points):
        pinn_field = pinn_data[i]
        fd_field = fd_data[i]
        
        # Extract velocity components
        pinn_vx, pinn_vy, pinn_vz = pinn_velocity_data[i]
        fd_vx, fd_vy, fd_vz = fd_velocity_data[i]
        
        # Calculate epsilon metric: ε = 2 * |PINN - FD| / (PINN + FD) * 100
        eps = 1e-6
        epsilon_metric = 200.0 * np.abs(pinn_field - fd_field) / (pinn_field + fd_field + eps)
        
        # Calculate percentile metrics
        median_eps = np.median(epsilon_metric)
        p75_eps = np.percentile(epsilon_metric, 75)
        p90_eps = np.percentile(epsilon_metric, 90)
        p99_eps = np.percentile(epsilon_metric, 99)
        
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
        
        # NOW ADD HISTOGRAM AND ANNOTATION (after ax_diff is created)
        axins = inset_axes(ax_diff, width="35%", height="30%", loc='upper right')
        hist_data = epsilon_metric.flatten()
        hist_data = hist_data[hist_data < np.percentile(hist_data, 99.5)]  # Clip outliers for viz
        axins.hist(hist_data, bins=40, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        axins.axvline(median_eps, color='red', linestyle='--', linewidth=2)
        axins.set_xlabel('ε (%)', fontsize=7)
        axins.set_ylabel('Pixels', fontsize=7)
        axins.tick_params(labelsize=6)
        axins.set_title(f'Med={median_eps:.1f}%, 90th={p90_eps:.1f}%', fontsize=7)
        
        # Add text annotation to main error plot
        ax_diff.text(0.02, 0.02, 
                    f'Median: {median_eps:.1f}%\n75th: {p75_eps:.1f}%\n90th: {p90_eps:.1f}%',
                    transform=ax_diff.transAxes, fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add axis labels
        if i == num_times - 1:
            ax_pinn.set_xlabel("x")
            ax_fd.set_xlabel("x")
            ax_diff.set_xlabel("x")
        
        ax_pinn.set_ylabel("y")
    
    #plt.tight_layout()
    
    # Save the figure if requested
    if save_plots:
        desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
        output_dir = os.path.join(desktop_path, "model test plots")
        os.makedirs(output_dir, exist_ok=True)
        time_str = "_".join([f"{t:.4f}" for t in time_points])
        save_path = os.path.join(output_dir, f"{which}_comparison_{num_times}x3_t{time_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    if show_plot:
        plt.show()
    return fig, axes


def _evaluate_pinn_density_3d(net, x_coords, y_coords, z_coords, time_value, batch_size=200000):
    """
    Evaluate the PINN density field on a full 3D grid using batched inference.
    """
    if DIMENSION < 3:
        raise ValueError("3D PINN evaluation requested but DIMENSION < 3.")
    
    X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    total_points = coords.shape[0]
    rho_values = np.zeros(total_points, dtype=np.float32)
    time_column = np.full((total_points, 1), time_value, dtype=np.float32)
    
    net.eval()
    with torch.no_grad():
        for start in range(0, total_points, batch_size):
            end = min(total_points, start + batch_size)
            chunk = coords[start:end]
            t_chunk = time_column[start:end]
            
            pt_x = torch.from_numpy(chunk[:, 0:1]).float().to(device)
            pt_y = torch.from_numpy(chunk[:, 1:2]).float().to(device)
            pt_z = torch.from_numpy(chunk[:, 2:3]).float().to(device)
            pt_t = torch.from_numpy(t_chunk).float().to(device)
            
            outputs = net([pt_x, pt_y, pt_z, pt_t])
            rho_chunk = outputs[:, 0:1].detach().cpu().numpy().reshape(-1)
            rho_values[start:end] = rho_chunk
    
    return rho_values.reshape(len(x_coords), len(y_coords), len(z_coords))


def _plot_cube_surface(ax, x_coords, y_coords, z_coords, values, cmap, norm):
    """
    Render a cube using opaque surfaces with proper lighting.
    """
    from matplotlib.colors import LightSource
    
    interp = RegularGridInterpolator(
        (x_coords, y_coords, z_coords),
        values,
        bounds_error=False,
        fill_value=None
    )

    def _sample_plane(axis, const_val, grid_a, grid_b):
        if axis == "z":
            A, B = np.meshgrid(grid_a, grid_b, indexing='ij')
            pts = np.column_stack([A.ravel(), B.ravel(), np.full(A.size, const_val)])
            data = interp(pts).reshape(A.shape)
            return A, B, np.full_like(A, const_val), data
        if axis == "y":
            A, B = np.meshgrid(grid_a, grid_b, indexing='ij')
            pts = np.column_stack([A.ravel(), np.full(A.size, const_val), B.ravel()])
            data = interp(pts).reshape(A.shape)
            return A, np.full_like(A, const_val), B, data
        A, B = np.meshgrid(grid_a, grid_b, indexing='ij')
        pts = np.column_stack([np.full(A.size, const_val), A.ravel(), B.ravel()])
        data = interp(pts).reshape(A.shape)
        return np.full_like(A, const_val), A, B, data

    # Higher density = smoother surfaces
    dense = 250
    xs = np.linspace(x_coords[0], x_coords[-1], dense)
    ys = np.linspace(y_coords[0], y_coords[-1], dense)
    zs = np.linspace(z_coords[0], z_coords[-1], dense)

    # Create light source with better parameters
    # Lower altdeg = less harsh lighting, blend controls color saturation
    ls = LightSource(azdeg=315, altdeg=35)
    
    planes = [
        ("z", z_coords[0], xs, ys),
        ("z", z_coords[-1], xs, ys),
        ("y", y_coords[0], xs, zs),
        ("y", y_coords[-1], xs, zs),
        ("x", x_coords[0], ys, zs),
        ("x", x_coords[-1], ys, zs),
    ]

    for axis_id, const_val, grid_a, grid_b in planes:
        Xa, Ya, Za, data_plane = _sample_plane(axis_id, const_val, grid_a, grid_b)
        
        # Map data to colors
        colors = cmap(norm(data_plane))
        
        # Apply lighting with blend mode to preserve color saturation
        # blend_mode='soft' or 'hsv' preserves colors better than default 'overlay'
        # fraction controls how much lighting vs original color (lower = more original color)
        rgb = np.array(colors[..., :3])
        shaded_rgb = ls.shade_rgb(rgb, elevation=Za, blend_mode='soft', fraction=0.7)
        
        # Reconstruct with alpha if present
        if colors.shape[-1] == 4:
            shaded_colors = np.dstack([shaded_rgb, colors[..., 3]])
        else:
            shaded_colors = shaded_rgb
        
        ax.plot_surface(Xa, Ya, Za, 
                       facecolors=shaded_colors,
                       rstride=1, cstride=1,
                       linewidth=0,
                       antialiased=True,
                       shade=False)

    # Set limits and aspect
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(y_coords[0], y_coords[-1])
    ax.set_zlim(z_coords[0], z_coords[-1])
    ax.set_box_aspect((
        x_coords[-1] - x_coords[0],
        y_coords[-1] - y_coords[0],
        z_coords[-1] - z_coords[0]
    ))
    
    # View settings
    ax.set_proj_type('persp')
    ax.view_init(elev=20, azim=-135)
    ax.dist = 10

    # Labels
    ax.set_xlabel("x", fontsize=11, labelpad=10)
    ax.set_ylabel("y", fontsize=11, labelpad=10)
    ax.set_zlabel("z", fontsize=11, labelpad=10)

    # Ticks
    tick_count = 5
    ax.set_xticks(np.linspace(x_coords[0], x_coords[-1], tick_count))
    ax.set_yticks(np.linspace(y_coords[0], y_coords[-1], tick_count))
    ax.set_zticks(np.linspace(z_coords[0], z_coords[-1], tick_count))
    
    formatter = FormatStrFormatter('%.2f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=9, pad=5)

    # Clean panes
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('lightgray')
        axis.pane.set_alpha(0.2)
        axis._axinfo["grid"]["linewidth"] = 0.4
        axis._axinfo["grid"]["linestyle"] = ':'
        axis._axinfo["grid"]["color"] = (0.6, 0.6, 0.6, 0.3)

    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.3)


def create_3d_sinusoidal_case_plot(net, initial_params, time_points, N=None, nu=0.5,
                                   save_plots=True, fd_backend="cpu", fd_cache=None):
    """
    Generate 3D block plots (density + epsilon) for sinusoidal 3D configurations.
    """
    if DIMENSION < 3 or str(PERTURBATION_TYPE).lower() != "sinusoidal":
        print("3D sinusoidal plotting is only available when DIMENSION=3 and PERTURBATION_TYPE='sinusoidal'.")
        return None
    if fd_cache is None:
        raise ValueError("fd_cache is required for 3D sinusoidal plotting.")
    
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax = initial_params
    num_of_waves = (xmax - xmin) / lam
    n_fd = int(N if N is not None else FD_N_3D)
    if fd_backend.lower() in ("gpu", "torch") and not torch.cuda.is_available():
        print("Warning: GPU backend requested for 3D plot but CUDA not available. Falling back to CPU.")
    
    cases = []
    density_min = np.inf
    density_max = -np.inf
    epsilon_max = 0.0
    
    for idx, t in enumerate(time_points):
        print(f"Collecting 3D data for case {idx+1} at t={t:.4f} (N={n_fd})")
        fd_solution = fd_cache.get_solution(t, N=n_fd, nu=nu, backend=fd_backend)
        if fd_solution["dimension"] != 3:
            raise ValueError("3D sinusoidal plot requested but FD cache did not return 3D data.")
        x_fd = fd_solution["x"]
        y_fd = fd_solution["y"]
        z_fd = fd_solution["z"]
        rho_fd = fd_solution["rho"]
        
        rho_pinn = _evaluate_pinn_density_3d(net, x_fd, y_fd, z_fd, t)
        epsilon_field = 200.0 * np.abs(rho_pinn - rho_fd) / (rho_pinn + rho_fd + 1e-6)
        
        cases.append({
            "rho": rho_pinn,
            "epsilon": epsilon_field,
            "x": x_fd,
            "y": y_fd,
            "z": z_fd,
            "label": f"Case {idx+1} (t={t:.2f})"
        })
        density_min = min(density_min, np.min(rho_pinn))
        density_max = max(density_max, np.max(rho_pinn))
        epsilon_max = max(epsilon_max, np.max(epsilon_field))
    
    if np.isclose(density_min, density_max):
        density_norm = Normalize(vmin=density_min - 1e-3, vmax=density_max + 1e-3)
    else:
        density_norm = Normalize(vmin=density_min, vmax=density_max)
    epsilon_norm = Normalize(vmin=0.0, vmax=epsilon_max if epsilon_max > 0 else 1.0)
    
    num_cases = len(cases)
    fig = plt.figure(figsize=(7 * num_cases, 10))
    grid = fig.add_gridspec(2, num_cases, hspace=0.22, wspace=0.15)
    density_axes = []
    epsilon_axes = []
    
    for idx, case in enumerate(cases):
        ax_density = fig.add_subplot(grid[0, idx], projection='3d')
        _plot_cube_surface(ax_density, case["x"], case["y"], case["z"], case["rho"], plt.get_cmap('YlOrBr'), density_norm)
        ax_density.set_title(case["label"])
        density_axes.append(ax_density)
        
        ax_eps = fig.add_subplot(grid[1, idx], projection='3d')
        _plot_cube_surface(ax_eps, case["x"], case["y"], case["z"], case["epsilon"], plt.get_cmap('Greys'), epsilon_norm)
        ax_eps.set_title(r"$\varepsilon$ (\%)", fontsize=12)
        epsilon_axes.append(ax_eps)
    
    density_sm = ScalarMappable(norm=density_norm, cmap='YlOrBr')
    density_sm.set_array([])
    fig.colorbar(density_sm, ax=density_axes, orientation='horizontal', fraction=0.05, pad=0.08, label=r"$\rho$")
    
    epsilon_sm = ScalarMappable(norm=epsilon_norm, cmap='Greys')
    epsilon_sm.set_array([])
    fig.colorbar(epsilon_sm, ax=epsilon_axes, orientation='horizontal', fraction=0.05, pad=0.12, label=r"$\varepsilon$ (\%)")
    
    if save_plots:
        desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
        output_dir = os.path.join(desktop_path, "model test plots")
        os.makedirs(output_dir, exist_ok=True)
        time_str = "_".join([f"{t:.4f}" for t in time_points])
        save_path = os.path.join(output_dir, f"3d_sinusoidal_cases_t{time_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D sinusoidal plot to {save_path}")
    
    plt.show()
    return fig


def create_1d_cross_section_plot(net, initial_params, time_points, y_fixed=0.6, N_fd=None, nu_fd=0.5,
                                 save_plots=True, fd_backend="cpu", fd_cache=None):
    """
    Create 1D cross-section plots at fixed y, comparing PINN vs FD solver.
    Always uses the dedicated 1D sinusoidal LAX solver (deterministic) to ensure clean references
    regardless of the global perturbation type or FD cache contents.
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: Array of time points to plot
        y_fixed: y value for the 1D slice through the 2D domain
        N_fd: grid size for FD solver
        nu_fd: Courant number for FD solver
        save_plots: Whether to save plots to disk
        fd_backend: Retained for CLI compatibility (not used; 1D solver is CPU-only).
        fd_cache: Retained for CLI compatibility (not used in the new 1D workflow).
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, _output_folder, _tmax = initial_params
    num_of_waves = (xmax - xmin) / lam
    
    # Set default N_fd to match spatial plots so FD cache can be reused
    if N_fd is None:
        if DIMENSION >= 3 and str(PERTURBATION_TYPE).lower() == "sinusoidal":
            N_fd = FD_N_3D
            print(f"1D cross-section: N_fd not provided, using default FD_N_3D={FD_N_3D} from config")
        else:
            N_fd = N_GRID
            print(f"1D cross-section: N_fd not provided, using default N_GRID={N_GRID} from config")
    else:
        print(f"1D cross-section: Using N_fd={N_fd} from command line argument")
    
    # Use baseline density for Linear Theory reference
    rho_base = rho_o
    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_base))
    k = np.sqrt(KX**2 + KY**2)
    v1_lt = (rho_1 / rho_base) * (alpha / k) if k > 0 else 0.0
    
    # Build x grid for PINN slice
    X = np.linspace(xmin, xmax, 1000).reshape(1000, 1)
    Y = y_fixed * np.ones_like(X)
    if DIMENSION >= 3:
        z_fixed = SLICE_Z
        Z = z_fixed * np.ones_like(X)
    else:
        Z = None
    
    # Create 4 rows x T columns panel layout
    T = len(time_points)
    fig = plt.figure(figsize=(6*T, 8), constrained_layout=False)
    grid = plt.GridSpec(4, T, figure=fig, hspace=0.12, wspace=0.18)
    
    # Cache for reused 1D FD solutions
    fd_1d_cache = {}

    for row_idx, t in enumerate(time_points):
        # PINN predictions at fixed y
        t_arr = t * np.ones_like(X)
        pt_x = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
        pt_z = Variable(torch.from_numpy(Z).float(), requires_grad=True).to(device) if Z is not None else None
        pt_t = Variable(torch.from_numpy(t_arr).float(), requires_grad=True).to(device)
        
        net_inputs = [pt_x]
        if DIMENSION >= 2:
            net_inputs.append(pt_y)
        if DIMENSION >= 3 and pt_z is not None:
            net_inputs.append(pt_z)
        net_inputs.append(pt_t)
        
        output_00 = net(net_inputs)
        rho_pinn = output_00[:, 0:1].data.cpu().numpy().reshape(-1)
        vx_pinn = output_00[:, 1:2].data.cpu().numpy().reshape(-1)
        
        # Linear Theory (only meaningful for KY == 0)
        rho_lt = None
        vx_lt = None
        if np.isclose(KY, 0.0):
            if lam >= jeans:
                # Gravitational instability case
                rho_lt = rho_base + rho_1*np.exp(alpha * t)*np.cos(KX * X[:, 0] + KY * y_fixed)
                vx_lt = -v1_lt*np.exp(alpha * t)*np.sin(KX * X[:, 0] + KY * y_fixed) * (KX / k) if k > 0 else 0.0
            else:
                # Oscillatory regime
                omega = np.sqrt(cs**2 * (KX**2 + KY**2) - const*G*rho_base)
                rho_lt = rho_base + rho_1*np.cos(omega * t - KX * X[:, 0] - KY * y_fixed)
                vx_lt = v1_lt*np.cos(omega * t - KX * X[:, 0] - KY * y_fixed) * (KX / k) if k > 0 else 0.0
        
        # Get FD 1D sinusoidal solution (always, to ensure clean comparison)
        cache_key = (round(float(t), 8), int(N_fd), round(float(nu_fd), 6))
        if cache_key not in fd_1d_cache:
            x_fd_1d, rho_fd_1d, v_fd_1d, _phi_fd_1d, _n, _rho_max = lax_solution1D_sinusoidal(
                time=t, N=N_fd, nu=nu_fd, lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
                gravity=True, isplot=False, comparison=False, animation=True
            )
            fd_1d_cache[cache_key] = (
                (x_fd_1d + xmin),
                rho_fd_1d,
                v_fd_1d
            )
        x_fd_line, rho_fd_1d, v_fd_1d = fd_1d_cache[cache_key]
        
        rho_fd_interp = interp1d(x_fd_line, rho_fd_1d, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])
        v_fd_interp = interp1d(x_fd_line, v_fd_1d, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])
        
        # Column index
        c = row_idx
        
        # Top row: density
        ax_rho = fig.add_subplot(grid[0, c])
        ax_rho.plot(X[:, 0], rho_pinn, label="GRINN", color='c', linewidth=2)
        if SHOW_LINEAR_THEORY and rho_lt is not None and np.isclose(KY, 0.0) and (a < 0.1):
            ax_rho.plot(X[:, 0], rho_lt, label="LT", linestyle='--', color='firebrick', linewidth=1.5)
        ax_rho.plot(X[:, 0], rho_fd_interp, label="FD (1D)", color='k', linewidth=1)
        ax_rho.set_title(f"t={t:.1f}")
        ax_rho.set_ylabel(r"$\rho$")
        ax_rho.grid(True)
        # Dynamic y-axis limits
        rho_all = [rho_pinn, rho_fd_interp]
        if SHOW_LINEAR_THEORY and rho_lt is not None:
            rho_all.append(rho_lt)
        rho_min = min(np.min(rho) for rho in rho_all)
        rho_max = max(np.max(rho) for rho in rho_all)
        rho_range = rho_max - rho_min
        padding = max(0.1 * rho_range, 0.05)
        ax_rho.set_ylim(rho_min - padding, rho_max + padding)
        if c == 0:
            ax_rho.legend(loc='upper right', fontsize=8)
        
        # Second row: epsilon for density
        eps_rho = 200.0 * np.abs(rho_pinn - rho_fd_interp) / (rho_pinn + rho_fd_interp + 1e-6)
        ax_eps_rho = fig.add_subplot(grid[1, c])
        ax_eps_rho.plot(X[:, 0], eps_rho, color='k', linewidth=1, label='FD')
        if SHOW_LINEAR_THEORY and rho_lt is not None and np.isclose(KY, 0.0) and (a < 0.1):
            eps_rho_lt = 200.0 * np.abs(rho_pinn - rho_lt) / (rho_pinn + rho_lt + 1e-6)
            ax_eps_rho.plot(X[:, 0], eps_rho_lt, color='firebrick', linestyle='--', linewidth=1, label='LT')
        ax_eps_rho.set_ylabel(r"$\varepsilon$")
        ax_eps_rho.grid(True)
        if c == 0:
            ax_eps_rho.legend(loc='upper right', fontsize=8)
        
        # Third row: velocity
        ax_v = fig.add_subplot(grid[2, c])
        ax_v.plot(X[:, 0], vx_pinn, label="GRINN", color='c', linewidth=2)
        if SHOW_LINEAR_THEORY and vx_lt is not None and np.isclose(KY, 0.0) and (a < 0.1):
            ax_v.plot(X[:, 0], vx_lt, label="LT", linestyle='--', color='firebrick', linewidth=1.5)
        ax_v.plot(X[:, 0], v_fd_interp, label="FD (1D)", color='k', linewidth=1)
        ax_v.set_ylabel(r"$v$")
        ax_v.grid(True)
        # Dynamic y-axis limits
        v_all = [vx_pinn, v_fd_interp]
        if SHOW_LINEAR_THEORY and vx_lt is not None:
            v_all.append(vx_lt)
        v_min = min(np.min(v) for v in v_all)
        v_max = max(np.max(v) for v in v_all)
        v_range = v_max - v_min
        padding = max(0.1 * v_range, 0.005)
        ax_v.set_ylim(v_min - padding, v_max + padding)
        if c == 0:
            ax_v.legend(loc='upper right', fontsize=8)
        
        # Fourth row: epsilon for velocity
        v_ref = v_fd_interp
        v_pred = vx_pinn
        eps_v = 200.0 * np.abs(v_pred - v_ref) / (v_pred + v_ref + 2.0)
        ax_eps_v = fig.add_subplot(grid[3, c])
        ax_eps_v.plot(X[:, 0], eps_v, color='k', linewidth=1, label='FD')
        if SHOW_LINEAR_THEORY and vx_lt is not None and np.isclose(KY, 0.0) and (a < 0.1):
            eps_v_lt = 200.0 * np.abs(v_pred - vx_lt) / (v_pred + vx_lt + 2.0)
            ax_eps_v.plot(X[:, 0], eps_v_lt, color='firebrick', linestyle='--', linewidth=1, label='LT')
        ax_eps_v.set_xlabel("x")
        ax_eps_v.set_ylabel(r"$\varepsilon$")
        ax_eps_v.grid(True)
        if c == 0:
            ax_eps_v.legend(loc='upper right', fontsize=8)
    
    # Reduce outer margins
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.12)
    
    # Save the figure if requested
    if save_plots:
        desktop_path = r"C:\Users\tirth\OneDrive\Desktop"
        output_dir = os.path.join(desktop_path, "model test plots")
        os.makedirs(output_dir, exist_ok=True)
        time_str = "_".join([f"{t:.4f}" for t in time_points])
        save_path = os.path.join(output_dir, f"1d_cross_section_t{time_str}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 1D cross-section plot to {save_path}")
    
    plt.show()
    return fig


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
                            fit_lognorm=True, fit_powerlaw_tail=True, powerlaw_threshold=0.8,
                            fd_backend="cpu", fd_cache=None):
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
        fd_cache: Instance of FDSolutionManager for reusing FD solutions.
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    if fd_cache is None:
        raise ValueError("fd_cache is required for density PDF plots.")
    
    if N is None:
        if DIMENSION >= 3 and str(PERTURBATION_TYPE).lower() == "sinusoidal":
            N = FD_N_3D
        else:
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
        
        # Get FD density field from cache and interpolate to PINN grid
        fd_solution = fd_cache.get_solution(t, N=N, nu=nu, backend=fd_backend)
        x_fd = fd_solution["x"]
        y_fd = fd_solution["y"]
        if fd_solution["dimension"] == 3:
            z_fd = fd_solution["z"]
            z_slice = SLICE_Z
            z_idx = np.argmin(np.abs(z_fd - z_slice))
            if np.abs(z_fd[z_idx] - z_slice) > 1e-6:
                print(f"Density PDF: using nearest z-slice {z_fd[z_idx]:.4f} for requested z={z_slice:.4f}")
            rho_fd = fd_solution["rho"][:, :, z_idx]
        else:
            rho_fd = fd_solution["rho"]
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
    parser.add_argument('--plot-type', type=str, default='spatial', choices=['spatial', 'pdf', '1d', 'cross-section', '3d'],
                       help='Plot selection: spatial (default), pdf (adds density PDF), 1d/cross-section (adds 1D plots), or 3d (adds 3D sinusoidal cubes).')
    parser.add_argument('--which', type=str, default='both', choices=['density', 'velocity', 'both'],
                       help='Which field to plot for spatial plots: density, velocity, or both (default: both)')
    parser.add_argument('--nu', type=float, default=0.5, help='Courant number for FD solver (default: 0.5)')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots to disk (only display them)')
    parser.add_argument('--no-fit', action='store_true', help='Do not fit distributions to PDF plots')
    parser.add_argument('--powerlaw-threshold', type=float, default=0.8, 
                       help='Minimum log density for power-law fit (default: 0.8)')
    parser.add_argument('--fd-backend', type=str, default='cpu', choices=['cpu', 'gpu', 'torch'],
                       help='FD solver backend: cpu (default) or gpu/torch for GPU-accelerated solver (works for both spatial and 1D cross-section plots)')
    parser.add_argument('--y-fixed', type=float, default=0.6,
                       help='y value for 1D cross-section slice (default: 0.6)')
    parser.add_argument('--N-fd', type=int, default=None,
                       help='Grid size for FD solver for all plots (default: FD_N_1D from config.py for 1D plots, N_GRID for 2D spatial plots)')
    parser.add_argument('--nu-fd', type=float, default=0.5,
                       help='Courant number for 1D LAX solver (default: 0.5)')
    parser.add_argument('--plot-growth', action='store_true',
                       help='Generate density growth plot (PINN vs LAX over time). Only works for power_spectrum perturbation type.')
    parser.add_argument('--growth-tmax', type=float, default=None,
                       help='Maximum time for density growth plot (default: GROWTH_PLOT_TMAX from config.py)')
    parser.add_argument('--growth-dt', type=float, default=None,
                       help='Time step for density growth plot (default: GROWTH_PLOT_DT from config.py)')
    
    args = parser.parse_args()
    
    # Debug: Print N_fd value if provided
    if args.N_fd is not None:
        print(f"Command line: --N-fd argument parsed as args.N_fd = {args.N_fd} (for all plots)")
    else:
        print(f"Command line: --N-fd not provided, will use defaults (N_GRID={N_GRID} for 2D runs, FD_N_3D={FD_N_3D} for 3D sinusoidal runs)")
    
    # Parse time points (handle spaces around commas)
    try:
        # Split by comma and strip whitespace from each element
        time_points = np.array([float(t.strip()) for t in args.time_points.replace(' ', '').split(',') if t.strip()])
    except ValueError as e:
        print(f"Error: time_points must be comma-separated numbers. Got: {args.time_points}")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Set up initial parameters (same as train.py) - need to calculate xmax/ymax before loading model
    # Use num_of_waves_config (imported at top level) to avoid scoping issues
    lam, rho_1, num_of_waves, tmax_calc, _, _, _ = input_taker(wave, a, num_of_waves_config, tmax, 0, 0, 0)
    jeans, alpha = req_consts_calc(lam, rho_1)
    
    xmax = xmin + lam * num_of_waves
    ymax = ymin + lam * num_of_waves
    zmax = zmin + lam * num_of_waves if DIMENSION >= 3 else None
    
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
        net = load_model(model_path, xmax, ymax, zmax=zmax)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize shared velocity fields for consistent PINN/FD initial conditions
    shared_vx_np = None
    shared_vy_np = None
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        v_1 = a * cs
        shared_vx_np, shared_vy_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
        set_shared_velocity_fields(shared_vx_np, shared_vy_np)
    
    initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)
    fd_cache = FDSolutionManager(initial_params, lam, num_of_waves, rho_1, PERTURBATION_TYPE, shared_vx_np, shared_vy_np)
    
    # Generate plots
    save_plots = not args.no_save
    
    # Check if GPU is available when GPU backend is requested
    if args.fd_backend.lower() in ['gpu', 'torch']:
        if not torch.cuda.is_available():
            print(f"Warning: GPU backend requested but CUDA not available. Falling back to CPU.")
            args.fd_backend = 'cpu'
        else:
            print(f"Using GPU-accelerated FD solver (CUDA available)")
    
    # Determine if we should create 1D cross-section plots
    create_1d_plots = (args.plot_type in ['1d', 'cross-section'])
    
    # Spatial plots are always generated by default (regardless of --plot-type)
    # The --plot-type flag only determines what ADDITIONAL plots to generate
    create_spatial_plots = True
    
    # Determine if we should create PDF or 3D plots
    create_pdf_plots = (args.plot_type == 'pdf')
    create_3d_plots = (args.plot_type == '3d')
    
    # Use --N-fd for all plots if provided, otherwise use defaults
    N_for_all = args.N_fd if args.N_fd is not None else None
    
    # Generate spatial comparison plots (if requested)
    if create_spatial_plots:
        if args.which == 'both':
            print(f"\nGenerating density and velocity comparison plots...")
            # Create both plots without showing them immediately
            fig_density, _ = create_comparison_plots(
                net, initial_params, time_points, which='density', N=N_for_all, nu=args.nu,
                save_plots=save_plots, fd_backend=args.fd_backend, show_plot=False, fd_cache=fd_cache
            )
            fig_velocity, _ = create_comparison_plots(
                net, initial_params, time_points, which='velocity', N=N_for_all, nu=args.nu,
                save_plots=save_plots, fd_backend=args.fd_backend, show_plot=False, fd_cache=fd_cache
            )
            # Show both plots together
            plt.show()
        else:
            print(f"\nGenerating {args.which} comparison plots...")
            create_comparison_plots(
                net, initial_params, time_points, which=args.which, N=N_for_all, nu=args.nu,
                save_plots=save_plots, fd_backend=args.fd_backend, fd_cache=fd_cache
            )
    
    # Generate 1D cross-section plots (if requested, as additional plots)
    if create_1d_plots:
        print(f"\nGenerating 1D cross-section plots (additional)...")
        print(f"Note: Using sinusoidal initial conditions regardless of PERTURBATION_TYPE setting")
        create_1d_cross_section_plot(
            net, initial_params, time_points, y_fixed=args.y_fixed,
            N_fd=N_for_all, nu_fd=args.nu_fd, save_plots=save_plots,
            fd_backend=args.fd_backend, fd_cache=fd_cache
        )
    
    # Generate PDF plots (if requested)
    if create_pdf_plots:
        print(f"\nGenerating density PDF plots...")
        create_density_pdf_plot(
            net, initial_params, time_points, N=N_for_all, nu=args.nu, save_plots=save_plots,
            fit_lognorm=not args.no_fit, fit_powerlaw_tail=not args.no_fit,
            powerlaw_threshold=args.powerlaw_threshold, fd_backend=args.fd_backend,
            fd_cache=fd_cache
        )
    
    # Generate 3D sinusoidal plots (if requested)
    if create_3d_plots:
        print(f"\nGenerating 3D sinusoidal comparison plots (additional)...")
        create_3d_sinusoidal_case_plot(
            net, initial_params, time_points, N=N_for_all, nu=args.nu, save_plots=save_plots,
            fd_backend=args.fd_backend, fd_cache=fd_cache
        )
    
    # Generate density growth plot (additional plot when --plot-growth is given, only for power spectrum case)
    if args.plot_growth:
        if str(PERTURBATION_TYPE).lower() == "power_spectrum":
            print(f"\nGenerating density growth plot (additional plot)...")
            tmax_growth = args.growth_tmax if args.growth_tmax is not None else GROWTH_PLOT_TMAX
            dt_growth = args.growth_dt if args.growth_dt is not None else GROWTH_PLOT_DT
            print(f"Using tmax={tmax_growth}, dt={dt_growth} for density growth plot")
            create_density_growth_plot(net, initial_params, tmax=tmax_growth, dt=dt_growth)
        else:
            print(f"\nWarning: --plot-growth is only available for power_spectrum perturbation type.")
            print(f"Current PERTURBATION_TYPE is '{PERTURBATION_TYPE}'. Skipping density growth plot.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
