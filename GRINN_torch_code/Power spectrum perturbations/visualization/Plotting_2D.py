from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.autograd import Variable
import torch
import scipy
import os
from numerical_solvers.LAX_2D import lax_solution, lax_solution_with_shared_velocity
from numerical_solvers.LAX_2D import lax_solution1D_sinusoidal as lax_solution1D_sin
from config import SAVE_STATIC_SNAPSHOTS, SNAPSHOT_DIR, PERTURBATION_TYPE, cs, const, G, rho_o, TIMES_1D, a, KX, KY, FD_N_1D, FD_N_2D, POWER_EXPONENT, FILTER_SCALE, N_GRID
from config import USE_XPINN, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y, SHOW_INTERFACE_LINES, INTERFACE_AVERAGING, RANDOM_SEED

# Global variable to store shared velocity fields for plotting
_shared_vx_np = None
_shared_vy_np = None

def set_shared_velocity_fields(vx_np, vy_np):
    """Set shared velocity fields for consistent FD plotting"""
    global _shared_vx_np, _shared_vy_np
    _shared_vx_np = vx_np
    _shared_vy_np = vy_np

def get_fd_default_params():
    """
    Get default FD parameters that match PINN training configuration.
    This ensures consistency between PINN and FD initial conditions.
    
    Returns:
        dict with keys: use_velocity_ps, ps_index, vel_rms, random_seed
    """
    return {
        'use_velocity_ps': (str(PERTURBATION_TYPE).lower() == "power_spectrum"),
        'ps_index': POWER_EXPONENT,
        'vel_rms': a * cs,
        'random_seed': RANDOM_SEED
    }

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() \
    else "cuda:0" if torch.cuda.is_available() else "cpu"


def predict_xpinn(nets, x, y, t, xmin, xmax, ymin, ymax):
    """
    Predict solution using XPINN with multiple networks.
    Automatically determines which subdomain(s) each point belongs to and evaluates accordingly.
    For points at interfaces, combines predictions using configured averaging method.
    
    Args:
        nets: List of neural networks (one per subdomain)
        x, y, t: Coordinate tensors [N, 1]
        xmin, xmax, ymin, ymax: Global domain bounds
    
    Returns:
        Combined predictions from all subdomains [N, 4] (rho, vx, vy, phi)
    """
    import methods.xpinn_decomposition as xpinn_utils
    
    N = x.shape[0]
    device = x.device
    
    # Initialize output
    output = torch.zeros(N, 4, device=device, dtype=x.dtype)
    
    # If single subdomain, just evaluate directly
    if len(nets) == 1:
        return nets[0]([x, y, t])
    
    # For each point, determine which subdomain(s) it belongs to
    for i in range(len(nets)):
        subdomain_bounds = xpinn_utils.get_subdomain_bounds(
            i, xmin, xmax, ymin, ymax, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y
        )
        x_min, x_max, y_min, y_max = subdomain_bounds
        
        # Find points in this subdomain (with small tolerance for boundaries)
        tol = 1e-6
        mask = ((x >= x_min - tol) & (x <= x_max + tol) & 
                (y >= y_min - tol) & (y <= y_max + tol)).squeeze()
        
        if mask.any():
            # Evaluate network for points in this subdomain
            x_sub = x[mask]
            y_sub = y[mask]
            t_sub = t[mask]
            
            # Move data to network's device
            net_device = next(nets[i].parameters()).device
            x_sub = x_sub.to(net_device)
            y_sub = y_sub.to(net_device)
            t_sub = t_sub.to(net_device)
            
            pred_sub = nets[i]([x_sub, y_sub, t_sub])
            
            # Move prediction back to output device
            pred_sub = pred_sub.to(device)
            
            # For simple mean averaging (default)
            if INTERFACE_AVERAGING == 'mean':
                # Add prediction (will average later if point is in multiple subdomains)
                output[mask] += pred_sub
            else:
                # For other strategies, just use first subdomain's prediction
                output[mask] = pred_sub
    
    # For mean averaging, divide by number of subdomains that covered each point
    # This automatically handles interface points by averaging
    if INTERFACE_AVERAGING == 'mean' and len(nets) > 1:
        # Count how many subdomains covered each point
        counts = torch.zeros(N, 1, device=device, dtype=x.dtype)
        for i in range(len(nets)):
            subdomain_bounds = xpinn_utils.get_subdomain_bounds(
                i, xmin, xmax, ymin, ymax, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y
            )
            x_min, x_max, y_min, y_max = subdomain_bounds
            tol = 1e-6
            mask = ((x >= x_min - tol) & (x <= x_max + tol) & 
                    (y >= y_min - tol) & (y <= y_max + tol)).squeeze()
            counts[mask] += 1
        
        # Avoid division by zero
        counts = torch.clamp(counts, min=1.0)
        output = output / counts
    
    return output


def add_interface_lines(ax, xmin, xmax, ymin, ymax):
    """
    Add interface lines to plot showing subdomain boundaries.
    
    Args:
        ax: Matplotlib axis
        xmin, xmax, ymin, ymax: Domain bounds
    """
    if not SHOW_INTERFACE_LINES or NUM_SUBDOMAINS_X == 1 and NUM_SUBDOMAINS_Y == 1:
        return
    
    # Calculate subdomain boundaries
    dx = (xmax - xmin) / NUM_SUBDOMAINS_X
    dy = (ymax - ymin) / NUM_SUBDOMAINS_Y
    
    # Vertical lines (constant x)
    for i in range(1, NUM_SUBDOMAINS_X):
        x_interface = xmin + i * dx
        ax.axvline(x=x_interface, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Interface' if i == 1 else '')
    
    # Horizontal lines (constant y)
    for j in range(1, NUM_SUBDOMAINS_Y):
        y_interface = ymin + j * dy
        ax.axhline(y=y_interface, color='white', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add legend only once
    if NUM_SUBDOMAINS_X > 1 or NUM_SUBDOMAINS_Y > 1:
        handles, labels = ax.get_legend_handles_labels()
        if 'Interface' in labels:
            # Only show interface label once
            unique_labels = []
            unique_handles = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=8)


def plot_function(net, time_array, initial_params, velocity=False, isplot=False, animation=False):
    """
    Plot function for 1D slices through 2D domain
    
    Args:
        net: Trained neural network OR list of networks (for XPINN)
        time_array: Array of times to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        velocity: Whether to plot velocity
        isplot: Whether to save plots
        animation: Whether this is for animation
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Handle both single network and list of networks
    if isinstance(net, list):
        nets = net
        use_xpinn = len(nets) > 1
    else:
        nets = [net]
        use_xpinn = False  
    # rho_o imported from config.py
    num_of_waves_x = (xmax-xmin)/lam
    num_of_waves_y = (ymax-ymin)/lam
    if animation:
        ## Converting the float (time-input) to an numpy array for animation
        ## Ignore this when the function is called in isolation
        time_array = np.array([time_array])
        # print("time",np.asarray(time_array))
    
    rho_max_Pinns = []    
    peak_lst=[]
    pert_xscale=[]
    for t in time_array:
        print("Plotting at t=", t)
        
        # Create 1D slice through the domain (like in notebook: Y = 0.6)
        X = np.linspace(xmin, xmax, 1000).reshape(1000, 1)
        Y = 0.6 * np.ones(1000).reshape(1000, 1)  # Fixed Y slice like in notebook
        t_ = t * np.ones(1000).reshape(1000, 1)
        
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        
        # Evaluate network(s)
        if use_xpinn:
            output_0 = predict_xpinn(nets, pt_x_collocation, pt_y_collocation, pt_t_collocation, xmin, xmax, ymin, ymax)
        else:
            output_0 = nets[0]([pt_x_collocation, pt_y_collocation, pt_t_collocation])
        
        rho_pred0 = output_0[:, 0:1].data.cpu().numpy()
        v_pred_x0 = output_0[:, 1:2].data.cpu().numpy()
        v_pred_y0 = output_0[:, 2:3].data.cpu().numpy()
        phi_pred0 = output_0[:, 3:4].data.cpu().numpy()
 
        rho_max_PN = np.max(rho_pred0)
        
        ## Theoretical Values
        #rho_theory = np.max(rho_o + rho_1*np.exp(alpha * t)*np.cos(2*np.pi*X[:, 0:1]/lam))
        #rho_theory0 = np.max(rho_o + rho_1*np.exp(alpha * 0)*np.cos(2*np.pi*X[:, 0:1]/lam)) ## at t =0 
        
        #diff=abs(rho_max_PN-rho_theory)/abs(rho_max_PN+rho_theory) * 2  ## since the den is rhomax+rhotheory

        
#         ### Difference between peaks for the PINNs solution
        
#         rho_pred0Flat=rho_pred0.reshape(-1)
#         peaks,_=scipy.signal.find_peaks(rho_pred0Flat)
#         peak_lst.append(peaks)
        
#         growth_pert=(rho_theory-rho_theory0)/rho_theory0*100 ## growth percentage
        
#         peak_diff=(rho_pred0Flat[peaks[1]]-rho_pred0Flat[peaks[0]])/(rho_pred0Flat[peaks[1]]+rho_pred0Flat[peaks[0]])

        #g_pred0=phi_x = dde.grad.jacobian(phi_pred0, X, i=0, j=0)
        if isplot:              
            # Create output folder if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Density plot
            plt.figure(1)
            plt.plot(X, rho_pred0, label="t={}".format(round(t,2)))
            plt.ylabel(r"$\rho$")
            plt.xlabel("x")
            plt.grid()
            plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
            plt.title(r"PINNs Solution for $\lambda$ = {} $\lambda_J$".format(round(lam/(2*np.pi),2)))
            #plt.savefig(output_folder+'/PINNS_density'+str(lam)+'_'+str(int(num_of_waves_x))+'_'+str(tmax)+'.png', dpi=300)

            if velocity == True:
                # Velocity plots
                plt.figure(2)
                plt.plot(X, v_pred_x0, '--', label="t={}".format(round(t,2)))
                plt.ylabel("$v_x$")
                plt.xlabel("x")
                plt.title("PINNs Solution Velocity")
                plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
                #plt.savefig(output_folder+'/PINNS_velocity'+str(lam)+'_'+str(int(num_of_waves_x))+'_'+str(tmax)+'.png', dpi=300)

            # Potential plot
            plt.figure(3)
            plt.plot(X, phi_pred0, '--', label="t={}".format(round(t,2)))
            plt.ylabel(r"$\phi$")
            plt.xlabel("x")
            plt.title("PINNs Solution Potential")
            plt.legend(numpoints=1, loc='upper right', fancybox=True, shadow=True)
            #plt.savefig(output_folder+'/phi'+str(lam)+'_'+str(int(num_of_waves_x))+'_'+str(tmax)+'.png', dpi=300)

        
        else:  
            if animation:
                return X, rho_pred0, v_pred_x0, v_pred_y0, phi_pred0, rho_max_PN
            else:
                return X, rho_pred0, rho_max_PN


def Two_D_surface_plots(net, time, initial_params, ax=None, which="density"):
    """
    Create 2D surface plots with velocity vectors

    Args:
        net: Trained neural network OR list of networks (for XPINN)
        time: Time to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        ax: Optional axis to plot on
        which: "density" or "velocity"
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Handle both single network and list of networks
    if isinstance(net, list):
        nets = net
        use_xpinn = len(nets) > 1
    else:
        nets = [net]
        use_xpinn = False
    
    # Use N_GRID for consistency with FD solver resolution
    # Exclude right boundary for periodic domains to avoid double-counting
    Q = N_GRID
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    tau, phi = np.meshgrid(xs, ys) 
    Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
    t_00 = time * np.ones(Q**2).reshape(Q**2, 1)
    
    # Convert to tensors
    pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    
    # Evaluate network(s)
    if use_xpinn:
        output_00 = predict_xpinn(nets, pt_x_collocation, pt_y_collocation, pt_t_collocation, xmin, xmax, ymin, ymax)
    else:
        output_00 = nets[0]([pt_x_collocation, pt_y_collocation, pt_t_collocation])
    
    rho = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
    U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
    V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)

    if ax is None:  # for single plot
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 

    # Clean velocity fields and avoid zero-length arrows
    U_clean = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    V_clean = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    Vmag = np.sqrt(U_clean**2 + V_clean**2)
    mask = Vmag > 1e-12

    if which == "density":
        pc = ax.pcolormesh(tau, phi, rho, shading='auto', cmap='YlOrBr', vmin=np.min(rho), vmax=np.max(rho))
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(
            tau[skip][mask[skip]], phi[skip][mask[skip]],
            U_clean[skip][mask[skip]], V_clean[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("Density, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:  # velocity magnitude surface plot
        pc = ax.pcolormesh(tau, phi, Vmag, shading='auto', cmap='viridis', vmin=np.min(Vmag), vmax=np.max(Vmag))
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(
            tau[skip][mask[skip]], phi[skip][mask[skip]],
            U_clean[skip][mask[skip]], V_clean[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("Velocity, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    # Add interface lines if using XPINN
    if use_xpinn:
        add_interface_lines(ax, xmin, xmax, ymin, ymax)
    
    return pc


def create_2d_animation(net, initial_params, time_points=None, which="density", fps=2, save_path=None, fixed_colorbar=True, verbose=False):
    """
    Create an animated 2D surface plot showing evolution over time

    Args:
        net: Trained neural network OR list of networks (for XPINN)
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: Array of time points for animation (default: 50 points from 0 to 2.0)
        which: "density" or "velocity"
        fps: Frames per second for animation
        save_path: Optional path to save the animation (e.g., 'animation.mp4')
    """
    if time_points is None:
        # Use the provided training tmax from initial_params to bound animation time
        _xmin, _xmax, _ymin, _ymax, _rho_1, _alpha, _lam, _output_folder, tmax = initial_params
        time_points = np.linspace(0.0, float(tmax), 80)
    
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Handle both single network and list of networks
    if isinstance(net, list):
        nets = net
        use_xpinn = len(nets) > 1
    else:
        nets = [net]
        use_xpinn = False
    
    if verbose:
        print(f"Creating 2D animation with {len(time_points)} frames...")
    
    # Create output directory for saving plots: always use config.SNAPSHOT_DIR/GRINN
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    
    # Get data for first frame to set up colorbar limits
    # Use N_GRID for consistency with FD solver resolution
    # Exclude right boundary for periodic domains to avoid double-counting
    Q = N_GRID
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    tau, phi = np.meshgrid(xs, ys) 
    Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
    t_00 = time_points[0] * np.ones(Q**2).reshape(Q**2, 1)
    
    # Convert to tensors for first frame
    pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    
    # Get first frame data to set colorbar limits
    if use_xpinn:
        output_00 = predict_xpinn(nets, pt_x_collocation, pt_y_collocation, pt_t_collocation, xmin, xmax, ymin, ymax)
    else:
        output_00 = nets[0]([pt_x_collocation, pt_y_collocation, pt_t_collocation])
    rho_first = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
    U_first = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
    V_first = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)
    
    # (removed temporary quick-check print of mean(U), mean(V))
    
    # Optionally precompute fixed color limits using first and last frames
    fixed_vmin = None
    fixed_vmax = None
    if which == "density" and fixed_colorbar:
        # Use N_GRID for consistency with FD solver resolution
        # Exclude right boundary for periodic domains to avoid double-counting
        Q = N_GRID
        xs = np.linspace(xmin, xmax, Q, endpoint=False)
        ys = np.linspace(ymin, ymax, Q, endpoint=False)
        tau, phi = np.meshgrid(xs, ys)
        Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
        # First frame
        t_first = time_points[0] * np.ones(Q**2).reshape(Q**2, 1)
        pt_x = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
        pt_t = Variable(torch.from_numpy(t_first).float(), requires_grad=True).to(device)
        if use_xpinn:
            pred_first = predict_xpinn(nets, pt_x, pt_y, pt_t, xmin, xmax, ymin, ymax)
            rho_first = pred_first[:, 0].data.cpu().numpy().reshape(Q, Q)
        else:
            rho_first = nets[0]([pt_x, pt_y, pt_t])[:, 0].data.cpu().numpy().reshape(Q, Q)
        # Last frame
        t_last = time_points[-1] * np.ones(Q**2).reshape(Q**2, 1)
        pt_t_last = Variable(torch.from_numpy(t_last).float(), requires_grad=True).to(device)
        if use_xpinn:
            pred_last = predict_xpinn(nets, pt_x, pt_y, pt_t_last, xmin, xmax, ymin, ymax)
            rho_last = pred_last[:, 0].data.cpu().numpy().reshape(Q, Q)
        else:
            rho_last = nets[0]([pt_x, pt_y, pt_t_last])[:, 0].data.cpu().numpy().reshape(Q, Q)
        fixed_vmin = min(np.min(rho_first), np.min(rho_last))
        fixed_vmax = max(np.max(rho_first), np.max(rho_last))
        if fixed_vmin == fixed_vmax:
            eps = 1e-6 if fixed_vmin == 0 else 1e-6 * abs(fixed_vmin)
            fixed_vmin, fixed_vmax = fixed_vmin - eps, fixed_vmax + eps

    # Set up initial plot
    if which == "density":
        # Handle flat initial frame by expanding color limits slightly
        if fixed_colorbar and fixed_vmin is not None:
            vmin_use, vmax_use = fixed_vmin, fixed_vmax
        else:
            rmin, rmax = np.min(rho_first), np.max(rho_first)
            if not np.isfinite(rmin) or not np.isfinite(rmax):
                rmin, rmax = 0.0, 1.0
            if rmin == rmax:
                eps = 1e-6 if rmin == 0 else 1e-6 * abs(rmin)
                rmin, rmax = rmin - eps, rmax + eps
            vmin_use, vmax_use = rmin, rmax
        pc = ax.pcolormesh(tau, phi, rho_first, shading='auto', cmap='YlOrBr', vmin=vmin_use, vmax=vmax_use)
        pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
        ax.set_title(f"{pert_str} Density, t={time_points[0]:.2f}")
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:  # velocity magnitude surface plot
        Vmag_first = np.sqrt(U_first**2 + V_first**2)
        pc = ax.pcolormesh(tau, phi, Vmag_first, shading='auto', cmap='viridis')
        pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
        ax.set_title(f"{pert_str} Velocity, t={time_points[0]:.2f}")
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Do not save the initial frame by default; animation frames and optional snapshots cover needs
    
    def animate(frame):
        t = time_points[frame]
        if verbose:
            print(f"Animating frame {frame+1}/{len(time_points)} at t={t:.2f}")
        
        # Get data for current time
        t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
        
        # Convert to tensors
        pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
        
        # Evaluate network(s)
        if use_xpinn:
            output_00 = predict_xpinn(nets, pt_x_collocation, pt_y_collocation, pt_t_collocation, xmin, xmax, ymin, ymax)
        else:
            output_00 = nets[0]([pt_x_collocation, pt_y_collocation, pt_t_collocation])
        
        rho = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
        U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
        V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)
        
        # Update plot data
        if which == "density":
            pc.set_array(rho.ravel())
            # Update color limits only if not fixed
            if not fixed_colorbar:
                rmin, rmax = np.min(rho), np.max(rho)
                if rmin == rmax:
                    eps = 1e-6 if rmin == 0 else 1e-6 * abs(rmin)
                    rmin, rmax = rmin - eps, rmax + eps
                pc.set_clim(vmin=rmin, vmax=rmax)
            # Add interface lines if using XPINN
            if use_xpinn:
                add_interface_lines(ax, xmin, xmax, ymin, ymax)
            pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
            ax.set_title(f"{pert_str} Density, t={t:.2f}")
        else:  # velocity magnitude surface plot
            Vmag = np.sqrt(U**2 + V**2)
            pc.set_array(Vmag.ravel())
            # Add interface lines if using XPINN
            if use_xpinn:
                add_interface_lines(ax, xmin, xmax, ymin, ymax)
            pert_str = "Sinusoidal" if str(PERTURBATION_TYPE).lower() == "sinusoidal" else "Power Spectrum"
            ax.set_title(f"{pert_str} Velocity, t={t:.2f}")
        
        # Save every 10th frame for static snapshots (disabled to avoid extra plots)
        # if frame % 10 == 0:
        #     save_path_frame = os.path.join(output_dir, f"{which}_t_{t:.2f}.png")
        #     plt.savefig(save_path_frame, dpi=300, bbox_inches='tight')
        #     if verbose:
        #         print(f"Saved frame to {save_path_frame}")
        
        return pc
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_points), 
                                 interval=1000/fps, blit=False, repeat=True)
    
    # Save animation with appropriate writer/extension
    try:
        if animation.writers.is_available('ffmpeg'):
            animation_path = os.path.join(output_dir, f"{which}_animation.mp4")
            if verbose:
                print(f"Saving animation to {animation_path} with ffmpeg...")
            anim.save(animation_path, writer='ffmpeg', fps=fps)
            saved_format = 'mp4'
        else:
            animation_path = os.path.join(output_dir, f"{which}_animation.gif")
            if verbose:
                print(f"ffmpeg not available. Saving animation to {animation_path} with Pillow...")
            anim.save(animation_path, writer='pillow', fps=fps)
            saved_format = 'gif'
        if verbose:
            print("Animation saved successfully!")
    except Exception as e:
        print(f"Animation save failed: {e}")
        raise
    
    # Optional static snapshots uniformly over [0, tmax]
    # Static snapshots disabled to prevent extra plots
    # if SAVE_STATIC_SNAPSHOTS:
    #     snapshot_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    #     os.makedirs(snapshot_dir, exist_ok=True)
    #     # Determine tmax from initial_params
    #     _xmin, _xmax, _ymin, _ymax, _rho_1, _alpha, _lam, _output_folder, tmax_val = initial_params
    #     times_static = np.linspace(0.0, float(tmax_val), 5)
    #     if verbose:
    #         print(f"Saving {len(times_static)} static snapshots to {snapshot_dir} over [0, {tmax_val}]...")
    #     for t in times_static:
    #         t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
    #         pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    #         pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
    #         pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    #         output_00 = net([pt_x_collocation, pt_y_collocation, pt_t_collocation])
    #         rho = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
    #         U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
    #         V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)
    #
    #         fig_static, ax_static = plt.subplots(figsize=(8, 8))
    #         if which == "density":
    #             pc_static = ax_static.pcolormesh(tau, phi, rho, shading='auto', cmap='YlOrBr')
    #             cbar_static = plt.colorbar(pc_static, shrink=0.6, location='right')
    #             cbar_static.formatter.set_powerlimits((0, 0))
    #             cbar_static.ax.set_title(r"$\rho$", fontsize=14)
    #         else:
    #             Vmag = np.sqrt(U**2 + V**2)
    #             pc_static = ax_static.pcolormesh(tau, phi, Vmag, shading='auto', cmap='viridis')
    #             cbar_static = plt.colorbar(pc_static, shrink=0.6, location='right')
    #             cbar_static.ax.set_title(r" $|v|$", fontsize=14)
    #         ax_static.set_xlim(xmin, xmax)
    #         ax_static.set_ylim(ymin, ymax)
    #         ax_static.set_xlabel("x")
    #         ax_static.set_ylabel("y")
    #         plt.tight_layout()
    #         static_save_path = os.path.join(snapshot_dir, f"{which}_static_t_{t:.2f}.png")
    #         plt.savefig(static_save_path, dpi=300, bbox_inches='tight')
    #         plt.close(fig_static)
    #         if verbose:
    #             print(f"Saved static snapshot to {static_save_path}")
    
    # Generate 5x3 comparison tables for both density and velocity (only once per animation call)
    if verbose:
        print("Generating 5x3 comparison tables...")
    
    # Only generate comparison tables if this is the density animation call
    # This prevents duplicate generation when both density and velocity animations are created
    if which == "density":
        print("Generating density comparison table...")
        # Create comparison table for density - use N_GRID to match PINN's power spectrum resolution
        create_5x3_comparison_table(net, initial_params, which="density", N=N_GRID, nu=0.5)
        
        print("Generating velocity comparison table...")
        # Create comparison table for velocity - use N_GRID to match PINN's power spectrum resolution
        create_5x3_comparison_table(net, initial_params, which="velocity", N=N_GRID, nu=0.5)
    
    # Display animation inline if in a notebook
    try:
        from IPython.display import HTML, display
        import base64
        with open(animation_path, 'rb') as f:
            data = f.read()
        data_base64 = base64.b64encode(data).decode('utf-8')
        if saved_format == 'mp4':
            video_html = f'''
    <div style="text-align: center;">
        <h3>{which.title()} Evolution Animation</h3>
        <video width="600" height="600" controls autoplay loop>
            <source src="data:video/mp4;base64,{data_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    '''
            display(HTML(video_html))
        else:
            img_html = f'''
    <div style="text-align: center;">
        <h3>{which.title()} Evolution Animation</h3>
        <img src="data:image/gif;base64,{data_base64}" width="600" height="600" />
    </div>
    '''
            display(HTML(img_html))
    except Exception as _:
        pass
    
    # Avoid displaying the figure in non-notebook runs
    plt.close(fig)
    
    return anim


def create_2d_surface_plots(net, initial_params, time_points=None, which="density"):
    """
    Create 2D surface plots at multiple time points

    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: List of time points to plot (default: [0.0, 0.5, 1.0, 1.5, 2.0])
        which: "density" or "velocity"
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]
    
    print("Creating 2D surface plots...")
    
    # Create subplots for multiple time points
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(time_points):
        if i < len(axes):
            print(f"Plotting at t = {t}")
            Two_D_surface_plots(net, t, initial_params, ax=axes[i], which=which)

    # Remove the last empty subplot if needed
    if len(time_points) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"2d_surface_plots_{which}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D surface plots ({which}) to {save_path}")
    
    return fig, axes


def create_1d_comparison_plots(net, initial_params, time_array_1d=None):
    """
    Create 1D cross section comparison plots between PINN, Linear Theory, and Finite Difference
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_array_1d: List of time points for 1D plots (default: [0.5, 1.0, 1.5])
    """
    if time_array_1d is None:
        time_array_1d = [0.5, 1.0, 1.5]
    
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    # rho_o imported from config.py
    num_of_waves = (xmax - xmin) / lam
    
    print("Creating 1D cross section comparison plots...")
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    # Collect all velocity data to determine appropriate y-axis limits
    all_velocity_data = []
    
    for i, time in enumerate(time_array_1d):
        print(f"Creating 1D comparison plots at t = {time}")
        
        # Get LAX solution (Finite Difference)
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max_FD, v_LT = lax_solution(
            time, FD_N_2D, 0.5, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=True, animation=True
        )
        
        # Get PINN solution
        X, rho_pred0, v_pred_x0, v_pred_y0, phi_pred0, rho_max_PN = plot_function(
            net, [time], initial_params, velocity=True, isplot=False, animation=True
        )
        
        # Interpolate LAX solutions to match PINN grid
        from scipy.interpolate import interp1d
        X_flat = X.flatten()
        
        # Interpolate Linear Theory and Finite Difference to PINN grid
        rho_LT_interp = interp1d(x, rho_LT, kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        rho_FD_interp = interp1d(x, rho[n-1,:], kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        v_LT_interp = interp1d(x, v_LT, kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        v_FD_interp = interp1d(x, v[n-1,:], kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        phi_FD_interp = interp1d(x, phi[n-1,:], kind='linear', bounds_error=False, fill_value='extrapolate')(X_flat)
        
        # Collect velocity data for limit calculation
        all_velocity_data.append(v_pred_x0)
        all_velocity_data.append(v_FD_interp)
        if (np.isclose(KY, 0.0)) and (a < 0.1):
            all_velocity_data.append(v_LT_interp)
        
        # Density comparison plots
        axes[i*3].plot(X, rho_pred0, color='c', linewidth=3, label="PINN")
        # Only plot Linear Theory when KY == 0 and amplitude is small
        if (np.isclose(KY, 0.0)) and (a < 0.1):
            axes[i*3].plot(X, rho_LT_interp, linestyle='dashed', color='firebrick', linewidth=2, label="Linear Theory")
        axes[i*3].plot(X, rho_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3].set_xlim(xmin, xmax)
        axes[i*3].set_title(f"Density at t={time:.1f}")
        axes[i*3].set_ylabel(r"$\rho$")
        axes[i*3].grid(True)
        axes[i*3].legend()
        axes[i*3].set_ylim(0.5*rho_o, 1.5*rho_o)
        
        # Velocity comparison plots
        axes[i*3+1].plot(X, v_pred_x0, color='c', linewidth=3, label="PINN")
        # Only plot Linear Theory when KY == 0 and amplitude is small
        if (np.isclose(KY, 0.0)) and (a < 0.1):
            axes[i*3+1].plot(X, v_LT_interp, linestyle='dashed', color='firebrick', linewidth=2, label="Linear Theory")
        axes[i*3+1].plot(X, v_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3+1].set_xlim(xmin, xmax)
        axes[i*3+1].set_title(f"Velocity at t={time:.1f}")
        axes[i*3+1].set_ylabel("$v_x$")
        axes[i*3+1].grid(True)
        axes[i*3+1].legend()
        
        # Potential comparison plots
        axes[i*3+2].plot(X, phi_pred0, color='c', linewidth=3, label="PINN")
        axes[i*3+2].plot(X, phi_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3+2].set_xlim(xmin, xmax)
        axes[i*3+2].set_title(f"Potential at t={time:.1f}")
        axes[i*3+2].set_ylabel(r"$\phi$")
        axes[i*3+2].set_xlabel("x")
        axes[i*3+2].grid(True)
        axes[i*3+2].legend()

    # Set velocity y-axis limits based on amplitude (same logic as create_1d_cross_sections_sinusoidal)
    # Calculate limits from all collected velocity data and apply consistently to all velocity plots
    if a < 0.1:
        # Default limits for small amplitude cases
        v_limu_default = 0.055
        v_liml_default = -0.055
    else:
        # Default limits for large amplitude cases
        v_limu_default = 0.6
        v_liml_default = -0.6
    
    # Calculate actual data range across all velocity data with padding
    if all_velocity_data:
        v_min = np.min([np.min(v_data) for v_data in all_velocity_data])
        v_max = np.max([np.max(v_data) for v_data in all_velocity_data])
        v_range = v_max - v_min
        # Add 10% padding to ensure all data stays within limits
        padding = max(0.01, 0.1 * v_range)
        v_liml_actual = v_min - padding
        v_limu_actual = v_max + padding
        
        # Ensure limits are at least as wide as default, but expand if data exceeds them
        v_liml_actual = min(v_liml_actual, v_liml_default)
        v_limu_actual = max(v_limu_actual, v_limu_default)
    else:
        v_liml_actual = v_liml_default
        v_limu_actual = v_limu_default
    
    # Apply consistent limits to all velocity plots
    for i in range(len(time_array_1d)):
        axes[i*3+1].set_ylim(v_liml_actual, v_limu_actual)

    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "1d_comparison_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 1D comparison plots to {save_path}")
    
    plt.show()


def create_growth_comparison_plot(net, initial_params, time_array_growth=None):
    """
    Create growth comparison plot showing density maximum evolution over time
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_array_growth: Array of time points for growth analysis (default: 10 points from 0.1 to tmax)
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    # rho_o imported from config.py
    num_of_waves = (xmax - xmin) / lam
    
    if time_array_growth is None:
        time_array_growth = np.linspace(0.1, tmax, 5)  # Reduced from 10 to 5 points
    
    print("Creating growth comparison plot...")
    
    Growth_LT_list = []
    Growth_FD_list = []
    Growth_PN_list = []

    for i, time in enumerate(time_array_growth):
        print(f"Processing growth point {i+1}/{len(time_array_growth)} at t={time:.2f}")
        # Get LAX solution with configured grid resolution
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max_FD, v_LT = lax_solution(
            time, FD_N_2D, 0.5, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=True, animation=True
        )
        
        # Get PINN solution
        X, rho_pred0, v_pred_x0, v_pred_y0, phi_pred0, rho_max_PN = plot_function(
            net, [time], initial_params, velocity=True, isplot=False, animation=True
        )
        
        Growth_LT = rho_LT_max - rho_o
        Growth_FD = rho_max_FD - rho_o  
        Growth_PN = rho_max_PN - rho_o
        
        Growth_LT_list.append(Growth_LT)
        Growth_FD_list.append(Growth_FD)
        Growth_PN_list.append(Growth_PN)

    # Plot growth comparison
    plt.figure(figsize=(8, 6))
    # Only plot Linear Theory when KY == 0 and amplitude is small
    if (np.isclose(KY, 0.0)) and (a < 0.1):
        plt.plot(time_array_growth, np.log(Growth_LT_list), marker='o', color='b', linewidth=2, label="Linear Theory")
    plt.plot(time_array_growth, np.log(Growth_FD_list), '--', marker='*', color='k', linewidth=3, label="Finite Difference")
    plt.plot(time_array_growth, np.log(Growth_PN_list), marker='^', markersize=8, linewidth=2, color='r', label="PINN")
    plt.xlabel("t", fontsize=14)
    plt.ylabel(r"$\log (\rho_{\rm max} - \rho_{0})$", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("Growth Comparison: PINN vs Linear Theory vs Finite Difference")
    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "growth_comparison_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved growth comparison plot to {save_path}")
    
    plt.show()


def create_all_plots(net, initial_params, include_growth=False,
                     fd_use_velocity_ps=None, fd_ps_index=None, fd_vel_rms=None, fd_random_seed=None):
    """
    Create only 2D surface plot grids (density and velocity). Optionally create FD grids and return figures.

    Args:
        net: Trained neural network
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        include_growth: Unused now; kept for API compatibility.
        fd_use_velocity_ps: Override for FD velocity power spectrum flag (defaults to config)
        fd_ps_index: Override for FD power spectrum index (defaults to POWER_EXPONENT)
        fd_vel_rms: Override for FD velocity RMS (defaults to a*cs)
        fd_random_seed: Override for FD random seed (defaults to 1234)
    Returns:
        dict with figures/axes: {"pinn_density": (fig, axes), "pinn_velocity": (fig, axes),
                                 "fd_density": (fig, axes), "fd_velocity": (fig, axes)}
    """
    print("="*60)
    print("CREATING GRID VISUALIZATION PLOTS")
    print("="*60)
    
    # Use config defaults if not overridden to ensure consistency with PINN training
    if fd_use_velocity_ps is None:
        fd_use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if fd_ps_index is None:
        fd_ps_index = POWER_EXPONENT
    if fd_vel_rms is None:
        fd_vel_rms = a * cs
    if fd_random_seed is None:
        fd_random_seed = RANDOM_SEED

    # 1. PINN density grid
    fig_den, axes_den = create_2d_surface_plots(net, initial_params, which="density")

    # 2. PINN velocity grid
    fig_vel, axes_vel = create_2d_surface_plots(net, initial_params, which="velocity")

    # 3. FD density grid - now uses consistent parameters with PINN training
    fig_fd_den, axes_fd_den = create_2d_surface_plots_FD(initial_params, which="density",
                                                         use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
                                                         vel_rms=fd_vel_rms, random_seed=fd_random_seed)

    # 4. FD velocity grid - now uses consistent parameters with PINN training
    fig_fd_vel, axes_fd_vel = create_2d_surface_plots_FD(initial_params, which="velocity",
                                                         use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
                                                         vel_rms=fd_vel_rms, random_seed=fd_random_seed)

    print("="*60)
    print("ALL GRID PLOTS COMPLETED!")
    print("="*60)

    # Display all created figures so they appear when called from train.py
    plt.show()

    result = {
        "pinn_density": (fig_den, axes_den),
        "pinn_velocity": (fig_vel, axes_vel),
        "fd_density": (fig_fd_den, axes_fd_den),
        "fd_velocity": (fig_fd_vel, axes_fd_vel),
    }
    
    return result


def create_density_growth_plot(net, initial_params, tmax, dt=0.1):
    """
    Create a PINN vs LAX density growth comparison plot.

    Plots over time: (1) maximum density, (2) log(rho_max - rho_o + eps).

    Args:
        net: trained PINN model
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax_train)
        tmax: maximum time to plot (independent of training tmax)
        dt: temporal spacing (default 0.1)
    """
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax_train = initial_params
    
    # Handle both single network and list of networks (XPINN)
    if isinstance(net, list):
        nets = net
        use_xpinn = len(nets) > 1
    else:
        nets = [net]
        use_xpinn = False
    num_of_waves = (xmax - xmin) / lam

    # Time grid (inclusive of tmax)
    time_points = np.arange(0.0, float(tmax) + 1e-9, float(dt))

    pinn_max_list = []
    fd_max_list = []

    # PINN grid sampling settings (use N_GRID for consistency with FD solver)
    # Exclude right boundary for periodic domains to avoid double-counting
    Q = N_GRID
    xs = np.linspace(xmin, xmax, Q, endpoint=False)
    ys = np.linspace(ymin, ymax, Q, endpoint=False)
    TAU, PHI = np.meshgrid(xs, ys)
    Xgrid = np.vstack([TAU.flatten(), PHI.flatten()]).T

    for idx, t in enumerate(time_points):
        # PINN evaluation on QxQ grid
        t_vec = t * np.ones(Q**2).reshape(Q**2, 1)
        pt_x = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
        pt_t = Variable(torch.from_numpy(t_vec).float(), requires_grad=True).to(device)
        if use_xpinn:
            out = predict_xpinn(nets, pt_x, pt_y, pt_t, xmin, xmax, ymin, ymax)
        else:
            out = nets[0]([pt_x, pt_y, pt_t])
        rho_pinn = out[:, 0].data.cpu().numpy().reshape(Q, Q)
        pinn_max_list.append(np.max(rho_pinn))

        # LAX/FD evaluation; use shared velocity fields when available
        if str(PERTURBATION_TYPE).lower() == "power_spectrum":
            if _shared_vx_np is not None and _shared_vy_np is not None:
                # Use the native resolution of the shared velocity fields to avoid shape mismatch
                n_fd_use = int(_shared_vx_np.shape[0])
                x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = lax_solution_with_shared_velocity(
                    t, n_fd_use, 0.5, lam, num_of_waves, rho_1, _shared_vx_np, _shared_vy_np,
                    gravity=True, isplot=False, comparison=False, animation=True
                )
            else:
                # Fallback: when shared fields absent, still use N_GRID for power spectrum LAX
                x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                    t, N_GRID, 0.5, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                    use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                )
        else:
            # Sinusoidal case (keep defaults)
            x_fd, rho_fd, _vx_fd, _vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                t, FD_N_2D, 0.5, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                use_velocity_ps=False
            )

        fd_max_list.append(np.max(rho_fd))

    # Build figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # (1) Max density vs time
    axes[0].plot(time_points, fd_max_list, label="LAX", color='k', linewidth=2)
    axes[0].plot(time_points, pinn_max_list, label="PINN", color='c', linewidth=2)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(r"$\rho_{\max}$")
    axes[0].set_title("Maximum Density vs Time")
    axes[0].grid(True)
    axes[0].legend()
    # Annotate parameters on the plot
    try:
        param_str = f"a={a}, power_index={POWER_EXPONENT}"
        axes[0].text(0.02, 0.95, param_str, transform=axes[0].transAxes,
                     fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6))
    except Exception:
        pass

    # (2) log growth vs time
    eps = 1e-12
    axes[1].plot(time_points, np.log(np.maximum(np.array(fd_max_list) - rho_o, 0.0) + eps), label="LAX", color='k', linewidth=2)
    axes[1].plot(time_points, np.log(np.maximum(np.array(pinn_max_list) - rho_o, 0.0) + eps), label="PINN", color='c', linewidth=2)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\log(\rho_{\max} - \rho_0)$")
    axes[1].set_title("Density Growth (log)")
    axes[1].grid(True)
    axes[1].legend()
    # Mirror annotation on second axis
    try:
        param_str = f"a={a}, power_index={POWER_EXPONENT}"
        axes[1].text(0.02, 0.95, param_str, transform=axes[1].transAxes,
                     fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.6))
    except Exception:
        pass

    plt.tight_layout()

    # Save figure
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "density_growth_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved density growth comparison plot to {save_path}")

    plt.show()

    return fig, axes


def create_1d_cross_sections_sinusoidal(net, initial_params, time_points=None, y_fixed=0.6, N_fd=1000, nu_fd=0.5):
    """
    Create 1D cross-section plots at fixed y for sinusoidal perturbations, comparing
    PINN vs Linear Theory vs 1D LAX (sinusoidal).

    Args:
        net: trained network
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: list of times to plot
        y_fixed: y value for the 1D slice through the 2D domain
        N_fd: grid size for 1D LAX solver
        nu_fd: Courant number for 1D LAX solver
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, _output_folder, _tmax = initial_params
    
    # Handle both single network and list of networks
    if isinstance(net, list):
        nets = net
        use_xpinn = len(nets) > 1
    else:
        nets = [net]
        use_xpinn = False
    num_of_waves = (xmax - xmin) / lam

    if time_points is None:
        time_points = TIMES_1D if isinstance(TIMES_1D, (list, tuple)) and len(TIMES_1D) > 0 else [0.5, 1.0, 1.5]

    # Use baseline density 1.0 for Linear Theory reference
    rho_base = 1.0
    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_base))
    k = np.sqrt(KX**2 + KY**2)
    v1_lt = (rho_1 / rho_base) * (alpha / k)

    # Build x grid for PINN slice
    X = np.linspace(xmin, xmax, 1000).reshape(1000, 1)
    Y = y_fixed * np.ones_like(X)

    # Create 2 rows x T columns panel layout matching target style
    T = len(time_points)
    fig = plt.figure(figsize=(6*T, 8), constrained_layout=False)
    grid = plt.GridSpec(4, T, figure=fig, hspace=0.12, wspace=0.18)

    for row_idx, t in enumerate(time_points):
        # PINN predictions at fixed y
        t_arr = t * np.ones_like(X)
        pt_x = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_y = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
        pt_t = Variable(torch.from_numpy(t_arr).float(), requires_grad=True).to(device)
        if use_xpinn:
            out = predict_xpinn(nets, pt_x, pt_y, pt_t, xmin, xmax, ymin, ymax)
        else:
            out = nets[0]([pt_x, pt_y, pt_t])
        rho_pinn = out[:, 0:1].data.cpu().numpy().reshape(-1)
        vx_pinn = out[:, 1:2].data.cpu().numpy().reshape(-1)
        # potential not used in cross-section plots

        # 2D Linear Theory (only meaningful for KY == 0 in current comparison policy)
        if np.isclose(KY, 0.0):
            if lam >= jeans:
                # Gravitational instability case
                rho_lt = rho_base + rho_1*np.exp(alpha * t)*np.cos(KX * X[:, 0] + KY * y_fixed)
                vx_lt = -v1_lt*np.exp(alpha * t)*np.sin(KX * X[:, 0] + KY * y_fixed) * (KX / np.sqrt(KX**2 + KY**2))
            else:
                # Oscillatory regime
                omega = np.sqrt(cs**2 * (KX**2 + KY**2) - const*G*rho_base)
                rho_lt = rho_base + rho_1*np.cos(omega * t - KX * X[:, 0] - KY * y_fixed)
                vx_lt = v1_lt*np.cos(omega * t - KX * X[:, 0] - KY * y_fixed) * (KX / np.sqrt(KX**2 + KY**2))

        # 2D LAX solver - get full 2D solution then extract slice
        x_fd_2d, rho_fd_2d, vx_fd_2d, vy_fd_2d, _phi_fd_2d, _n, _rho_max = lax_solution(
            t, FD_N_2D, nu_fd, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True
        )
        
        # Extract 1D slice from 2D solution at y = y_fixed
        # Exclude right boundary for periodic domains to avoid double-counting
        y_fd_2d = np.linspace(0, lam * num_of_waves, rho_fd_2d.shape[1], endpoint=False)
        y_idx = np.argmin(np.abs(y_fd_2d - y_fixed))
        
        # Extract the slice
        rho_fd = rho_fd_2d[:, y_idx]
        v_fd = vx_fd_2d[:, y_idx]  # Use x-component of velocity
        # Interpolate FD results to PINN X grid for comparison
        from scipy.interpolate import interp1d
        rho_fd_interp = interp1d(x_fd_2d, rho_fd, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])
        v_fd_interp = interp1d(x_fd_2d, v_fd, kind='linear', bounds_error=False, fill_value='extrapolate')(X[:, 0])

        # Column index
        c = row_idx
        # Top row: density
        ax_rho = fig.add_subplot(grid[0, c])
        ax_rho.plot(X[:, 0], rho_pinn, label="GRINN", color='c', linewidth=2)
        # Only plot Linear Theory when KY == 0 and amplitude is small
        if np.isclose(KY, 0.0) and (a < 0.1):
            ax_rho.plot(X[:, 0], rho_lt, label="LT", linestyle='--', color='firebrick', linewidth=1.5)
        ax_rho.plot(X[:, 0], rho_fd_interp, label="FD", color='k', linewidth=1)
        ax_rho.set_title(f"t={t:.1f}")
        ax_rho.set_ylabel(r"$\rho$")
        ax_rho.grid(True)
        if a < 0.1:
            limu = 1.2*rho_o
            liml = .8*rho_o
        else:
            limu = 3.0*rho_o
            liml = -1.0*rho_o
        ax_rho.set_ylim(liml,limu)
        if c == 0:
            ax_rho.legend(loc='upper right', fontsize=8)

        # Second row: epsilon for density using symmetric percent with absolute numerator
        # ε = 200 * |G - R| / (G + R) with more robust denominator
        eps_rho = 200.0 * np.abs(rho_pinn - rho_fd_interp) / (rho_pinn + rho_fd_interp + 1e-6)
        ax_eps_rho = fig.add_subplot(grid[1, c])
        ax_eps_rho.plot(X[:, 0], eps_rho, color='k', linewidth=1, label='FD')
        # Only plot Linear Theory epsilon when KY == 0 and amplitude is small
        if np.isclose(KY, 0.0) and (a < 0.1):
            eps_rho_lt = 200.0 * np.abs(rho_pinn - rho_lt) / (rho_pinn + rho_lt + 1e-6)
            ax_eps_rho.plot(X[:, 0], eps_rho_lt, color='firebrick', linestyle='--', linewidth=1, label='LT')
        ax_eps_rho.set_ylabel(r"$\varepsilon$")
        ax_eps_rho.grid(True)
        if c == 0:
            ax_eps_rho.legend(loc='upper right', fontsize=8)

        # Third row: velocity
        ax_v = fig.add_subplot(grid[2, c])
        ax_v.plot(X[:, 0], vx_pinn, label="GRINN", color='c', linewidth=2)
        # Only plot Linear Theory when KY == 0 and amplitude is small
        if np.isclose(KY, 0.0) and (a < 0.1):
            ax_v.plot(X[:, 0], vx_lt, label="LT", linestyle='--', color='firebrick', linewidth=1.5)
        ax_v.plot(X[:, 0], v_fd_interp, label="FD", color='k', linewidth=1)
        ax_v.set_ylabel(r"$v$")
        ax_v.grid(True)
        if a < 0.1:
            limu = 0.055
            liml = -0.055
        else:
            limu = 0.6
            liml = -0.6
        ax_v.set_ylim(liml,limu)
        if c == 0:
            ax_v.legend(loc='upper right', fontsize=8)

        # Fourth row: epsilon for velocity using notebook-style +1 offset (symmetric percent with shift)
        # ε = 200 * |(v_pred+1) - (v_ref+1)| / ((v_pred+1) + (v_ref+1)) = 200 * |v_pred - v_ref| / (v_pred + v_ref + 2)
        v_ref = v_fd_interp
        v_pred = vx_pinn
        eps_v = 200.0 * np.abs(v_pred - v_ref) / (v_pred + v_ref + 2.0)
        ax_eps_v = fig.add_subplot(grid[3, c])
        ax_eps_v.plot(X[:, 0], eps_v, color='k', linewidth=1, label='FD')
        # Only plot Linear Theory epsilon when KY == 0 and amplitude is small
        if np.isclose(KY, 0.0) and (a < 0.1):
            eps_v_lt = 200.0 * np.abs(v_pred - vx_lt) / (v_pred + vx_lt + 2.0)
            ax_eps_v.plot(X[:, 0], eps_v_lt, color='firebrick', linestyle='--', linewidth=1, label='LT')
        ax_eps_v.set_xlabel("x")
        ax_eps_v.set_ylabel(r"$\varepsilon$")
        ax_eps_v.grid(True)
        if c == 0:
            ax_eps_v.legend(loc='upper right', fontsize=8)

        # No potential subplot per request

    # Reduce outer margins similar to notebook style
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10, wspace=0.18, hspace=0.12)
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "1d_cross_sections_sinusoidal.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 1D cross sections plot to {save_path}")
    
    plt.show()

    return fig


def Two_D_surface_plots_FD(time, initial_params, N=200, nu=0.5, ax=None, which="density",
                           use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None):
    """
    Create 2D surface plots with velocity vectors using the Finite Difference (LAX) solver

    Args:
        time: Time to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        N: Grid resolution for LAX solver (Nx = Ny = N)
        nu: Courant number for LAX solver
        ax: Optional matplotlib axis to plot on
        which: "density" or "velocity"
        use_velocity_ps: Whether to use velocity power spectrum (defaults to config)
        ps_index: Power spectrum index (defaults to POWER_EXPONENT)
        vel_rms: Velocity RMS amplitude (defaults to a*cs)
        random_seed: Random seed (defaults to 1234)

    Returns:
        The QuadMesh object from pcolormesh
    """
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax = initial_params

    # Use config defaults if not specified to ensure consistency with PINN training
    if use_velocity_ps is None:
        use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if ps_index is None:
        ps_index = POWER_EXPONENT
    if vel_rms is None:
        vel_rms = a * cs
    if random_seed is None:
        random_seed = RANDOM_SEED

    # Domain properties for LAX_2D (Lx = Ly and Nx = Ny in solver)
    num_of_waves = (xmax - xmin) / lam

    # Decide grid resolution policy: use N_GRID for power spectrum; FD_N_2D for sinusoidal when N is None
    if str(PERTURBATION_TYPE).lower() == "power_spectrum":
        N_use = N_GRID
    else:
        N_use = FD_N_2D if N is None else N

    # Run LAX solver (finite difference) with self-gravity enabled to obtain 2D fields
    # Prefer using the exact shared velocity fields (if available) to ensure identical realization
    # across PINN ICs and all FD visualizations.
    # Returns (gravity=True, comparison=False, animation=True):
    #   x (Nx,), rho (Nx,Ny), vx (Nx,Ny), vy (Nx,Ny), phi (Nx,Ny), n, rho_max
    if (str(PERTURBATION_TYPE).lower() == "power_spectrum" \
        and _shared_vx_np is not None and _shared_vy_np is not None):
        # Use native resolution of shared fields to avoid resampling artifacts
        n_fd_use = int(_shared_vx_np.shape[0])
        x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_with_shared_velocity(
            time, n_fd_use, nu, lam, num_of_waves, rho_1, _shared_vx_np, _shared_vy_np,
            gravity=True, isplot=False, comparison=False, animation=True
        )
    else:
        x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
            time, N_use, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
            use_velocity_ps=use_velocity_ps, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed
        )

    # Build y-array consistent with solver setup (square domain with same resolution)
    Lx = lam * num_of_waves
    Nx = x_fd.shape[0]
    Ny = rho_fd.shape[1]
    y_fd = np.linspace(0.0, Lx, Ny)

    # Create meshgrid for plotting
    X, Y = np.meshgrid(x_fd, y_fd, indexing='ij')

    if ax is None:  # for single plot
        plt.figure(figsize=(5, 5))
        ax = plt.gca()

    # Clean FD velocity fields and avoid zero-length arrows
    vx_c = np.nan_to_num(vx_fd, nan=0.0, posinf=0.0, neginf=0.0)
    vy_c = np.nan_to_num(vy_fd, nan=0.0, posinf=0.0, neginf=0.0)
    Vmag_fd = np.sqrt(vx_c**2 + vy_c**2)
    mask = Vmag_fd > 1e-12

    if which == "density":
        pc = ax.pcolormesh(X, Y, rho_fd, shading='auto', cmap='YlOrBr', vmin=np.min(rho_fd), vmax=np.max(rho_fd))
        skip = (slice(None, None, max(1, Nx // 20)), slice(None, None, max(1, Ny // 20)))
        ax.quiver(
            X[skip][mask[skip]], Y[skip][mask[skip]],
            vx_c[skip][mask[skip]], vy_c[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("FD Density, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:
        pc = ax.pcolormesh(X, Y, Vmag_fd, shading='auto', cmap='viridis', vmin=np.min(Vmag_fd), vmax=np.max(Vmag_fd))
        skip = (slice(None, None, max(1, Nx // 20)), slice(None, None, max(1, Ny // 20)))
        ax.quiver(
            X[skip][mask[skip]], Y[skip][mask[skip]],
            vx_c[skip][mask[skip]], vy_c[skip][mask[skip]],
            color='k', headwidth=3.0, width=0.003,
            scale_units='xy', angles='xy', scale=1.0, minlength=0.0, pivot='mid'
        )
        ax.set_title("FD Velocity, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return pc


def create_2d_surface_plots_FD(initial_params, time_points=None, which="density", N=200, nu=0.5,
                               use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None):
    """
    Create 2D surface plots at multiple time points using the LAX FD solver

    Args:
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: list of times
        which: "density" or "velocity"
        N: grid size
        nu: Courant number
        use_velocity_ps: Whether to use velocity power spectrum (defaults to config)
        ps_index: Power spectrum index (defaults to POWER_EXPONENT)
        vel_rms: Velocity RMS amplitude (defaults to a*cs)
        random_seed: Random seed (defaults to 1234)
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Use config defaults if not specified to ensure consistency with PINN training
    if use_velocity_ps is None:
        use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if ps_index is None:
        ps_index = POWER_EXPONENT
    if vel_rms is None:
        vel_rms = a * cs
    if random_seed is None:
        random_seed = RANDOM_SEED

    print("Creating 2D FD surface plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(time_points):
        if i < len(axes):
            print(f"FD plotting at t = {t}")
            # Enforce grid policy: use N_GRID for power spectrum; else pass through N
            if str(PERTURBATION_TYPE).lower() == "power_spectrum":
                N_call = N_GRID
            else:
                N_call = N
            Two_D_surface_plots_FD(t, initial_params, N=N_call, nu=nu, ax=axes[i], which=which,
                                   use_velocity_ps=use_velocity_ps, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed)

    if len(time_points) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    
    # Save the figure to output folder
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"2d_surface_plots_FD_{which}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 2D FD surface plots ({which}) to {save_path}")
    
    return fig, axes


def create_5x3_comparison_table(net, initial_params, which="density", N=200, nu=0.5,
                                use_velocity_ps=None, ps_index=None, vel_rms=None, random_seed=None):
    """
    Create 5x3 comparison table showing PINN, FD, and epsilon metric at 5 time snapshots
    
    Args:
        net: Trained neural network
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        which: "density" or "velocity"
        N: Grid resolution for LAX solver
        nu: Courant number for LAX solver
        use_velocity_ps: Whether to use velocity power spectrum for FD (defaults to config)
        ps_index: Power spectrum index for FD (defaults to POWER_EXPONENT)
        vel_rms: Velocity RMS for FD (defaults to a*cs)
        random_seed: Random seed for FD (defaults to 1234)
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    # Use config defaults if not specified to ensure consistency with PINN training
    if use_velocity_ps is None:
        use_velocity_ps = (str(PERTURBATION_TYPE).lower() == "power_spectrum")
    if ps_index is None:
        ps_index = POWER_EXPONENT
    if vel_rms is None:
        vel_rms = a * cs
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    # Handle both single network and list of networks
    if isinstance(net, list):
        nets = net
        use_xpinn = len(nets) > 1
    else:
        nets = [net]
        use_xpinn = False
    
    # Generate 5 time points uniformly distributed over [0, tmax]
    time_points = np.linspace(0.0, float(tmax), 5)
    
    print(f"Creating 5x3 comparison table for {which}...")
    
    # Create 5x3 subplot grid
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    
    # Store data for consistent color limits
    pinn_data = []
    fd_data = []
    pinn_velocity_data = []  # Store velocity components separately
    fd_velocity_data = []    # Store velocity components separately
    
    # First pass: collect data to determine consistent color limits
    for i, t in enumerate(time_points):
        # print(f"Collecting data for t = {t:.2f}")  # Commented out to reduce output noise
        
        # Get PINN data - use N_GRID for consistency with FD solver
        # Exclude right boundary for periodic domains to avoid double-counting
        Q = N_GRID
        xs = np.linspace(xmin, xmax, Q, endpoint=False)
        ys = np.linspace(ymin, ymax, Q, endpoint=False)
        tau, phi = np.meshgrid(xs, ys) 
        Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
        t_00 = t * np.ones(Q**2).reshape(Q**2, 1)
        
        pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
        
        if use_xpinn:
            output_00 = predict_xpinn(nets, pt_x_collocation, pt_y_collocation, pt_t_collocation, xmin, xmax, ymin, ymax)
        else:
            output_00 = nets[0]([pt_x_collocation, pt_y_collocation, pt_t_collocation])
        
        if which == "density":
            pinn_field = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
            # Extract velocity components for density plots too
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
        
        # Debug output (commented out to reduce noise)
        # print(f"  PINN {which} range: [{np.min(pinn_field):.6f}, {np.max(pinn_field):.6f}], std: {np.std(pinn_field):.6f}")
        
        # Get FD data - use same parameters as PINN for power spectrum
        num_of_waves = (xmax - xmin) / lam
        if str(PERTURBATION_TYPE).lower() == "power_spectrum":
            # For power spectrum, use shared velocity fields if available
            if _shared_vx_np is not None and _shared_vy_np is not None:
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution_with_shared_velocity(
                    t, N, nu, lam, num_of_waves, rho_1, _shared_vx_np, _shared_vy_np,
                    gravity=True, isplot=False, comparison=False, animation=True
                )
            else:
                # Fallback to original method
                x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                    t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                    use_velocity_ps=True, ps_index=POWER_EXPONENT, vel_rms=a*cs, random_seed=RANDOM_SEED
                )
            # Debug: Check FD density range (commented out to reduce output noise)
            # print(f"  FD {which} range: [{np.min(rho_fd):.6f}, {np.max(rho_fd):.6f}], std: {np.std(rho_fd):.6f}")
        else:
            # For sinusoidal, use original parameters
            x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
                t, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
                use_velocity_ps=False, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed
            )
        
        # Build y-array consistent with solver setup
        # Exclude right boundary for periodic domains to avoid double-counting
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
        
        # Interpolate FD data to PINN grid for comparison using single robust method
        from scipy.interpolate import RegularGridInterpolator
        points_fd = np.column_stack([X_fd.ravel(), Y_fd.ravel()])
        points_pinn = np.column_stack([tau.ravel(), phi.ravel()])
        
        # Use RegularGridInterpolator for robust interpolation with proper boundary handling
        # This avoids the interpolation cascade issues and provides consistent results
        interpolator = RegularGridInterpolator(
            (x_fd, y_fd), fd_field, 
            method='linear', 
            bounds_error=False, 
            fill_value=None  # extrapolate
        )
        fd_field_interp = interpolator(points_pinn)
        
        fd_field_interp = fd_field_interp.reshape(Q, Q)
        
        # Interpolate FD velocity components for vector plots using single robust method
        vx_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), vx_fd, 
            method='linear', 
            bounds_error=False, 
            fill_value=None  # extrapolate
        )
        fd_vx_interp = vx_interpolator(points_pinn)
        
        vy_interpolator = RegularGridInterpolator(
            (x_fd, y_fd), vy_fd, 
            method='linear', 
            bounds_error=False, 
            fill_value=None  # extrapolate
        )
        fd_vy_interp = vy_interpolator(points_pinn)
        
        fd_vx_interp = fd_vx_interp.reshape(Q, Q)
        fd_vy_interp = fd_vy_interp.reshape(Q, Q)
        
        pinn_data.append(pinn_field)
        fd_data.append(fd_field_interp)
        
        # Store velocity components for vector plots (both density and velocity plots)
        pinn_velocity_data.append((pinn_vx, pinn_vy))
        fd_velocity_data.append((fd_vx_interp, fd_vy_interp))
    
    # Use individual color limits for each plot (like animation) to show dynamic range
    # This allows collapse features to be visible, rather than using global limits
    
    # Second pass: create plots
    for i, t in enumerate(time_points):
        pinn_field = pinn_data[i]
        fd_field = fd_data[i]
        
        # Extract velocity components for vector plots
        pinn_vx, pinn_vy = pinn_velocity_data[i]
        fd_vx, fd_vy = fd_velocity_data[i]
        
        # Calculate epsilon metric: ε = 2 * |PINN - FD| / (PINN + FD) * 100
        # Use a more robust denominator to reduce sensitivity to small values
        eps = 1e-6  # Increased from 1e-12 to reduce sensitivity
        epsilon_metric = 200.0 * np.abs(pinn_field - fd_field) / (pinn_field + fd_field + eps)
        
        # Column 1: PINN - use individual color limits like animation
        ax_pinn = axes[i, 0]
        if which == "density":
            pc_pinn = ax_pinn.pcolormesh(tau, phi, pinn_field, shading='auto', cmap='YlOrBr', 
                                       vmin=np.min(pinn_field), vmax=np.max(pinn_field))
        else:
            pc_pinn = ax_pinn.pcolormesh(tau, phi, pinn_field, shading='auto', cmap='viridis', 
                                       vmin=np.min(pinn_field), vmax=np.max(pinn_field))
        
        # Add velocity vectors for both density and velocity plots
        if pinn_vx is not None and pinn_vy is not None:
            # Subsample vectors for clarity (similar to analyze_lax.py)
            skip_x = max(1, Q // 20)
            skip_y = max(1, Q // 20)
            skip = (slice(None, None, skip_x), slice(None, None, skip_y))
            ax_pinn.quiver(tau[skip], phi[skip], pinn_vx[skip], pinn_vy[skip], 
                          color='k', headwidth=3.0, width=0.003, alpha=0.7)
        
        ax_pinn.set_title(f"PINN {which.title()}, t={t:.2f}")
        ax_pinn.set_xlim(xmin, xmax)
        ax_pinn.set_ylim(ymin, ymax)
        
        # Add interface lines for XPINN
        if use_xpinn:
            add_interface_lines(ax_pinn, xmin, xmax, ymin, ymax)
        
        cbar_pinn = plt.colorbar(pc_pinn, ax=ax_pinn, shrink=0.6)
        cbar_pinn.ax.set_title(r"$\rho$" if which == "density" else r"$|v|$", fontsize=14)
        
        # Column 2: FD - use individual color limits
        ax_fd = axes[i, 1]
        if which == "density":
            pc_fd = ax_fd.pcolormesh(tau, phi, fd_field, shading='auto', cmap='YlOrBr', 
                                    vmin=np.min(fd_field), vmax=np.max(fd_field))
        else:
            pc_fd = ax_fd.pcolormesh(tau, phi, fd_field, shading='auto', cmap='viridis', 
                                    vmin=np.min(fd_field), vmax=np.max(fd_field))
        
        # Add velocity vectors for both density and velocity plots
        if fd_vx is not None and fd_vy is not None:
            # Subsample vectors for clarity (similar to analyze_lax.py)
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
        
        # Add x-axis labels only on bottom row
        if i == 4:
            ax_pinn.set_xlabel("x")
            ax_fd.set_xlabel("x")
            ax_diff.set_xlabel("x")
        
        # Add y-axis labels only on leftmost column
        ax_pinn.set_ylabel("y")
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{which}_comparison_5x3.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved 5x3 comparison table to {save_path}")
    
    plt.show()
    return fig, axes
