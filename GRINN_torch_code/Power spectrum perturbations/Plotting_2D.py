from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import scipy
import os
from LAX_2D import lax_solution

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() \
    else "cuda:0" if torch.cuda.is_available() else "cpu"


def plot_function(net, time_array, initial_params, velocity=False, isplot=False, animation=False):
    """
    Plot function for 1D slices through 2D domain
    
    Args:
        net: Trained neural network
        time_array: Array of times to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        velocity: Whether to plot velocity
        isplot: Whether to save plots
        animation: Whether this is for animation
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params  
    rho_o = 1.0          # zeroth order density
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
        
        # PINN model expects a list of tensors [x, y, t]
        output_0 = net([pt_x_collocation, pt_y_collocation, pt_t_collocation])
        
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
        net: Trained neural network
        time: Time to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        ax: Optional axis to plot on
        which: "density" or "velocity"
    """
    xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax = initial_params
    
    Q = 100
    xs = np.linspace(xmin, xmax, Q)
    ys = np.linspace(ymin, ymax, Q)
    tau, phi = np.meshgrid(xs, ys) 
    Xgrid = np.vstack([tau.flatten(), phi.flatten()]).T
    t_00 = time * np.ones(Q**2).reshape(Q**2, 1)
    
    # Convert to tensors
    pt_x_collocation = Variable(torch.from_numpy(Xgrid[:, 0:1]).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(Xgrid[:, 1:2]).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_00).float(), requires_grad=True).to(device)
    
    # PINN model expects a list of tensors [x, y, t]
    output_00 = net([pt_x_collocation, pt_y_collocation, pt_t_collocation])
    
    rho = output_00[:, 0].data.cpu().numpy().reshape(Q, Q)
    U = output_00[:, 1].data.cpu().numpy().reshape(Q, Q)
    V = output_00[:, 2].data.cpu().numpy().reshape(Q, Q)

    if ax is None:  # for single plot
        plt.figure(figsize=(5, 5))
        ax = plt.gca() 

    if which == "density":
        pc = ax.pcolormesh(tau, phi, rho, shading='auto', cmap='YlOrBr', vmin=np.min(rho), vmax=np.max(rho))
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(tau[skip], phi[skip], U[skip], V[skip], color='k', headwidth=3.0, width=0.003)
        ax.set_title("Density, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:  # velocity magnitude surface plot
        Vmag = np.sqrt(U**2 + V**2)
        pc = ax.pcolormesh(tau, phi, Vmag, shading='auto', cmap='viridis', vmin=np.min(Vmag), vmax=np.max(Vmag))
        skip = (slice(None, None, 5), slice(None, None, 5))
        ax.quiver(tau[skip], phi[skip], U[skip], V[skip], color='k', headwidth=3.0, width=0.003)
        ax.set_title("Velocity, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    return pc


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
    rho_o = 1.0
    num_of_waves = (xmax - xmin) / lam
    
    print("Creating 1D cross section comparison plots...")
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, time in enumerate(time_array_1d):
        print(f"Creating 1D comparison plots at t = {time}")
        
        # Get LAX solution (Finite Difference) with reduced grid for speed
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max_FD, v_LT = lax_solution(
            time, 1000, 0.5, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=True, animation=True
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
        
        # Density comparison plots
        axes[i*3].plot(X, rho_pred0, color='c', linewidth=3, label="PINN")
        axes[i*3].plot(X, rho_LT_interp, linestyle='dashed', color='firebrick', linewidth=2, label="Linear Theory")
        axes[i*3].plot(X, rho_FD_interp, linestyle='solid', color='black', linewidth=1, label="Finite Difference")
        axes[i*3].set_xlim(xmin, xmax)
        axes[i*3].set_title(f"Density at t={time:.1f}")
        axes[i*3].set_ylabel(r"$\rho$")
        axes[i*3].grid(True)
        axes[i*3].legend()
        axes[i*3].set_ylim(0.8*rho_o, 1.2*rho_o)
        
        # Velocity comparison plots
        axes[i*3+1].plot(X, v_pred_x0, color='c', linewidth=3, label="PINN")
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

    plt.tight_layout()
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
    rho_o = 1.0
    num_of_waves = (xmax - xmin) / lam
    
    if time_array_growth is None:
        time_array_growth = np.linspace(0.1, tmax, 5)  # Reduced from 10 to 5 points
    
    print("Creating growth comparison plot...")
    
    Growth_LT_list = []
    Growth_FD_list = []
    Growth_PN_list = []

    for i, time in enumerate(time_array_growth):
        print(f"Processing growth point {i+1}/{len(time_array_growth)} at t={time:.2f}")
        # Get LAX solution with reduced grid resolution for speed
        x, rho, v, phi, n, rho_LT, rho_LT_max, rho_max_FD, v_LT = lax_solution(
            time, 1000, 0.5, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=True, animation=True
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
    plt.plot(time_array_growth, np.log(Growth_LT_list), marker='o', color='b', linewidth=2, label="Linear Theory")
    plt.plot(time_array_growth, np.log(Growth_FD_list), '--', marker='*', color='k', linewidth=3, label="Finite Difference")
    plt.plot(time_array_growth, np.log(Growth_PN_list), marker='^', markersize=8, linewidth=2, color='r', label="PINN")
    plt.xlabel("t", fontsize=14)
    plt.ylabel(r"$\log (\rho_{\rm max} - \rho_{0})$", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title("Growth Comparison: PINN vs Linear Theory vs Finite Difference")
    plt.tight_layout()
    plt.show()


def create_all_plots(net, initial_params, include_growth=False,
                     fd_use_velocity_ps=True, fd_ps_index=-3.0, fd_vel_rms=0.02, fd_random_seed=None):
    """
    Create only 2D surface plot grids (density and velocity). Optionally create FD grids and return figures.

    Args:
        net: Trained neural network
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        include_growth: Unused now; kept for API compatibility.
    Returns:
        dict with figures/axes: {"pinn_density": (fig, axes), "pinn_velocity": (fig, axes),
                                 "fd_density": (fig, axes), "fd_velocity": (fig, axes)}
    """
    print("="*60)
    print("CREATING GRID VISUALIZATION PLOTS")
    print("="*60)

    # 1. PINN density grid
    fig_den, axes_den = create_2d_surface_plots(net, initial_params, which="density")

    # 2. PINN velocity grid
    fig_vel, axes_vel = create_2d_surface_plots(net, initial_params, which="velocity")

    # 3. FD density grid
    fig_fd_den, axes_fd_den = create_2d_surface_plots_FD(initial_params, which="density",
                                                         use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
                                                         vel_rms=fd_vel_rms, random_seed=fd_random_seed)

    # 4. FD velocity grid
    fig_fd_vel, axes_fd_vel = create_2d_surface_plots_FD(initial_params, which="velocity",
                                                         use_velocity_ps=fd_use_velocity_ps, ps_index=fd_ps_index,
                                                         vel_rms=fd_vel_rms, random_seed=fd_random_seed)

    print("="*60)
    print("ALL GRID PLOTS COMPLETED!")
    print("="*60)

    # Display all created figures so they appear when called from train.py
    plt.show()

    return {
        "pinn_density": (fig_den, axes_den),
        "pinn_velocity": (fig_vel, axes_vel),
        "fd_density": (fig_fd_den, axes_fd_den),
        "fd_velocity": (fig_fd_vel, axes_fd_vel),
    }


def Two_D_surface_plots_FD(time, initial_params, N=200, nu=0.5, ax=None, which="density",
                           use_velocity_ps=False, ps_index=-3.0, vel_rms=0.02, random_seed=None):
    """
    Create 2D surface plots with velocity vectors using the Finite Difference (LAX) solver

    Args:
        time: Time to plot
        initial_params: Tuple containing (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        N: Grid resolution for LAX solver (Nx = Ny = N)
        nu: Courant number for LAX solver
        ax: Optional matplotlib axis to plot on
        which: "density" or "velocity"

    Returns:
        The QuadMesh object from pcolormesh
    """
    xmin, xmax, ymin, ymax, rho_1, _alpha, lam, _output_folder, _tmax = initial_params

    # Domain properties for LAX_2D (Lx = Ly and Nx = Ny in solver)
    num_of_waves = (xmax - xmin) / lam

    # Run LAX solver (finite difference) with self-gravity enabled to obtain 2D fields
    # Returns (gravity=True, comparison=False, animation=True):
    #   x (Nx,), rho (Nx,Ny), vx (Nx,Ny), vy (Nx,Ny), phi (Nx,Ny), n, rho_max
    x_fd, rho_fd, vx_fd, vy_fd, _phi_fd, _n, _rho_max = lax_solution(
        time, N, nu, lam, num_of_waves, rho_1, gravity=True, isplot=False, comparison=False, animation=True,
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

    if which == "density":
        pc = ax.pcolormesh(X, Y, rho_fd, shading='auto', cmap='YlOrBr', vmin=np.min(rho_fd), vmax=np.max(rho_fd))
        skip = (slice(None, None, max(1, Nx // 20)), slice(None, None, max(1, Ny // 20)))
        ax.quiver(X[skip], Y[skip], vx_fd[skip], vy_fd[skip], color='k', headwidth=3.0, width=0.003)
        ax.set_title("FD Density, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.formatter.set_powerlimits((0, 0))
        cbar.ax.set_title(r"$\rho$", fontsize=14)
    else:
        Vmag_fd = np.sqrt(vx_fd**2 + vy_fd**2)
        pc = ax.pcolormesh(X, Y, Vmag_fd, shading='auto', cmap='viridis', vmin=np.min(Vmag_fd), vmax=np.max(Vmag_fd))
        skip = (slice(None, None, max(1, Nx // 20)), slice(None, None, max(1, Ny // 20)))
        ax.quiver(X[skip], Y[skip], vx_fd[skip], vy_fd[skip], color='k', headwidth=3.0, width=0.003)
        ax.set_title("FD Velocity, t={}".format(round(time, 2)))
        cbar = plt.colorbar(pc, shrink=0.6, location='right')
        cbar.ax.set_title(r" $|v|$", fontsize=14)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return pc


def create_2d_surface_plots_FD(initial_params, time_points=None, which="density", N=200, nu=0.5,
                               use_velocity_ps=False, ps_index=-3.0, vel_rms=0.02, random_seed=None):
    """
    Create 2D surface plots at multiple time points using the LAX FD solver

    Args:
        initial_params: (xmin, xmax, ymin, ymax, rho_1, alpha, lam, output_folder, tmax)
        time_points: list of times
        which: "density" or "velocity"
        N: grid size
        nu: Courant number
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0, 1.5, 2.0]

    print("Creating 2D FD surface plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, t in enumerate(time_points):
        if i < len(axes):
            print(f"FD plotting at t = {t}")
            Two_D_surface_plots_FD(t, initial_params, N=N, nu=nu, ax=axes[i], which=which,
                                   use_velocity_ps=use_velocity_ps, ps_index=ps_index, vel_rms=vel_rms, random_seed=random_seed)

    if len(time_points) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    return fig, axes
