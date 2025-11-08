import os
import sys
import shutil
import numpy as np
import time
from typing import Tuple
import torch
import torch.nn as nn
from core.data_generator import input_taker, req_consts_calc
from training.trainer import train, train_xpinn
from core.initial_conditions import initialize_shared_velocity_fields
from config import BATCH_SIZE, NUM_BATCHES, N_0, N_r, DIMENSION
from config import a, wave, cs, xmin, ymin, tmin, tmax as TMAX_CFG, iteration_adam_2D, iteration_lbgfs_2D, harmonics, PERTURBATION_TYPE, rho_o
from config import num_neurons, num_layers
from config import USE_XPINN, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y, DEFAULT_ACTIVATION, RANDOM_SEED
from config import N_INTERFACE, XPINN_OPTIMIZER_STRATEGY, USE_MULTI_GPU, CACHE_IC_VALUES, STARTUP_DT
from config import USE_CAUSAL_TRAINING, CAUSAL_WEIGHTING_MODE, USE_CAUSAL_CURRICULUM
from config import CAUSAL_NUM_WINDOWS, CAUSAL_WINDOW_SCHEDULE, CAUSAL_USE_RESTARTS
from config import CAUSAL_CUSTOM_WINDOWS
from config import CAUSAL_GAMMA_MAX, CAUSAL_GAMMA_MIN
from config import CAUSAL_EPSILON, CAUSAL_EPSILON_FLOOR, CAUSAL_NUM_TIME_BINS
from config import USE_EPSILON_ANNEALING, CAUSAL_EPSILON_MIN, CAUSAL_EPSILON_MAX
from config import CAUSAL_ADAM_PER_WINDOW, CAUSAL_LBFGS_PER_WINDOW
from core.losses import ASTPN, XPINN_Loss
from core.model_architecture import PINN
from visualization.Plotting_2D import create_2d_animation
from visualization.Plotting_2D import create_1d_cross_sections_sinusoidal
from visualization.Plotting_2D import create_density_growth_plot
from config import PLOT_DENSITY_GROWTH, GROWTH_PLOT_TMAX, GROWTH_PLOT_DT
from config import FD_N_2D
import methods.xpinn_decomposition as xpinn_utils
from methods.xpinn_decomposition import (setup_xpinn_devices, setup_xpinn_networks, 
                                         setup_xpinn_collocation, setup_xpinn_interfaces, 
                                         cache_xpinn_initial_conditions)


def clean_pycache(root_dir: str) -> Tuple[int, int]:
    """Remove all __pycache__ directories and .pyc files under ``root_dir``."""
    removed_dirs = 0
    removed_files = 0

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed: {pycache_path}")
                removed_dirs += 1
            except Exception as exc:  # pragma: no cover - cleanup best effort
                print(f"Error removing {pycache_path}: {exc}")

        for filename in filenames:
            if filename.endswith(".pyc"):
                pyc_path = os.path.join(dirpath, filename)
                try:
                    os.remove(pyc_path)
                    print(f"Removed: {pyc_path}")
                    removed_files += 1
                except Exception as exc:  # pragma: no cover - cleanup best effort
                    print(f"Error removing {pyc_path}: {exc}")

    print("=" * 60)
    print("Cleanup complete!")
    print(f"  Removed {removed_dirs} __pycache__ directories")
    print(f"  Removed {removed_files} .pyc files")
    print("=" * 60)

    return removed_dirs, removed_files


if "--clean-pycache" in sys.argv or "--clean-cache" in sys.argv:
    flag = "--clean-pycache" if "--clean-pycache" in sys.argv else "--clean-cache"
    script_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Cleaning Python cache files from: {script_root}")
    clean_pycache(script_root)
    # Prevent the rest of the training script from running in cleanup-only mode
    sys.exit(0)

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Clear GPU memory if using CUDA
if device.startswith('cuda'):
    torch.cuda.empty_cache()


lam, rho_1, num_of_waves, tmax, _, _, _ = input_taker(wave, a, 2, TMAX_CFG, N_0, 0, N_r)

jeans, alpha = req_consts_calc(lam, rho_1)
# Set initial velocity amplitude per perturbation type
if str(PERTURBATION_TYPE).lower() == "sinusoidal":
    k = 2*np.pi/lam
    v_1 = (rho_1 / (rho_o if rho_o != 0 else 1.0)) * (alpha / k)
else:
    v_1 = a * cs

xmax = xmin + lam * num_of_waves
ymax = ymin + lam * num_of_waves

# Initialize shared velocity fields for consistent PINN/FD initial conditions
vx_np, vy_np = None, None  # Default values for sinusoidal case
if str(PERTURBATION_TYPE).lower() == "power_spectrum":
    vx_np, vy_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
    
    # Set shared velocity fields for plotting
    from visualization.Plotting_2D import set_shared_velocity_fields
    set_shared_velocity_fields(vx_np, vy_np)

# ==================== MODE SWITCHING: ORIGINAL PINN vs XPINN ====================

if not USE_XPINN:
    # ==================== ORIGINAL SINGLE PINN MODE ====================
    print("Running in original PINN mode (single network)...")
    
    net = PINN(n_harmonics=harmonics)
    net = net.to(device)
    mse_cost_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    optimizerL = torch.optim.LBFGS(net.parameters(), line_search_fn='strong_wolfe')

    model_2D = ASTPN(rmin=[xmin, ymin, tmin], rmax=[xmax, ymax, tmax], N_0=N_0, N_b=0, N_r=N_r, dimension=DIMENSION)

    # Set domain on the network so periodic embeddings enforce hard BCs
    net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=DIMENSION)

    # IC collocation stays at t=0 throughout
    collocation_IC_2D = model_2D.geo_time_coord(option="IC")

    start_time = time.time()
    
    if not USE_CAUSAL_TRAINING:
        # Standard training (no causal features)
        print("Using standard training (no causal curriculum)...")
        collocation_domain_2D = model_2D.geo_time_coord(option="Domain")
        
        train(
            net=net,
            model=model_2D,
            collocation_domain=collocation_domain_2D,
            collocation_IC=collocation_IC_2D,
            optimizer=optimizer,
            optimizerL=optimizerL,
            closure=None,
            mse_cost_function=mse_cost_function,
            iteration_adam=iteration_adam_2D,
            iterationL=iteration_lbgfs_2D,
            rho_1=rho_1,
            lam=lam,
            jeans=jeans,
            v_1=v_1,
            device=device,
            causal_gamma=0.0,
            causal_mode="none"
        )
    else:
        # Causal training using CausalTrainer module
        from methods.causal_training import CausalTrainer
        
        # Build configuration dictionary
        causal_config = {
            'weighting_mode': CAUSAL_WEIGHTING_MODE,
            'use_curriculum': USE_CAUSAL_CURRICULUM,
            'num_windows': CAUSAL_NUM_WINDOWS,
            'window_schedule': CAUSAL_WINDOW_SCHEDULE,
            'use_restarts': CAUSAL_USE_RESTARTS,
            'custom_windows': CAUSAL_CUSTOM_WINDOWS,
            'gamma_max': CAUSAL_GAMMA_MAX,
            'gamma_min': CAUSAL_GAMMA_MIN,
            'epsilon': CAUSAL_EPSILON,
            'epsilon_floor': CAUSAL_EPSILON_FLOOR,
            'epsilon_min': CAUSAL_EPSILON_MIN,
            'epsilon_max': CAUSAL_EPSILON_MAX,
            'use_epsilon_annealing': USE_EPSILON_ANNEALING,
            'num_time_bins': CAUSAL_NUM_TIME_BINS,
            'adam_per_window': CAUSAL_ADAM_PER_WINDOW,
            'lbfgs_per_window': CAUSAL_LBFGS_PER_WINDOW,
            'iteration_adam': iteration_adam_2D,
            'iteration_lbfgs': iteration_lbgfs_2D,
            'xmin': xmin,
            'xmax': xmax,
            'ymin': ymin,
            'ymax': ymax,
            'tmin': tmin,
            'tmax': tmax,
            'startup_dt': STARTUP_DT
        }
        
        # Initialize causal trainer
        causal_trainer = CausalTrainer(
            model=model_2D,
            net=net,
            optimizer=optimizer,
            optimizerL=optimizerL,
            mse_cost_function=mse_cost_function,
            train_func=train,
            config=causal_config,
            device=device
        )
        
        # Train with or without curriculum
        train_kwargs = {'rho_1': rho_1, 'lam': lam, 'jeans': jeans, 'v_1': v_1}
        
        if USE_CAUSAL_CURRICULUM:
            net = causal_trainer.train_with_curriculum(collocation_IC_2D, **train_kwargs)
        else:
            net = causal_trainer.train_without_curriculum(collocation_IC_2D, **train_kwargs)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Store net in a list for compatibility with plotting functions
    nets = [net]

else:
    # ==================== XPINN MULTI-NETWORK MODE ====================
    print("Running in XPINN mode (domain decomposition)...")
    
    # Calculate total number of subdomains
    num_subdomains = xpinn_utils.get_num_subdomains(NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y)
    print(f"Total subdomains: {num_subdomains}")
    
    # Setup device assignment
    subdomain_devices, num_gpus = setup_xpinn_devices(num_subdomains, device, USE_MULTI_GPU)
    
    # Initialize networks
    nets = setup_xpinn_networks(num_subdomains, subdomain_devices, xmin, xmax, ymin, ymax, 
                                 DIMENSION, num_neurons, num_layers, harmonics, 
                                 DEFAULT_ACTIVATION, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y)
    
    # Generate collocation points
    subdomain_collocs, subdomain_ic_collocs = setup_xpinn_collocation(
        num_subdomains, subdomain_devices, xmin, xmax, ymin, ymax, tmin, tmax, N_r, N_0,
        STARTUP_DT, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y
    )
    
    # Setup interfaces
    interfaces, interface_collocs = setup_xpinn_interfaces(
        subdomain_devices, xmin, xmax, ymin, ymax, tmin, tmax, N_INTERFACE,
        NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y
    )
    
    # Create XPINN loss model
    xpinn_loss_model = XPINN_Loss(rmin=[xmin, ymin, tmin], rmax=[xmax, ymax, tmax], dimension=DIMENSION)
    
    # Create initial condition functions - branch by perturbation type
    from core.initial_conditions import generate_power_spectrum_field, generate_power_spectrum_field_vy, fun_rho_0, fun_vx_0, fun_vy_0, func
    
    # Set IC functions based on perturbation type
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use sinusoidal IC functions
        ic_functions = {
            'rho': lambda colloc: fun_rho_0(rho_1, lam, colloc),
            'vx': lambda colloc: fun_vx_0(lam, jeans, v_1, colloc),
            'vy': lambda colloc: fun_vy_0(lam, jeans, v_1, colloc),
            'phi': lambda colloc: func(colloc)
        }
    else:
        # Use power spectrum IC functions
        ic_functions = {
            'rho': lambda colloc: fun_rho_0(rho_1, lam, colloc),
            'vx': lambda colloc: generate_power_spectrum_field(lam, v_1, colloc),
            'vy': lambda colloc: generate_power_spectrum_field_vy(lam, v_1, colloc),
            'phi': lambda colloc: func(colloc)
        }
    
    # Cache IC values for each subdomain to avoid recomputation
    cached_ic_values = None
    if CACHE_IC_VALUES:
        cached_ic_values = cache_xpinn_initial_conditions(num_subdomains, subdomain_ic_collocs, ic_functions)
    
    # Placeholder for exterior boundaries (to be implemented properly with periodic BC)
    exterior_boundaries = {}  # TODO: implement proper exterior boundary handling
    
    # Setup optimizer based on strategy and device configuration
    if USE_MULTI_GPU and len(set(subdomain_devices)) > 1:
        # Multi-GPU: use per-subdomain optimizers to avoid cross-device gradient issues
        print("Multi-GPU detected: using per-subdomain Adam optimizers")
        optimizer = None  # Will use per-subdomain optimizers in train_xpinn
        optimizerL = None
    else:
        # Single device: use unified optimizer (original approach)
        if XPINN_OPTIMIZER_STRATEGY == 'unified':
            all_params = []
            for net in nets:
                all_params.extend(list(net.parameters()))
            optimizer = torch.optim.Adam(all_params, lr=0.001)
            optimizerL = torch.optim.LBFGS(all_params, line_search_fn='strong_wolfe')
        else:
            raise NotImplementedError("Separate optimizer strategy not yet implemented")
    
    # Train XPINN
    start_time = time.time()
    train_xpinn(
        nets=nets,
        subdomain_collocs=subdomain_collocs,
        interface_collocs=interface_collocs,
        subdomain_ic_collocs=subdomain_ic_collocs,
        ic_functions=ic_functions,
        interfaces=interfaces,
        exterior_boundaries=exterior_boundaries,
        xpinn_loss_model=xpinn_loss_model,
        optimizer=optimizer,
        optimizerL=optimizerL,
        iteration_adam=iteration_adam_2D,
        iterationL=iteration_lbgfs_2D,
        device=device,
        subdomain_devices=subdomain_devices,
        cached_ic_values=cached_ic_values
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"XPINN training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Clear GPU memory after training
if device.startswith('cuda'):
    torch.cuda.empty_cache()
    print("GPU memory cleared after training")

# ==================== VISUALIZATION ====================
initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)

if not USE_XPINN:
    # Original single network visualization
    net = nets[0]  # Extract single network from list
    anim_density = create_2d_animation(net, initial_params, which="density", fps=10, verbose=False)
    anim_velocity = create_2d_animation(net, initial_params, which="velocity", fps=10, verbose=False)

    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        create_1d_cross_sections_sinusoidal(net, initial_params, time_points=None, y_fixed=0.6, N_fd=600, nu_fd=0.5)

    if PLOT_DENSITY_GROWTH:
        try:
            tmax_growth = float(GROWTH_PLOT_TMAX)
        except Exception:
            tmax_growth = float(TMAX_CFG)
        dt_growth = float(GROWTH_PLOT_DT)
        create_density_growth_plot(net, initial_params, tmax=tmax_growth, dt=dt_growth)

else:
    # XPINN multi-network visualization
    print("Creating XPINN visualizations...")
    anim_density = create_2d_animation(nets, initial_params, which="density", fps=10, verbose=False)
    anim_velocity = create_2d_animation(nets, initial_params, which="velocity", fps=10, verbose=False)
    print("XPINN visualizations created successfully!")
    
    # Sinusoidal cross-section plot for XPINN
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        create_1d_cross_sections_sinusoidal(nets, initial_params, time_points=None, y_fixed=0.6, N_fd=600, nu_fd=0.5)

    # Density growth plot for XPINN as well
    if PLOT_DENSITY_GROWTH:
        try:
            tmax_growth = float(GROWTH_PLOT_TMAX)
        except Exception:
            tmax_growth = float(TMAX_CFG)
        dt_growth = float(GROWTH_PLOT_DT)
        create_density_growth_plot(nets, initial_params, tmax=tmax_growth, dt=dt_growth)

# ==================== MODEL SAVING ====================
try:
    from config import SNAPSHOT_DIR
    model_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(model_dir, exist_ok=True)
    
    if not USE_XPINN:
        # Save single network
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(nets[0].state_dict(), model_path)
        print(f"Saved model to {model_path}")
    else:
        # Save all subdomain networks
        for i, net in enumerate(nets):
            model_path = os.path.join(model_dir, f"model_subdomain_{i}.pth")
            torch.save(net.state_dict(), model_path)
        print(f"Saved {len(nets)} subdomain models to {model_dir}")
except Exception as e:
    print(f"Warning: failed to save model: {e}")

# ==================== FINAL CACHE CLEANUP ====================
script_root = os.path.dirname(os.path.abspath(__file__))
print("Performing final Python cache cleanup...")
clean_pycache(script_root)

