import numpy as np
import time
import torch
import torch.nn as nn
from solver import input_taker, req_consts_calc, train, initialize_shared_velocity_fields, train_xpinn, distribute_collocation_points
from config import BATCH_SIZE, NUM_BATCHES, N_0, N_r, DIMENSION
from config import a, wave, cs, xmin, ymin, tmin, tmax as TMAX_CFG, iteration_adam_2D, iteration_lbgfs_2D, harmonics, PERTURBATION_TYPE, rho_o
from config import num_neurons, num_layers
from config import USE_XPINN, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y, DEFAULT_ACTIVATION, RANDOM_SEED
from config import N_INTERFACE, XPINN_OPTIMIZER_STRATEGY, USE_MULTI_GPU, CACHE_IC_VALUES, STARTUP_DT
from losses import ASTPN, XPINN_Loss
from model_architecture import PINN
from Plotting_2D import create_2d_animation
from Plotting_2D import create_1d_cross_sections_sinusoidal
from Plotting_2D import create_density_growth_plot
from config import PLOT_DENSITY_GROWTH, GROWTH_PLOT_TMAX, GROWTH_PLOT_DT
import xpinn_decomposition as xpinn_utils

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Clear GPU memory if using CUDA
if device.startswith('cuda'):
    torch.cuda.empty_cache()
else:
    pass

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
if str(PERTURBATION_TYPE).lower() == "power_spectrum":
    vx_np, vy_np = initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=RANDOM_SEED)
    
    # Set shared velocity fields for plotting
    from Plotting_2D import set_shared_velocity_fields
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

    collocation_domain_2D = model_2D.geo_time_coord(option="Domain") 
    collocation_IC_2D = model_2D.geo_time_coord(option="IC")

    start_time = time.time()
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
        device=device
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Store net in a list for compatibility with plotting functions
    nets = [net]

else:
    # ==================== XPINN MULTI-NETWORK MODE ====================
    
    # Calculate total number of subdomains
    num_subdomains = xpinn_utils.get_num_subdomains(NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y)
    print(f"Total subdomains: {num_subdomains}")
    
    # Multi-GPU setup
    if USE_MULTI_GPU and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Multi-GPU enabled: {num_gpus} GPUs available")
        devices = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        num_gpus = 1
        devices = [device] * num_subdomains
    
    # Load subdomain-specific configurations
    from config import SUBDOMAIN_CONFIGS
    
    # Validate SUBDOMAIN_CONFIGS
    if SUBDOMAIN_CONFIGS and len(SUBDOMAIN_CONFIGS) != num_subdomains:
        print(f"WARNING: SUBDOMAIN_CONFIGS has {len(SUBDOMAIN_CONFIGS)} entries but {num_subdomains} subdomains expected.")
        print(f"         Using global defaults for all subdomains.")
        SUBDOMAIN_CONFIGS = None
    
    # Initialize multiple networks (one per subdomain)
    nets = []
    subdomain_devices = []
    for i in range(num_subdomains):
        # Get subdomain-specific configuration or use defaults
        if SUBDOMAIN_CONFIGS and i < len(SUBDOMAIN_CONFIGS):
            config = SUBDOMAIN_CONFIGS[i]
            sub_neurons = config.get('num_neurons', num_neurons)
            sub_layers = config.get('num_layers', num_layers)
            sub_harmonics = config.get('n_harmonics', harmonics)
            sub_activation = config.get('activation', DEFAULT_ACTIVATION)
        else:
            # Use global defaults
            sub_neurons = num_neurons
            sub_layers = num_layers
            sub_harmonics = harmonics
            sub_activation = DEFAULT_ACTIVATION
        
        # Assign device (round-robin across GPUs)
        subdomain_device = devices[i % len(devices)] if USE_MULTI_GPU else device
        subdomain_devices.append(subdomain_device)
        
        # Create network with subdomain-specific architecture
        net = PINN(num_neurons=sub_neurons, num_layers=sub_layers, 
                  n_harmonics=sub_harmonics, activation_type=sub_activation)
        
        # Get subdomain bounds for logging and data generation
        subdomain_bounds = xpinn_utils.get_subdomain_bounds(i, xmin, xmax, ymin, ymax, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y)
        
        # Use GLOBAL domain for periodic embeddings (not subdomain bounds)
        # This ensures all nets use the same spatial basis for IC representation
        # while still training on their respective subdomain data
        net.set_domain(rmin=[xmin, ymin], 
                      rmax=[xmax, ymax], 
                      dimension=DIMENSION)
        net = net.to(subdomain_device)
        nets.append(net)
        
        # Print configuration
        print(f"  Subdomain {i}: bounds={subdomain_bounds}")
        print(f"    Architecture: neurons={sub_neurons}, layers={sub_layers}, harmonics={sub_harmonics}, activation={sub_activation}, device={subdomain_device}")
    
    # Distribute collocation points across subdomains
    n_r_per_subdomain = distribute_collocation_points(N_r, num_subdomains)
    n_0_per_subdomain = distribute_collocation_points(N_0, num_subdomains)
    
    # Generate collocation points for each subdomain (on corresponding device)
    subdomain_collocs = []
    subdomain_ic_collocs = []
    
    for i in range(num_subdomains):
        subdomain_bounds = xpinn_utils.get_subdomain_bounds(i, xmin, xmax, ymin, ymax, NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y)
        subdomain_device = subdomain_devices[i]
        colloc_domain, colloc_ic = xpinn_utils.generate_subdomain_collocation(
            subdomain_bounds, n_r_per_subdomain[i], n_0_per_subdomain[i],
            tmin, tmax, STARTUP_DT, device=subdomain_device  # Using STARTUP_DT from config
        )
        subdomain_collocs.append(colloc_domain)
        subdomain_ic_collocs.append(colloc_ic)
    
    # Get interfaces between adjacent subdomains
    interfaces = xpinn_utils.get_interfaces(NUM_SUBDOMAINS_X, NUM_SUBDOMAINS_Y)
    
    # Generate interface collocation points (on device of first subdomain in pair)
    interface_collocs = {}
    for interface in interfaces:
        subdomain_i, subdomain_j, _, _ = interface
        # Use device of first subdomain in the interface pair
        interface_device = subdomain_devices[subdomain_i]
        interface_points = xpinn_utils.generate_interface_points(
            interface, xmin, xmax, ymin, ymax, tmin, tmax, N_INTERFACE, device=interface_device
        )
        interface_collocs[(subdomain_i, subdomain_j)] = interface_points
    
    # Create XPINN loss model
    xpinn_loss_model = XPINN_Loss(rmin=[xmin, ymin, tmin], rmax=[xmax, ymax, tmax], dimension=DIMENSION)
    
    # Create initial condition functions (using power spectrum)
    from solver import generate_power_spectrum_field, generate_power_spectrum_field_vy, fun_rho_0, func
    
    ic_functions = {
        'rho': lambda colloc: fun_rho_0(rho_1, lam, colloc),
        'vx': lambda colloc: generate_power_spectrum_field(lam, v_1, colloc),
        'vy': lambda colloc: generate_power_spectrum_field_vy(lam, v_1, colloc),
        'phi': lambda colloc: func(colloc)
    }
    
    # Cache IC values for each subdomain to avoid recomputation
    cached_ic_values = None
    if CACHE_IC_VALUES:
        print("Caching IC values for all subdomains...")
        cached_ic_values = []
        for i in range(num_subdomains):
            colloc_ic = subdomain_ic_collocs[i]
            subdomain_device = subdomain_devices[i]
            ic_cache = {
                'rho': ic_functions['rho'](colloc_ic),
                'vx': ic_functions['vx'](colloc_ic),
                'vy': ic_functions['vy'](colloc_ic),
                'phi': ic_functions['phi'](colloc_ic)
            }
            cached_ic_values.append(ic_cache)
        print("IC values cached successfully!")
    
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
    import os
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