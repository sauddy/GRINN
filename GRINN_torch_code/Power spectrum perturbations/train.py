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
from config import USE_CAUSAL_TRAINING, CAUSAL_WEIGHTING_MODE, USE_CAUSAL_CURRICULUM
from config import CAUSAL_NUM_WINDOWS, CAUSAL_WINDOW_SCHEDULE, CAUSAL_USE_RESTARTS
from config import CAUSAL_CUSTOM_WINDOWS
from config import CAUSAL_GAMMA_MAX, CAUSAL_GAMMA_MIN
from config import CAUSAL_EPSILON, CAUSAL_EPSILON_FLOOR, CAUSAL_NUM_TIME_BINS
from config import USE_EPSILON_ANNEALING, CAUSAL_EPSILON_MIN, CAUSAL_EPSILON_MAX
from config import CAUSAL_ADAM_PER_WINDOW, CAUSAL_LBFGS_PER_WINDOW
from config import USE_ADAPTIVE_COLLOCATION, ADAPTIVE_COLLOCATION_FREQUENCY
from config import ADAPTIVE_COLLOCATION_THRESHOLD_MODE, ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL
from config import ADAPTIVE_COLLOCATION_PERCENTILE_FINAL, ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START
from config import ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END, ADAPTIVE_COLLOCATION_RATIO_MODE, ADAPTIVE_COLLOCATION_LBFGS_MODE
from config import ADAPTIVE_COLLOCATION_CAUSAL_COMPATIBLE
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
vx_np, vy_np = None, None  # Default values for sinusoidal case
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

    # IC collocation stays at t=0 throughout
    collocation_IC_2D = model_2D.geo_time_coord(option="IC")

    # ==================== ADAPTIVE COLLOCATION ALLOCATION ====================
    adaptive_allocator = None
    if USE_ADAPTIVE_COLLOCATION:
        print("Setting up adaptive collocation allocation...")
        from adaptive_collocation import create_adaptive_allocator
        
        # Create adaptive allocator with shared velocity fields
        adaptive_allocator = create_adaptive_allocator(
            lam=lam, num_of_waves=num_of_waves, rho_1=rho_1,
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
            tmin=tmin, tmax=tmax, total_points=N_r,
            device=device, vx0_shared=vx_np, vy0_shared=vy_np
        )
        
        if adaptive_allocator is not None:
            print("Adaptive collocation allocation enabled!")
            print(f"  Update frequency: every {ADAPTIVE_COLLOCATION_FREQUENCY} Adam iterations")
            print(f"  Threshold mode: {ADAPTIVE_COLLOCATION_THRESHOLD_MODE}")
            if ADAPTIVE_COLLOCATION_THRESHOLD_MODE == "percentile":
                print(f"  Percentile: {ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL} -> {ADAPTIVE_COLLOCATION_PERCENTILE_FINAL}")
            print(f"  Ratio mode: {ADAPTIVE_COLLOCATION_RATIO_MODE}")
            print(f"  LBFGS mode: {ADAPTIVE_COLLOCATION_LBFGS_MODE}")
        else:
            print("Adaptive collocation allocation disabled.")
    else:
        print("Adaptive collocation allocation disabled.")

    start_time = time.time()
    
    if not USE_CAUSAL_TRAINING:
        # Standard training (original behavior)
        print("Using standard training (no causal curriculum)...")
        collocation_domain_2D = model_2D.geo_time_coord(option="Domain")
        
        # Use adaptive collocation if available
        if adaptive_allocator is not None:
            collocation_domain_2D = adaptive_allocator.current_points
        
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
            causal_gamma=0.0,  # No causal weighting
            causal_mode="none",
            adaptive_allocator=adaptive_allocator
        )
    else:
        # Causal training with temporal curriculum and/or causal weighting
        if USE_CAUSAL_CURRICULUM:
            print(f"Using causal training with {CAUSAL_NUM_WINDOWS} temporal windows...")
        else:
            print(f"Using causal training (no curriculum, full domain)...")
        
        if CAUSAL_WEIGHTING_MODE == "static":
            print(f"  Weighting mode: Static (gamma range: [{CAUSAL_GAMMA_MAX}, {CAUSAL_GAMMA_MIN}])")
        elif CAUSAL_WEIGHTING_MODE == "adaptive":
            print(f"  Weighting mode: Adaptive (epsilon: {CAUSAL_EPSILON}, time bins: {CAUSAL_NUM_TIME_BINS})")
        else:
            raise ValueError(f"Unknown CAUSAL_WEIGHTING_MODE: {CAUSAL_WEIGHTING_MODE}")
        
        # Initialize residual tracker for adaptive weighting
        from solver import ResidualTracker
        residual_tracker = None
        if CAUSAL_WEIGHTING_MODE == "adaptive":
            t_start = max(tmin, STARTUP_DT)
            residual_tracker = ResidualTracker(
                t_min=t_start,
                t_max=tmax,
                num_bins=CAUSAL_NUM_TIME_BINS,
                epsilon=CAUSAL_EPSILON,
                device=device
            )
        
        if USE_CAUSAL_CURRICULUM:
            # Temporal curriculum with windows
            adam_per_window = CAUSAL_ADAM_PER_WINDOW if CAUSAL_ADAM_PER_WINDOW is not None else iteration_adam_2D // CAUSAL_NUM_WINDOWS
            lbfgs_per_window = CAUSAL_LBFGS_PER_WINDOW if CAUSAL_LBFGS_PER_WINDOW is not None else iteration_lbgfs_2D // CAUSAL_NUM_WINDOWS
            
            t_start = max(tmin, STARTUP_DT)
            t_range = tmax - t_start
            
            # Validate custom windows if specified
            if CAUSAL_WINDOW_SCHEDULE == "custom":
                if CAUSAL_CUSTOM_WINDOWS is None:
                    raise ValueError("CAUSAL_CUSTOM_WINDOWS must be provided when CAUSAL_WINDOW_SCHEDULE='custom'")
                if not isinstance(CAUSAL_CUSTOM_WINDOWS, (list, tuple)):
                    raise ValueError("CAUSAL_CUSTOM_WINDOWS must be a list or tuple")
                if len(CAUSAL_CUSTOM_WINDOWS) != CAUSAL_NUM_WINDOWS:
                    raise ValueError(f"CAUSAL_CUSTOM_WINDOWS must have {CAUSAL_NUM_WINDOWS} entries, got {len(CAUSAL_CUSTOM_WINDOWS)}")
                for i, window in enumerate(CAUSAL_CUSTOM_WINDOWS):
                    if not isinstance(window, (list, tuple)) or len(window) != 2:
                        raise ValueError(f"CAUSAL_CUSTOM_WINDOWS[{i}] must be a [t_min, t_max] pair")
                    if window[1] <= window[0]:
                        raise ValueError(f"CAUSAL_CUSTOM_WINDOWS[{i}]: t_max ({window[1]}) must be > t_min ({window[0]})")
                
                # Check that the final window's maximum time equals tmax
                final_window = CAUSAL_CUSTOM_WINDOWS[-1]
                final_t_max = final_window[1]
                if abs(final_t_max - tmax) > 1e-10:  # Use small epsilon for floating point comparison
                    raise ValueError(
                        f"Custom windows must end at tmax ({tmax}), but final window ends at {final_t_max}. "
                        f"Ensure the last window's t_max equals tmax."
                    )
                
                print(f"Using custom windows: {CAUSAL_CUSTOM_WINDOWS}")
            
            for window_idx in range(CAUSAL_NUM_WINDOWS):
                # Compute time window bounds
                if CAUSAL_WINDOW_SCHEDULE == "custom":
                    # Use user-provided custom windows
                    t_window_min, t_window_max = CAUSAL_CUSTOM_WINDOWS[window_idx]
                    # Note: With custom windows, CAUSAL_USE_RESTARTS is ignored - custom windows control everything
                elif CAUSAL_WINDOW_SCHEDULE == "linear":
                    t_window_max = t_start + t_range * (window_idx + 1) / CAUSAL_NUM_WINDOWS
                    
                    # For restart marching: compute window minimum (previous window's max)
                    if CAUSAL_USE_RESTARTS:
                        # Start first window from STARTUP_DT, subsequent windows from previous t_window_max
                        if window_idx == 0:
                            t_window_min = t_start  # t_start is already max(tmin, STARTUP_DT)
                        else:
                            # Previous window's max becomes current window's min
                            t_window_min = t_start + t_range * window_idx / CAUSAL_NUM_WINDOWS
                    else:
                        # Expanding windows: always start from STARTUP_DT (t_start)
                        t_window_min = t_start
                else:
                    raise ValueError(f"Unknown CAUSAL_WINDOW_SCHEDULE: {CAUSAL_WINDOW_SCHEDULE}")
                
                # Compute causal gamma for static mode (decays linearly to 0)
                if CAUSAL_WEIGHTING_MODE == "static":
                    causal_gamma = CAUSAL_GAMMA_MAX * (1.0 - window_idx / max(1, CAUSAL_NUM_WINDOWS - 1))
                    if window_idx == CAUSAL_NUM_WINDOWS - 1:
                        causal_gamma = CAUSAL_GAMMA_MIN
                else:
                    causal_gamma = 0.0  # Not used in adaptive mode
                
                # Anneal epsilon for adaptive mode
                if CAUSAL_WEIGHTING_MODE == "adaptive" and residual_tracker is not None:
                    # Choose epsilon based on annealing strategy
                    if USE_EPSILON_ANNEALING:
                        # Automatic interpolation between MIN and MAX (ensures monotonic increase)
                        progress = window_idx / max(1, CAUSAL_NUM_WINDOWS - 1)
                        eps_k = CAUSAL_EPSILON_MIN + (CAUSAL_EPSILON_MAX - CAUSAL_EPSILON_MIN) * progress
                        print(f"  Epsilon annealing: interpolated value ε = {eps_k:.3f} (progress: {progress:.2f})")
                    else:
                        # Linear decay (original method): strong early → moderate late
                        eps_k = CAUSAL_EPSILON_FLOOR + (CAUSAL_EPSILON - CAUSAL_EPSILON_FLOOR) * (1.0 - window_idx / max(1, CAUSAL_NUM_WINDOWS - 1))
                        print(f"  Epsilon linear decay: ε = {eps_k:.3f}")
                    
                    # For restart mode with local bins: reinitialize tracker for this window's time range
                    if CAUSAL_USE_RESTARTS:
                        residual_tracker = ResidualTracker(
                            t_min=t_window_min,
                            t_max=t_window_max,
                            num_bins=CAUSAL_NUM_TIME_BINS,
                            epsilon=eps_k,
                            device=device
                        )
                        print(f"  Adaptive epsilon: {eps_k:.3f} (local bins: [{t_window_min:.3f}, {t_window_max:.3f}])")
                    else:
                        residual_tracker.epsilon = eps_k
                        print(f"  Adaptive epsilon: {eps_k:.3f}")
                
                print(f"\n=== Causal Window {window_idx + 1}/{CAUSAL_NUM_WINDOWS} ===")
                if CAUSAL_USE_RESTARTS:
                    print(f"  Mode: Restart marching (warm start from previous window)")
                else:
                    print(f"  Mode: Expanding windows (retrain from t=0)")
                print(f"  Time range: [{t_window_min:.3f}, {t_window_max:.3f}]")
                if CAUSAL_WEIGHTING_MODE == "static":
                    print(f"  Causal gamma: {causal_gamma:.3f}")
                print(f"  Iterations: {adam_per_window} Adam + {lbfgs_per_window} LBFGS")
                
                # Update model's temporal bounds for this window
                model_2D.rmin = [xmin, ymin, t_window_min]
                model_2D.rmax = [xmax, ymax, t_window_max]
                
                # Regenerate domain collocation for this window
                collocation_domain_window = model_2D.geo_time_coord(option="Domain")
                
                # Use adaptive collocation if available (but filter by time window)
                if adaptive_allocator is not None and ADAPTIVE_COLLOCATION_CAUSAL_COMPATIBLE:
                    # Filter adaptive points to current time window
                    adaptive_points = adaptive_allocator.current_points
                    time_mask = (adaptive_points[:, 2] >= t_window_min) & (adaptive_points[:, 2] <= t_window_max)
                    if torch.sum(time_mask) > 0:
                        collocation_domain_window = adaptive_points[time_mask]
                        print(f"  Using {torch.sum(time_mask).item()} adaptive points in time window [{t_window_min:.3f}, {t_window_max:.3f}]")
                
                # Train on this window
                train(
                    net=net,
                    model=model_2D,
                    collocation_domain=collocation_domain_window,
                    collocation_IC=collocation_IC_2D,
                    optimizer=optimizer,
                    optimizerL=optimizerL,
                    closure=None,
                    mse_cost_function=mse_cost_function,
                    iteration_adam=adam_per_window,
                    iterationL=lbfgs_per_window,
                    rho_1=rho_1,
                    lam=lam,
                    jeans=jeans,
                    v_1=v_1,
                    device=device,
                    causal_gamma=causal_gamma,
                    causal_mode=CAUSAL_WEIGHTING_MODE,
                    residual_tracker=residual_tracker,
                    window_idx=window_idx,
                    adaptive_allocator=adaptive_allocator
                )
            
            # Restore full time/space range for final evaluation/plotting
            model_2D.rmin = [xmin, ymin, tmin]
            model_2D.rmax = [xmax, ymax, tmax]
            if CAUSAL_USE_RESTARTS:
                print(f"\nCausal training completed (restart marching). Final time range: [{tmin}, {tmax}]")
            else:
                print(f"\nCausal training completed (expanding windows). Final time range: [{tmin}, {tmax}]")
        
        else:
            # No curriculum - train on full domain with adaptive weighting
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
                causal_gamma=0.0,  # Not used in adaptive mode
                causal_mode=CAUSAL_WEIGHTING_MODE,
                residual_tracker=residual_tracker,
                window_idx=None  # No curriculum
            )
            print(f"\nCausal training completed (full domain)")
    
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
    
    # Create initial condition functions - branch by perturbation type
    from solver import generate_power_spectrum_field, generate_power_spectrum_field_vy, fun_rho_0, fun_vx_0, fun_vy_0, func
    
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