# Config.py Parameter Reference

## Basic Physics Parameters
- `PERTURBATION_TYPE`: Type of initial perturbation ("power_spectrum" or "sinusoidal")
- `xmin`, `ymin`: Minimum spatial coordinates of the domain
- `cs`: Sound speed parameter
- `rho_o`: Reference density
- `const`: Physical constant
- `G`: Gravitational constant

## Collocation Points
- `N_0`: Number of initial condition collocation points
- `N_r`: Number of residual/collocation points for PDE enforcement
- `DIMENSION`: Spatial dimension (1, 2, or 3)

## Training Configuration
- `BATCH_SIZE`: Number of collocation points per mini-batch
- `NUM_BATCHES`: Number of mini-batches to aggregate per optimizer step
- `a`: Training parameter
- `tmin`, `tmax`: Time domain bounds
- `num_neurons`: Number of neurons per hidden layer
- `harmonics`: Number of harmonic features for periodic boundary conditions
- `num_layers`: Total number of linear layers in the network
- `iteration_adam_2D`: Number of Adam optimizer iterations
- `iteration_lbgfs_2D`: Number of L-BFGS optimizer iterations
- `IC_WEIGHT`: Weight for initial condition loss (float, default 1.0)

## Wave Parameters
- `wave`: Wavelength for sinusoidal perturbations
- `k`: Wavenumber (2π/wave)
- `KX`, `KY`: Wave vector components

## Output Controls
- `SAVE_STATIC_SNAPSHOTS`: Whether to save static plot snapshots
- `SNAPSHOT_DIR`: Directory for saving model snapshots
- `TIMES_1D`: Time points for 1D cross-section plots
- `FD_N_1D`: Grid points for 1D finite difference comparison
- `FD_N_2D`: Grid points per dimension for 2D finite difference comparison

## Power Spectrum Parameters
- `N_GRID`: Grid resolution for power spectrum generation
- `POWER_EXPONENT`: Power spectrum slope exponent
- `FILTER_SCALE`: Filter scale parameter (Rf)
- `STARTUP_DT`: Time offset after which PDE is enforced (ICs remain at t=0)
- `CONTINUITY_IC_WEIGHT`: Weight for enforcing continuity at t=0
- `DECAY_PORTION`: Fraction of training steps over which to fully decay

## Density Growth Plotting
- `PLOT_DENSITY_GROWTH`: Whether to plot PINN vs LAX density growth comparison
- `GROWTH_PLOT_TMAX`: Maximum time for growth plot
- `GROWTH_PLOT_DT`: Time step for growth plot

## XPINN Domain Decomposition
- `USE_XPINN`: Toggle XPINN on/off (False = original single PINN)
- `NUM_SUBDOMAINS_X`: Number of subdomain splits in x-direction
- `NUM_SUBDOMAINS_Y`: Number of subdomain splits in y-direction
- `N_INTERFACE`: Number of interface collocation points per interface
- `N_r_PER_SUBDOMAIN`: Residual points per subdomain (None = auto-distribute)
- `N_0_PER_SUBDOMAIN`: IC points per subdomain (None = auto-distribute)

## Interface Loss Weights
- `INTERFACE_SOLUTION_WEIGHT`: Weight for solution continuity at interfaces
- `INTERFACE_RESIDUAL_WEIGHT`: Weight for residual continuity at interfaces
- `INTERFACE_SOLUTION_COMPONENTS`: Which solution components to enforce continuity for

## Per-Subdomain Network Architecture
- `SUBDOMAIN_CONFIGS`: List of dicts configuring each subdomain's network architecture (list or None)
  - Format: `[{config_0}, {config_1}, ..., {config_N-1}]` where N = NUM_SUBDOMAINS_X × NUM_SUBDOMAINS_Y
  - Each dict can contain: `num_neurons`, `num_layers`, `n_harmonics`, `activation`
  - Missing keys use global defaults (`num_neurons`, `num_layers`, `harmonics`, `DEFAULT_ACTIVATION`)
  - Set to `None` to use global defaults for all subdomains
  - Subdomain ordering (for 2×2 grid): bottom-left [0], bottom-right [2], top-left [1], top-right [3]
- `DEFAULT_ACTIVATION`: Default activation function used when not specified in SUBDOMAIN_CONFIGS

## Training Strategy
- `XPINN_OPTIMIZER_STRATEGY`: Optimizer strategy ('unified' or 'separate')
- `XPINN_ALTERNATING_TRAINING`: Whether to alternate subdomain training
- `USE_XPINN_BATCHING`: Whether to use mini-batch processing for XPINN (reduces GPU memory)
- `USE_MULTI_GPU`: Whether to distribute subdomains across available GPUs
  - If `True`: Automatically detects available GPUs and assigns subdomains round-robin
  - Networks stay on assigned devices during Adam; L-BFGS optimizes per-subdomain on its device
  - Works with any number of GPUs (1, 2, 4, 8, etc.)
- `CACHE_IC_VALUES`: Whether to precompute and cache initial condition values for faster training

## Visualization
- `SHOW_INTERFACE_LINES`: Whether to draw subdomain boundaries in plots
- `INTERFACE_AVERAGING`: Method to combine overlapping predictions at interfaces ('mean', 'weighted', 'subdomain1', 'subdomain2')

## Causal Training Configuration
Temporal curriculum learning and time-weighted PDE residuals for long-time predictions.

- `USE_CAUSAL_TRAINING`: Enable causal training (bool)
- `CAUSAL_WEIGHTING_MODE`: Weighting mode ("static" or "adaptive")
- `USE_CAUSAL_CURRICULUM`: Enable temporal curriculum windows (bool)
- `CAUSAL_NUM_WINDOWS`: Number of progressive time windows (int, 2-10 recommended)
- `CAUSAL_WINDOW_SCHEDULE`: Window schedule ("linear" or "custom")
- `CAUSAL_USE_RESTARTS`: Use restart marching instead of expanding windows (bool)
- `CAUSAL_CUSTOM_WINDOWS`: User-specified time windows (list of [t_min, t_max] pairs or None)
- `CAUSAL_GAMMA_MAX`: Maximum gamma for exponential weighting (float, 0.5-5.0 recommended)
- `CAUSAL_GAMMA_MIN`: Minimum gamma for final window (float, typically 0.0)
- `CAUSAL_EPSILON`: Causality parameter for adaptive weighting (float, 0.1-2.0 recommended)
- `CAUSAL_EPSILON_FLOOR`: Minimum epsilon value (float, 0.0-0.3)
- `USE_EPSILON_ANNEALING`: Enable automatic epsilon interpolation (bool)
- `CAUSAL_EPSILON_MIN`: Starting epsilon for interpolation (float, 0.1-2.0)
- `CAUSAL_EPSILON_MAX`: Final epsilon for interpolation (float, 5.0-50.0)
- `CAUSAL_NUM_TIME_BINS`: Number of time bins for tracking residuals (int, 5-20 recommended)
- `CAUSAL_ADAM_PER_WINDOW`: Adam iterations per window (int or None for auto-split)
- `CAUSAL_LBFGS_PER_WINDOW`: L-BFGS iterations per window (int or None for auto-split)

---

## Random Seed
- `RANDOM_SEED`: Random seed for reproducibility across all functions