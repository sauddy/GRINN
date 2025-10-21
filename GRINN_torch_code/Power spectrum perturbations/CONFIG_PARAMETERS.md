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
- `SUBDOMAIN_CONFIGS`: List of dicts configuring each subdomain's network architecture
  - **Format**: `[{config_0}, {config_1}, ..., {config_N-1}]` where N = NUM_SUBDOMAINS_X × NUM_SUBDOMAINS_Y
  - **Each dict can contain**:
    - `num_neurons`: Number of neurons per hidden layer (int)
    - `num_layers`: Total number of linear layers (int)
    - `n_harmonics`: Number of harmonic features for periodic BCs (int)
    - `activation`: Activation function ('sin', 'tanh', 'relu', 'elu')
  - **Fallback**: Missing keys use global defaults (`num_neurons`, `num_layers`, `harmonics`, `DEFAULT_ACTIVATION`)
  - **Disable**: Set to `None` to use global defaults for all subdomains
  
  **Subdomain Ordering** (for 2×2 grid):
  ```
  [1] | [3]    (top row, y: ymax/2 → ymax)
  ----+----
  [0] | [2]    (bottom row, y: ymin → ymax/2)
  ```
  
  **Example 1 - Full specification**:
  ```python
  SUBDOMAIN_CONFIGS = [
      {'num_neurons': 64, 'num_layers': 4, 'n_harmonics': 2, 'activation': 'tanh'},
      {'num_neurons': 64, 'num_layers': 4, 'n_harmonics': 2, 'activation': 'tanh'},
      {'num_neurons': 64, 'num_layers': 4, 'n_harmonics': 2, 'activation': 'tanh'},
      {'num_neurons': 128, 'num_layers': 6, 'n_harmonics': 4, 'activation': 'sin'},
  ]
  ```
  
  **Example 2 - Only specify activation**:
  ```python
  SUBDOMAIN_CONFIGS = [
      {'activation': 'tanh'},  # Uses global num_neurons, num_layers, harmonics
      {'activation': 'tanh'},
      {'activation': 'tanh'},
      {'activation': 'sin'},
  ]
  ```
  
  **Example 3 - Use global defaults**:
  ```python
  SUBDOMAIN_CONFIGS = None  # All subdomains use global settings
  ```
  
  **Use Cases**:
  - **Gravitational collapse**: Use stronger network (more neurons/layers/harmonics, 'sin' activation) in collapse region; lighter networks ('tanh') elsewhere
  - **Varying complexity**: Allocate compute resources based on local physics complexity
  - **Faster convergence**: Use 'tanh' for smooth regions (faster), 'sin' for high-frequency/nonlinear regions
  
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
