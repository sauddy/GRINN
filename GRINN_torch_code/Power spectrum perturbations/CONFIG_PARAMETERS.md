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

## Causal Training Configuration
Causal training improves long-time predictions by using temporal curriculum learning and time-weighted PDE residuals. This approach trains the network progressively from early to late times, respecting the causality inherent in physical systems.

### Core Settings
- `USE_CAUSAL_TRAINING`: Enable/disable causal training (True/False)
  - **True**: Uses temporal curriculum and/or causal weighting
  - **False**: Uses standard training (original behavior)
  - **Default**: True

- `CAUSAL_WEIGHTING_MODE`: Type of causal weighting ("static" or "adaptive")
  - **"static"**: Simple exp(-gamma*t) weighting (simpler, predictable)
  - **"adaptive"**: Residual-based weighting w_i = exp(-epsilon * Σ L_r(t_k)) (paper's full method, progress-aware)
  - **Default**: "adaptive"
  - **Recommendation**: Use "adaptive" for best results, "static" for simplicity

### Temporal Curriculum Settings
- `USE_CAUSAL_CURRICULUM`: Enable temporal curriculum windows (True/False)
  - **True**: Train progressively on expanding time windows
  - **False**: Train on full temporal domain from start
  - **Default**: True
  - **Recommendation**: True with adaptive mode for best stability, optional with static mode

- `CAUSAL_NUM_WINDOWS`: Number of progressive time windows (int)
  - **Purpose**: Controls granularity of temporal curriculum
  - **Range**: 2-10 recommended (more windows = more gradual progression)
  - **Default**: 8 (increased for tmax=2.0)
  - **Example**: With tmax=0.5 and 5 windows: [0.01,0.108], [0.01,0.206], [0.01,0.304], [0.01,0.402], [0.01,0.500]

- `CAUSAL_WINDOW_SCHEDULE`: Time window progression schedule (string)
  - **Options**: "linear" (currently only supported option)
  - **Linear**: Each window covers 1/NUM_WINDOWS of the total time range
  - **Default**: "linear"
  - **Future**: Could support "exponential", "custom" schedules

### Static Weighting Settings (CAUSAL_WEIGHTING_MODE = "static")
- `CAUSAL_GAMMA_MAX`: Maximum gamma for exponential time-weighting (float)
  - **Purpose**: Controls strength of early-time emphasis in PDE residuals
  - **Formula**: w(t) = exp(-gamma * t) where higher gamma = stronger early-time focus
  - **Range**: 0.5-5.0 recommended (0.0 = uniform weighting)
  - **Default**: 1.5 (adjusted for tmax=2.0)
  - **Effect**: Early times get weight ~1.0, later times get weight ~0.37 (for gamma=2.0, t=0.5)
  - **Tuning**: Lower for longer tmax (e.g., 1.0-1.5 for tmax=2.0, 2.0-3.0 for tmax=0.5)

- `CAUSAL_GAMMA_MIN`: Minimum gamma for final window (float)
  - **Purpose**: Ensures final window uses uniform weighting (no bias)
  - **Range**: 0.0 recommended (uniform weighting)
  - **Default**: 0.0
  - **Behavior**: Gamma decays linearly across windows: gamma_k = GAMMA_MAX * (1 - k/NUM_WINDOWS)

### Adaptive Weighting Settings (CAUSAL_WEIGHTING_MODE = "adaptive")
- `CAUSAL_EPSILON`: Causality parameter epsilon for adaptive weighting (float)
  - **Purpose**: Controls how strongly past residuals suppress future time weights
  - **Formula**: w_i = exp(-epsilon * Σ_{k=1}^{i-1} L_r(t_k, θ))
  - **Range**: 0.1-2.0 recommended
  - **Default**: 0.5
  - **Tuning**: Lower for longer tmax (0.1-0.3 for tmax=2.0, 0.5-1.0 for tmax=0.5)
  - **Effect**: Higher epsilon = weights shift forward only after early times converge well

- `CAUSAL_NUM_TIME_BINS`: Number of time bins for tracking residuals (int)
  - **Purpose**: Temporal resolution for adaptive weight calculation
  - **Range**: 5-20 recommended
  - **Default**: 10
  - **Effect**: More bins = finer control but slightly more overhead
  - **Recommendation**: 10-15 for most cases

### Training Iterations per Window
- `CAUSAL_ADAM_PER_WINDOW`: Adam iterations per temporal window (int or None)
  - **None**: Auto-splits `iteration_adam_2D` equally across windows
  - **Custom**: Specify exact iterations per window
  - **Default**: None (auto-split)
  - **Example**: With iteration_adam_2D=800 and 5 windows → 160 Adam iterations per window

- `CAUSAL_LBFGS_PER_WINDOW`: L-BFGS iterations per temporal window (int or None)
  - **None**: Auto-splits `iteration_lbgfs_2D` equally across windows
  - **Custom**: Specify exact iterations per window
  - **Default**: None (auto-split)
  - **Example**: With iteration_lbgfs_2D=160 and 5 windows → 32 L-BFGS iterations per window

### How Causal Training Works

1. **Temporal Curriculum** (when USE_CAUSAL_CURRICULUM = True):
   - Network trains on progressively longer time windows
   - Window 1: [STARTUP_DT, t₁], Window 2: [STARTUP_DT, t₂], ..., Window N: [STARTUP_DT, tmax]
   - Each window builds on previous windows' learned weights
   - Provides stable initialization for longer time intervals

2. **Static Causal Weighting** (when CAUSAL_WEIGHTING_MODE = "static"):
   - PDE residuals weighted by w(t) = exp(-gamma * t)
   - Early times (t≈0): w(t) ≈ 1.0 (full weight)
   - Later times (t≈tmax): w(t) ≈ exp(-gamma * tmax) (reduced weight)
   - Gamma decays across windows to gradually equalize weights
   - Simple, predictable, easy to tune

3. **Adaptive Causal Weighting** (when CAUSAL_WEIGHTING_MODE = "adaptive"):
   - PDE residuals weighted by w_i = exp(-epsilon * Σ_{k=1}^{i-1} L_r(t_k, θ))
   - Time domain divided into bins, residuals tracked per bin
   - Weight for time bin i depends on cumulative residuals from earlier bins
   - Automatically shifts focus forward as earlier times converge
   - Progress-aware: only emphasizes later times after earlier times are well-learned
   - More sophisticated, self-adapting, better for difficult problems

4. **Combined Benefits**:
   - Temporal curriculum provides stable progression
   - Adaptive weighting ensures proper causality enforcement
   - Together they provide robust training for long-time evolution

### Usage Examples

**Adaptive Mode with Curriculum (Recommended for tmax=2.0)**:
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = True
CAUSAL_NUM_WINDOWS = 8
CAUSAL_EPSILON = 0.5
CAUSAL_NUM_TIME_BINS = 10
CAUSAL_ADAM_PER_WINDOW = None  # Auto-split
CAUSAL_LBFGS_PER_WINDOW = None  # Auto-split
```

**Adaptive Mode without Curriculum** (for well-behaved problems):
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = False  # Train on full domain
CAUSAL_EPSILON = 0.3  # Lower epsilon for stability
CAUSAL_NUM_TIME_BINS = 10
```

**Static Mode with Curriculum** (simpler alternative):
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "static"
USE_CAUSAL_CURRICULUM = True
CAUSAL_NUM_WINDOWS = 5
CAUSAL_GAMMA_MAX = 2.0
CAUSAL_GAMMA_MIN = 0.0
```

**Disable Causal Training**:
```python
USE_CAUSAL_TRAINING = False  # Uses original training method
```

### When to Use Causal Training

**Recommended for**:
- Long-time evolution problems (tmax > 0.3)
- Problems with temporal instabilities
- Gravitational collapse simulations
- Any case where early-time accuracy is critical

**May not help**:
- Very short-time problems (tmax < 0.1)
- Problems already well-solved by standard training
- When computational cost is primary concern (adds ~20% overhead)

### Console Output

**Adaptive Mode with Curriculum**:
```
Using causal training with 8 temporal windows...
  Weighting mode: Adaptive (epsilon: 0.5, time bins: 10)

=== Causal Window 1/8 ===
  Time range: [0.010, 0.260]
  Iterations: 100 Adam + 20 LBFGS
Training Loss at 0 for Adam (batched) in 2D system = 1.23e-01
...

=== Causal Window 2/8 ===
  Time range: [0.010, 0.510]
  Iterations: 100 Adam + 20 LBFGS
...
```

**Static Mode with Curriculum**:
```
Using causal training with 5 temporal windows...
  Weighting mode: Static (gamma range: [2.0, 0.0])

=== Causal Window 1/5 ===
  Time range: [0.010, 0.108]
  Causal gamma: 2.000
  Iterations: 160 Adam + 32 LBFGS
...
```

**Adaptive Mode without Curriculum**:
```
Using causal training (no curriculum, full domain)...
  Weighting mode: Adaptive (epsilon: 0.3, time bins: 10)
Training Loss at 0 for Adam (batched) in 2D system = 1.45e-01
...
```