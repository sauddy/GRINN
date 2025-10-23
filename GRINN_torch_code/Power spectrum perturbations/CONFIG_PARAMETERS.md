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

- `CAUSAL_USE_RESTARTS`: Use restart marching instead of expanding windows (bool)
  - **False** (default): Expanding windows - train on [0, t₁], [0, t₂], ..., [0, tₘₐₓ]
    - Network repeatedly trains on early times, reinforcing early-time accuracy
    - Better for problems where early-time consistency is critical
  - **True**: Restart marching - train on [t₀, t₁], [t₁, t₂], ..., [tₙ₋₁, tₘₐₓ]
    - Network focuses on fresh time slabs with warm starts from previous window
    - More stable for stiff/long-time dynamics (recommended for rapid growth)
    - Automatically uses local bins (tracker reinitialized per window)
    - Each window has internal early→late resolution
  - **Default**: False
  - **Recommendation**: Use True for stiff PDEs with rapid late-time growth (a=0.1, tmax≥3.0)

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
  - **Default**: 0.8
  - **Tuning**: Lower for longer tmax (0.1-0.3 for tmax=2.0, 0.5-1.0 for tmax=0.5)

- `CAUSAL_EPSILON_FLOOR`: Minimum epsilon value to prevent flat weights (float)
  - **Purpose**: Prevents epsilon from going to 0, maintaining causality in late windows (only used if USE_EPSILON_ANNEALING = False)
  - **Range**: 0.0-0.3 (0.0 = no floor, weights can become uniform)
  - **Default**: 0.15
  - **Effect**: Keeps causal weighting active throughout training instead of reverting to uniform
  - **Higher values**: Stronger causality maintained in final windows
  - **Lower values**: Allows weights to become more uniform in late training
  - **0.0**: Original behavior (epsilon → 0, weights become flat in final windows)

- `USE_EPSILON_ANNEALING`: Enable automatic epsilon interpolation (bool)
  - **Purpose**: Use automatic interpolation between MIN and MAX epsilon values
  - **True**: Use CAUSAL_EPSILON_MIN and CAUSAL_EPSILON_MAX with automatic interpolation (recommended)
  - **False**: Use linear decay from CAUSAL_EPSILON to CAUSAL_EPSILON_FLOOR
  - **Default**: True
  - **Paper recommendation**: Avoids hyper-parameter tuning by using automatic interpolation
  - **Theory**: Small ε (early) = easy optimization, Large ε (later) = strong causality enforcement

- `CAUSAL_EPSILON_MIN`: Starting epsilon value for interpolation (float)
  - **Purpose**: Small epsilon value for early training windows (easy optimization)
  - **Default**: 0.5
  - **Range**: 0.1-2.0 recommended
  - **Effect**: Lower values = easier early optimization, higher values = stronger early causality

- `CAUSAL_EPSILON_MAX`: Final epsilon value for interpolation (float)
  - **Purpose**: Large epsilon value for final training windows (strong causality enforcement)
  - **Default**: 10.0
  - **Range**: 5.0-50.0 recommended
  - **Effect**: Higher values = stronger final causality enforcement
  - **Formula**: ε_k = MIN + (MAX-MIN) * (k / (NUM_WINDOWS-1))

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

### Diagnostic Settings

- `CAUSAL_PRINT_DIAGNOSTICS`: Enable detailed causal weighting diagnostics (bool)
  - **True**: Print per-bin residual tracking, effective weights, and attention analysis
  - **False**: Disable detailed diagnostics (only basic loss info)
  - **Default**: True
  - **Output**: Shows bin-by-bin residual averages, update counts, effective weights, and attention distribution
  - **Use**: Helps debug causal weighting effectiveness and identify flat weight issues

- `CAUSAL_PRINT_RESIDUAL_SCALES`: Enable verbose residual scale diagnostics (bool)
  - **True**: Print detailed residual statistics (mean, std, max) for each PDE term
  - **False**: Disable verbose residual diagnostics
  - **Default**: False (very verbose)
  - **Output**: Shows residual scales, causal weight statistics, and time-weighted residuals
  - **Use**: Helps identify scale imbalances between PDE terms and gradient issues

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

### Interpreting Diagnostic Output

When `CAUSAL_PRINT_DIAGNOSTICS = True`, you'll see output like:

```
=== Causal Weighting Diagnostics (Step 200) ===
Window: 2, Epsilon: 0.400
Bin | Time Range    | Avg Residual | Count | Effective Weight | Attention
----|---------------|--------------|-------|-----------------|----------
  0 |  0.010- 0.259 |     2.34e-03 |   156 |           1.000 | ACTIVE
  1 |  0.259- 0.508 |     3.45e-03 |   142 |           0.847 | ACTIVE
  2 |  0.508- 0.757 |     4.12e-03 |   138 |           0.712 | ACTIVE
  3 |  0.757- 1.006 |     5.67e-03 |   134 |           0.598 | ACTIVE
  4 |  1.006- 1.255 |     7.23e-03 |   130 |           0.501 | ACTIVE

Total Attention: 245.3
Weight Range: [0.501, 1.000]
Weight Std: 0.199
```

**What to look for:**
- **Effective Weight**: Should decrease with time (early bins > later bins)
- **Weight Std**: Should be > 0.1 (if < 0.01, weights are nearly flat - problematic!)
- **Attention**: Should be highest for early bins initially, then migrate forward
- **Avg Residual**: Should decrease over training (convergence indicator)
- **Count**: Should be roughly uniform across bins (sampling balance)

**Warning signs:**
- ⚠️ "Weights are nearly flat" → Increase `CAUSAL_EPSILON` or check residual scales
- ⚠️ "Weights have low variance" → Consider adjusting epsilon or time bin count
- All bins showing "INACTIVE" → Check time range and collocation point generation

## Log-Density Transformation

- `USE_LOG_DENSITY`: Enable log-density prediction (bool)
  - **False** (default): Network predicts ρ directly
  - **True**: Network predicts s = log(ρ), PDEs are transformed accordingly
  - **Purpose**: Better numerical conditioning for exponential density growth
  - **How it works**:
    - Network outputs: s (log-density), vx, vy, φ instead of ρ, vx, vy, φ
    - Continuity PDE becomes: s_t + v·∇s + ∇·v = 0 (linear in s)
    - Momentum PDE becomes: v_t + (v·∇)v = -cs²∇s - ∇φ (for isothermal EOS)
    - Poisson equation: ∇²φ = const·(exp(s) - ρ₀)
    - Initial conditions: Network trained on s₀ = log(ρ₀)
  - **Benefits**:
    - Exponential growth (ρ: 1→100) becomes linear growth (s: 0→4.6)
    - Better gradient scaling and optimizer stability
    - Automatic positivity (ρ = exp(s) > 0 always)
  - **When to use**: Stiff problems with rapid exponential growth (a=0.1, tmax≥3.0)
  - **Note**: Only implemented for single PINN, not XPINN
  - **Plotting**: Network outputs s; apply exp(s) to recover ρ for visualization

### Usage Examples

**Automatic Epsilon Interpolation (Recommended - from Paper)**:
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = True
CAUSAL_NUM_WINDOWS = 12
USE_EPSILON_ANNEALING = True  # Use automatic interpolation
CAUSAL_EPSILON_MIN = 0.5      # Starting value (easy optimization)
CAUSAL_EPSILON_MAX = 10.0     # Final value (strong causality)
CAUSAL_NUM_TIME_BINS = 10
CAUSAL_ADAM_PER_WINDOW = None  # Auto-split
CAUSAL_LBFGS_PER_WINDOW = None  # Auto-split
```
*This automatically interpolates: ε = [0.5, 1.3, 2.1, 2.9, 3.7, 4.5, 5.3, 6.1, 6.9, 7.7, 8.5, 10.0]*

**Conservative Epsilon Range (for difficult/stiff problems)**:
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = True
CAUSAL_NUM_WINDOWS = 10
USE_EPSILON_ANNEALING = True
CAUSAL_EPSILON_MIN = 0.1      # Very gentle start
CAUSAL_EPSILON_MAX = 5.0      # Moderate final strength
CAUSAL_NUM_TIME_BINS = 10
```

**Aggressive Epsilon Range (for easier problems)**:
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = True
CAUSAL_NUM_WINDOWS = 8
USE_EPSILON_ANNEALING = True
CAUSAL_EPSILON_MIN = 1.0      # Strong start
CAUSAL_EPSILON_MAX = 20.0     # Very strong final
CAUSAL_NUM_TIME_BINS = 10
```

**Adaptive Mode with Linear Decay (Original Method)**:
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = True
CAUSAL_NUM_WINDOWS = 8
USE_EPSILON_ANNEALING = False  # Use linear decay instead
CAUSAL_EPSILON = 0.8  # Starting value
CAUSAL_EPSILON_FLOOR = 0.15  # Ending value
CAUSAL_NUM_TIME_BINS = 10
```

**Adaptive Mode without Curriculum** (for well-behaved problems):
```python
USE_CAUSAL_TRAINING = True
CAUSAL_WEIGHTING_MODE = "adaptive"
USE_CAUSAL_CURRICULUM = False  # Train on full domain
USE_EPSILON_ANNEALING = False
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

## **Adaptive Collocation Allocation (Single PINN Only)**

This section describes the adaptive collocation allocation system that automatically redistributes collocation points based on PDE residuals to focus computational resources on high-error regions.

### **Core Concept**

Adaptive collocation allocation monitors PDE residuals during training and redistributes collocation points to focus on regions where the PINN is struggling to learn the physics. This is particularly useful for problems with sharp gradients, shocks, or complex multi-scale structures.

### **Configuration Parameters**

- `USE_ADAPTIVE_COLLOCATION`: Enable adaptive collocation allocation (bool)
  - **Purpose**: Master switch for adaptive collocation system
  - **True**: Enable adaptive point redistribution
  - **False**: Use standard uniform collocation points
  - **Default**: True
  - **Note**: Only works with single PINN, not XPINN

- `ADAPTIVE_FD_SPATIAL_GRID`: Spatial grid resolution for FD reference (int)
  - **Purpose**: Grid resolution for generating FD reference solution
  - **Default**: N_GRID (300)
  - **Usage**: Reuses existing grid resolution from power spectrum generation
  - **Effect**: Higher resolution = more accurate FD reference, but slower generation

- `ADAPTIVE_FD_TIME_STEP`: Time step for FD reference solution (float)
  - **Purpose**: Time step for generating FD reference solution
  - **Default**: 0.1 (same as GROWTH_PLOT_DT)
  - **Range**: 0.05-0.2 recommended
  - **Effect**: Smaller step = finer temporal resolution, but slower generation
  - **Note**: Number of time points is automatically calculated as int((tmax-tmin)/step) + 1

- `ADAPTIVE_COLLOCATION_FREQUENCY`: Update frequency for point redistribution (int)
  - **Purpose**: How often to redistribute collocation points
  - **Default**: 1000 (every 1000 Adam iterations)
  - **Range**: 500-2000 recommended
  - **Effect**: Higher frequency = more responsive, but more computational overhead
  - **0**: Disable adaptive updates

- `ADAPTIVE_COLLOCATION_THRESHOLD_MODE`: Threshold strategy (str)
  - **Purpose**: How to determine which points are "high error"
  - **Options**: "percentile" (recommended) or "absolute"
  - **Default**: "percentile"
  - **Percentile mode**: Automatically adapts to residual scale (e.g., "top 25% worst residuals")
  - **Absolute mode**: Uses fixed threshold values (requires manual tuning)
  - **Effect**: Percentile mode is simpler and more robust across different problems

### Percentile-Based Thresholds (Recommended)

- `ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL`: Initial percentile for high-error identification (float)
  - **Purpose**: Starting percentile threshold (lenient, focuses on worst residuals)
  - **Default**: 75.0 (identifies top 25% worst residuals as "high error")
  - **Range**: 50.0-95.0 recommended
  - **Guidance**:
    - **90.0-95.0**: Very lenient (top 5-10% worst residuals)
    - **75.0-85.0**: Lenient (top 15-25% worst residuals) - **RECOMMENDED START**
    - **60.0-70.0**: Moderate (top 30-40% worst residuals)
    - **50.0-60.0**: Strict (top 40-50% worst residuals)
  - **Effect**: Higher percentile = more focused on extreme errors only

- `ADAPTIVE_COLLOCATION_PERCENTILE_FINAL`: Final percentile for high-error identification (float)
  - **Purpose**: Final percentile after progressive decay (strict, broader focus)
  - **Default**: 50.0 (identifies top 50% worst residuals as "high error")
  - **Range**: 30.0-70.0 recommended
  - **Guidance**: Should be lower than initial to progressively broaden focus
  - **Effect**: Lower percentile = more points identified as high-error in later training

- `ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START`: Iteration to start percentile decay (int)
  - **Purpose**: When to start making identification stricter
  - **Default**: 400 (after basic physics learned)
  - **Range**: 200-600 recommended
  - **Effect**: Earlier start = faster transition to broader focus

- `ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END`: Iteration to finish percentile decay (int)
  - **Purpose**: When percentile reaches final value
  - **Default**: 900 (gradual transition, completes before LBFGS)
  - **Range**: 800-1200 recommended
  - **Effect**: Longer decay = smoother transition

### Absolute Thresholds (Advanced Users Only)

- `ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE`: Initial absolute PDE residual threshold (float)
  - **Purpose**: Starting threshold for identifying high-error regions (only used if MODE = "absolute")
  - **Default**: 0.01
  - **Units**: PDE residual magnitude (L2 norm of continuity + momentum equations)
  - **Note**: Requires problem-specific tuning; percentile mode is easier

- `ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE_FINAL`: Final absolute PDE residual threshold (float)
  - **Purpose**: Final threshold after progressive decay
  - **Default**: 0.001
  - **Note**: Should be lower than initial for progressive refinement

- `ADAPTIVE_COLLOCATION_RATIO_MODE`: Point redistribution strategy (str)
  - **Purpose**: How to determine how many points to redistribute
  - **Options**: "fixed" or "adaptive"
  - **Default**: "adaptive"
  - **"fixed"**: Use ADAPTIVE_COLLOCATION_FIXED_RATIO
  - **"adaptive"**: Scale redistribution based on error distribution

- `ADAPTIVE_COLLOCATION_FIXED_RATIO`: Fixed fraction of points to redistribute (float)
  - **Purpose**: Fraction of points to redistribute in fixed mode
  - **Default**: 0.3 (30%)
  - **Range**: 0.1-0.5 recommended
  - **Effect**: Higher ratio = more aggressive redistribution

- `ADAPTIVE_COLLOCATION_MIN_POINTS`: Minimum points per region (int)
  - **Purpose**: Prevent regions from being completely depopulated
  - **Default**: 100
  - **Range**: 50-200 recommended
  - **Effect**: Ensures minimum coverage of all regions

- `ADAPTIVE_COLLOCATION_MAX_POINTS`: Maximum points per region (int)
  - **Purpose**: Prevent over-concentration in single regions
  - **Default**: 5000
  - **Range**: 1000-10000 recommended
  - **Effect**: Prevents excessive point concentration

- `ADAPTIVE_COLLOCATION_LBFGS_MODE`: LBFGS phase behavior (str)
  - **Purpose**: How to handle collocation points during LBFGS phase
  - **Options**: "uniform" or "adaptive"
  - **Default**: "uniform"
  - **"uniform"**: Redistribute points uniformly before LBFGS
  - **"adaptive"**: Keep adaptive distribution during LBFGS

- `ADAPTIVE_COLLOCATION_CAUSAL_COMPATIBLE`: Causal training compatibility (bool)
  - **Purpose**: Ensure compatibility with causal training
  - **Default**: True
  - **Effect**: Filters adaptive points by time windows in causal training

- `ADAPTIVE_COLLOCATION_RESIDUAL_BATCH_SIZE`: Batch size for residual computation (int)
  - **Purpose**: Control memory usage during residual computation
  - **Default**: 10000
  - **Range**: 1000-20000 recommended
  - **Effect**: Smaller batch = less memory, larger batch = faster computation
  - **Memory Management**: Reduces GPU memory usage from ~24GB to manageable levels

### **Debug Settings**

- `ADAPTIVE_COLLOCATION_VERBOSE`: Enable detailed debug output for adaptive collocation troubleshooting (bool)
  - **Purpose**: Control verbose debug output for adaptive collocation system
  - **True**: Print detailed gradient flow, residual computation, and threshold information
  - **False**: Print only essential information (recommended for normal use)
  - **Default**: False
  - **Debug Output Includes**:
    - Gradient flow verification (requires_grad, is_leaf, torch.is_grad_enabled)
    - Network output statistics (shape, range, gradient status)
    - PDE residual statistics (min, max, mean for each equation)
    - Manual gradient computation tests
    - Batch-by-batch processing details
  - **When to Enable**: When troubleshooting gradient issues, zero residuals, or unexpected behavior
  - **Performance Impact**: Minimal (only affects print statements)

### **How It Works**

1. **FD Reference Generation**: 
   - Generates FD solution using LAX solver on coarse grid
   - Uses time step (ADAPTIVE_FD_TIME_STEP) to determine temporal resolution
   - Creates interpolation functions for all fields (ρ, vx, vy, φ)
   - Uses same grid resolution as power spectrum generation

2. **Residual Monitoring**:
   - Computes PINN-FD epsilon errors at current collocation points
   - Uses same formula as plotting: ε = 200 * |PINN-FD| / (PINN+FD)
   - Density: ε_ρ = 200 * |ρ_PINN - ρ_FD| / (ρ_PINN + ρ_FD)
   - Velocity: ε_v = 200 * |v_PINN - v_FD| / (v_PINN + v_FD + 2)
   - Combined: ε_total = 0.5*ε_ρ + 0.25*ε_vx + 0.25*ε_vy
   - Identifies high-error regions using threshold
   - Tracks epsilon error statistics over time

3. **Point Redistribution**:
   - Removes points from low-error regions
   - Adds points near existing high-error points
   - Maintains total point count
   - Applies min/max point constraints

4. **Integration with Training**:
   - Updates points every N Adam iterations
   - Prepares points for LBFGS phase
   - Compatible with causal training windows

### **Expected Benefits**

- **Better Accuracy**: Focuses computational resources on difficult regions
- **Efficient Training**: Reduces wasted computation on well-learned regions
- **Automatic Adaptation**: No manual tuning of point distribution
- **Problem-Agnostic**: Works for various types of physics problems

### **Usage Examples**

**Conservative Settings (for stable problems)**:
```python
USE_ADAPTIVE_COLLOCATION = True
ADAPTIVE_FD_TIME_STEP = 0.2  # Coarser temporal resolution
ADAPTIVE_COLLOCATION_FREQUENCY = 2000
ADAPTIVE_COLLOCATION_THRESHOLD_MODE = "percentile"
ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL = 85.0  # Top 15% worst residuals
ADAPTIVE_COLLOCATION_PERCENTILE_FINAL = 60.0    # Top 40% worst residuals
ADAPTIVE_COLLOCATION_RATIO_MODE = "fixed"
ADAPTIVE_COLLOCATION_FIXED_RATIO = 0.2
ADAPTIVE_COLLOCATION_LBFGS_MODE = "uniform"
```

**Aggressive Settings (for difficult problems)**:
```python
USE_ADAPTIVE_COLLOCATION = True
ADAPTIVE_FD_TIME_STEP = 0.05  # Finer temporal resolution
ADAPTIVE_COLLOCATION_FREQUENCY = 500
ADAPTIVE_COLLOCATION_THRESHOLD_MODE = "percentile"
ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL = 60.0  # Top 40% worst residuals
ADAPTIVE_COLLOCATION_PERCENTILE_FINAL = 30.0    # Top 70% worst residuals
ADAPTIVE_COLLOCATION_RATIO_MODE = "adaptive"
ADAPTIVE_COLLOCATION_LBFGS_MODE = "adaptive"
```

**High Accuracy Settings (for critical applications)**:
```python
USE_ADAPTIVE_COLLOCATION = True
ADAPTIVE_FD_TIME_STEP = 0.05  # Fine temporal resolution
ADAPTIVE_COLLOCATION_FREQUENCY = 1000
ADAPTIVE_COLLOCATION_THRESHOLD_MODE = "percentile"
ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL = 50.0  # Top 50% worst residuals
ADAPTIVE_COLLOCATION_PERCENTILE_FINAL = 20.0    # Top 80% worst residuals
ADAPTIVE_COLLOCATION_RATIO_MODE = "adaptive"
ADAPTIVE_COLLOCATION_LBFGS_MODE = "adaptive"
```

**Disabled (for comparison)**:
```python
USE_ADAPTIVE_COLLOCATION = False
```

### **Compatibility Notes**

- **Single PINN Only**: Not implemented for XPINN
- **Causal Training**: Compatible with causal training windows
- **Memory Usage**: FD solution generation requires additional memory
- **Computational Overhead**: Initial FD generation + periodic updates
Training Loss at 0 for Adam (batched) in 2D system = 1.45e-01
...
```