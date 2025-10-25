import numpy as np

# Random seed for reproducibility across all functions
RANDOM_SEED = 1234

# Perturbation selection: "power_spectrum" or "sinusoidal"
PERTURBATION_TYPE = "power_spectrum"

xmin = 0.
ymin = 0.
cs = 1.0
rho_o = 1.0
const = 1.0
G = 1.0

# Collocation point parameters
N_0 = 10000  # Number of initial condition points
N_r = 80000 # Number of residual/collocation points
DIMENSION = 2  # Spatial dimension

# Number of collocation/IC points per mini-batch and
# how many such mini-batches to aggregate in a single optimizer step
BATCH_SIZE = 100000
NUM_BATCHES = 1

a = 0.1

tmin = 0.
tmax = 3.0

num_neurons = 64
harmonics = 3
num_layers = 5

wave = 7.0
k = 2 * np.pi / wave

iteration_adam_2D = 1001
iteration_lbgfs_2D = 201

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = False

# Directory to save snapshots; default keeps Kaggle working dir
SNAPSHOT_DIR = "/kaggle/working/"

KX = k
KY = 0
TIMES_1D = [0.5, 1.5, 2.5] # 1D cross-section times to plot (used for sinusoidal panel plots
FD_N_1D = 1000  # Grid points for 1D LAX (when used)
FD_N_2D = 300  # Grid points per dimension for 2D LAX

# Power spectrum parameters
N_GRID = 300  # Grid resolution for power spectrum generation
POWER_EXPONENT = -4  # Power spectrum exponent
FILTER_SCALE = 0  # Filter scale (Rf)
STARTUP_DT = 0.01 # Time offset after which PDE is enforced (ICs remain at t=0)
CONTINUITY_IC_WEIGHT = 0.0 # Weight for enforcing continuity at t=0: rho_t(0) = -rho0 * div v0
DECAY_PORTION = 0.5 # Fraction of total training steps over which to fully decay

# ==================== Causal Training Configuration ====================
# Enable causal training with temporal curriculum + time-weighted residuals
USE_CAUSAL_TRAINING = False

# Causal Weighting Mode
CAUSAL_WEIGHTING_MODE = "adaptive"  # "static" or "adaptive"
# - "static": Simple exp(-gamma*t) weighting
# - "adaptive": Residual-based weighting w_i = exp(-epsilon * Σ L_r(t_k)) (paper's full method)

# Temporal Curriculum Settings
USE_CAUSAL_CURRICULUM = False    # Enable temporal curriculum windows (recommended with adaptive, optional otherwise)
CAUSAL_NUM_WINDOWS = 12          # Number of progressive time windows
CAUSAL_WINDOW_SCHEDULE = "linear"  # Time window schedule type (currently only "linear" supported)
CAUSAL_USE_RESTARTS = False      # Use restart marching [t_k, t_{k+1}] instead of expanding windows [0, t_k]
# Restart marching: train on non-overlapping time slabs with warm starts from previous slab
# More stable for stiff/long-time dynamics but doesn't reinforce early-time learning

# Static Weighting Settings (only used when CAUSAL_WEIGHTING_MODE = "static")
CAUSAL_GAMMA_MAX = 1.5          # Maximum gamma for exp(-gamma*t) weighting (applied in early windows)
CAUSAL_GAMMA_MIN = 0.0          # Minimum gamma (applied in final window, 0 = uniform weighting)
# Gamma decays linearly across windows: gamma_k = GAMMA_MAX * (1 - k / NUM_WINDOWS)

# Adaptive Weighting Settings (only used when CAUSAL_WEIGHTING_MODE = "adaptive")
CAUSAL_EPSILON = 0.8           # Causality parameter epsilon for adaptive weighting
# Controls how strongly past residuals suppress future time weights
# Recommended range: 0.1-2.0 (higher = stronger suppression, lower for longer tmax)
CAUSAL_EPSILON_FLOOR = 0.15    # Minimum epsilon value (prevents weights from becoming flat)
# Keep causality active in late windows by maintaining minimum epsilon
# Recommended range: 0.1-0.3 (0.0 = no floor, weights can become uniform)

# Epsilon Annealing Strategy (from "Respecting Causality is all you need for PINNs")
USE_EPSILON_ANNEALING = True  # Use automatic epsilon interpolation instead of linear decay
# If True, interpolates between EPSILON_MIN and EPSILON_MAX across all windows
# If False, uses linear decay from CAUSAL_EPSILON to CAUSAL_EPSILON_FLOOR

# Automatic Epsilon Interpolation Settings
CAUSAL_EPSILON_MIN = 0.1      # Starting epsilon value (small, easy optimization)
CAUSAL_EPSILON_MAX = 1.0     # Final epsilon value (large, strong causality enforcement)
# The system automatically interpolates between MIN and MAX across CAUSAL_NUM_WINDOWS
# This ensures monotonic increase: ε_k = MIN + (MAX-MIN) * (k / (NUM_WINDOWS-1))
# Paper recommendation: Start small → increase gradually to strengthen causality enforcement

CAUSAL_NUM_TIME_BINS = 20       # Number of time bins for tracking residuals within each window
# More bins = finer temporal resolution for adaptive weights

# Iterations per Window (if None, splits total iterations equally)
CAUSAL_ADAM_PER_WINDOW = None   # Adam iterations per window (None = auto-split iteration_adam_2D)
CAUSAL_LBFGS_PER_WINDOW = None  # LBFGS iterations per window (None = auto-split iteration_lbgfs_2D)

# ==================== Adaptive Collocation Allocation (Single PINN Only) ====================
# Enable adaptive redistribution of collocation points based on PDE residuals
USE_ADAPTIVE_COLLOCATION = False

# Adaptive Allocation Parameters
ADAPTIVE_COLLOCATION_FREQUENCY = 200  # Update every N Adam iterations (0 = disable)

# Threshold Strategy: "percentile" (recommended) or "absolute"
ADAPTIVE_COLLOCATION_THRESHOLD_MODE = "percentile"

# Percentile-based thresholds (simpler, automatically adapts to residual scale)
ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL = 75.0  # Initial: focus on top 25% worst residuals (lenient)
ADAPTIVE_COLLOCATION_PERCENTILE_FINAL = 50.0    # Final: focus on top 50% worst residuals (strict)
ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START = 400  # Start decaying percentile after this many iterations
ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END = 900    # Finish decaying percentile at this iteration

# Absolute thresholds (only used if THRESHOLD_MODE = "absolute")
ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE = 0.01        # Initial absolute threshold
ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE_FINAL = 0.001 # Final absolute threshold

ADAPTIVE_COLLOCATION_RATIO_MODE = "adaptive"  # "fixed" or "adaptive" point redistribution
ADAPTIVE_COLLOCATION_FIXED_RATIO = 0.3       # Fraction of points to redistribute (fixed mode)
ADAPTIVE_COLLOCATION_MIN_POINTS = 300         # Minimum points per region
ADAPTIVE_COLLOCATION_MAX_POINTS = 6000       # Maximum points per region

# LBFGS Behavior
ADAPTIVE_COLLOCATION_LBFGS_MODE = "uniform"    # "uniform" or "adaptive" for LBFGS phase
# If "uniform": redistribute points uniformly before LBFGS
# If "adaptive": keep adaptive distribution during LBFGS

# Integration with Causal Training
ADAPTIVE_COLLOCATION_CAUSAL_COMPATIBLE = True  # Ensure compatibility with causal training

# Memory Management
ADAPTIVE_COLLOCATION_RESIDUAL_BATCH_SIZE = 10000  # Batch size for residual computation (reduce if OOM)

# Diagnostic Settings
CAUSAL_PRINT_DIAGNOSTICS = False  # Print detailed causal weighting diagnostics
CAUSAL_PRINT_RESIDUAL_SCALES = False  # Print detailed residual scale diagnostics (verbose)

# Adaptive Collocation Debug Settings
ADAPTIVE_COLLOCATION_VERBOSE = False  # Print detailed debug output for adaptive collocation troubleshooting

# ==================== Log-Density Transformation ====================
# Predict s = log(rho) instead of rho to handle exponential growth better
USE_LOG_DENSITY = False  # Enable log-density prediction (only for single PINN, not XPINN)
# - Transforms continuity equation to: s_t + v·∇s + ∇·v = 0 (linear in s)
# - Transforms momentum to: v_t + (v·∇)v = -cs²∇s + g (for isothermal EOS)
# - Poisson remains: ∇²φ = 4πG·exp(s)

# ==================== Spectral Poisson Consistency ====================
# Enforce global Poisson coupling using FFT-based spectral consistency loss
USE_SPECTRAL_POISSON = False  # Enable spectral Poisson consistency enforcement
# - Evaluates network on regular grid and enforces Poisson equation in Fourier space
# - Provides stronger global gravitational coupling than pointwise residuals alone
# - Helps prevent under-prediction of density growth and velocity magnitudes

# Spectral Poisson Parameters
SPECTRAL_POISSON_GRID_SIZE = 64
SPECTRAL_POISSON_WEIGHT = 5e-5
SPECTRAL_POISSON_FREQUENCY = 100
SPECTRAL_POISSON_TIMES = [0.0, 0.75, 1.5, 2.25, 3.0]
SPECTRAL_POISSON_WEIGHT_HIGH_K = True
SPECTRAL_POISSON_IN_LBFGS = False  # Disable spectral loss during LBFGS for stability

# ==================== FFT-Based Poisson Solver ====================
# Compute gravitational potential φ via FFT Poisson solve instead of learning it
USE_FFT_PHI = False  # Enable FFT-based φ computation (disables φ head loss)
# - Computes φ from ρ using differentiable FFT Poisson solver each forward pass
# - Enforces exact global Poisson coupling: ∇²φ = const·(ρ - ρ₀)
# - Eliminates φ-ρ inconsistency and improves velocity field accuracy

# FFT Poisson Parameters
FFT_GRID_SIZE = 64           # Grid resolution for FFT solve (32, 64, 96, 128)
FFT_PHI_SUPERVISE_WEIGHT = 1e-3  # Weight for optional φ head supervision (0 = disable φ loss entirely)
FFT_PHI_IN_LBFGS = False     # Enable FFT Poisson during LBFGS (memory intensive)
FFT_PHI_MEMORY_CLEANUP = True # Clear GPU cache after FFT operations

# Density growth comparison plot controls
# Plot PINN vs LAX density growth (max density over time)
PLOT_DENSITY_GROWTH = True
GROWTH_PLOT_TMAX = 4.0
GROWTH_PLOT_DT = 0.1

# ==================== XPINN Domain Decomposition Configuration ====================

# Domain Decomposition
USE_XPINN = False  # Toggle XPINN on/off (False = original single PINN)
NUM_SUBDOMAINS_X = 2  # Subdomain splits in x-direction (1 = no split)
NUM_SUBDOMAINS_Y = 2  # Subdomain splits in y-direction (1 = no split)

# Collocation Points
N_INTERFACE = 500  # Interface collocation points per interface
N_r_PER_SUBDOMAIN = None  # Residual points per subdomain (None = auto-distribute N_r)
N_0_PER_SUBDOMAIN = None  # IC points per subdomain (None = auto-distribute N_0)

# Interface Loss Weights
INTERFACE_SOLUTION_WEIGHT = 0.1  # Solution continuity weight
INTERFACE_RESIDUAL_WEIGHT = 0.1  # Residual continuity weight
INTERFACE_SOLUTION_COMPONENTS = ['rho', 'vx', 'vy', 'phi']  # Which components to enforce

# Per-Subdomain Network Architecture (see CONFIG_PARAMETERS.md for details)
SUBDOMAIN_CONFIGS = [
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
]
# Set to None to use global defaults for all subdomains

DEFAULT_ACTIVATION = 'sin'  # Default activation (if not specified in SUBDOMAIN_CONFIGS)

# Training Strategy
XPINN_OPTIMIZER_STRATEGY = 'unified'  # 'unified' = single optimizer, 'separate' = one per subdomain
XPINN_ALTERNATING_TRAINING = False  # True = alternate subdomains, False = simultaneous
USE_XPINN_BATCHING = False  # True = use mini-batch processing for XPINN (reduces GPU memory)
USE_MULTI_GPU = True  # True = distribute subdomains across available GPUs
CACHE_IC_VALUES = True  # True = precompute and cache IC values for faster training

# Visualization
SHOW_INTERFACE_LINES = True  # Draw subdomain boundaries in plots
INTERFACE_AVERAGING = 'mean'  # Combine overlapping predictions: 'mean', 'weighted', 'subdomain1', 'subdomain2'