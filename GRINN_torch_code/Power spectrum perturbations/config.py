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
tmax = 4.0

num_neurons = 64
harmonics = 3
num_layers = 5

wave = 7.0
k = 2 * np.pi / wave

iteration_adam_2D = 1212
iteration_lbgfs_2D = 252

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = False

# Directory to save snapshots; default keeps Kaggle working dir
SNAPSHOT_DIR = "/kaggle/working/"

KX = k
KY = 0
TIMES_1D = [1.0, 2.0, 3.0] # 1D cross-section times to plot (used for sinusoidal panel plots
FD_N_1D = 1000  # Grid points for 1D LAX (when used)
FD_N_2D = 1000  # Grid points per dimension for 2D LAX

# Power spectrum parameters
N_GRID = 300  # Grid resolution for power spectrum generation
POWER_EXPONENT = -4  # Power spectrum exponent
FILTER_SCALE = 0  # Filter scale (Rf)
STARTUP_DT = 0.01 # Time offset after which PDE is enforced (ICs remain at t=0)
CONTINUITY_IC_WEIGHT = 0.0 # Weight for enforcing continuity at t=0: rho_t(0) = -rho0 * div v0
DECAY_PORTION = 0.5 # Fraction of total training steps over which to fully decay

# ==================== Causal Training Configuration ====================
# Enable causal training with temporal curriculum + time-weighted residuals
USE_CAUSAL_TRAINING = True

# Causal Weighting Mode
CAUSAL_WEIGHTING_MODE = "adaptive"  # "static" or "adaptive"
# - "static": Simple exp(-gamma*t) weighting
# - "adaptive": Residual-based weighting w_i = exp(-epsilon * Σ L_r(t_k)) (paper's full method)

# Temporal Curriculum Settings
USE_CAUSAL_CURRICULUM = True    # Enable temporal curriculum windows (recommended with adaptive, optional otherwise)
CAUSAL_NUM_WINDOWS = 12          # Number of progressive time windows
CAUSAL_WINDOW_SCHEDULE = "linear"  # Time window schedule type (currently only "linear" supported)

# Static Weighting Settings (only used when CAUSAL_WEIGHTING_MODE = "static")
CAUSAL_GAMMA_MAX = 1.5          # Maximum gamma for exp(-gamma*t) weighting (applied in early windows)
CAUSAL_GAMMA_MIN = 0.0          # Minimum gamma (applied in final window, 0 = uniform weighting)
# Gamma decays linearly across windows: gamma_k = GAMMA_MAX * (1 - k / NUM_WINDOWS)

# Adaptive Weighting Settings (only used when CAUSAL_WEIGHTING_MODE = "adaptive")
CAUSAL_EPSILON = 0.8           # Causality parameter epsilon for adaptive weighting
# Controls how strongly past residuals suppress future time weights
# Recommended range: 0.1-2.0 (higher = stronger suppression, lower for longer tmax)
CAUSAL_NUM_TIME_BINS = 25       # Number of time bins for tracking residuals within each window
# More bins = finer temporal resolution for adaptive weights

# Iterations per Window (if None, splits total iterations equally)
CAUSAL_ADAM_PER_WINDOW = None   # Adam iterations per window (None = auto-split iteration_adam_2D)
CAUSAL_LBFGS_PER_WINDOW = None  # LBFGS iterations per window (None = auto-split iteration_lbgfs_2D)

# Density growth comparison plot controls
# Plot PINN vs LAX density growth (max density over time)
PLOT_DENSITY_GROWTH = True
GROWTH_PLOT_TMAX = 5.0
GROWTH_PLOT_DT = 0.1

# ==================== XPINN Domain Decomposition Configuration ====================

# Domain Decomposition
USE_XPINN = False  # Toggle XPINN on/off (False = original single PINN)
NUM_SUBDOMAINS_X = 2  # Subdomain splits in x-direction (1 = no split)
NUM_SUBDOMAINS_Y = 2  # Subdomain splits in y-direction (1 = no split)

# Collocation Points
N_INTERFACE = 400  # Interface collocation points per interface
N_r_PER_SUBDOMAIN = None  # Residual points per subdomain (None = auto-distribute N_r)
N_0_PER_SUBDOMAIN = None  # IC points per subdomain (None = auto-distribute N_0)

# Interface Loss Weights
INTERFACE_SOLUTION_WEIGHT = 10.0  # Solution continuity weight
INTERFACE_RESIDUAL_WEIGHT = 1.0  # Residual continuity weight
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