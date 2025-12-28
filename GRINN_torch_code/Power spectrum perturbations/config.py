import numpy as np

# Random seed for reproducibility across all functions
RANDOM_SEED = 93

# Perturbation selection: "power_spectrum" or "sinusoidal"
PERTURBATION_TYPE = "power_spectrum"

xmin = 0.0
ymin = 0.0
zmin = 0.0
cs = 1.0
rho_o = 1.0
const = 1.0
G = 1.0

# Collocation point parameters
N_0 = 70000  # Number of initial condition points
N_r = 70000 # Number of residual/collocation points
DIMENSION = 2  # Spatial dimension

# Number of collocation/IC points per mini-batch and
# how many such mini-batches to aggregate in a single optimizer step
BATCH_SIZE = (N_0 + N_r) if N_0 + N_r < 70000 else 70000
NUM_BATCHES = 1

a = 0.1

tmin = 0.
tmax = 3.0

num_neurons = 64
harmonics = 3
num_layers = 5

wave = 7.0
k = 2 * np.pi / wave
num_of_waves = 2.0

iteration_adam_2D = 1001
iteration_lbgfs_2D = 201

IC_WEIGHT = 1.0

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = False

# Directory to save snapshots; default keeps Kaggle working dir
SNAPSHOT_DIR = "/kaggle/working/"

# Training diagnostics
ENABLE_TRAINING_DIAGNOSTICS = True  # Enable automatic training diagnostics plots and logging

# Density growth comparison plot controls
PLOT_DENSITY_GROWTH = True
GROWTH_PLOT_TMAX = 4.0
GROWTH_PLOT_DT = 0.1

KX = k
KY = 0
KZ = 0
TIMES_1D = [2.5, 5.0, 7.5] # 1D cross-section times to plot (used for sinusoidal panel plots
FD_N_1D = 300  # Grid points for 1D LAX (when used)
FD_N_2D = 300  # Grid points per dimension for 2D LAX
FD_N_3D = 300   # Grid points per dimension for 3D LAX slices
SLICE_Y = 0.6  # Default y slice for visualization/cross-sections
SLICE_Z = 0.6  # Default z slice for visualization/cross-sections
SHOW_LINEAR_THEORY = False

# Power spectrum parameters
N_GRID = 400  # Grid resolution for power spectrum generation
POWER_EXPONENT = -4  # Power spectrum exponent
STARTUP_DT = 0.01 # Time offset after which PDE is enforced (ICs remain at t=0)
USE_PARAMETERIZATION = "none"  # Options: "exponential", "linear", "none"
# - "exponential": ρ = ρ₀ * exp(t_eff * ρ̂) - strictly positive
# - "linear": ρ = ρ₀ + t_eff * ρ̂ - can go negative
# - "none": ρ = ρ̂ directly for t ≥ STARTUP_DT, ρ = ρ₀ for t < STARTUP_DT

# ==================== Causal Training Configuration ====================
USE_CAUSAL_TRAINING = False

CAUSAL_WEIGHTING_MODE = "static"  # "static" or "adaptive"

USE_CAUSAL_CURRICULUM = True    # Enable temporal curriculum windows
CAUSAL_NUM_WINDOWS = 2          # Number of progressive time windows
CAUSAL_WINDOW_SCHEDULE = "custom"  # Time window schedule type: "linear" (automatic) or "custom" (user-specified)
CAUSAL_USE_RESTARTS = False      # Use restart marching [t_k, t_{k+1}] instead of expanding windows [0, t_k]
CAUSAL_CUSTOM_WINDOWS = [[0.0, 6.0], [0.0, 8.0]]  # List of [t_min, t_max] pairs for each window

CAUSAL_GAMMA_MAX = 0.0          # Maximum gamma for exp(-gamma*t) weighting (applied in early windows)
CAUSAL_GAMMA_MIN = 0.0          # Minimum gamma (applied in final window, 0 = uniform weighting)

# Adaptive Weighting Settings (only used when CAUSAL_WEIGHTING_MODE = "adaptive")
CAUSAL_EPSILON = 0.0           # Causality parameter epsilon for adaptive weighting
CAUSAL_EPSILON_FLOOR = 0.0    # Minimum epsilon value (prevents weights from becoming flat)

# Epsilon Annealing Strategy (from "Respecting Causality is all you need for PINNs")
USE_EPSILON_ANNEALING = False  # Use automatic epsilon interpolation instead of linear decay

# Automatic Epsilon Interpolation Settings
CAUSAL_EPSILON_MIN = 0.1     # Starting epsilon value (small, easy optimization)
CAUSAL_EPSILON_MAX = 1.0     # Final epsilon value (large, strong causality enforcement)

CAUSAL_NUM_TIME_BINS = 20    # Number of time bins for tracking residuals within each window

# Iterations per Window (if None, splits total iterations equally)
CAUSAL_ADAM_PER_WINDOW = None   # Adam iterations per window (None = auto-split iteration_adam_2D)
CAUSAL_LBFGS_PER_WINDOW = None  # LBFGS iterations per window (None = auto-split iteration_lbgfs_2D)

# ==================== XPINN Domain Decomposition Configuration ====================
USE_XPINN = False  # Toggle XPINN on/off (False = original single PINN)
NUM_SUBDOMAINS_X = 2  # Subdomain splits in x-direction (1 = no split)
NUM_SUBDOMAINS_Y = 2  # Subdomain splits in y-direction (1 = no split)

N_INTERFACE = 50  # Interface collocation points per interface
N_r_PER_SUBDOMAIN = None  # Residual points per subdomain (None = auto-distribute N_r)
N_0_PER_SUBDOMAIN = None  # IC points per subdomain (None = auto-distribute N_0)

INTERFACE_SOLUTION_WEIGHT = 0.1  # Solution continuity weight
INTERFACE_RESIDUAL_WEIGHT = 0.1  # Residual continuity weight
INTERFACE_SOLUTION_COMPONENTS = ['rho', 'vx', 'vy', 'phi']  # Which components to enforce

SUBDOMAIN_CONFIGS = [
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
    {'num_neurons': 64, 'num_layers': 5, 'n_harmonics': 3, 'activation': 'sin'},
]
# Set to None to use global defaults for all subdomains

DEFAULT_ACTIVATION = 'sin'  # Default activation (if not specified in SUBDOMAIN_CONFIGS)

XPINN_OPTIMIZER_STRATEGY = 'unified'  # 'unified' = single optimizer, 'separate' = one per subdomain
XPINN_ALTERNATING_TRAINING = False  # True = alternate subdomains, False = simultaneous
USE_XPINN_BATCHING = False  # True = use mini-batch processing for XPINN (reduces GPU memory)
USE_MULTI_GPU = True  # True = distribute subdomains across available GPUs
CACHE_IC_VALUES = True  # True = precompute and cache IC values for faster training
# Visualization
SHOW_INTERFACE_LINES = True  # Draw subdomain boundaries in plots
INTERFACE_AVERAGING = 'mean'  # Combine overlapping predictions: 'mean', 'weighted', 'subdomain1', 'subdomain2'