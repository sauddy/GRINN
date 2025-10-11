import numpy as np

# Perturbation selection: "power_spectrum" or "sinusoidal"
PERTURBATION_TYPE = "power_spectrum"

xmin = 0.
ymin = 0.
cs = 1.0
rho_o = 1.0
const = 4*np.pi      # 4π for Poisson equation ∇²φ = 4πGρ
G = 1.0/(4*np.pi)    # Gravitational Constant (paper's unit system: 4πG = 1)

a = 0.01

tmin = 0.
tmax = 2.0

num_neurons = 64
harmonics = 3
num_layers = 4

wave = 7.0
k = 2 * np.pi / wave

KX = k # x-component of wave vector (2π/λx)
KY = 0 # y-component of wave vector (2π/λy))
TIMES_1D = [1.0, 2.0, 3.0] # 1D cross-section times to plot (used for sinusoidal panel plots
FD_N_1D = 1000  # Grid points for 1D LAX (when used)
FD_N_2D = 1000  # Grid points per dimension for 2D LAX

# Power spectrum parameters
N_GRID = 500  # Grid resolution for power spectrum generation
POWER_EXPONENT = -4  # Power spectrum exponent (more interesting than 0)
FILTER_SCALE = 0  # Filter scale (Rf) - small but non-zero
STARTUP_DT = 0.01 # Time offset after which PDE is enforced (ICs remain at t=0)
CONTINUITY_IC_WEIGHT = 1.0 # Weight for enforcing continuity at t=0: rho_t(0) = -rho0 * div v0
DECAY_PORTION = 0.5 # Fraction of total training steps over which to fully decay

iteration_adam_2D = 800
iteration_lbgfs_2D = 200

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = True

# Directory to save snapshots; default keeps Kaggle working dir
SNAPSHOT_DIR = "/kaggle/working/"

# Number of collocation/IC points per mini-batch and
# how many such mini-batches to aggregate in a single optimizer step
BATCH_SIZE = 40000
NUM_BATCHES = 3