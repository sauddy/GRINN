import numpy as np

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
BATCH_SIZE = 30000
NUM_BATCHES = 3

a = 0.5

tmin = 0.
tmax = 2.0

num_neurons = 64
harmonics = 3
num_layers = 5

wave = 7.0
k = 2 * np.pi / wave

iteration_adam_2D = 800
iteration_lbgfs_2D = 200

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = True

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

# Density growth comparison plot controls
# Plot PINN vs LAX density growth (max density over time)
PLOT_DENSITY_GROWTH = True
GROWTH_PLOT_TMAX = 4.0
GROWTH_PLOT_DT = 0.1