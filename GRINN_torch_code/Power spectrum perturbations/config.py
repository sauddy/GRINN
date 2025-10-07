# Perturbation selection: "power_spectrum" or "sinusoidal"
PERTURBATION_TYPE = "sinusoidal"

xmin = 0.
ymin = 0.

cs = 1.0
rho_o = 1.0
const = 1
G = 1
a = 0.3
wave = 7.0

tmin = 0.
tmax = 1.5

num_neurons = 64
harmonics = 3
num_layers = 4

# Time offset after which PDE is enforced (ICs remain at t=0)
STARTUP_DT = 0.01

# Weight for enforcing continuity at t=0: rho_t(0) = -rho0 * div v0
CONTINUITY_IC_WEIGHT = 1.0

# Fraction of total training steps over which to fully decay
DECAY_PORTION = 0.5

iteration_adam_2D = 800
iteration_lbgfs_2D = 200

# Power spectrum parameters
N_GRID = 1000  # Grid resolution for power spectrum generation
POWER_EXPONENT = 0  # Power spectrum exponent
FILTER_SCALE = 0  # Filter scale (Rf)

# 1D cross-section times to plot (used for sinusoidal panel plots)
TIMES_1D = [0.5, 1.0, 1.5]

# Output/snapshot controls
SAVE_STATIC_SNAPSHOTS = True

# Directory to save snapshots; default keeps Kaggle working dir
SNAPSHOT_DIR = "/kaggle/working/"