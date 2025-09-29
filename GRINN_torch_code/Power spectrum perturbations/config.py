tmin = 0.
xmin = 0.
ymin= 0.
zmin= 0.
zmax= 1.

cs = 1.0
rho_o = 1
const = 1
G = 1
a = 0.1

# Time offset after which PDE is enforced (ICs remain at t=0)
STARTUP_DT = 0.01

# Weight for enforcing continuity at t=0: rho_t(0) = -rho0 * div v0
CONTINUITY_IC_WEIGHT = 0.01

iteration_adam_1D = 200
iteration_lbgfs_1D = 100

iteration_adam_2D = 800
iteration_lbgfs_2D = 200

iteration_adam_3D = 1000
iteration_lbgfs_3D = 400

# Power spectrum parameters
N_GRID = 1000  # Grid resolution for power spectrum generation
POWER_EXPONENT = 0  # Power spectrum exponent
FILTER_SCALE = 0  # Filter scale (Rf)