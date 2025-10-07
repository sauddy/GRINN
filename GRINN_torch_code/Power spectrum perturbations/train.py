import numpy as np
import time
import torch
import torch.nn as nn
from solver import input_taker, req_consts_calc, closure, train
from config import a, wave, cs, xmin, ymin, tmin, tmax as TMAX_CFG, iteration_adam_2D, iteration_lbgfs_2D, harmonics, PERTURBATION_TYPE, rho_o
from losses import ASTPN
from model_architecture import PINN
from Plotting_2D import create_2d_animation
from Plotting_2D import create_1d_cross_sections_sinusoidal

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Clear GPU memory if using CUDA
if device.startswith('cuda'):
    torch.cuda.empty_cache()
else:
    pass

lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r = input_taker(wave, a, 2, TMAX_CFG, 7000, 7000, 55000)

jeans, alpha = req_consts_calc(lam, rho_1)
# Set initial velocity amplitude per perturbation type
if str(PERTURBATION_TYPE).lower() == "sinusoidal":
    k = 2*np.pi/lam
    v_1 = (rho_1 / (rho_o if rho_o != 0 else 1.0)) * (alpha / k)
else:
    v_1 = a * cs

xmax = xmin + lam * num_of_waves
ymax = ymin + lam * num_of_waves

net = PINN(n_harmonics=harmonics)
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,)
optimizerL = torch.optim.LBFGS(net.parameters(),line_search_fn='strong_wolfe')

model_2D = ASTPN(rmin=[xmin, ymin, tmin],rmax=[xmax, ymax, tmax], N_0= N_0,N_b=N_b,N_r= N_r,dimension=2)

# Set domain on the network so periodic embeddings enforce hard BCs
net.set_domain(rmin=[xmin, ymin], rmax=[xmax, ymax], dimension=2)

collocation_domain_2D = model_2D.geo_time_coord(option= "Domain") 
collocation_IC_2D = model_2D.geo_time_coord(option= "IC")

start_time = time.time()
train(
    net=net,
    model=model_2D,
    collocation_domain=collocation_domain_2D,
    collocation_IC=collocation_IC_2D,
    optimizer=optimizer,
    optimizerL=optimizerL,
    closure=closure,
    mse_cost_function=mse_cost_function,
    iteration_adam=iteration_adam_2D,
    iterationL=iteration_lbgfs_2D,
    rho_1=rho_1,
    lam=lam,
    jeans=jeans,
    v_1=v_1,
    device=device
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

# Clear GPU memory after training
if device.startswith('cuda'):
    torch.cuda.empty_cache()
    print("GPU memory cleared after training")

# Create output folder for plots
#output_folder = f"PowerSpectrum_2D_fig_{lam}_{tmax}_{num_of_waves}_{rho_1}"
#os.makedirs(output_folder, exist_ok=True)
#print(f"Created output folder: {output_folder}")

# Create animated visualization plots
initial_params = (xmin, xmax, ymin, ymax, rho_1, alpha, lam, "temp", tmax)

anim_density = create_2d_animation(net, initial_params, which="density", fps=10, verbose=False)
anim_velocity = create_2d_animation(net, initial_params, which="velocity", fps=10, verbose=False)

# If using sinusoidal perturbations, also plot 1D cross-sections at fixed y
if str(PERTURBATION_TYPE).lower() == "sinusoidal":
    # Use config.TIMES_1D when time_points is None
    create_1d_cross_sections_sinusoidal(net, initial_params, time_points=None, y_fixed=0.6, N_fd=600, nu_fd=0.5)

# Always save the trained model to SNAPSHOT_DIR/GRINN
try:
    import os
    from config import SNAPSHOT_DIR
    model_dir = os.path.join(SNAPSHOT_DIR, "GRINN")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pth")
    torch.save(net.state_dict(), model_path)
    print(f"Saved model to {model_path}")
except Exception as e:
    print(f"Warning: failed to save model: {e}")