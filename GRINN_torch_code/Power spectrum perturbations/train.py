import numpy as np
import time
import torch
import torch.nn as nn
from solver import input_taker, req_consts_calc, closure, train
from config import a, cs, xmin, ymin, tmin, iteration_adam_2D, iteration_lbgfs_2D, rho_o
from losses import ASTPN
from model_architecture import PINN
from Plotting_2D import create_2d_animation

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

# Clear GPU memory if using CUDA
if device.startswith('cuda'):
    torch.cuda.empty_cache()
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"Using device: {device}")

lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r = input_taker(7.0, 0.03, 2, 2.0, 10000, 10000, 100000)

jeans, alpha = req_consts_calc(lam, rho_1)
#v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam))
v_1 = a*cs

xmax = xmin+lam*num_of_waves
ymax= ymin+lam*num_of_waves

net = PINN()
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

print("Creating density animation...")
anim_density = create_2d_animation(net, initial_params, which="density", fps=10)

print("Creating velocity animation...")
anim_velocity = create_2d_animation(net, initial_params, which="velocity", fps=10)