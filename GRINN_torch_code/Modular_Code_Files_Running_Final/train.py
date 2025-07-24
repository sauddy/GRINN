import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#torch.cuda.empty_cache()
import time

from data_generator import alpha_generator
from solver import input_taker, req_consts_calc, closure_batched, train_batched
from config import xmin, ymin, zmin, tmin, iteration_adam_1D, iteration_lbgfs_1D
from config import iteration_adam_2D, iteration_lbgfs_2D, iteration_adam_3D, iteration_lbgfs_3D
from config import rho_o
from losses import ASTPN
from model_architecture import PINN
from visualisation import plot_function, rel_misfit
#from LAX import fft_solver, lax_solution1D

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

alpha_min, alpha_max, alpha_list = alpha_generator(alpha_min = 0.01, alpha_max = 0.08, N = 30)
alpha_list = alpha_list.to(device)

lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r = input_taker(7.0, 0.03, 2, 1.5, 2000, 2000, 20000)

jeans, alpha = req_consts_calc(lam)
#v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam))

xmax = xmin+lam*num_of_waves
ymax= ymin+lam*num_of_waves
zmax = zmin + lam*num_of_waves

net = PINN()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,)
optimizerL = torch.optim.LBFGS(net.parameters(),line_search_fn='strong_wolfe')

model_1D = ASTPN(rmin = [xmin, tmin], rmax = [xmax, tmax], 
                 N_0 = N_0, N_b = N_b, N_r = N_r, dimension = 1)

collocation_domain_1D = model_1D.geo_time_coord(option = "Domain") 
collocation_IC_1D = model_1D.geo_time_coord(option = "IC")

'''model_2D = ASTPN(rmin = [xmin, ymin, tmin], rmax = [xmax, ymax, tmax], 
                 N_0 = N_0, N_b = N_b, N_r = N_r, dimension = 2)
model_3D = ASTPN(rmin = [xmin, ymin, zmin, tmin], rmax = [xmax, ymax, zmax, tmax], 
                 N_0 = N_0, N_b = N_b, N_r = N_r, dimension = 3)

collocation_domain_2D = model_2D.geo_time_coord(option = "Domain")
collocation_IC_2D = model_2D.geo_time_coord(option = "IC")

collocation_domain_3D = model_3D.geo_time_coord(option = "Domain")
collocation_IC_3D = model_3D.geo_time_coord(option = "IC")'''

#print("alpha_list: ", alpha_list)

v_1 = ((alpha/(rho_o*2*np.pi/lam)) * alpha_list)
#print("v1: ", v_1)

start_time = time.time()
train_batched(
    net=net,
    model=model_1D,
    alpha=alpha_list,
    collocation_domain=collocation_domain_1D,
    collocation_IC=collocation_IC_1D,
    optimizer=optimizer,
    optimizerL=optimizerL,
    closure=closure_batched,
    mse_cost_function=mse_cost_function,
    iteration_adam=iteration_adam_1D,
    iterationL=iteration_lbgfs_1D,
    lam=lam,
    jeans=jeans,
    v_1=v_1,
    batch_size = 20000,
    num_batches = 20
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

plot_rho = 0.021
plot_rho_1 = 0.067

plot_v1 = ((alpha/(rho_o*2*np.pi/lam)) * plot_rho)
plot_v1_1 = ((alpha/(rho_o*2*np.pi/lam)) * plot_rho_1)

time_array_plot = np.linspace(0.5,int(tmax),int(tmax)+3)
time_array_misfit = np.array([0.5, 1.0, 1.5, 2.0, 2.5])

initial_params_1 = xmin, xmax, rho_1, plot_rho, plot_v1, jeans, lam, tmax, device
initial_params_2 = xmin, xmax, rho_1, plot_rho_1, plot_v1_1, jeans, lam, tmax, device

nu = 0.5
N = 1000

plot_function(net,time_array_plot,initial_params_1, N, velocity=True,isplot =True)
plot_function(net,time_array_plot,initial_params_2, N, velocity=True,isplot =True)

rel_misfit(net, time_array_misfit, initial_params_1, N, nu, num_of_waves, rho_1)
rel_misfit(net, time_array_misfit, initial_params_2, N, nu, num_of_waves, rho_1)

# Save the trained model
MODEL_PATH = 'Running_Final_pinn_model.pth'
torch.save(net.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")