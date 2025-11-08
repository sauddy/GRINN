import numpy as np
import time
import torch
import torch.nn as nn
from solver import input_taker, req_consts_calc, closure, train
from config import xmin, ymin, tmin, iteration_adam_1D, iteration_lbgfs_1D, rho_o
from losses import ASTPN
from model_architecture import PINN

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() else "cuda:0" if torch.cuda.is_available() else "cpu"

lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r = input_taker(7.0, 0.03, 2, 6.0, 2000, 2000, 20000)

jeans, alpha = req_consts_calc(lam, rho_1)
v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam))

xmax = xmin+lam*num_of_waves
ymax= ymin+lam*num_of_waves

net = PINN()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr=0.001,)
optimizerL = torch.optim.LBFGS(net.parameters(),line_search_fn='strong_wolfe')

model_1D = ASTPN(rmin=[xmin, tmin],rmax=[xmax, tmax], N_0= N_0,N_b=N_b,N_r= N_r,dimension=1)

collocation_domain_1D = model_1D.geo_time_coord(option= "Domain") 
collocation_IC_1D = model_1D.geo_time_coord(option= "IC")

start_time = time.time()
train(
    net=net,
    model=model_1D,
    collocation_domain=collocation_domain_1D,
    collocation_IC=collocation_IC_1D,
    optimizer=optimizer,
    optimizerL=optimizerL,
    closure=closure,
    mse_cost_function=mse_cost_function,
    iteration_adam=iteration_adam_1D,
    iterationL=iteration_lbgfs_1D,
    rho_1=rho_1,
    lam=lam,
    jeans=jeans,
    v_1=v_1,
    device=device
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")