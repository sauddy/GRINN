import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from model_architecture import PINN
from config import cs, const, G, rho_o

def input_taker(lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r):
    lam = float(lam)  # Wavelength
    rho_1 = float(rho_1)  # Amplitude of perturbation
    num_of_waves = int(num_of_waves)  # Number of waves
    tmax = float(tmax)  # Maximum time
    N_0 = int(N_0)  # Number of initial condition points
    N_b = int(N_b)  # Number of boundary condition points
    N_r = int(N_r)  # Number of collocation points
    
    return lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r

def req_consts_calc(lam, rho_1):

    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_o))

    if lam > jeans:
        alpha = np.sqrt(const*G*rho_o-cs**2*(2*np.pi/lam)**2)
    else:
        alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*rho_o)

    return jeans, alpha

def fun_rho_0(rho_1, lam, x):
    ''' Define initial condition for density Returning Eq (11a)'''
    # print('wavelength',lam)
    rho_0 = rho_o + rho_1 * torch.cos(2*np.pi*x[0]/lam)    
    return rho_0

def fun_v_0(lam, jeans, v_1, x):
    '''initial condition for velocity -- Returning Eq 11b'''
    
    if lam > jeans:
        v_0 = - v_1 * torch.sin(2*np.pi*x[0]/lam)## This is for sound wave ## refer to the paper for details
    else:
        v_0 = v_1 * torch.cos(2*np.pi*x[0]/lam)  ## This is for the gravity wave
    return v_0

def func(x):
    return x[0]*0

### (3) Training / Fitting
from tqdm import tqdm

#start = time.time()
def closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1):

    ############## Loss based on initial conditions ###############
    rho_0 = fun_rho_0(rho_1, lam, collocation_IC)
    vx_0  = fun_v_0(lam, jeans, v_1, collocation_IC)

    if model.dimension == 2:
        vy_0  = func(collocation_IC)

    elif model.dimension == 3:
        vy_0  = func(collocation_IC)
        vz_0  = func(collocation_IC)
    
    net_ic_out = net(collocation_IC)

    rho_ic_out = net_ic_out[:,0:1]
    vx_ic_out  = net_ic_out[:,1:2]

    if model.dimension == 2:
        vy_ic_out  = net_ic_out[:,2:3]
    elif model.dimension == 3:
        vy_ic_out  = net_ic_out[:,2:3]
        vz_ic_out  = net_ic_out[:,3:4]

    mse_rho_ic =  mse_cost_function(rho_ic_out, rho_0)
    mse_vx_ic  =  mse_cost_function(vx_ic_out, vx_0)

    if model.dimension == 2:
        mse_vy_ic  =  mse_cost_function(vy_ic_out, vy_0)

    elif model.dimension == 3:
        mse_vy_ic  =  mse_cost_function(vy_ic_out, vy_0)
        mse_vz_ic  =  mse_cost_function(vz_ic_out, vz_0)

    ############# Loss based on boundary conditions #################

    
    rhox_b     = model.periodic_BC(net,coordinate=1,derivative_order=0,component=0)
    
    vx_xb      = model.periodic_BC(net,coordinate=1,derivative_order=0,component=1)
       
    phi_xb     = model.periodic_BC(net,coordinate=1,derivative_order=0,component=2)
    phi_xx_b   = model.periodic_BC(net,coordinate=1,derivative_order=1,component=2)

    if model.dimension == 2 or model.dimension == 3:
        rhoy_b     = model.periodic_BC(net,coordinate=2,derivative_order=0,component=0)

        vx_yb      = model.periodic_BC(net,coordinate=2,derivative_order=0,component=1)
        vy_xb      = model.periodic_BC(net,coordinate=1,derivative_order=0,component=2)
        vy_yb      = model.periodic_BC(net,coordinate=2,derivative_order=0,component=2)

        phi_yb     = model.periodic_BC(net,coordinate=2,derivative_order=0,component=3)
        phi_yy_b   = model.periodic_BC(net,coordinate=2,derivative_order=1,component=3)

        if model.dimension == 3:
            rhoz_b     = model.periodic_BC(net,coordinate=3,derivative_order=0,component=0)

            vx_zb      = model.periodic_BC(net,coordinate=3,derivative_order=0,component=1)
            vy_yb      = model.periodic_BC(net,coordinate=2,derivative_order=0,component=2)
            vy_zb      = model.periodic_BC(net,coordinate=3,derivative_order=0,component=2)
    
            vz_xb      = model.periodic_BC(net,coordinate=1,derivative_order=0,component=3)
            vz_yb      = model.periodic_BC(net,coordinate=2,derivative_order=0,component=3)
            vz_zb      = model.periodic_BC(net,coordinate=3,derivative_order=0,component=3)

            phi_zb     = model.periodic_BC(net,coordinate=3,derivative_order=0,component=4)
            phi_zz_b   = model.periodic_BC(net,coordinate=3,derivative_order=1,component=4)


    ############## Loss based on PDE ###################################
    
    if model.dimension == 1:
        rho_r,vx_r,phi_r = pde_residue(collocation_domain, net, dimension = 1)

    elif model.dimension == 2:
        rho_r,vx_r,vy_r,phi_r = pde_residue(collocation_domain, net, dimension = 2)

    elif model.dimension == 3:
        rho_r,vx_r,vy_r,vz_r,phi_r = pde_residue(collocation_domain, net, dimension = 3)
    

    mse_rho  = torch.mean(rho_r ** 2)
    mse_velx = torch.mean(vx_r  ** 2)

    if model.dimension == 2:
        mse_vely = torch.mean(vy_r  ** 2)

    elif model.dimension == 3:
        mse_vely = torch.mean(vy_r  ** 2)
        mse_velz = torch.mean(vz_r  ** 2)
    
    mse_phi  = torch.mean(phi_r ** 2)

    ################### Combining the loss functions ####################
    if model.dimension == 1:
        loss = mse_rho_ic + mse_vx_ic + mse_rho + mse_velx + mse_phi + rhox_b + vx_xb + phi_xb + phi_xx_b

    elif model.dimension == 2:
        loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_rho + mse_velx + mse_vely + mse_phi + rhox_b + rhoy_b + \
        vx_xb + vx_yb + vy_xb + vy_yb + phi_xb + phi_yb + phi_xx_b  + phi_yy_b

    elif model.dimension == 3:
        loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_vz_ic + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi + \
        rhox_b + rhoy_b + rhoz_b + vx_xb + vx_yb + vx_zb + vy_xb + vy_yb + vy_zb + vz_xb + vz_yb + vz_zb + \
        phi_xb + phi_yb + phi_zb + phi_xx_b  + phi_yy_b  + phi_zz_b

    
        #loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_vz_ic + \
        #rhox_b + rhoy_b + rhoz_b + vx_xb + vx_yb + vx_zb +  vy_xb + vy_yb + vy_zb + vz_xb + vz_yb + vz_zb + \
        #phi_xb + phi_xx_b + phi_yb + phi_yy_b +  phi_zb + phi_zz_b + mse_rho + mse_velx +  mse_vely + mse_velz + mse_phi 

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    
    return loss

def train(model, net, collocation_domain, collocation_IC, optimizer, optimizerL, iteration_adam, iterationL, mse_cost_function, closure, rho_1, lam, jeans, v_1, device):

    for i in range(iteration_adam):

        optimizer.zero_grad() # to make the gradients zero

        loss = optimizer.step(lambda: closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1))

        with torch.autograd.no_grad():
            
            if i % 250 == 0:
                print(f"Training Loss at {i} for Adam in 1D system = {loss.item():.2e}", flush=True)

    for i in range(iterationL):
        
        optimizer.zero_grad() # to make the gradients zero

        loss = optimizerL.step(lambda: closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1))

        with torch.autograd.no_grad():
            
            if i % 250 == 0:
                print(f"Training Loss at {i} for LBGFS in 1D system = {loss.item():.2e}", flush=True)