import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from data_generator import diff
from model_architecture import PINN
from config import cs, const, G, rho_o, N_GRID, POWER_EXPONENT, FILTER_SCALE, CONTINUITY_IC_WEIGHT

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

    #rho_0 = rho_o + rho_1 * torch.cos(2*np.pi*x[0]/lam)
    rho_0 = torch.full_like(x[0], rho_o)
    # Ensure correct shape [N, 1]
    if rho_0.dim() == 1:
        return rho_0.unsqueeze(-1)
    else:
        return rho_0

def generate_power_spectrum_field(lam, v_1, x, seed=1234):
    '''Generate 2D Gaussian random field with power spectrum'''
    
    Lx = lam * 2  # Domain size
    dx = Lx / N_GRID
    
    # Create coordinate grids
    x_coords = torch.linspace(0, Lx, N_GRID, device=x[0].device)
    y_coords = torch.linspace(0, Lx, N_GRID, device=x[0].device)
    
    # Calculate wave numbers
    kx = 2 * np.pi * torch.fft.fftfreq(N_GRID, dx, device=x[0].device)
    ky = 2 * np.pi * torch.fft.fftfreq(N_GRID, dx, device=x[0].device)
    KX, KY = torch.meshgrid(kx, ky, indexing='ij')
    
    # Calculate magnitude of wave number
    K = torch.sqrt(KX**2 + KY**2)
    
    # Power spectrum: P(k) ~ k^expon * exp((-k*Rf)^2)
    K_safe = torch.where(K == 0, torch.tensor(1e-10, device=x[0].device), K)
    power_spectrum = K_safe**POWER_EXPONENT * torch.exp(-(K_safe * FILTER_SCALE)**2)
    
    # Safety check: limit extreme values
    power_spectrum = torch.clamp(power_spectrum, 0, 1e6)
    
    # Generate random phases
    torch.manual_seed(seed)
    random_phases = torch.randn(N_GRID, N_GRID, device=x[0].device) + 1j * torch.randn(N_GRID, N_GRID, device=x[0].device)
    
    # Create complex field in Fourier space and transform to real space
    field_fourier = torch.sqrt(power_spectrum) * random_phases
    field_real = torch.real(torch.fft.ifft2(field_fourier))
    
    # Normalize the field
    field_real = field_real / torch.std(field_real) * v_1
    
    # Interpolate to the actual collocation points
    x_norm = torch.clamp((x[0] / Lx) * (N_GRID - 1), 0, N_GRID - 1)
    y_norm = torch.clamp((x[1] / Lx) * (N_GRID - 1), 0, N_GRID - 1)
    
    x_idx = torch.round(x_norm).long()
    y_idx = torch.round(y_norm).long()
    
    # Ensure correct tensor shape [N, 1]
    result = field_real[x_idx, y_idx]
    if result.dim() == 1:
        return result.unsqueeze(-1)
    else:
        return result

def fun_vx_0(lam, jeans, v_1, x):
    '''initial condition for x-velocity -- Power spectrum Gaussian random field'''
    return generate_power_spectrum_field(lam, v_1, x, seed=1234)

def fun_vy_0(lam, jeans, v_1, x):
    '''initial condition for y-velocity -- Power spectrum Gaussian random field'''
    return generate_power_spectrum_field(lam, v_1, x, seed=5678)

def func(x):
    return x[0]*0

### (3) Training / Fitting
from tqdm import tqdm

#start = time.time()
def closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1):

    ############## Loss based on initial conditions ###############
    rho_0 = fun_rho_0(rho_1, lam, collocation_IC)
    vx_0  = fun_vx_0(lam, jeans, v_1, collocation_IC)

    if model.dimension == 2:
        vy_0  = fun_vy_0(lam, jeans, v_1, collocation_IC)

    elif model.dimension == 3:
        vy_0  = fun_vy_0(lam, jeans, v_1, collocation_IC)
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

    ############# Boundary conditions enforced by construction #############
    # With periodic input embeddings, outputs and derivatives match at boundaries,
    # so no explicit BC loss terms are needed.


    ############## Continuity at t=0 to seed early-time evolution ###############
    # Enforce rho_t(0) = -rho0 * div v0 at IC points
    x_ic = collocation_IC[0]
    if model.dimension == 1:
        t_ic = collocation_IC[1]
    elif model.dimension == 2:
        y_ic = collocation_IC[1]
        t_ic = collocation_IC[2]
    elif model.dimension == 3:
        y_ic = collocation_IC[1]
        z_ic = collocation_IC[2]
        t_ic = collocation_IC[3]

    # Temporarily enable gradients for t to differentiate rho wrt t at t=0
    t_ic = t_ic.clone().detach().requires_grad_(True)
    ic_inputs = [x_ic]
    if model.dimension >= 2:
        ic_inputs.append(y_ic)
    if model.dimension == 3:
        ic_inputs.append(z_ic)
    ic_inputs.append(t_ic)
    ic_outputs = net(ic_inputs)
    rho_ic = ic_outputs[:,0:1]
    if model.dimension == 1:
        vx_ic = ic_outputs[:,1:2]
        rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
        # Use prescribed initial velocity field divergence for target
        vx0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
        div_v0 = diff(vx0, x_ic, order=1)
        continuity_ic_loss = mse_cost_function(rho_t_ic, -rho_o * div_v0)
    elif model.dimension == 2:
        rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
        # Use prescribed initial velocity field divergence for target
        vx0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
        vy0 = fun_vy_0(lam, jeans, v_1, collocation_IC)
        dvx_dx = diff(vx0, x_ic, order=1)
        dvy_dy = diff(vy0, y_ic, order=1)
        div_v0 = dvx_dx + dvy_dy
        continuity_ic_loss = mse_cost_function(rho_t_ic, -rho_o * div_v0)
    else: # dimension == 3
        rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
        vx0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
        vy0 = fun_vy_0(lam, jeans, v_1, collocation_IC)
        vz0 = func(collocation_IC)
        dvx_dx = diff(vx0, x_ic, order=1)
        dvy_dy = diff(vy0, y_ic, order=1)
        dvz_dz = diff(vz0, z_ic, order=1)
        div_v0 = dvx_dx + dvy_dy + dvz_dz
        continuity_ic_loss = mse_cost_function(rho_t_ic, -rho_o * div_v0)

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
        loss = mse_rho_ic + mse_vx_ic + CONTINUITY_IC_WEIGHT * continuity_ic_loss + mse_rho + mse_velx + mse_phi

    elif model.dimension == 2:
        loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + CONTINUITY_IC_WEIGHT * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_phi

    elif model.dimension == 3:
        loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_vz_ic + CONTINUITY_IC_WEIGHT * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi

    
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
            
            if i % 100 == 0:
                print(f"Training Loss at {i} for Adam in 2D system = {loss.item():.2e}", flush=True)

    for i in range(iterationL):
        
        optimizer.zero_grad() # to make the gradients zero

        loss = optimizerL.step(lambda: closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1))

        with torch.autograd.no_grad():
            
            if i % 50 == 0:
                print(f"Training Loss at {i} for LBGFS in 2D system = {loss.item():.2e}", flush=True)