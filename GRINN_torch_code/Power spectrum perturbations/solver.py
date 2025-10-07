import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from data_generator import diff
from model_architecture import PINN
from config import cs, const, G, rho_o, N_GRID, POWER_EXPONENT, FILTER_SCALE, CONTINUITY_IC_WEIGHT, STARTUP_DT, DECAY_PORTION, PERTURBATION_TYPE

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

    if rho_o != 0:
        jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_o))
    else:
        jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*(rho_o + 1)))

    if lam > jeans:
        if rho_o != 0:
            alpha = np.sqrt(const*G*rho_o-cs**2*(2*np.pi/lam)**2)
        else:
            alpha = np.sqrt(const*G*(rho_o + 1)-cs**2*(2*np.pi/lam)**2)
    else:
        if rho_o != 0:
            alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*rho_o)
        else:
            alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*(rho_o + 1))

    return jeans, alpha

def fun_rho_0(rho_1, lam, x):
    ''' Define initial condition for density Returning Eq (11a)'''
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        coord = x[0]
        u = coord if coord.dim() > 1 else coord.unsqueeze(-1)
        rho_0 = rho_o + rho_1 * torch.cos(2*np.pi*u/lam)
        return rho_0
    else:
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
    
    # Remove DC (uniform) mode to avoid bulk drift
    power_spectrum[K == 0] = 0.0
    
    # Safety check: limit extreme values
    power_spectrum = torch.clamp(power_spectrum, 0, 1e6)
    
    # Generate random phases
    torch.manual_seed(seed)
    random_phases = torch.randn(N_GRID, N_GRID, device=x[0].device) + 1j * torch.randn(N_GRID, N_GRID, device=x[0].device)
    
    # Create complex field in Fourier space and transform to real space
    field_fourier = torch.sqrt(power_spectrum) * random_phases
    field_real = torch.real(torch.fft.ifft2(field_fourier))
    
    # Remove any residual mean (bulk flow) and normalize rms to v_1
    field_real = field_real - torch.mean(field_real)
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

def _sinusoidal_component(coord, lam, jeans, v_1):
    # coord: tensor [N,1] or [N]
    u = coord if coord.dim() > 1 else coord.unsqueeze(-1)
    if lam > jeans:
        return - v_1 * torch.sin(2*np.pi*u/lam)
    else:
        return v_1 * torch.cos(2*np.pi*u/lam)

def fun_vx_0(lam, jeans, v_1, x):
    '''initial condition for x-velocity -- branch by PERTURBATION_TYPE'''
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        return _sinusoidal_component(x[0], lam, jeans, v_1)
    else:
        return generate_power_spectrum_field(lam, v_1, x, seed=1234)

def fun_vy_0(lam, jeans, v_1, x):
    '''initial condition for y-velocity -- branch by PERTURBATION_TYPE'''
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # use y coordinate for vy sinusoid
        if len(x) >= 2:
            return _sinusoidal_component(x[1], lam, jeans, v_1)
        else:
            # fallback to x if y is unavailable (1D)
            return _sinusoidal_component(x[0], lam, jeans, v_1)
    else:
        return generate_power_spectrum_field(lam, v_1, x, seed=5678)

def func(x):
    return x[0]*0

### (3) Training / Fitting
from tqdm import tqdm

#start = time.time()
def closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, continuity_weight, startup_dt):

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

    # For sinusoidal: enforce only explicit sinusoidal ICs; skip continuity seeding
    is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
    if is_sin:
        x_ic_for_ic = collocation_IC[0]
        rho_ic_target = rho_o + rho_1 * torch.cos(2*np.pi*x_ic_for_ic/lam)
        mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
    else:
        mse_rho_ic = 0.0 * torch.mean(rho_ic_out*0)

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

    # For sinusoidal testing, do not apply continuity seeding at t=0
    t_ic = t_ic.clone().detach().requires_grad_(not is_sin)
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
        if is_sin:
            continuity_ic_loss = torch.tensor(0.0, device= rho_ic.device, dtype=rho_ic.dtype)
        else:
            rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
            vx0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
            div_v0 = diff(vx0, x_ic, order=1)
            rho0_field = rho_o * torch.ones_like(div_v0)
            continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)
    elif model.dimension == 2:
        if is_sin:
            continuity_ic_loss = torch.tensor(0.0, device= rho_ic.device, dtype=rho_ic.dtype)
        else:
            rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
            vx0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
            vy0 = fun_vy_0(lam, jeans, v_1, collocation_IC)
            dvx_dx = diff(vx0, x_ic, order=1)
            dvy_dy = diff(vy0, y_ic, order=1)
            div_v0 = dvx_dx + dvy_dy
            rho0_field = rho_o * torch.ones_like(div_v0)
            continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)
    else: # dimension == 3
        if is_sin:
            continuity_ic_loss = torch.tensor(0.0, device= rho_ic.device, dtype=rho_ic.dtype)
        else:
            rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
            vx0 = fun_vx_0(lam, jeans, v_1, collocation_IC)
            vy0 = fun_vy_0(lam, jeans, v_1, collocation_IC)
            vz0 = func(collocation_IC)
            dvx_dx = diff(vx0, x_ic, order=1)
            dvy_dy = diff(vy0, y_ic, order=1)
            dvz_dz = diff(vz0, z_ic, order=1)
            div_v0 = dvx_dx + dvy_dy + dvz_dz
            rho0_field = rho_o * torch.ones_like(div_v0)
            continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)

    ############## Loss based on PDE ###################################
    
    # Apply startup time offset to PDE collocation time only (IC remains at t=0)
    if isinstance(collocation_domain, (list, tuple)):
        colloc_shifted = list(collocation_domain)
    else:
        colloc_shifted = collocation_domain

    if model.dimension == 1:
        # time is at index 1
        if not is_sin:
            colloc_shifted[1] = colloc_shifted[1] + startup_dt
        rho_r,vx_r,phi_r = pde_residue(colloc_shifted, net, dimension = 1)

    elif model.dimension == 2:
        # time is at index 2
        if not is_sin:
            colloc_shifted[2] = colloc_shifted[2] + startup_dt
        rho_r,vx_r,vy_r,phi_r = pde_residue(colloc_shifted, net, dimension = 2)

    elif model.dimension == 3:
        # time is at index 3
        if not is_sin:
            colloc_shifted[3] = colloc_shifted[3] + startup_dt
        rho_r,vx_r,vy_r,vz_r,phi_r = pde_residue(colloc_shifted, net, dimension = 3)
    

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
        base = mse_vx_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    elif model.dimension == 2:
        base = mse_vx_ic + mse_vy_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    elif model.dimension == 3:
        base = mse_vx_ic + mse_vy_ic + mse_vz_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    
        #loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_vz_ic + \
        #rhox_b + rhoy_b + rhoz_b + vx_xb + vx_yb + vx_zb +  vy_xb + vy_yb + vy_zb + vz_xb + vz_yb + vz_zb + \
        #phi_xb + phi_xx_b + phi_yb + phi_yy_b +  phi_zb + phi_zz_b + mse_rho + mse_velx +  mse_vely + mse_velz + mse_phi 

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    
    return loss

def train(model, net, collocation_domain, collocation_IC, optimizer, optimizerL, iteration_adam, iterationL, mse_cost_function, closure, rho_1, lam, jeans, v_1, device):

    # Cosine schedules for continuity weight and startup dt
    total_steps = iteration_adam + iterationL
    total_for_decay = max(1, int(total_steps * DECAY_PORTION))

    def cosine_schedule(step, total, start_value, end_value):
        if total_for_decay <= 1:
            return end_value
        s = min(step, total_for_decay - 1)
        cos_term = (1 + np.cos(np.pi * s / (total_for_decay - 1))) / 2.0
        return end_value + (start_value - end_value) * cos_term

    for i in range(iteration_adam):

        optimizer.zero_grad() # to make the gradients zero

        global_step = i
        continuity_weight = cosine_schedule(global_step, total_steps, CONTINUITY_IC_WEIGHT, 0.0)
        startup_dt = cosine_schedule(global_step, total_steps, STARTUP_DT, 0.0)

        loss = optimizer.step(lambda: closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, continuity_weight, startup_dt))

        with torch.autograd.no_grad():
            
            if i % 100 == 0:
                print(f"Training Loss at {i} for Adam in 2D system = {loss.item():.2e}", flush=True)

    for i in range(iterationL):
        
        optimizer.zero_grad() # to make the gradients zero

        global_step = iteration_adam + i
        continuity_weight = cosine_schedule(global_step, total_steps, CONTINUITY_IC_WEIGHT, 0.0)
        startup_dt = cosine_schedule(global_step, total_steps, STARTUP_DT, 0.0)

        loss = optimizerL.step(lambda: closure(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1, continuity_weight, startup_dt))

        with torch.autograd.no_grad():
            
            if i % 50 == 0:
                print(f"Training Loss at {i} for LBGFS in 2D system = {loss.item():.2e}", flush=True)