import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from data_generator import diff
from model_architecture import PINN
from config import cs, const, G, rho_o, N_GRID, POWER_EXPONENT, FILTER_SCALE, CONTINUITY_IC_WEIGHT, STARTUP_DT, DECAY_PORTION, PERTURBATION_TYPE, KX, KY, BATCH_SIZE, NUM_BATCHES

def input_taker(lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r):
    lam = float(lam)  # Wavelength
    rho_1 = float(rho_1)  # Amplitude of perturbation
    num_of_waves = int(num_of_waves)  # Number of waves
    tmax = float(tmax)  # Maximum time
    N_0 = int(N_0)  # Number of initial condition points
    # N_b is no longer used due to hard constraints but kept for compatibility
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

# Global shared velocity fields for consistent initial conditions
_shared_vx_interp = None
_shared_vy_interp = None

def initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=1234):
    """
    Initialize shared velocity fields for consistent PINN/FD initial conditions.
    This should be called once at the beginning of training.
    """
    global _shared_vx_interp, _shared_vy_interp
    
    # Import LAX_2D functions
    from LAX_2D import generate_shared_velocity_field
    
    # Calculate domain size to match FD solver
    Lx = lam * num_of_waves
    Ly = lam * num_of_waves
    
    # Generate shared velocity fields
    vx_np, vy_np, vx_interp, vy_interp = generate_shared_velocity_field(
        N_GRID, N_GRID, Lx, Ly, 
        power_index=POWER_EXPONENT, 
        amplitude=v_1, 
        random_seed=seed
    )
    
    # Store interpolation functions globally
    _shared_vx_interp = vx_interp
    _shared_vy_interp = vy_interp
    
    return vx_np, vy_np

def generate_power_spectrum_field(lam, v_1, x, seed=1234):
    '''Generate 2D Gaussian random field with power spectrum using shared fields if available'''
    
    # Use shared velocity fields if available
    if _shared_vx_interp is not None and _shared_vy_interp is not None:
        # Convert tensor coordinates to numpy for interpolation
        x_np = x[0].detach().cpu().numpy()
        y_np = x[1].detach().cpu().numpy()
        
        # Create coordinate pairs for interpolation
        coords = np.stack([x_np.flatten(), y_np.flatten()], axis=1)
        
        # Interpolate shared velocity field
        vx_interp = _shared_vx_interp(coords)
        vy_interp = _shared_vy_interp(coords)
        
        # Convert back to tensor and reshape
        vx_tensor = torch.from_numpy(vx_interp).float().to(x[0].device)
        vy_tensor = torch.from_numpy(vy_interp).float().to(x[0].device)
        
        # Return vx component (vy will be handled separately)
        if vx_tensor.dim() == 1:
            return vx_tensor.unsqueeze(-1)
        else:
            return vx_tensor
    
    # Fallback to original method if shared fields not available
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

def _sinusoidal_component(coord, lam, jeans, v_1, k_component=None):
    # coord: tensor [N,1] or [N]
    u = coord if coord.dim() > 1 else coord.unsqueeze(-1)
    if k_component is not None:
        # Use specific k component for 2D case
        wave_phase = k_component * u
    else:
        # Fallback to original behavior for 1D case
        wave_phase = 2*np.pi*u/lam
    
    if lam > jeans:
        return - v_1 * torch.sin(wave_phase)
    else:
        return v_1 * torch.cos(wave_phase)

def _coupled_2d_velocity_components(x, lam, jeans, v_1):
    '''Generate coupled 2D velocity components from the same wave pattern'''
    if len(x) < 2:
        # Fallback to 1D case
        return _sinusoidal_component(x[0], lam, jeans, v_1)
    
    x_coord = x[0] if x[0].dim() > 1 else x[0].unsqueeze(-1)
    y_coord = x[1] if x[1].dim() > 1 else x[1].unsqueeze(-1)
    
    # Calculate wave vector magnitude (convert to tensor)
    k_magnitude = torch.sqrt(torch.tensor(KX**2 + KY**2, device=x_coord.device, dtype=x_coord.dtype))
    
    # Generate the coupled wave pattern
    wave_phase = KX * x_coord + KY * y_coord
    
    if lam > jeans:
        # Gravitational instability case
        wave_field = -v_1 * torch.sin(wave_phase)
    else:
        # Oscillatory case
        wave_field = v_1 * torch.cos(wave_phase)
    
    # Coupled velocity components
    if k_magnitude > 0:
        vx = wave_field * (KX / k_magnitude)
        vy = wave_field * (KY / k_magnitude)
    else:
        vx = wave_field
        vy = torch.zeros_like(wave_field)
    
    return vx, vy

def fun_rho_0(rho_1, lam, x):
    ''' Define initial condition for density Returning Eq (11a)'''
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use separate kx and ky components for 2D wave vector
        if len(x) >= 2:  # 2D case
            x_coord = x[0] if x[0].dim() > 1 else x[0].unsqueeze(-1)
            y_coord = x[1] if x[1].dim() > 1 else x[1].unsqueeze(-1)
            rho_0 = rho_o + rho_1 * torch.cos(KX * x_coord + KY * y_coord)
        else:  # 1D case - fallback to original behavior
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

def generate_power_spectrum_field_vy(lam, v_1, x, seed=5678):
    '''Generate vy component using shared fields if available'''
    
    # Use shared velocity fields if available
    if _shared_vx_interp is not None and _shared_vy_interp is not None:
        # Convert tensor coordinates to numpy for interpolation
        x_np = x[0].detach().cpu().numpy()
        y_np = x[1].detach().cpu().numpy()
        
        # Create coordinate pairs for interpolation
        coords = np.stack([x_np.flatten(), y_np.flatten()], axis=1)
        
        # Interpolate shared velocity field
        vy_interp = _shared_vy_interp(coords)
        
        # Convert back to tensor and reshape
        vy_tensor = torch.from_numpy(vy_interp).float().to(x[0].device)
        
        # Return vy component
        if vy_tensor.dim() == 1:
            return vy_tensor.unsqueeze(-1)
        else:
            return vy_tensor
    
    # Fallback to original method if shared fields not available
    return generate_power_spectrum_field(lam, v_1, x, seed=seed)

def fun_vx_0(lam, jeans, v_1, x):
    '''initial condition for x-velocity -- branch by PERTURBATION_TYPE'''
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use coupled 2D velocity components for proper 2D wave physics
        if len(x) >= 2:  # 2D case
            vx, _ = _coupled_2d_velocity_components(x, lam, jeans, v_1)
            return vx
        else:  # 1D case
            return _sinusoidal_component(x[0], lam, jeans, v_1)
    else:
        return generate_power_spectrum_field(lam, v_1, x, seed=1234)

def fun_vy_0(lam, jeans, v_1, x):
    '''initial condition for y-velocity -- branch by PERTURBATION_TYPE'''
    if str(PERTURBATION_TYPE).lower() == "sinusoidal":
        # Use coupled 2D velocity components for proper 2D wave physics
        if len(x) >= 2:  # 2D case
            _, vy = _coupled_2d_velocity_components(x, lam, jeans, v_1)
            return vy
        else:
            # fallback to x if y is unavailable (1D)
            return _sinusoidal_component(x[0], lam, jeans, v_1)
    else:
        return generate_power_spectrum_field_vy(lam, v_1, x, seed=5678)

def func(x):
    return x[0]*0

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
        if len(collocation_IC) >= 2:  # 2D case
            y_ic_for_ic = collocation_IC[1]
            rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic_for_ic + KY * y_ic_for_ic)
        else:  # 1D case
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
        # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
        rho_r,vx_r,phi_r = pde_residue(colloc_shifted, net, dimension = 1)

    elif model.dimension == 2:
        # time is at index 2
        # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
        rho_r,vx_r,vy_r,phi_r = pde_residue(colloc_shifted, net, dimension = 2)

    elif model.dimension == 3:
        # time is at index 3
        # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
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
    # Batched training is the default
    total_steps = iteration_adam + iterationL
    total_for_decay = max(1, int(total_steps * DECAY_PORTION))

    def cosine_schedule(step, total, start_value, end_value):
        if total_for_decay <= 1:
            return end_value
        s = min(step, total_for_decay - 1)
        cos_term = (1 + np.cos(np.pi * s / (total_for_decay - 1))) / 2.0
        return end_value + (start_value - end_value) * cos_term

    bs = int(BATCH_SIZE)
    nb = int(NUM_BATCHES)

    for i in range(iteration_adam):
        optimizer.zero_grad()
        global_step = i
        continuity_weight = cosine_schedule(global_step, total_steps, CONTINUITY_IC_WEIGHT, 0.0)
        startup_dt = cosine_schedule(global_step, total_steps, STARTUP_DT, 0.0)

        loss = optimizer.step(lambda: closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, continuity_weight, startup_dt, bs, nb))

        with torch.autograd.no_grad():
            if i % 100 == 0:
                print(f"Training Loss at {i} for Adam (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)

    for i in range(iterationL):
        optimizer.zero_grad()
        global_step = iteration_adam + i
        continuity_weight = cosine_schedule(global_step, total_steps, CONTINUITY_IC_WEIGHT, 0.0)
        startup_dt = cosine_schedule(global_step, total_steps, STARTUP_DT, 0.0)

        loss = optimizerL.step(lambda: closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1, continuity_weight, startup_dt, bs, nb))

        with torch.autograd.no_grad():
            if i % 50 == 0:
                print(f"Training Loss at {i} for LBGFS (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)


def _random_batch_indices(total_count, batch_size, device):
    actual = int(min(batch_size, total_count))
    return torch.randperm(total_count, device=device)[:actual]


def _make_batch_tensors(tensors_list, indices):
    return [t[indices].clone() for t in tensors_list]


def closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                    rho_1, lam, jeans, v_1, continuity_weight, startup_dt, batch_size, num_batches):

    # Aggregate losses across mini-batches
    total_loss = 0.0
    num_effective_batches = 0

    # Determine counts and devices
    dom_n = collocation_domain[0].size(0)
    ic_n = collocation_IC[0].size(0)
    device = collocation_domain[0].device

    for _ in range(int(max(1, num_batches))):
        dom_idx = _random_batch_indices(dom_n, batch_size, device)
        ic_idx = _random_batch_indices(ic_n, batch_size, device)

        batch_dom = _make_batch_tensors(collocation_domain, dom_idx)
        batch_ic = _make_batch_tensors(collocation_IC, ic_idx)

        # IC loss terms
        rho_0 = fun_rho_0(rho_1, lam, batch_ic)
        vx_0  = fun_vx_0(lam, jeans, v_1, batch_ic)

        net_ic_out = net(batch_ic)
        rho_ic_out = net_ic_out[:,0:1]
        vx_ic_out  = net_ic_out[:,1:2]

        if model.dimension == 2:
            vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic)
            vy_ic_out = net_ic_out[:,2:3]
        elif model.dimension == 3:
            vy_0 = fun_vy_0(lam, jeans, v_1, batch_ic)
            vz_0 = func(batch_ic)
            vy_ic_out = net_ic_out[:,2:3]
            vz_ic_out = net_ic_out[:,3:4]

        is_sin = str(PERTURBATION_TYPE).lower() == "sinusoidal"
        if is_sin:
            x_ic_for_ic = batch_ic[0]
            if len(batch_ic) >= 2:
                y_ic_for_ic = batch_ic[1]
                rho_ic_target = rho_o + rho_1 * torch.cos(KX * x_ic_for_ic + KY * y_ic_for_ic)
            else:
                rho_ic_target = rho_o + rho_1 * torch.cos(2*np.pi*x_ic_for_ic/lam)
            mse_rho_ic = mse_cost_function(rho_ic_out, rho_ic_target)
        else:
            mse_rho_ic = 0.0 * torch.mean(rho_ic_out*0)

        mse_vx_ic  = mse_cost_function(vx_ic_out, vx_0)
        if model.dimension == 2:
            mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        elif model.dimension == 3:
            mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
            mse_vz_ic = mse_cost_function(vz_ic_out, vz_0)

        # Continuity seeding at t=0 on IC points
        x_ic = batch_ic[0]
        if model.dimension == 1:
            t_ic = batch_ic[1]
        elif model.dimension == 2:
            y_ic = batch_ic[1]
            t_ic = batch_ic[2]
        elif model.dimension == 3:
            y_ic = batch_ic[1]
            z_ic = batch_ic[2]
            t_ic = batch_ic[3]

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
            if is_sin:
                continuity_ic_loss = torch.tensor(0.0, device=rho_ic.device, dtype=rho_ic.dtype)
            else:
                rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
                vx0 = fun_vx_0(lam, jeans, v_1, batch_ic)
                div_v0 = diff(vx0, x_ic, order=1)
                rho0_field = rho_o * torch.ones_like(div_v0)
                continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)
        elif model.dimension == 2:
            if is_sin:
                continuity_ic_loss = torch.tensor(0.0, device=rho_ic.device, dtype=rho_ic.dtype)
            else:
                rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
                vx0 = fun_vx_0(lam, jeans, v_1, batch_ic)
                vy0 = fun_vy_0(lam, jeans, v_1, batch_ic)
                dvx_dx = diff(vx0, x_ic, order=1)
                dvy_dy = diff(vy0, y_ic, order=1)
                div_v0 = dvx_dx + dvy_dy
                rho0_field = rho_o * torch.ones_like(div_v0)
                continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)
        else:
            if is_sin:
                continuity_ic_loss = torch.tensor(0.0, device=rho_ic.device, dtype=rho_ic.dtype)
            else:
                rho_t_ic = torch.autograd.grad(rho_ic, t_ic, grad_outputs=torch.ones_like(rho_ic), create_graph=True)[0]
                vx0 = fun_vx_0(lam, jeans, v_1, batch_ic)
                vy0 = fun_vy_0(lam, jeans, v_1, batch_ic)
                vz0 = func(batch_ic)
                dvx_dx = diff(vx0, x_ic, order=1)
                dvy_dy = diff(vy0, y_ic, order=1)
                dvz_dz = diff(vz0, z_ic, order=1)
                div_v0 = dvx_dx + dvy_dy + dvz_dz
                rho0_field = rho_o * torch.ones_like(div_v0)
                continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)

        # PDE residuals on batched domain with startup shift
        if isinstance(batch_dom, (list, tuple)):
            colloc_shifted = list(batch_dom)
        else:
            colloc_shifted = batch_dom
        if model.dimension == 1:
            # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
            rho_r, vx_r, phi_r = pde_residue(colloc_shifted, net, dimension=1)
        elif model.dimension == 2:
            # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
            rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_shifted, net, dimension=2)
        else:
            # Note: Domain collocation points already start from STARTUP_DT, no need to shift further
            rho_r, vx_r, vy_r, vz_r, phi_r = pde_residue(colloc_shifted, net, dimension=3)

        mse_rho  = torch.mean(rho_r ** 2)
        mse_velx = torch.mean(vx_r  ** 2)
        if model.dimension == 2:
            mse_vely = torch.mean(vy_r  ** 2)
        elif model.dimension == 3:
            mse_vely = torch.mean(vy_r  ** 2)
            mse_velz = torch.mean(vz_r  ** 2)
        mse_phi  = torch.mean(phi_r ** 2)

        if model.dimension == 1:
            base = mse_vx_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
        elif model.dimension == 2:
            base = mse_vx_ic + mse_vy_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)
        else:
            base = mse_vx_ic + mse_vy_ic + mse_vz_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi
            loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

        total_loss = total_loss + loss
        num_effective_batches += 1

    optimizer.zero_grad()
    avg_loss = total_loss / max(1, num_effective_batches)
    avg_loss.backward(retain_graph=True)
    return avg_loss


def distribute_collocation_points(n_total, num_subdomains):
    """
    Distribute collocation points across subdomains.
    
    Args:
        n_total: Total number of collocation points
        num_subdomains: Number of subdomains
    
    Returns:
        List of point counts per subdomain
    """
    from config import N_r_PER_SUBDOMAIN, N_0_PER_SUBDOMAIN
    
    # If per-subdomain count is specified, use it
    if n_total == N_r_PER_SUBDOMAIN and N_r_PER_SUBDOMAIN is not None:
        return [N_r_PER_SUBDOMAIN] * num_subdomains
    elif n_total == N_0_PER_SUBDOMAIN and N_0_PER_SUBDOMAIN is not None:
        return [N_0_PER_SUBDOMAIN] * num_subdomains
    
    # Otherwise, auto-distribute evenly
    base_count = n_total // num_subdomains
    remainder = n_total % num_subdomains
    
    counts = [base_count] * num_subdomains
    # Distribute remainder to first few subdomains
    for i in range(remainder):
        counts[i] += 1
    
    return counts


def closure_xpinn(xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                   subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                   optimizer):
    """
    Closure function for XPINN training with multiple networks.
    
    Args:
        xpinn_loss_model: XPINN_Loss instance
        nets: List of neural networks
        subdomain_collocs: List of subdomain collocation points
        interface_collocs: Dict of interface collocation points
        subdomain_ic_collocs: List of IC collocation points per subdomain
        ic_functions: Initial condition functions
        interfaces: List of interface tuples
        exterior_boundaries: Dict of exterior boundary info
        optimizer: Optimizer instance
    
    Returns:
        Total loss (scalar tensor)
    """
    optimizer.zero_grad()
    
    # Compute total XPINN loss
    total_loss, loss_dict = xpinn_loss_model.compute_total_loss(
        nets, subdomain_collocs, interface_collocs,
        subdomain_ic_collocs, ic_functions, interfaces,
        exterior_boundaries
    )
    
    # Backward pass
    total_loss.backward(retain_graph=True)
    
    return total_loss


def closure_xpinn_batched(xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                           subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                           optimizer, batch_size, num_batches):
    """
    Batched closure function for XPINN training with multiple networks.
    Processes collocation points in mini-batches with gradient accumulation.
    
    Args:
        xpinn_loss_model: XPINN_Loss instance
        nets: List of neural networks
        subdomain_collocs: List of subdomain collocation points
        interface_collocs: Dict of interface collocation points
        subdomain_ic_collocs: List of IC collocation points per subdomain
        ic_functions: Initial condition functions
        interfaces: List of interface tuples
        exterior_boundaries: Dict of exterior boundary info
        optimizer: Optimizer instance
        batch_size: Maximum points per mini-batch
        num_batches: Number of mini-batches to aggregate
    
    Returns:
        Total loss (scalar tensor)
    """
    optimizer.zero_grad()
    
    # Aggregate losses across mini-batches
    total_loss = 0.0
    num_effective_batches = 0
    
    # Get device from first subdomain's collocation points
    device = subdomain_collocs[0][0].device if len(subdomain_collocs) > 0 else 'cuda'
    
    for batch_idx in range(int(max(1, num_batches))):
        # Create batched subdomain collocation points
        batched_subdomain_collocs = []
        for subdomain_colloc in subdomain_collocs:
            dom_n = subdomain_colloc[0].size(0)
            dom_idx = _random_batch_indices(dom_n, batch_size, device)
            batch_dom = _make_batch_tensors(subdomain_colloc, dom_idx)
            batched_subdomain_collocs.append(batch_dom)
        
        # Create batched subdomain IC points
        batched_subdomain_ic_collocs = []
        for subdomain_ic_colloc in subdomain_ic_collocs:
            ic_n = subdomain_ic_colloc[0].size(0)
            ic_idx = _random_batch_indices(ic_n, batch_size, device)
            batch_ic = _make_batch_tensors(subdomain_ic_colloc, ic_idx)
            batched_subdomain_ic_collocs.append(batch_ic)
        
        # Create batched interface collocation points
        batched_interface_collocs = {}
        for interface_key, interface_colloc in interface_collocs.items():
            if_n = interface_colloc[0].size(0)
            if_idx = _random_batch_indices(if_n, batch_size, device)
            batch_if = _make_batch_tensors(interface_colloc, if_idx)
            batched_interface_collocs[interface_key] = batch_if
        
        # Compute loss for this mini-batch
        batch_loss, batch_loss_dict = xpinn_loss_model.compute_total_loss(
            nets, batched_subdomain_collocs, batched_interface_collocs,
            batched_subdomain_ic_collocs, ic_functions, interfaces,
            exterior_boundaries
        )
        
        # Accumulate loss
        total_loss += batch_loss
        num_effective_batches += 1
    
    # Average loss across batches
    if num_effective_batches > 0:
        total_loss = total_loss / num_effective_batches
    
    # Backward pass
    total_loss.backward(retain_graph=True)
    
    return total_loss


def train_xpinn(nets, subdomain_collocs, interface_collocs, subdomain_ic_collocs,
                ic_functions, interfaces, exterior_boundaries, xpinn_loss_model,
                optimizer, optimizerL, iteration_adam, iterationL, device):
    """
    Train XPINN with multiple networks.
    
    Args:
        nets: List of neural networks (one per subdomain)
        subdomain_collocs: List of subdomain collocation points
        interface_collocs: Dict of interface collocation points
        subdomain_ic_collocs: List of IC collocation points per subdomain
        ic_functions: Initial condition functions dictionary
        interfaces: List of interface tuples
        exterior_boundaries: Dict of exterior boundary information
        xpinn_loss_model: XPINN_Loss instance
        optimizer: Adam optimizer
        optimizerL: L-BFGS optimizer
        iteration_adam: Number of Adam iterations
        iterationL: Number of L-BFGS iterations
        device: PyTorch device
    
    Returns:
        None (trains networks in-place)
    """
    from config import XPINN_ALTERNATING_TRAINING, USE_XPINN_BATCHING, BATCH_SIZE, NUM_BATCHES
    
    print(f"Starting XPINN training with {len(nets)} subdomains...")
    print(f"Interfaces: {len(interfaces)}")
    print(f"Optimizer strategy: {'unified' if optimizer is not None else 'separate'}")
    print(f"Batching: {'enabled' if USE_XPINN_BATCHING else 'disabled'}")
    
    # Choose closure function based on batching setting
    if USE_XPINN_BATCHING:
        bs_global = int(BATCH_SIZE)
        nb = int(NUM_BATCHES)
        num_sub = len(nets)
        bs = max(1, bs_global // max(1, num_sub))  # per-subdomain batch size
        print(f"Batching: global={bs_global}, per_subdomain={bs}, num_subdomains={num_sub}, num_batches={nb}")
        
        def make_closure(opt):
            return lambda: closure_xpinn_batched(
                xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                opt, bs, nb
            )
    else:
        def make_closure(opt):
            return lambda: closure_xpinn(
                xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                opt
            )
    
    # Adam training
    for i in range(iteration_adam):
        if XPINN_ALTERNATING_TRAINING:
            # Alternate between subdomains (not yet implemented - would need separate optimizers)
            raise NotImplementedError("Alternating training not yet implemented")
        else:
            # Train all networks simultaneously
            loss = optimizer.step(make_closure(optimizer))
        
        if i % 100 == 0:
            with torch.autograd.no_grad():
                print(f"XPINN Training Loss at {i} (Adam) = {loss.item():.2e}", flush=True)
    
    # L-BFGS training
    for i in range(iterationL):
        if XPINN_ALTERNATING_TRAINING:
            raise NotImplementedError("Alternating training not yet implemented")
        else:
            loss = optimizerL.step(make_closure(optimizerL))
        
        if i % 50 == 0:
            with torch.autograd.no_grad():
                print(f"XPINN Training Loss at {i} (L-BFGS) = {loss.item():.2e}", flush=True)
    
    print("XPINN training completed.")