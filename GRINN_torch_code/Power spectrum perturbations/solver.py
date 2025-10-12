import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from data_generator import diff
from model_architecture import PINN

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

# Global variable to store shared velocity field interpolators
_shared_vx_interp = None
_shared_vy_interp = None

def initialize_shared_velocity_field(lam, v_1, domain_size=None):
    """Initialize shared velocity field for consistent PINN and FD initial conditions"""
    global _shared_vx_interp, _shared_vy_interp
    
    if domain_size is None:
        domain_size = lam * 2  # Default domain size
    
    # Import here to avoid circular imports
    from LAX_2D import generate_shared_velocity_field
    
    # Generate shared velocity field with same parameters as PINN power spectrum
    # Use N_GRID for power spectrum generation (not FD_N_2D which is for sinusoidal)
    vx_np, vy_np, vx_interp, vy_interp = generate_shared_velocity_field(
        nx=N_GRID, ny=N_GRID, Lx=domain_size, Ly=domain_size,
        power_index=POWER_EXPONENT, amplitude=v_1, random_seed=1234
    )
    
    _shared_vx_interp = vx_interp
    _shared_vy_interp = vy_interp
    
    return vx_np, vy_np

def generate_power_spectrum_field(lam, v_1, x, seed=1234):
    '''Generate 2D velocity field using shared interpolator for consistent ICs'''
    
    global _shared_vx_interp, _shared_vy_interp
    
    # Initialize shared field if not done yet
    if _shared_vx_interp is None or _shared_vy_interp is None:
        initialize_shared_velocity_field(lam, v_1)
    
    # Convert torch tensors to numpy for interpolation
    x_np = x[0].detach().cpu().numpy()
    y_np = x[1].detach().cpu().numpy()
    
    # Create coordinate pairs for interpolation
    coords = np.column_stack([x_np.flatten(), y_np.flatten()])
    
    # Interpolate velocity field
    vx_values = _shared_vx_interp(coords)
    vy_values = _shared_vy_interp(coords)
    
    # Convert back to torch tensors with correct shape and device
    vx_tensor = torch.tensor(vx_values, device=x[0].device, dtype=x[0].dtype)
    vy_tensor = torch.tensor(vy_values, device=x[0].device, dtype=x[0].dtype)
    
    # Reshape to match input shape
    if x[0].dim() > 1:
        vx_tensor = vx_tensor.reshape(x[0].shape)
        vy_tensor = vy_tensor.reshape(x[0].shape)
    
    # Ensure correct tensor shape [N, 1]
    if vx_tensor.dim() == 1:
        vx_tensor = vx_tensor.unsqueeze(-1)
    if vy_tensor.dim() == 1:
        vy_tensor = vy_tensor.unsqueeze(-1)
    
    return vx_tensor

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

def generate_vy_power_spectrum_field(lam, v_1, x, seed=1234):
    '''Generate 2D y-velocity field using shared interpolator for consistent ICs'''
    
    global _shared_vx_interp, _shared_vy_interp
    
    # Initialize shared field if not done yet
    if _shared_vx_interp is None or _shared_vy_interp is None:
        initialize_shared_velocity_field(lam, v_1)
    
    # Convert torch tensors to numpy for interpolation
    x_np = x[0].detach().cpu().numpy()
    y_np = x[1].detach().cpu().numpy()
    
    # Create coordinate pairs for interpolation
    coords = np.column_stack([x_np.flatten(), y_np.flatten()])
    
    # Interpolate velocity field
    vy_values = _shared_vy_interp(coords)
    
    # Convert back to torch tensors with correct shape and device
    vy_tensor = torch.tensor(vy_values, device=x[0].device, dtype=x[0].dtype)
    
    # Reshape to match input shape
    if x[0].dim() > 1:
        vy_tensor = vy_tensor.reshape(x[0].shape)
    
    # Ensure correct tensor shape [N, 1]
    if vy_tensor.dim() == 1:
        vy_tensor = vy_tensor.unsqueeze(-1)
    
    return vy_tensor

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
        return generate_vy_power_spectrum_field(lam, v_1, x, seed=1234)

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


    if model.dimension == 2:

    elif model.dimension == 3:

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

        if model.dimension == 2:
            mse_vy_ic = VELOCITY_IC_WEIGHT * mse_cost_function(vy_ic_out, vy_0)
        elif model.dimension == 3:

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
            if not is_sin:
                colloc_shifted[1] = colloc_shifted[1] + startup_dt
            rho_r, vx_r, phi_r = pde_residue(colloc_shifted, net, dimension=1)
        elif model.dimension == 2:
            if not is_sin:
                colloc_shifted[2] = colloc_shifted[2] + startup_dt
            rho_r, vx_r, vy_r, phi_r = pde_residue(colloc_shifted, net, dimension=2)
        else:
            if not is_sin:
                colloc_shifted[3] = colloc_shifted[3] + startup_dt
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


    return net