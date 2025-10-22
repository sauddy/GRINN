import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from data_generator import diff
from model_architecture import PINN
from config import cs, const, G, rho_o, N_GRID, POWER_EXPONENT, FILTER_SCALE, CONTINUITY_IC_WEIGHT, STARTUP_DT, DECAY_PORTION, PERTURBATION_TYPE, KX, KY, BATCH_SIZE, NUM_BATCHES, RANDOM_SEED

class ResidualTracker:
    """
    Tracks cumulative residuals across time bins for adaptive causal weighting.
    Implements w_i = exp(-epsilon * Σ_{k=1}^{i-1} L_r(t_k, θ))
    """
    def __init__(self, t_min, t_max, num_bins, epsilon, device='cuda'):
        """
        Args:
            t_min: Minimum time value
            t_max: Maximum time value
            num_bins: Number of time bins for tracking residuals
            epsilon: Causality parameter (controls weight suppression strength)
            device: PyTorch device
        """
        self.t_min = t_min
        self.t_max = t_max
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.device = device
        
        # Bin edges for time discretization
        self.bin_edges = torch.linspace(t_min, t_max, num_bins + 1, device=device)
        self.bin_width = (t_max - t_min) / num_bins
        
        # Cumulative residuals per bin (initialized to zero)
        self.cumulative_residuals = torch.zeros(num_bins, device=device)
        
        # Counter for number of updates per bin (for averaging)
        self.update_counts = torch.zeros(num_bins, device=device)
    
    def get_bin_indices(self, t_values):
        """
        Get bin indices for given time values.
        
        Args:
            t_values: Tensor of time values [N, 1]
        
        Returns:
            Bin indices [N] (clamped to valid range)
        """
        t_flat = t_values.flatten()
        # Compute bin index: floor((t - t_min) / bin_width)
        bin_idx = ((t_flat - self.t_min) / self.bin_width).long()
        # Clamp to valid range [0, num_bins-1]
        bin_idx = torch.clamp(bin_idx, 0, self.num_bins - 1)
        return bin_idx
    
    def update_residuals(self, t_values, residuals):
        """
        Update cumulative residuals for time bins based on current batch.
        Vectorized for speed.
        
        Args:
            t_values: Time values [N, 1]
            residuals: PDE residuals [N, 1] or list of residuals
        """
        # Convert residuals to single scalar per point if it's a list
        if isinstance(residuals, (list, tuple)):
            # Aggregate all residual components
            total_residual = sum(r.flatten() ** 2 for r in residuals)
            residual_values = torch.sqrt(total_residual)
        else:
            residual_values = residuals.flatten().abs()
        
        bin_idx = self.get_bin_indices(t_values)
        
        # Vectorized accumulation using bincount
        bin_sums = torch.bincount(bin_idx, weights=residual_values, minlength=self.num_bins)
        bin_counts = torch.bincount(bin_idx, minlength=self.num_bins).to(bin_sums.dtype)
        
        self.cumulative_residuals += bin_sums.detach()
        self.update_counts += bin_counts.detach()
    
    def get_adaptive_weights(self, t_values):
        """
        Compute adaptive causal weights based on cumulative past residuals.
        w_i = exp(-epsilon * Σ_{k=1}^{i-1} L_r(t_k))
        Vectorized for speed with normalization to prevent weight collapse.
        
        Args:
            t_values: Time values [N, 1]
        
        Returns:
            Weights [N, 1]
        """
        eps = 1e-12
        bin_idx = self.get_bin_indices(t_values)
        
        # Compute average residual per bin (point-averaged)
        avg_residuals = self.cumulative_residuals / (self.update_counts + eps)
        
        # Normalize by early-time scale (bin 0) to keep magnitude stable
        # This prevents the cumulative sum from growing too large with more bins
        ref_scale = avg_residuals[0].clamp_min(eps)
        avg_residuals_norm = avg_residuals / ref_scale
        
        # Compute cumulative sum of normalized average residuals
        cumsum_avg = torch.cumsum(avg_residuals_norm, dim=0)
        
        # For bin i, we want sum from bins 0 to i-1, so shift cumsum by 1
        # cumsum_shifted[i] = sum of bins 0 to i-1
        cumsum_shifted = torch.cat([torch.zeros(1, device=self.device), cumsum_avg[:-1]], dim=0)
        
        # Gather the appropriate cumulative sum for each point based on its bin
        bin_idx_flat = bin_idx.clamp(min=0, max=self.num_bins-1)
        past_residual_sum = cumsum_shifted[bin_idx_flat]
        
        # Apply exponential suppression with floor to prevent starving later times
        weights = torch.exp(-self.epsilon * past_residual_sum).clamp_min(0.05)
        
        return weights.unsqueeze(-1) if weights.dim() == 1 else weights
    
    def reset(self):
        """Reset cumulative residuals and counts."""
        self.cumulative_residuals.zero_()
        self.update_counts.zero_()
    
    def get_stats(self):
        """Get current statistics for logging."""
        avg_residuals = torch.where(
            self.update_counts > 0,
            self.cumulative_residuals / self.update_counts,
            torch.zeros_like(self.cumulative_residuals)
        )
        return {
            'cumulative': self.cumulative_residuals.cpu().numpy(),
            'counts': self.update_counts.cpu().numpy(),
            'average': avg_residuals.cpu().numpy()
        }

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

def initialize_shared_velocity_fields(lam, num_of_waves, v_1, seed=None):
    """
    Initialize shared velocity fields for consistent PINN/FD initial conditions.
    This should be called once at the beginning of training.
    
    IMPORTANT: The parameters used here (POWER_EXPONENT, v_1=a*cs, seed) MUST match
    the parameters used in FD plotting functions to ensure identical initial conditions.
    All FD visualization functions should use the same defaults.
    """
    global _shared_vx_interp, _shared_vy_interp
    
    if seed is None:
        seed = RANDOM_SEED
    
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

def generate_power_spectrum_field(lam, v_1, x, seed=None):
    '''Generate 2D Gaussian random field with power spectrum using shared fields if available'''
    
    if seed is None:
        seed = RANDOM_SEED
    
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

def generate_power_spectrum_field_vy(lam, v_1, x, seed=None):
    '''Generate vy component using shared fields if available'''
    
    if seed is None:
        seed = RANDOM_SEED
    
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
        return generate_power_spectrum_field(lam, v_1, x, seed=RANDOM_SEED)

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
        return generate_power_spectrum_field_vy(lam, v_1, x, seed=RANDOM_SEED)

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
        # Note: Domain collocation points now start from STARTUP_DT (set in data_generator.py)
        rho_r,vx_r,phi_r = pde_residue(colloc_shifted, net, dimension = 1)

    elif model.dimension == 2:
        # time is at index 2
        # Note: Domain collocation points now start from STARTUP_DT (set in data_generator.py)
        rho_r,vx_r,vy_r,phi_r = pde_residue(colloc_shifted, net, dimension = 2)

    elif model.dimension == 3:
        # time is at index 3
        # Note: Domain collocation points now start from STARTUP_DT (set in data_generator.py)
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
    
    # Create loss breakdown dictionary
    loss_breakdown = {}
    
    # IC losses (grouped together)
    ic_loss = mse_vx_ic.item()
    if isinstance(mse_rho_ic, torch.Tensor) and mse_rho_ic.item() > 0:
        ic_loss += mse_rho_ic.item()
    
    if model.dimension == 2:
        ic_loss += mse_vy_ic.item()
    elif model.dimension == 3:
        ic_loss += mse_vy_ic.item()
        ic_loss += mse_vz_ic.item()
    
    loss_breakdown['IC'] = ic_loss
    
    # Continuity loss
    if continuity_weight > 0 and continuity_ic_loss.item() > 0:
        loss_breakdown['Continuity'] = (continuity_weight * continuity_ic_loss).item()
    
    # PDE losses (grouped together)
    pde_loss = mse_rho.item() + mse_velx.item()
    
    if model.dimension == 2:
        pde_loss += mse_vely.item()
    elif model.dimension == 3:
        pde_loss += mse_vely.item()
        pde_loss += mse_velz.item()
    
    pde_loss += mse_phi.item()
    loss_breakdown['PDE'] = pde_loss
    
    return loss, loss_breakdown

def train(model, net, collocation_domain, collocation_IC, optimizer, optimizerL, iteration_adam, iterationL, mse_cost_function, closure, rho_1, lam, jeans, v_1, device, causal_gamma=0.0, causal_mode="none", residual_tracker=None):
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

        loss, loss_breakdown = optimizer.step(lambda: closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, continuity_weight, startup_dt, bs, nb, causal_gamma, causal_mode, residual_tracker, update_tracker=True))

        with torch.autograd.no_grad():
            if i % 200 == 0:
                print(f"Training Loss at {i} for Adam (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)
                # Print loss breakdown
                breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
                if breakdown_str:
                    print(f"  Loss breakdown: {breakdown_str}", flush=True)

    for i in range(iterationL):
        optimizer.zero_grad()
        global_step = iteration_adam + i
        continuity_weight = cosine_schedule(global_step, total_steps, CONTINUITY_IC_WEIGHT, 0.0)
        startup_dt = cosine_schedule(global_step, total_steps, STARTUP_DT, 0.0)

        # L-BFGS expects a closure that returns only scalar loss
        # Store loss_breakdown in a list so we can access it after the step
        loss_breakdown_holder = [None]
        
        def lbfgs_closure():
            loss, loss_breakdown = closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1, continuity_weight, startup_dt, bs, nb, causal_gamma, causal_mode, residual_tracker, update_tracker=False)
            loss_breakdown_holder[0] = loss_breakdown
            return loss
        
        loss = optimizerL.step(lbfgs_closure)
        loss_breakdown = loss_breakdown_holder[0]

        with torch.autograd.no_grad():
            if i % 20 == 0:
                print(f"Training Loss at {i} for LBGFS (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)
                # Print loss breakdown
                breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
                if breakdown_str:
                    print(f"  Loss breakdown: {breakdown_str}", flush=True)


def _random_batch_indices(total_count, batch_size, device):
    actual = int(min(batch_size, total_count))
    return torch.randperm(total_count, device=device)[:actual]


def _make_batch_tensors(tensors_list, indices):
    """Create batch tensors by indexing. No cloning for speed."""
    return [t[indices] for t in tensors_list]


def closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                    rho_1, lam, jeans, v_1, continuity_weight, startup_dt, batch_size, num_batches, causal_gamma=0.0, causal_mode="none", residual_tracker=None, update_tracker=True):

    def _causal_weight_static(t_values, gamma):
        """Compute static causal weights: w(t) = exp(-gamma * t)"""
        if gamma == 0.0:
            return torch.ones_like(t_values)
        return torch.exp(-gamma * torch.clamp(t_values, min=0.0))

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

        # Extract time values from batch_dom
        if model.dimension == 1:
            t_dom = batch_dom[1]
        elif model.dimension == 2:
            t_dom = batch_dom[2]
        elif model.dimension == 3:
            t_dom = batch_dom[3]

        # Compute causal weights for PDE residuals based on mode
        causal_weights = None
        if causal_mode == "static" and causal_gamma > 0.0:
            # Static exponential weighting
            causal_weights = _causal_weight_static(t_dom, causal_gamma)
        elif causal_mode == "adaptive" and residual_tracker is not None:
            # Adaptive residual-based weighting
            # Get adaptive weights based on PAST residuals (before updating)
            # This ensures weights reflect only history up to previous iterations
            causal_weights = residual_tracker.get_adaptive_weights(t_dom)
        
        # Apply weights to PDE residuals
        if causal_weights is not None:
            mse_rho  = torch.mean(causal_weights * (rho_r ** 2))
            mse_velx = torch.mean(causal_weights * (vx_r ** 2))
            if model.dimension == 2:
                mse_vely = torch.mean(causal_weights * (vy_r ** 2))
            elif model.dimension == 3:
                mse_vely = torch.mean(causal_weights * (vy_r ** 2))
                mse_velz = torch.mean(causal_weights * (vz_r ** 2))
            mse_phi  = torch.mean(causal_weights * (phi_r ** 2))
        else:
            # Standard uniform weighting
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

        # Update residual tracker AFTER computing loss (only during Adam)
        # This ensures the tracker uses only past history for weight computation
        if causal_mode == "adaptive" and residual_tracker is not None and update_tracker:
            with torch.no_grad():
                if model.dimension == 1:
                    residual_tracker.update_residuals(t_dom, [rho_r, vx_r, phi_r])
                elif model.dimension == 2:
                    residual_tracker.update_residuals(t_dom, [rho_r, vx_r, vy_r, phi_r])
                else:
                    residual_tracker.update_residuals(t_dom, [rho_r, vx_r, vy_r, vz_r, phi_r])

        total_loss = total_loss + loss
        num_effective_batches += 1

    optimizer.zero_grad()
    avg_loss = total_loss / max(1, num_effective_batches)
    avg_loss.backward(retain_graph=True)
    
    # Create loss breakdown dictionary (averaged across batches)
    loss_breakdown = {}
    
    # For batched version, we need to compute breakdown from the last batch
    # This is an approximation since we can't easily track individual terms across batches
    # IC losses (grouped together)
    ic_loss = mse_vx_ic.item()
    if isinstance(mse_rho_ic, torch.Tensor) and mse_rho_ic.item() > 0:
        ic_loss += mse_rho_ic.item()
    
    if model.dimension == 2:
        ic_loss += mse_vy_ic.item()
    elif model.dimension == 3:
        ic_loss += mse_vy_ic.item()
        ic_loss += mse_vz_ic.item()
    
    loss_breakdown['IC'] = ic_loss
    
    # Continuity loss
    if continuity_weight > 0 and continuity_ic_loss.item() > 0:
        loss_breakdown['Continuity'] = (continuity_weight * continuity_ic_loss).item()
    
    # PDE losses (grouped together)
    pde_loss = mse_rho.item() + mse_velx.item()
    
    if model.dimension == 2:
        pde_loss += mse_vely.item()
    elif model.dimension == 3:
        pde_loss += mse_vely.item()
        pde_loss += mse_velz.item()
    
    pde_loss += mse_phi.item()
    loss_breakdown['PDE'] = pde_loss
    
    return avg_loss, loss_breakdown


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
                   optimizer, subdomain_devices=None, cached_ic_values=None):
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
        subdomain_devices: List of devices per subdomain
        cached_ic_values: Precomputed IC values
    
    Returns:
        Total loss (scalar tensor), loss_dict
    """
    optimizer.zero_grad()
    
    # Compute total XPINN loss (passing cached IC if available)
    total_loss, loss_dict = xpinn_loss_model.compute_total_loss(
        nets, subdomain_collocs, interface_collocs,
        subdomain_ic_collocs, ic_functions, interfaces,
        exterior_boundaries, cached_ic_values
    )
    
    # Backward pass
    total_loss.backward(retain_graph=True)
    
    return total_loss, loss_dict


def closure_xpinn_batched(xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                           subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                           optimizer, batch_size, num_batches, subdomain_devices=None, cached_ic_values=None):
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
        subdomain_devices: List of devices per subdomain
        cached_ic_values: Precomputed IC values
    
    Returns:
        Total loss (scalar tensor), loss_dict
    """
    if subdomain_devices is None:
        subdomain_devices = [subdomain_collocs[0][0].device] * len(nets)
    if cached_ic_values is None:
        cached_ic_values = [None] * len(nets)
    optimizer.zero_grad()
    
    # Aggregate losses across mini-batches
    total_loss = 0.0
    num_effective_batches = 0
    
    # Get device from first subdomain's collocation points
    device = subdomain_collocs[0][0].device if len(subdomain_collocs) > 0 else 'cuda'
    
    for batch_idx in range(int(max(1, num_batches))):
        # Create batched subdomain collocation points (on correct devices)
        batched_subdomain_collocs = []
        for i, subdomain_colloc in enumerate(subdomain_collocs):
            dom_n = subdomain_colloc[0].size(0)
            sub_device = subdomain_devices[i]
            dom_idx = _random_batch_indices(dom_n, batch_size, sub_device)
            batch_dom = _make_batch_tensors(subdomain_colloc, dom_idx)
            batched_subdomain_collocs.append(batch_dom)
        
        # Create batched subdomain IC points and batched cached IC values
        batched_subdomain_ic_collocs = []
        batched_cached_ic = []
        for i, subdomain_ic_colloc in enumerate(subdomain_ic_collocs):
            ic_n = subdomain_ic_colloc[0].size(0)
            sub_device = subdomain_devices[i]
            ic_idx = _random_batch_indices(ic_n, batch_size, sub_device)
            batch_ic = _make_batch_tensors(subdomain_ic_colloc, ic_idx)
            batched_subdomain_ic_collocs.append(batch_ic)
            
            # Batch cached IC values if available
            if cached_ic_values[i] is not None:
                batched_ic_cache = {
                    key: val[ic_idx] for key, val in cached_ic_values[i].items()
                }
                batched_cached_ic.append(batched_ic_cache)
            else:
                batched_cached_ic.append(None)
        
        # Create batched interface collocation points (use device of first subdomain in pair)
        batched_interface_collocs = {}
        for interface_key, interface_colloc in interface_collocs.items():
            if_n = interface_colloc[0].size(0)
            # Use device of first subdomain in the interface pair
            sub_device = subdomain_devices[interface_key[0]]
            if_idx = _random_batch_indices(if_n, batch_size, sub_device)
            
            # Move interface collocation points to the correct device before indexing
            interface_colloc_on_device = [t.to(sub_device) for t in interface_colloc]
            batch_if = _make_batch_tensors(interface_colloc_on_device, if_idx)
            batched_interface_collocs[interface_key] = batch_if
        
        # Compute loss for this mini-batch (with cached IC if available)
        batch_loss, batch_loss_dict = xpinn_loss_model.compute_total_loss(
            nets, batched_subdomain_collocs, batched_interface_collocs,
            batched_subdomain_ic_collocs, ic_functions, interfaces,
            exterior_boundaries, batched_cached_ic
        )
        
        # Accumulate loss
        total_loss += batch_loss
        num_effective_batches += 1
    
    # Average loss across batches
    if num_effective_batches > 0:
        total_loss = total_loss / num_effective_batches
    
    # Backward pass
    total_loss.backward(retain_graph=True)
    
    # For batched XPINN, we need to compute breakdown from the last batch
    # This is an approximation since we can't easily track individual terms across batches
    # Note: batch_loss_dict is already computed from the last batch iteration above
    
    return total_loss, batch_loss_dict


def train_xpinn(nets, subdomain_collocs, interface_collocs, subdomain_ic_collocs,
                ic_functions, interfaces, exterior_boundaries, xpinn_loss_model,
                optimizer, optimizerL, iteration_adam, iterationL, device,
                subdomain_devices=None, cached_ic_values=None):
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
        subdomain_devices: List of devices per subdomain (for multi-GPU)
        cached_ic_values: Precomputed IC values (list of dicts per subdomain)
    
    Returns:
        None (trains networks in-place)
    """
    # Set defaults
    if subdomain_devices is None:
        subdomain_devices = [device] * len(nets)
    if cached_ic_values is None:
        cached_ic_values = [None] * len(nets)
    from config import XPINN_ALTERNATING_TRAINING, USE_XPINN_BATCHING, BATCH_SIZE, NUM_BATCHES
    
    # Determine if we need per-subdomain optimizers (multi-GPU case)
    use_per_subdomain_adam = (optimizer is None)
    
    if use_per_subdomain_adam:
        # Multi-GPU: create separate Adam optimizer for each subdomain
        print("Creating per-subdomain Adam optimizers for multi-GPU training...")
        adam_optimizers = [torch.optim.Adam(net.parameters(), lr=0.001) for net in nets]
    else:
        # Single-GPU: use unified optimizer
        adam_optimizers = None
        
        # Choose closure function based on batching setting
        if USE_XPINN_BATCHING:
            bs_global = int(BATCH_SIZE)
            nb = int(NUM_BATCHES)
            num_sub = len(nets)
            bs = max(1, bs_global // max(1, num_sub))  # per-subdomain batch size
            
            def make_closure(opt):
                return lambda: closure_xpinn_batched(
                    xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                    subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                    opt, bs, nb, subdomain_devices, cached_ic_values
                )
        else:
            def make_closure(opt):
                return lambda: closure_xpinn(
                    xpinn_loss_model, nets, subdomain_collocs, interface_collocs,
                    subdomain_ic_collocs, ic_functions, interfaces, exterior_boundaries,
                    opt, subdomain_devices, cached_ic_values
                )
    
    # Adam training
    for i in range(iteration_adam):
        if use_per_subdomain_adam:
            # Per-subdomain Adam training (multi-GPU)
            # Process each subdomain completely independently to avoid graph sharing issues
            subdomain_losses = []
            
            # Track component losses across all subdomains
            total_pde_loss = 0.0
            total_ic_loss = 0.0
            total_interface_sol_loss = 0.0
            total_interface_res_loss = 0.0
            
            for sub_idx in range(len(nets)):
                try:
                    adam_optimizers[sub_idx].zero_grad()
                    
                    # Compute PDE and IC loss for this subdomain
                    # Create fresh copies of collocation points to avoid graph reuse across iterations
                    colloc_orig = subdomain_collocs[sub_idx]
                    colloc_ic_orig = subdomain_ic_collocs[sub_idx]
                    
                    colloc = [t.clone().detach().requires_grad_(True) for t in colloc_orig]
                    colloc_ic = [t.clone().detach().requires_grad_(False) for t in colloc_ic_orig]
                    
                    # Detach cached IC values to avoid graph reuse across iterations
                    cached_ic = None
                    if cached_ic_values and cached_ic_values[sub_idx] is not None:
                        cached_ic = {
                            key: val.detach() for key, val in cached_ic_values[sub_idx].items()
                        }
                    
                    pde_loss = xpinn_loss_model.compute_pde_loss(colloc, nets[sub_idx])
                    ic_loss = xpinn_loss_model.compute_ic_loss(colloc_ic, nets[sub_idx], ic_functions, cached_ic)
                    
                    total_loss = pde_loss + ic_loss
                    
                    # Track component losses
                    total_pde_loss += pde_loss.item()
                    total_ic_loss += ic_loss.item()
                    
                    # Accumulate all interface losses for this subdomain
                    interface_sol_loss_sum = 0.0
                    interface_res_loss_sum = 0.0
                    for interface in interfaces:
                        subdomain_i, subdomain_j, _, _ = interface
                        interface_key = (subdomain_i, subdomain_j)
                        
                        # Only compute interface losses where this subdomain participates
                        if sub_idx == subdomain_i or sub_idx == subdomain_j:
                            if interface_key in interface_collocs:
                                # Create fresh copies of interface collocation points to avoid graph sharing
                                colloc_interface = interface_collocs[interface_key]
                                colloc_interface_fresh = [t.clone().detach().requires_grad_(False) for t in colloc_interface]
                                
                                net1 = nets[subdomain_i]
                                net2 = nets[subdomain_j]
                                
                                # Backprop only to current subdomain's network
                                if sub_idx == subdomain_i:
                                    sol_loss = xpinn_loss_model.compute_interface_solution_loss(
                                        colloc_interface_fresh, net1, net2, grad_to='net1')
                                    res_loss = xpinn_loss_model.compute_interface_residual_loss(
                                        colloc_interface_fresh, net1, net2, grad_to='net1')
                                else:  # sub_idx == subdomain_j
                                    sol_loss = xpinn_loss_model.compute_interface_solution_loss(
                                        colloc_interface_fresh, net1, net2, grad_to='net2')
                                    res_loss = xpinn_loss_model.compute_interface_residual_loss(
                                        colloc_interface_fresh, net1, net2, grad_to='net2')
                                
                                interface_sol_loss_sum += sol_loss.item()
                                interface_res_loss_sum += res_loss.item()
                                
                                total_loss = total_loss + sol_loss + res_loss
                    
                    # Track interface losses
                    total_interface_sol_loss += interface_sol_loss_sum
                    total_interface_res_loss += interface_res_loss_sum
                    
                    # Single backward pass for this subdomain
                    total_loss.backward()
                    adam_optimizers[sub_idx].step()
                    subdomain_losses.append(total_loss.item())
                    
                except Exception as e:
                    print(f"Error in subdomain {sub_idx}: {e}")
                    raise
            
            if i % 200 == 0:
                avg_loss = sum(subdomain_losses) / len(subdomain_losses)
                # Average component losses across subdomains
                avg_pde = total_pde_loss / len(nets)
                avg_ic = total_ic_loss / len(nets)
                avg_if_sol = total_interface_sol_loss / len(nets)
                avg_if_res = total_interface_res_loss / len(nets)
                
                print(f"XPINN Training Loss at {i} (Adam, per-subdomain) = {avg_loss:.2e}", flush=True)
                print(f"  Component losses: PDE={avg_pde:.2e} | IC={avg_ic:.2e} | IF_sol={avg_if_sol:.2e} | IF_res={avg_if_res:.2e}", flush=True)
        else:
            # Unified Adam training (single-GPU, original approach)
            if XPINN_ALTERNATING_TRAINING:
                # Alternate between subdomains (not yet implemented - would need separate optimizers)
                raise NotImplementedError("Alternating training not yet implemented")
            else:
                # Train all networks simultaneously
                loss, loss_dict = optimizer.step(make_closure(optimizer))
            
            if i % 200 == 0:
                with torch.autograd.no_grad():
                    print(f"XPINN Training Loss at {i} (Adam) = {loss.item():.2e}", flush=True)
                    # Print loss breakdown
                    breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_dict.items() if v > 0])
                    if breakdown_str:
                        print(f"  Loss breakdown: {breakdown_str}", flush=True)
    
    # L-BFGS training - per-subdomain optimization
    # Each subdomain network is optimized separately while others are frozen
    # This avoids multi-GPU gradient gathering issues and memory constraints
    if iterationL > 0:
        print("\nStarting per-subdomain L-BFGS training...")
        
        # Optimize each subdomain network separately
        for subdomain_idx in range(len(nets)):
            print(f"  Optimizing subdomain {subdomain_idx}...")
            
            net = nets[subdomain_idx]
            net_device = subdomain_devices[subdomain_idx] if subdomain_devices else device
            
            # Create optimizer for this subdomain only
            optimizer_sub = torch.optim.LBFGS(
                net.parameters(),
                lr=1.0,
                max_iter=20,
                max_eval=None,
                tolerance_grad=1e-11,
                tolerance_change=1e-11,
                history_size=200,
                line_search_fn='strong_wolfe'
            )
            
            # Create closure for this subdomain
            # Computes loss for this subdomain + its interface losses
            def make_subdomain_lbfgs_closure(sub_idx, current_net):
                def subdomain_closure():
                    optimizer_sub.zero_grad()
                    
                    # Compute PDE and IC loss for this subdomain
                    colloc = subdomain_collocs[sub_idx]
                    colloc_ic = subdomain_ic_collocs[sub_idx]
                    cached_ic = cached_ic_values[sub_idx] if cached_ic_values else None
                    
                    pde_loss = xpinn_loss_model.compute_pde_loss(colloc, current_net)
                    ic_loss = xpinn_loss_model.compute_ic_loss(colloc_ic, current_net, ic_functions, cached_ic)
                    
                    total_loss = pde_loss + ic_loss
                    
                    # Add interface losses where this subdomain is involved
                    for interface in interfaces:
                        subdomain_i, subdomain_j, _, _ = interface
                        interface_key = (subdomain_i, subdomain_j)
                        
                        # Only compute interface losses where this subdomain participates
                        if sub_idx == subdomain_i or sub_idx == subdomain_j:
                            if interface_key in interface_collocs:
                                colloc_interface = interface_collocs[interface_key]
                                net1 = nets[subdomain_i]
                                net2 = nets[subdomain_j]
                                
                                # Compute interface losses (gradients only flow to this net)
                                sol_loss = xpinn_loss_model.compute_interface_solution_loss(
                                    colloc_interface, net1, net2)
                                res_loss = xpinn_loss_model.compute_interface_residual_loss(
                                    colloc_interface, net1, net2)
                                
                                # Move interface losses to current net's device before adding
                                current_device = next(current_net.parameters()).device
                                sol_loss = sol_loss.to(current_device)
                                res_loss = res_loss.to(current_device)
                                
                                total_loss = total_loss + sol_loss + res_loss
                    
                    # L-BFGS will call this closure multiple times for line search
                    # Use retain_graph=True to allow multiple backward passes
                    total_loss.backward(retain_graph=True)
                    return total_loss
                
                return subdomain_closure
            
            # Helper to compute and print a detailed loss breakdown for this subdomain (no grad)
            def _lbfgs_subdomain_breakdown(sub_idx, current_net):
                with torch.no_grad():
                    colloc = subdomain_collocs[sub_idx]
                    colloc_ic = subdomain_ic_collocs[sub_idx]
                    cached_ic = cached_ic_values[sub_idx] if cached_ic_values else None
                    pde_loss = xpinn_loss_model.compute_pde_loss(colloc, current_net)
                    ic_loss = xpinn_loss_model.compute_ic_loss(colloc_ic, current_net, ic_functions, cached_ic)
                    if len(nets) > 1:
                        device0 = next(current_net.parameters()).device
                        pde_loss = pde_loss.to(device0)
                        ic_loss = ic_loss.to(device0)
                    if_sol = torch.tensor(0.0, device=next(current_net.parameters()).device)
                    if_res = torch.tensor(0.0, device=next(current_net.parameters()).device)
                    for interface in interfaces:
                        subdomain_i, subdomain_j, _, _ = interface
                        interface_key = (subdomain_i, subdomain_j)
                        if sub_idx == subdomain_i or sub_idx == subdomain_j:
                            if interface_key in interface_collocs:
                                colloc_interface = interface_collocs[interface_key]
                                net1 = nets[subdomain_i]
                                net2 = nets[subdomain_j]
                                sol_loss = xpinn_loss_model.compute_interface_solution_loss(colloc_interface, net1, net2)
                                res_loss = xpinn_loss_model.compute_interface_residual_loss(colloc_interface, net1, net2)
                                current_device = next(current_net.parameters()).device
                                if_sol = if_sol + sol_loss.to(current_device)
                                if_res = if_res + res_loss.to(current_device)
                    total = pde_loss + ic_loss + if_sol + if_res
                    return total.item(), pde_loss.item(), ic_loss.item(), if_sol.item(), if_res.item()

            # Run L-BFGS for this subdomain
            for i in range(iterationL):
                loss = optimizer_sub.step(make_subdomain_lbfgs_closure(subdomain_idx, net))
                
                if i % 20 == 0:
                    total_v, pde_v, ic_v, if_sol_v, if_res_v = _lbfgs_subdomain_breakdown(subdomain_idx, net)
                    print(
                        f"    Subdomain {subdomain_idx} L-BFGS step {i}: "
                        f"Total={total_v:.2e} | PDE={pde_v:.2e} | IC={ic_v:.2e} | "
                        f"IF_sol={if_sol_v:.2e} | IF_res={if_res_v:.2e}",
                        flush=True
                    )
        
        print("Per-subdomain L-BFGS training completed.")
    
    print("XPINN training completed.")