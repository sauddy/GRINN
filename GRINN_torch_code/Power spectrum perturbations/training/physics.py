import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from core.losses import ASTPN, pde_residue
from utilities.training_diagnostics import TrainingDiagnostics
from core.data_generator import diff
from core.model_architecture import PINN
from methods.causal_training import compute_causal_weights_static
from core.initial_conditions import (initialize_shared_velocity_fields, generate_power_spectrum_field, 
                                     generate_power_spectrum_field_vy, fun_rho_0, fun_vx_0, fun_vy_0, func)
from config import cs, const, G, rho_o, CONTINUITY_IC_WEIGHT, STARTUP_DT, DECAY_PORTION, PERTURBATION_TYPE, KX, KY, BATCH_SIZE, NUM_BATCHES, RANDOM_SEED
from config import IC_WEIGHT, ENABLE_TRAINING_DIAGNOSTICS


# ==================== Physics Calculations and Loss Functions ====================

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
            # For standard density: rho_t ≈ -rho_o * ∇·v at t=0
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
            # For standard density: rho_t ≈ -rho_o * ∇·v at t=0
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
            # For standard density: rho_t ≈ -rho_o * ∇·v at t=0
            rho0_field = rho_o * torch.ones_like(div_v0)
            continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)

    ############## Loss based on PDE ###################################
    
    # Apply startup time offset to PDE collocation time only (IC remains at t=0)
    if isinstance(collocation_domain, (list, tuple)):
        colloc_shifted = list(collocation_domain)
    else:
        # Single tensor format: split into [x, y, t]
        colloc_shifted = [collocation_domain[:, i:i+1] for i in range(collocation_domain.shape[1])]

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

    ic_weight = IC_WEIGHT

    ################### Combining the loss functions ####################
    if model.dimension == 1:
        base = ic_weight * mse_vx_ic + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    elif model.dimension == 2:
        base = ic_weight * (mse_vx_ic + mse_vy_ic) + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_phi
        loss = base + (mse_rho_ic if isinstance(mse_rho_ic, torch.Tensor) else 0.0)

    elif model.dimension == 3:
        base = ic_weight * (mse_vx_ic + mse_vy_ic + mse_vz_ic) + continuity_weight * continuity_ic_loss + mse_rho + mse_velx + mse_vely + mse_velz + mse_phi
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

# Note: train() and train_xpinn() functions have been moved to training/trainer.py

def _random_batch_indices(total_count, batch_size, device):
    actual = int(min(batch_size, total_count))
    return torch.randperm(total_count, device=device)[:actual]


def _make_batch_tensors(tensors_list, indices):
    """Create batch tensors by indexing. No cloning for speed."""
    return [t[indices] for t in tensors_list]


def closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer,
                    rho_1, lam, jeans, v_1, continuity_weight, startup_dt, batch_size, num_batches, causal_gamma=0.0, causal_mode="none", residual_tracker=None, update_tracker=True, iteration=0, use_fft_poisson=None):
    """
    Batched closure function for training with optional causal weighting.
    
    This function computes losses over mini-batches and optionally applies causal weights.
    """
    # Aggregate losses across mini-batches
    total_loss = 0.0
    num_effective_batches = 0

    # Determine counts and devices
    # Handle both formats: list of tensors [x, y, t] or single tensor [N, 3]
    if isinstance(collocation_domain, (list, tuple)):
        dom_n = collocation_domain[0].size(0)
        device = collocation_domain[0].device
    else:
        dom_n = collocation_domain.size(0)
        device = collocation_domain.device
    
    ic_n = collocation_IC[0].size(0)

    for _ in range(int(max(1, num_batches))):
        dom_idx = _random_batch_indices(dom_n, batch_size, device)
        ic_idx = _random_batch_indices(ic_n, batch_size, device)

        # Handle both formats for collocation_domain
        if isinstance(collocation_domain, (list, tuple)):
            batch_dom = _make_batch_tensors(collocation_domain, dom_idx)
        else:
            # Single tensor format: split into [x, y, t]
            batch_dom = [collocation_domain[dom_idx, i:i+1] for i in range(collocation_domain.shape[1])]
        
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
                # For standard density: rho_t ≈ -rho_o * ∇·v at t=0
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
                # For standard density: rho_t ≈ -rho_o * ∇·v at t=0
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
                # For standard density: rho_t ≈ -rho_o * ∇·v at t=0
                rho0_field = rho_o * torch.ones_like(div_v0)
                continuity_ic_loss = mse_cost_function(rho_t_ic, -rho0_field * div_v0)

        # PDE residuals on batched domain with startup shift
        if isinstance(batch_dom, (list, tuple)):
            colloc_shifted = list(batch_dom)
        else:
            # Single tensor format: split into [x, y, t]
            colloc_shifted = [batch_dom[:, i:i+1] for i in range(batch_dom.shape[1])]
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
            causal_weights = compute_causal_weights_static(t_dom, causal_gamma)
        elif causal_mode == "adaptive" and residual_tracker is not None:
            # Adaptive residual-based weighting
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
    
    # Aggressive memory cleanup after backward pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
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


# ==================== XPINN Physics Calculations ====================

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
