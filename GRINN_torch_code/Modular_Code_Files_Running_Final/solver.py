import numpy as np
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from losses import ASTPN, pde_residue
from model_architecture import PINN
from config import cs, const, G, rho_o

def input_taker(lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r):
    lam = float(lam)
    rho_1 = float(rho_1)
    num_of_waves = int(num_of_waves)  
    tmax = float(tmax)
    N_0 = int(N_0)
    N_b = int(N_b)
    N_r = int(N_r)
    
    return lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r

def req_consts_calc(lam):

    jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_o))

    if lam > jeans:
        alpha = np.sqrt(const*G*rho_o-cs**2*(2*np.pi/lam)**2)
    else:
        alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*rho_o)

    return jeans, alpha

def fun_rho_0(lam, x, alpha):
    """
    Compute initial density with proper tensor sizes
    """
    x_input = x[0]
    alpha = alpha
    
    # Ensure inputs have the same size
    assert x_input.size() == alpha.size(), f"Size mismatch in fun_rho_0: x_input {x_input.size()} vs alpha {alpha.size()}"
    
    rho_0 = rho_o + alpha * torch.cos(2*np.pi*x_input/lam)
    return rho_0

def fun_v_0(lam, jeans, x, v_1):
    """
    Compute initial velocity with proper tensor sizes
    """
    x_input = x[0]
    v_1 = v_1
    
    # Ensure inputs have the same size
    assert x_input.size() == v_1.size(), f"Size mismatch in fun_v_0: x_input {x_input.size()} vs v_1 {v_1.size()}"
    
    if lam > jeans:
        v_0 = - v_1 * torch.sin(2*np.pi*x_input/lam)
    else:
        v_0 = v_1 * torch.cos(2*np.pi*x_input/lam)
    return v_0

def func(x):
    return x[0]*0

from tqdm import tqdm

start = time.time()

def process_batch(batch_size, collocation_domain, collocation_IC, alpha, net, model, 
                  lam, jeans, v_1, mse_cost_function, alpha_idx, alpha_val):
    """
    Process a single batch of data
    """
    # Get total number of points
    total_domain_points = collocation_domain[0].size(0)
    total_ic_points = collocation_IC[0].size(0)
    num_alphas = alpha.size(0)
    
    # Use the smaller of batch_size and total_ic_points to ensure we don't exceed available data
    actual_batch_size = min(batch_size, total_ic_points)
    
    # Create batch indices
    domain_indices = torch.randperm(total_domain_points)[:actual_batch_size]
    ic_indices = torch.randperm(total_ic_points)[:actual_batch_size]
    
    # Get batch data
    batch_domain = [t[domain_indices].clone() for t in collocation_domain]
    batch_ic = [t[ic_indices].clone() for t in collocation_IC]
    
    # Prepare network inputs
    if model.dimension == 1:
        x_ic = batch_ic[0]
        t_ic = batch_ic[1]
    elif model.dimension == 2:
        x_ic = batch_ic[0]
        y_ic = batch_ic[1]
        t_ic = batch_ic[2]
    elif model.dimension == 3:
        x_ic = batch_ic[0]
        y_ic = batch_ic[1]
        z_ic = batch_ic[2]
        t_ic = batch_ic[3]
    
    # Create network inputs for the selected alpha value
    alpha_input = torch.full((actual_batch_size, 1), alpha_val, device=x_ic.device, dtype=x_ic.dtype)
    v1_input = torch.full((actual_batch_size, 1), v_1[alpha_idx].item(), device=x_ic.device, dtype=x_ic.dtype)
    
    # Network inputs
    if model.dimension == 1:
        net_ic_inputs = [x_ic, t_ic, alpha_input]
        net_dom_inputs = [batch_domain[0], batch_domain[1], alpha_input]
    elif model.dimension == 2:
        net_ic_inputs = [x_ic, y_ic, t_ic, alpha_input]
        net_dom_inputs = [batch_domain[0], batch_domain[1], batch_domain[2], alpha_input]
    elif model.dimension == 3:
        net_ic_inputs = [x_ic, y_ic, z_ic, t_ic, alpha_input]
        net_dom_inputs = [batch_domain[0], batch_domain[1], batch_domain[2], batch_domain[3], alpha_input]
    
    # Compute network outputs
    net_ic_output = net(net_ic_inputs)
    
    # Compute targets
    if model.dimension == 1:
        rho_0 = fun_rho_0(lam, [x_ic, t_ic], alpha_input)
        vx_0 = fun_v_0(lam, jeans, [x_ic, t_ic], v1_input)
    elif model.dimension == 2:
        rho_0 = fun_rho_0(lam, [x_ic, y_ic, t_ic], alpha_input)
        vx_0 = fun_v_0(lam, jeans, [x_ic, y_ic, t_ic], v1_input)
        vy_0 = func([x_ic, y_ic, t_ic])
    elif model.dimension == 3:
        rho_0 = fun_rho_0(lam, [x_ic, y_ic, z_ic, t_ic], alpha_input)
        vx_0 = fun_v_0(lam, jeans, [x_ic, y_ic, z_ic, t_ic], v1_input)
        vy_0 = func([x_ic, y_ic, z_ic, t_ic])
        vz_0 = func([x_ic, y_ic, z_ic, t_ic])
    
    # Extract outputs
    if model.dimension == 1:
        rho_ic_out = net_ic_output[:,0:1]
        vx_ic_out = net_ic_output[:,1:2]
    elif model.dimension == 2:
        rho_ic_out = net_ic_output[:,0:1]
        vx_ic_out = net_ic_output[:,1:2]
        vy_ic_out = net_ic_output[:,2:3]
    elif model.dimension == 3:
        rho_ic_out = net_ic_output[:,0:1]
        vx_ic_out = net_ic_output[:,1:2]
        vy_ic_out = net_ic_output[:,2:3]
        vz_ic_out = net_ic_output[:,3:4]
    
    # Compute losses for this batch
    mse_rho_ic = mse_cost_function(rho_ic_out, rho_0)
    mse_vx_ic = mse_cost_function(vx_ic_out, vx_0)
    if model.dimension == 2:
        mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
    elif model.dimension == 3:
        mse_vy_ic = mse_cost_function(vy_ic_out, vy_0)
        mse_vz_ic = mse_cost_function(vz_ic_out, vz_0)
    
    # Compute PDE residuals
    if model.dimension == 1:
        residuals = pde_residue(net_dom_inputs, net, dimension=1)
    elif model.dimension == 2:
        residuals = pde_residue(net_dom_inputs, net, dimension=2)
    elif model.dimension == 3:
        residuals = pde_residue(net_dom_inputs, net, dimension=3)
    
    # Extract PDE residuals
    if model.dimension == 1:
        rho_r, vx_r, phi_r = residuals
    elif model.dimension == 2:
        rho_r, vx_r, vy_r, phi_r = residuals
    elif model.dimension == 3:
        rho_r, vx_r, vy_r, vz_r, phi_r = residuals
    
    # Compute PDE losses
    mse_rho = torch.mean(rho_r ** 2)
    mse_velx = torch.mean(vx_r ** 2)
    if model.dimension == 2:
        mse_vely = torch.mean(vy_r ** 2)
    elif model.dimension == 3:
        mse_vely = torch.mean(vy_r ** 2)
        mse_velz = torch.mean(vz_r ** 2)
    mse_phi = torch.mean(phi_r ** 2)
    
    # Create alpha tensor for boundary conditions
    alpha_bc = torch.tensor([[alpha_val]], device=x_ic.device, dtype=x_ic.dtype)  # [1, 1]
    
    # Compute boundary conditions
    rhox_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=0)
    vx_xb = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=1)
    phi_xb = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=2)
    phi_xx_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=1, component=2)
    if model.dimension == 2:
        vy_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=2)
    elif model.dimension == 3:
        vy_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=2)
        vz_b = model.periodic_BC(net, alpha_bc, 1, coordinate=1, derivative_order=0, component=3)
    
    # Combine losses
    if model.dimension == 1:
        ic_loss = mse_rho_ic + mse_vx_ic
        bc_loss = rhox_b + vx_xb + phi_xb + phi_xx_b
        pde_loss = mse_rho + mse_velx + mse_phi
    elif model.dimension == 2:
        ic_loss = mse_rho_ic + mse_vx_ic + mse_vy_ic
        bc_loss = rhox_b + vx_xb + phi_xb + phi_xx_b + vy_b
        pde_loss = mse_rho + mse_velx + mse_vely + mse_phi
    elif model.dimension == 3:
        ic_loss = mse_rho_ic + mse_vx_ic + mse_vy_ic + mse_vz_ic
        bc_loss = rhox_b + vx_xb + phi_xb + phi_xx_b + vy_b + vz_b
        pde_loss = mse_rho + mse_velx + mse_vely + mse_velz + mse_phi

    return ic_loss, bc_loss, pde_loss

def closure_batched(model, net, alpha, mse_cost_function, collocation_domain, collocation_IC, 
                    optimizer, lam, jeans, v_1, w_IC, w_BC, w_PDE, batch_size, num_batches):
    """
    Closure function for batched training with stochastic alpha sampling
    """
    optimizer.zero_grad()
    
    # Randomly sample alpha indices for this optimization step
    num_alphas = alpha.size(0)
    if num_batches >= num_alphas:
        # If num_batches >= total alphas, use all alphas
        alpha_indices = torch.arange(num_alphas)
    else:
        # Otherwise, randomly sample without replacement
        alpha_indices = torch.randperm(num_alphas)[:num_batches]
    
    total_ic_loss = 0
    total_bc_loss = 0
    total_pde_loss = 0
    
    for batch_idx, alpha_idx in enumerate(alpha_indices):
        alpha_val = alpha[alpha_idx].item()
        ic_loss, bc_loss, pde_loss = process_batch(
            batch_size, collocation_domain, collocation_IC, alpha,
            net, model, lam, jeans, v_1, mse_cost_function, alpha_idx, alpha_val
        )
        
        total_ic_loss += ic_loss
        total_bc_loss += bc_loss
        total_pde_loss += pde_loss
    
    # Average losses across batches
    ic_loss = total_ic_loss / len(alpha_indices)
    bc_loss = total_bc_loss / len(alpha_indices)
    pde_loss = total_pde_loss / len(alpha_indices)
    
    # Compute total loss with weights
    loss = w_IC * ic_loss + w_BC * bc_loss + w_PDE * pde_loss
    
    # Backward pass
    loss.backward()
    
    return loss.item(), ic_loss.item(), bc_loss.item(), pde_loss.item()

def train_batched(net, model, alpha, collocation_domain, collocation_IC, optimizer, optimizerL, closure, mse_cost_function, 
                  iteration_adam, iterationL, lam, jeans, v_1, batch_size, num_batches):
    """
    Training function with batched processing
    """
    # Enable debug prints only for the first iteration
    net.debug_print = True
    model.debug_print = True
    
    # Training loop for Adam
    for i in range(iteration_adam):
        # Disable debug prints after first iteration
        if i == 1:
            net.debug_print = False
            model.debug_print = False
            
        # Update weights based on iteration
        if i < 200:
            w_IC, w_BC, w_PDE = 1, 1, 2
        elif 200 <= i < 400:
            w_IC, w_BC, w_PDE = 1, 1, 4
        elif 400 <= i < 600:
            w_IC, w_BC, w_PDE = 1, 1, 6
        elif 600 <= i < 800:
            w_IC, w_BC, w_PDE = 1, 1, 8
        else:
            w_IC, w_BC, w_PDE = 1, 1, 10
        
        # Compute loss and update parameters
        loss, ic_loss, bc_loss, pde_loss = closure_batched(
            model, net, alpha, mse_cost_function, collocation_domain,
            collocation_IC, optimizer, lam, jeans, v_1,
            w_IC, w_BC, w_PDE, batch_size, num_batches
        )
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Training Loss at {i} for Adam in 1D system = {loss:.2e}", flush=True)
        
        optimizer.step()
    
    # L-BFGS optimization
    if iterationL > 0:
        print("\nStarting L-BFGS optimization...")
        
        # Create a closure for L-BFGS that captures all required variables
        def lbfgs_closure():
            nonlocal w_IC, w_BC, w_PDE
            # Use the final weights from Adam optimization
            w_IC, w_BC, w_PDE = 1, 1, 10
            
            # Compute loss using closure_batched
            loss, ic_loss, bc_loss, pde_loss = closure_batched(
                model, net, alpha, mse_cost_function, collocation_domain,
                collocation_IC, optimizerL, lam, jeans, v_1,
                w_IC, w_BC, w_PDE, batch_size, num_batches
            )
            
            # Return only the total loss for L-BFGS
            return w_IC * ic_loss + w_BC * bc_loss + w_PDE * pde_loss
        
        # L-BFGS optimization loop
        for i in range(iterationL):
            # Compute loss and update parameters
            loss = optimizerL.step(lbfgs_closure)
            
            # Print progress every 50 iterations
            if i % 50 == 0:
                # Compute detailed losses for printing
                _, ic_loss, bc_loss, pde_loss = closure_batched(
                    model, net, alpha, mse_cost_function, collocation_domain,
                    collocation_IC, optimizerL, lam, jeans, v_1,
                    w_IC, w_BC, w_PDE, batch_size, num_batches
                )
                print(f"Training Loss at {i} for LBGFS in 1D system = {loss:.2e}", flush=True)
    
    return net