"""
Training loop management for PINN and XPINN.

This module handles the optimization loops (Adam and L-BFGS) for both
single PINN and XPINN decomposition training.
"""
import numpy as np
import torch
from utilities.training_diagnostics import TrainingDiagnostics
from config import BATCH_SIZE, NUM_BATCHES, ENABLE_TRAINING_DIAGNOSTICS
from training.physics import closure_batched


def train(model, net, collocation_domain, collocation_IC, optimizer, optimizerL, iteration_adam, iterationL, mse_cost_function, closure, rho_1, lam, jeans, v_1, device, causal_gamma=0.0, causal_mode="none", residual_tracker=None, window_idx=None, data_terms=None):
    """
    Standard training loop for single PINN.
    
    Manages Adam and L-BFGS optimization phases with cosine scheduling for
    continuity weight and startup_dt parameters.
    
    Args:
        model: Collocation model
        net: Neural network
        collocation_domain: Domain collocation points
        collocation_IC: Initial condition points
        optimizer: Adam optimizer
        optimizerL: L-BFGS optimizer
        iteration_adam: Number of Adam iterations
        iterationL: Number of L-BFGS iterations
        mse_cost_function: MSE loss function
        closure: Closure function (unused, kept for compatibility)
        rho_1: Perturbation amplitude
        lam: Wavelength
        jeans: Jeans length
        v_1: Velocity amplitude
        device: PyTorch device
        causal_gamma: Causal weighting parameter
        causal_mode: Causal training mode
        residual_tracker: Residual tracker for adaptive causal weighting
        window_idx: Current curriculum window index
        data_terms: Optional list of additional supervised datasets with weights
    """
    bs = int(BATCH_SIZE)
    nb = int(NUM_BATCHES)

    diagnostics = TrainingDiagnostics(save_dir='./diagnostics/') if ENABLE_TRAINING_DIAGNOSTICS else None

    for i in range(iteration_adam):
        optimizer.zero_grad()

        loss, loss_breakdown = optimizer.step(lambda: closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizer, rho_1, lam, jeans, v_1, bs, nb, causal_gamma, causal_mode, residual_tracker, update_tracker=True, iteration=i, use_fft_poisson=True, data_terms=data_terms))

        with torch.autograd.no_grad():
            # Diagnostics logging every 50 iterations
            if i % 50 == 0 and diagnostics is not None:
                try:
                    diagnostics.log_iteration(
                        iteration=i,
                        model=net,
                        loss_dict={
                            'total': loss.item(),
                            'pde': float(loss_breakdown.get('PDE', 0.0)),
                            'ic': float(loss_breakdown.get('IC', 0.0))
                        },
                        geomtime_col=collocation_domain
                    )
                except Exception as _diag_err:
                    # Keep training robust if diagnostics fail
                    print(f"[WARN] Diagnostics logging failed at {i}: {_diag_err}")

            if i % 100 == 0:
                print(f"Training Loss at {i} for Adam (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)
                # Print loss breakdown
                breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
                if breakdown_str:
                    print(f"  Loss breakdown: {breakdown_str}", flush=True)

    for i in range(iterationL):
        optimizer.zero_grad()
        global_step = iteration_adam + i

        # L-BFGS expects a closure that returns only scalar loss
        # Store loss_breakdown in a list so we can access it after the step
        loss_breakdown_holder = [None]
        
        def lbfgs_closure():
            loss, loss_breakdown = closure_batched(model, net, mse_cost_function, collocation_domain, collocation_IC, optimizerL, rho_1, lam, jeans, v_1, bs, nb, causal_gamma, causal_mode, residual_tracker, update_tracker=False, iteration=global_step, use_fft_poisson=False, data_terms=data_terms)
            loss_breakdown_holder[0] = loss_breakdown
            return loss
        
        loss = optimizerL.step(lbfgs_closure)
        loss_breakdown = loss_breakdown_holder[0]

        with torch.autograd.no_grad():
            # Diagnostics logging every 50 iterations in LBFGS too
            if i % 50 == 0 and loss_breakdown is not None and diagnostics is not None:
                try:
                    diagnostics.log_iteration(
                        iteration=iteration_adam + i,
                        model=net,
                        loss_dict={
                            'total': loss.item() if hasattr(loss, 'item') else float(loss),
                            'pde': float(loss_breakdown.get('PDE', 0.0)),
                            'ic': float(loss_breakdown.get('IC', 0.0))
                        },
                        geomtime_col=collocation_domain
                    )
                except Exception as _diag_err:
                    print(f"[WARN] Diagnostics logging (LBFGS) failed at {i}: {_diag_err}")
            if i % 50 == 0:
                print(f"Training Loss at {i} for LBGFS (batched) in {model.dimension}D system = {loss.item():.2e}", flush=True)
                # Print loss breakdown
                breakdown_str = " | ".join([f"{k}: {v:.2e}" for k, v in loss_breakdown.items() if v > 0])
                if breakdown_str:
                    print(f"  Loss breakdown: {breakdown_str}", flush=True)
    
    # Generate diagnostic plots at the end of training
    if diagnostics is not None:
        try:
            final_iteration = iteration_adam + iterationL - 1
            diagnostics.plot_diagnostics(final_iteration)
        except Exception as _diag_err:
            print(f"[WARN] Final diagnostics plotting failed: {_diag_err}")


def train_xpinn(nets, subdomain_collocs, interface_collocs, subdomain_ic_collocs,
                ic_functions, interfaces, exterior_boundaries, xpinn_loss_model,
                optimizer, optimizerL, iteration_adam, iterationL, device,
                subdomain_devices=None, cached_ic_values=None):
    """
    Training loop for XPINN with multiple networks.
    
    Manages Adam and L-BFGS optimization for domain decomposition, with support
    for multi-GPU training and per-subdomain optimization.
    
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
    from training.physics import closure_xpinn, closure_xpinn_batched, _random_batch_indices, _make_batch_tensors
    
    # Set defaults
    if subdomain_devices is None:
        subdomain_devices = [device] * len(nets)
    if cached_ic_values is None:
        cached_ic_values = [None] * len(nets)
    from config import XPINN_ALTERNATING_TRAINING, USE_XPINN_BATCHING
    
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
    xpinn_diagnostics = TrainingDiagnostics(save_dir='./diagnostics/') if ENABLE_TRAINING_DIAGNOSTICS else None
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
            
            # Diagnostics for XPINN (aggregate) every 50 iterations
            if i % 50 == 0 and len(nets) > 0 and len(subdomain_collocs) > 0 and xpinn_diagnostics is not None:
                try:
                    xpinn_diagnostics.log_iteration(
                        iteration=i,
                        model=nets[0],
                        loss_dict={
                            'total': float(np.mean(subdomain_losses)) if subdomain_losses else np.nan,
                            'pde': float(total_pde_loss / max(1, len(nets))),
                            'ic': float(total_ic_loss / max(1, len(nets)))
                        },
                        geomtime_col=subdomain_collocs[0]
                    )
                except Exception as _diag_err:
                    print(f"[WARN] XPINN diagnostics logging failed at {i}: {_diag_err}")

            if i % 100 == 0:
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
            
            # Diagnostics for XPINN unified every 50 iterations
            if i % 50 == 0 and len(nets) > 0 and len(subdomain_collocs) > 0 and xpinn_diagnostics is not None:
                try:
                    xpinn_diagnostics.log_iteration(
                        iteration=i,
                        model=nets[0],
                        loss_dict={
                            'total': loss.item(),
                            'pde': float(loss_dict.get('PDE', 0.0)),
                            'ic': float(loss_dict.get('IC', 0.0))
                        },
                        geomtime_col=subdomain_collocs[0]
                    )
                except Exception as _diag_err:
                    print(f"[WARN] XPINN diagnostics logging failed at {i}: {_diag_err}")

            if i % 100 == 0:
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
                net.parameters(), line_search_fn='strong_wolfe')
            
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
                
                if i % 50 == 0:
                    total_v, pde_v, ic_v, if_sol_v, if_res_v = _lbfgs_subdomain_breakdown(subdomain_idx, net)
                    print(
                        f"    Subdomain {subdomain_idx} L-BFGS step {i}: "
                        f"Total={total_v:.2e} | PDE={pde_v:.2e} | IC={ic_v:.2e} | "
                        f"IF_sol={if_sol_v:.2e} | IF_res={if_res_v:.2e}",
                        flush=True
                    )
        
        print("Per-subdomain L-BFGS training completed.")
    
    # Generate diagnostic plots at the end of XPINN training
    if xpinn_diagnostics is not None:
        try:
            final_iteration = iteration_adam + iterationL - 1
            xpinn_diagnostics.plot_diagnostics(final_iteration)
        except Exception as _diag_err:
            print(f"[WARN] Final XPINN diagnostics plotting failed: {_diag_err}")
    
    print("XPINN training completed.")

