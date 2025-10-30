"""
Adaptive Collocation Allocation for PINNs

This module implements adaptive collocation point allocation based on PDE residuals.
Points are redistributed to focus on high-error regions during training.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import torch
# scipy.interpolate.RegularGridInterpolator no longer needed - FD solutions removed
from config import (
    USE_ADAPTIVE_COLLOCATION, ADAPTIVE_COLLOCATION_FREQUENCY, ADAPTIVE_COLLOCATION_THRESHOLD_MODE,
    ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL, ADAPTIVE_COLLOCATION_PERCENTILE_FINAL,
    ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START, ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END,
    ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE, ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE_FINAL,
    ADAPTIVE_COLLOCATION_RATIO_MODE, ADAPTIVE_COLLOCATION_FIXED_RATIO, 
    ADAPTIVE_COLLOCATION_MIN_POINTS, ADAPTIVE_COLLOCATION_MAX_POINTS, 
    ADAPTIVE_COLLOCATION_LBFGS_MODE, ADAPTIVE_COLLOCATION_CAUSAL_COMPATIBLE, 
    ADAPTIVE_COLLOCATION_RESIDUAL_BATCH_SIZE, USE_LOG_DENSITY, ADAPTIVE_COLLOCATION_VERBOSE
)
# FD solution generation removed - no longer needed for PDE residual-based adaptive collocation
# FD solutions are now generated on-demand for plotting/comparison only


class AdaptiveCollocationAllocator:
    """
    Adaptive collocation point allocation based on PDE residuals.
    Redistributes points to focus on high-error regions.
    """
    
    def __init__(self, fd_reference, xmin, xmax, ymin, ymax, tmin, tmax, 
                 total_points, device='cpu'):
        """
        Initialize adaptive collocation allocator.
        
        Args:
            fd_reference: FDReferenceSolution instance (can be None if using PDE residuals)
            xmin, xmax, ymin, ymax: Spatial domain bounds
            tmin, tmax: Temporal domain bounds
            total_points: Total number of collocation points
            device: PyTorch device
        """
        self.fd_reference = fd_reference  # Can be None for PDE residual mode
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.tmin, self.tmax = tmin, tmax
        self.total_points = total_points
        self.device = device
        
        # Current point distribution
        self.current_points = None
        self.point_weights = None
        
        # Tracking for adaptive allocation
        self.allocation_history = []
        self.residual_history = []
    
    def generate_initial_points(self):
        """
        Generate initial uniform collocation points.
        
        Returns:
            Tensor: Initial collocation points [N, 3] with (x, y, t)
        """
        print(f"Generating initial uniform collocation points: {self.total_points}")
        
        # Generate random points in domain
        x = torch.rand(self.total_points, device=self.device) * (self.xmax - self.xmin) + self.xmin
        y = torch.rand(self.total_points, device=self.device) * (self.ymax - self.ymin) + self.ymin
        t = torch.rand(self.total_points, device=self.device) * (self.tmax - self.tmin) + self.tmin
        
        # Combine into single tensor
        points = torch.stack([x, y, t], dim=1)
        
        # Store current points
        self.current_points = points
        self.point_weights = torch.ones(self.total_points, device=self.device)
        
        return points
    
    def compute_residuals(self, net, points=None, batch_size=None):
        """
        Compute actual PDE residuals at given points using PINN network.
        Uses batching to avoid GPU memory issues.
        
        Args:
            net: PINN network
            points: Points to evaluate (default: current_points)
            batch_size: Batch size for residual computation (default: config value)
            
        Returns:
            Tensor: PDE residual magnitudes
        """
        if points is None:
            points = self.current_points
        
        if batch_size is None:
            batch_size = ADAPTIVE_COLLOCATION_RESIDUAL_BATCH_SIZE
        
        total_points = points.shape[0]
        all_residuals = []
        
        print(f"  Computing PDE residuals for {total_points} points in batches of {batch_size}")
        
        # Import PDE residue function
        from losses import pde_residue
        
        # CRITICAL: Ensure network is in training mode for gradient computation
        was_training = net.training
        net.train()  # Enable gradient tracking through network
        
        # Debug: Check if network parameters have requires_grad
        params_with_grad = sum(p.requires_grad for p in net.parameters())
        total_params = sum(1 for _ in net.parameters())
        print(f"  Network has {params_with_grad}/{total_params} parameters with requires_grad=True")
        
        # Process points in batches to avoid memory issues
        # NOTE: Must enable gradients for PDE residual computation (derivatives needed)
        # CRITICAL: Use torch.enable_grad() context to ensure gradients flow
        with torch.enable_grad():  # Force enable gradients even during eval/inference
            for i in range(0, total_points, batch_size):
                end_idx = min(i + batch_size, total_points)
                batch_points = points[i:end_idx]
                
                # Debug: Check if points have requires_grad
                if i == 0 and ADAPTIVE_COLLOCATION_VERBOSE:
                    print(f"    DEBUG - batch_points.requires_grad: {batch_points.requires_grad}")
                    print(f"    DEBUG - batch_points.is_leaf: {batch_points.is_leaf}")
                    print(f"    DEBUG - torch.is_grad_enabled(): {torch.is_grad_enabled()}")
                
                # Convert points to network format [x, y, t] WITH gradients enabled
                # CRITICAL: Must create NEW tensors with gradients from scratch
                if isinstance(batch_points, torch.Tensor) and batch_points.dim() == 2:
                    # Single tensor format [N, 3] -> [x, y, t]
                    # Create fresh tensors with gradients enabled
                    x_inp = batch_points[:, 0:1].clone().requires_grad_(True)
                    y_inp = batch_points[:, 1:2].clone().requires_grad_(True)
                    t_inp = batch_points[:, 2:3].clone().requires_grad_(True)
                    network_inputs = [x_inp, y_inp, t_inp]
                else:
                    # Already in list format [x, y, t]
                    network_inputs = [p.clone().requires_grad_(True) for p in batch_points]
                
                # Debug: Verify gradients are enabled
                if i == 0 and ADAPTIVE_COLLOCATION_VERBOSE:
                    print(f"    DEBUG - network_inputs[0].requires_grad: {network_inputs[0].requires_grad}")
                    print(f"    DEBUG - network_inputs[0].is_leaf: {network_inputs[0].is_leaf}")
                
                # Debug: Test network output before PDE computation
                if i == 0 and ADAPTIVE_COLLOCATION_VERBOSE:
                    test_out = net(network_inputs)
                    print(f"    DEBUG - network output shape: {test_out.shape}")
                    print(f"    DEBUG - network output requires_grad: {test_out.requires_grad}")
                    print(f"    DEBUG - network output range: [{test_out.min().item():.2e}, {test_out.max().item():.2e}]")
                else:
                    test_out = None
                
                # Compute PDE residuals for this batch (requires gradients!)
                rho_r, vx_r, vy_r, phi_r = pde_residue(network_inputs, net, dimension=2, use_log_density=USE_LOG_DENSITY)
                
                # Debug: Check if residuals are being computed
                if i == 0 and ADAPTIVE_COLLOCATION_VERBOSE:  # Only print for first batch
                    print(f"    DEBUG - PDE residuals:")
                    print(f"      rho_r: min={rho_r.min().item():.2e}, max={rho_r.max().item():.2e}, mean={rho_r.mean().item():.2e}")
                    print(f"      vx_r: min={vx_r.min().item():.2e}, max={vx_r.max().item():.2e}, mean={vx_r.mean().item():.2e}")
                    print(f"      vy_r: min={vy_r.min().item():.2e}, max={vy_r.max().item():.2e}, mean={vy_r.mean().item():.2e}")
                    
                    # Test if we can manually compute a gradient
                    if test_out is not None:
                        test_rho = test_out[:, 0:1]
                        try:
                            test_grad = torch.autograd.grad(test_rho.sum(), network_inputs[0], create_graph=True, allow_unused=True)[0]
                            if test_grad is None:
                                print(f"    DEBUG - Manual gradient test: FAILED (returned None)")
                            else:
                                print(f"    DEBUG - Manual gradient test: SUCCESS (mean={test_grad.mean().item():.2e})")
                        except Exception as e:
                            print(f"    DEBUG - Manual gradient test: ERROR - {e}")
                
                # Compute magnitude of residuals (combined L2 norm)
                batch_residuals = torch.sqrt(rho_r**2 + vx_r**2 + vy_r**2)
                batch_residuals = batch_residuals.squeeze()
                
                # Detach only the final residuals (not intermediate computations)
                all_residuals.append(batch_residuals.detach())
                
                # Clear intermediate tensors to free memory
                del rho_r, vx_r, vy_r, phi_r, batch_residuals, network_inputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Restore original training mode
        if not was_training:
            net.eval()
        
        # Concatenate all batch results
        residuals = torch.cat(all_residuals, dim=0)
        
        # Ensure residuals have correct shape [N]
        if residuals.dim() == 2 and residuals.shape[1] == 1:
            residuals = residuals.squeeze(1)
        
        return residuals
    
    def identify_high_error_regions(self, residuals, threshold=None, iteration=None):
        """
        Identify high-error regions based on residual threshold.
        Uses percentile-based (adaptive to scale) or absolute threshold with progressive decay.
        
        Args:
            residuals: Residual magnitudes [N]
            threshold: Error threshold (if None, uses progressive threshold)
            iteration: Current training iteration (for progressive threshold)
            
        Returns:
            Tensor: Boolean mask for high-error points
        """
        if threshold is None:
            threshold = self.get_progressive_threshold(residuals, iteration)
        
        high_error_mask = residuals > threshold
        return high_error_mask
    
    def get_progressive_threshold(self, residuals, iteration):
        """
        Get progressive threshold that starts lenient and becomes stricter over time.
        Supports both percentile-based (automatic scaling) and absolute thresholds.
        
        Args:
            residuals: Residual magnitudes [N] (needed for percentile mode)
            iteration: Current training iteration
            
        Returns:
            float: Current threshold value
        """
        if ADAPTIVE_COLLOCATION_THRESHOLD_MODE == "percentile":
            # Percentile-based: automatically adapts to residual scale
            if iteration is None:
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL
            elif iteration < ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START:
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL
            elif iteration > ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END:
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_FINAL
            else:
                # Linear decay: start lenient (high percentile), end strict (low percentile)
                progress = (iteration - ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START) / \
                          (ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END - ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START)
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL + progress * \
                           (ADAPTIVE_COLLOCATION_PERCENTILE_FINAL - ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL)
            
            # Compute threshold as the Nth percentile of residuals
            # Higher percentile = more lenient (e.g., 75th = top 25% are "high error")
            # Lower percentile = stricter (e.g., 50th = top 50% are "high error")
            if isinstance(residuals, torch.Tensor):
                threshold = torch.quantile(residuals, percentile / 100.0).item()
            else:
                threshold = np.percentile(residuals, percentile)
                
        else:  # "absolute" mode
            # Absolute threshold: uses fixed values
            if iteration is None:
                threshold = ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE
            elif iteration < ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START:
                threshold = ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE
            elif iteration > ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END:
                threshold = ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE_FINAL
            else:
                # Linear interpolation
                progress = (iteration - ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START) / \
                          (ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END - ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START)
                threshold = ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE + progress * \
                           (ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE_FINAL - ADAPTIVE_COLLOCATION_THRESHOLD_ABSOLUTE)
        
        return threshold
    
    def redistribute_points(self, residuals, ratio_mode=None, fixed_ratio=None, iteration=None):
        '''
        Redistribute collocation points based on residuals.
        FIXED VERSION: Preserves temporal coverage via stratified removal.

        Args:
            residuals: Residual magnitudes
            ratio_mode: "fixed" or "adaptive" (default: ADAPTIVE_COLLOCATION_RATIO_MODE)
            fixed_ratio: Fixed ratio for redistribution (default: ADAPTIVE_COLLOCATION_FIXED_RATIO)
            iteration: Current training iteration (for progressive threshold)

        Returns:
            Tensor: New collocation points
        '''
        if ratio_mode is None:
            ratio_mode = ADAPTIVE_COLLOCATION_RATIO_MODE
        if fixed_ratio is None:
            fixed_ratio = ADAPTIVE_COLLOCATION_FIXED_RATIO
        
        # Identify high-error regions
        high_error_mask = self.identify_high_error_regions(residuals, iteration=iteration)
        n_high_error = torch.sum(high_error_mask).item()
        
        current_threshold = self.get_progressive_threshold(residuals, iteration)
        
        # Print diagnostic information
        print(f"  High-error points: {n_high_error}/{self.total_points} ({n_high_error/self.total_points*100:.1f}%)")
        print(f"  Residual range: [{residuals.min():.2e}, {residuals.max():.2e}]")
        
        if ADAPTIVE_COLLOCATION_THRESHOLD_MODE == "percentile":
            # Calculate and display current percentile
            if iteration is None:
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL
            elif iteration < ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START:
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL
            elif iteration > ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END:
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_FINAL
            else:
                progress = (iteration - ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START) / \
                          (ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_END - ADAPTIVE_COLLOCATION_PERCENTILE_DECAY_START)
                percentile = ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL + progress * \
                           (ADAPTIVE_COLLOCATION_PERCENTILE_FINAL - ADAPTIVE_COLLOCATION_PERCENTILE_INITIAL)
            print(f"  Threshold: {current_threshold:.2e} (P{percentile:.1f} - top {100-percentile:.1f}% residuals)")
        else:
            print(f"  Threshold: {current_threshold:.2e} (absolute)")
        
        # Determine redistribution ratio
        if ratio_mode == "adaptive":
            # Adaptive ratio based on error distribution
            error_ratio = n_high_error / self.total_points
            
            # More sophisticated adaptive logic
            if error_ratio > 0.8:  # Most points are high-error
                # Focus on the highest error points only
                redistribution_ratio = 0.2  # Redistribute fewer points but more strategically
            elif error_ratio > 0.5:  # Moderate error distribution
                redistribution_ratio = 0.3
            else:  # Few high-error points
                redistribution_ratio = min(0.4, error_ratio * 3)  # Scale up for sparse errors
            
            redistribution_ratio = max(0.1, redistribution_ratio)  # Minimum 10%
        else:
            # Fixed ratio
            redistribution_ratio = fixed_ratio
        
        n_redistribute = int(self.total_points * redistribution_ratio)
        print(f"  Redistributing {n_redistribute} points ({redistribution_ratio*100:.1f}%)")
        
        # ============================================================================
        # CRITICAL FIX: TEMPORAL STRATIFICATION
        # Preserve temporal coverage by removing points proportionally from each time bin
        # ============================================================================
        
        NUM_TIME_BINS = 5  # Divide temporal domain into 5 bins
        points_per_bin = n_redistribute // NUM_TIME_BINS
        
        low_error_mask = ~high_error_mask
        low_error_indices = torch.where(low_error_mask)[0]
        
        if len(low_error_indices) == 0:
            print("  WARNING: No low-error points available for redistribution!")
            return self.current_points
        
        # Extract time coordinates (assuming time is the 3rd column, index 2)
        t_vals = self.current_points[:, 2]
        
        # Stratified removal across time bins
        remove_indices_list = []
        
        for i in range(NUM_TIME_BINS):
            # Define time bin boundaries
            t_low = self.tmin + i * (self.tmax - self.tmin) / NUM_TIME_BINS
            t_high = self.tmin + (i + 1) * (self.tmax - self.tmin) / NUM_TIME_BINS
            
            # Find low-error points in this time bin
            in_bin = (t_vals >= t_low) & (t_vals < t_high)
            bin_low_error_mask = in_bin & low_error_mask
            bin_low_error_indices = torch.where(bin_low_error_mask)[0]
            
            if len(bin_low_error_indices) > 0:
                # Remove proportionally from each bin
                n_remove_bin = min(points_per_bin, len(bin_low_error_indices))
                
                # Randomly select points to remove from this bin
                selected_indices = bin_low_error_indices[torch.randperm(len(bin_low_error_indices))[:n_remove_bin]]
                remove_indices_list.append(selected_indices)
                
                print(f"    Time bin {i+1} [{t_low:.2f}, {t_high:.2f}]: Removing {n_remove_bin}/{len(bin_low_error_indices)} points")
            else:
                print(f"    Time bin {i+1} [{t_low:.2f}, {t_high:.2f}]: No low-error points to remove")
        
        if len(remove_indices_list) == 0:
            print("  WARNING: No points could be removed from any time bin!")
            return self.current_points
        
        # Combine all removal indices
        remove_indices = torch.cat(remove_indices_list)
        n_remove = len(remove_indices)
        
        print(f"  Total points marked for removal: {n_remove}")
        
        # ============================================================================
        # Generate new points near high-error regions
        # ============================================================================
        
        high_error_points = self.current_points[high_error_mask]
        high_error_residuals = residuals[high_error_mask]
        
        if len(high_error_points) > 0 and n_remove > 0:
            # Focus on the highest error points
            if len(high_error_points) > n_remove:
                # Select the top highest-error points
                _, top_error_indices = torch.topk(high_error_residuals, n_remove)
                top_error_points = high_error_points[top_error_indices]
                new_points = self._generate_points_near_existing(top_error_points, n_remove)
            else:
                # Use all high-error points as seeds
                new_points = self._generate_points_near_existing(high_error_points, n_remove)
        else:
            # Fallback: generate random points if no high-error regions
            print("  WARNING: No high-error points found, generating random points")
            new_points = self._generate_random_points(n_remove)
        
        # ============================================================================
        # Update point distribution
        # ============================================================================
        
        new_current_points = self.current_points.clone()
        new_current_points[remove_indices] = new_points
        self.current_points = new_current_points
        
        print(f"  [OK] Redistributed {n_remove} points with temporal stratification preserved")
        
        # ============================================================================
        # DIAGNOSTIC: Verify temporal coverage is maintained
        # ============================================================================
        
        if ADAPTIVE_COLLOCATION_VERBOSE:
            print("  Temporal coverage after redistribution:")
            t_vals_new = self.current_points[:, 2].cpu().numpy()
            for i in range(NUM_TIME_BINS):
                t_low = self.tmin + i * (self.tmax - self.tmin) / NUM_TIME_BINS
                t_high = self.tmin + (i + 1) * (self.tmax - self.tmin) / NUM_TIME_BINS
                count = np.sum((t_vals_new >= t_low) & (t_vals_new < t_high))
                percentage = count / self.total_points * 100
                print(f"    Bin {i+1} [{t_low:.2f}, {t_high:.2f}]: {count} points ({percentage:.1f}%)")
        
        return self.current_points
    
    def _generate_points_near_existing(self, existing_points, n_points):
        """
        Generate new points near existing high-error points.
        Only perturbs spatial coordinates, preserves time structure.
        
        Args:
            existing_points: Existing high-error points
            n_points: Number of new points to generate
            
        Returns:
            Tensor: New points
        """
        # Add small random perturbations to existing points
        noise_scale = 0.15  # 15% of spatial domain size
        
        # Randomly select existing points to perturb
        n_existing = len(existing_points)
        if n_existing == 0:
            return self._generate_random_points(n_points)
        
        selected_indices = torch.randint(0, n_existing, (n_points,), device=self.device)
        selected_points = existing_points[selected_indices]
        
        # Add noise ONLY to spatial coordinates (x, y), NOT time
        x_noise = torch.randn(n_points, device=self.device) * noise_scale * (self.xmax - self.xmin)
        y_noise = torch.randn(n_points, device=self.device) * noise_scale * (self.ymax - self.ymin)
        
        new_points = selected_points.clone()
        new_points[:, 0] += x_noise  # Perturb x
        new_points[:, 1] += y_noise  # Perturb y
        # new_points[:, 2] unchanged - preserve time!
        
        # Clamp spatial coordinates to domain bounds
        new_points[:, 0] = torch.clamp(new_points[:, 0], self.xmin, self.xmax)
        new_points[:, 1] = torch.clamp(new_points[:, 1], self.ymin, self.ymax)
        # Time already valid, no need to clamp
        
        return new_points
    
    def _generate_random_points(self, n_points):
        """
        Generate random points in domain.
        
        Args:
            n_points: Number of points to generate
            
        Returns:
            Tensor: Random points [n_points, 3]
        """
        x = torch.rand(n_points, device=self.device) * (self.xmax - self.xmin) + self.xmin
        y = torch.rand(n_points, device=self.device) * (self.ymax - self.ymin) + self.ymin
        t = torch.rand(n_points, device=self.device) * (self.tmax - self.tmin) + self.tmin
        
        return torch.stack([x, y, t], dim=1)
    
    def prepare_for_lbfgs(self):
        """
        Prepare collocation points for LBFGS phase.
        
        Returns:
            Tensor: Points prepared for LBFGS
        """
        if ADAPTIVE_COLLOCATION_LBFGS_MODE == "uniform":
            print("Preparing uniform distribution for LBFGS phase...")
            return self.generate_initial_points()
        else:  # "adaptive"
            print("Keeping adaptive distribution for LBFGS phase...")
            return self.current_points
    
    def should_update(self, iteration):
        """
        Check if adaptive allocation should be updated.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            bool: Whether to update
        """
        return ADAPTIVE_COLLOCATION_FREQUENCY > 0 and iteration % ADAPTIVE_COLLOCATION_FREQUENCY == 0


def create_adaptive_allocator(lam, num_of_waves, rho_1, xmin, xmax, ymin, ymax, 
                             tmin, tmax, total_points, device, vx0_shared=None, vy0_shared=None):
    """
    Create an adaptive collocation allocator using PDE residuals only.
    FD solutions are no longer needed since we use actual PDE residuals for threshold calculation.
    
    Args:
        lam: Wavelength parameter (unused, kept for compatibility)
        num_of_waves: Number of waves (unused, kept for compatibility)
        rho_1: Reference density (unused, kept for compatibility)
        xmin, xmax, ymin, ymax: Spatial domain bounds
        tmin, tmax: Temporal domain bounds
        total_points: Total number of collocation points
        device: PyTorch device
        vx0_shared, vy0_shared: Shared velocity fields (unused, kept for compatibility)
        
    Returns:
        AdaptiveCollocationAllocator instance
    """
    print("Creating adaptive collocation allocator...")
    print("Using PDE residuals for adaptive allocation (no FD solution needed)")
    
    # Create adaptive allocator without FD reference
    allocator = AdaptiveCollocationAllocator(
        fd_reference=None,  # No FD reference needed
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        tmin=tmin, tmax=tmax, total_points=total_points, device=device
    )
    
    # Generate initial points
    allocator.generate_initial_points()
    
    print("Adaptive collocation allocator created successfully!")
    
    return allocator
