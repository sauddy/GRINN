"""
Causal Training Module for Physics-Informed Neural Networks

This module provides infrastructure for causal training with temporal curriculum
and adaptive/static weighting schemes. Similar to xpinn_decomposition.py structure.

Key Components:
- ResidualTracker: Tracks residuals across time bins for adaptive weighting
- CausalTrainer: Orchestrates the complete causal training workflow
- Helper functions for window scheduling, weight computation, etc.
"""

import numpy as np
import torch


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
        """Get bin indices for given time values."""
        t_flat = t_values.flatten()
        bin_idx = ((t_flat - self.t_min) / self.bin_width).long()
        return torch.clamp(bin_idx, 0, self.num_bins - 1)
    
    def update_residuals(self, t_values, residuals):
        """
        Update cumulative residuals for time bins based on current batch.
        
        Args:
            t_values: Time values [N, 1]
            residuals: PDE residuals [N, 1] or list of residuals
        """
        # Convert residuals to single scalar per point if it's a list
        if isinstance(residuals, (list, tuple)):
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
        ref_scale = avg_residuals[0].clamp_min(eps)
        avg_residuals_norm = avg_residuals / ref_scale
        
        # Compute cumulative sum of normalized average residuals
        cumsum_avg = torch.cumsum(avg_residuals_norm, dim=0)
        
        # For bin i, we want sum from bins 0 to i-1, so shift cumsum by 1
        cumsum_shifted = torch.cat([torch.zeros(1, device=self.device), cumsum_avg[:-1]], dim=0)
        
        # Gather the appropriate cumulative sum for each point based on its bin
        bin_idx_flat = bin_idx.clamp(min=0, max=self.num_bins-1)
        past_residual_sum = cumsum_shifted[bin_idx_flat]
        
        # Apply exponential suppression with floor to prevent starving later times
        weights = torch.exp(-self.epsilon * past_residual_sum).clamp_min(0.02)
        
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


def compute_causal_weights_static(t_values, gamma):
    """
    Compute static causal weights: w(t) = exp(-gamma * t)
    
    Args:
        t_values: Time values tensor [N, 1]
        gamma: Exponential decay parameter
    
    Returns:
        Weights [N, 1]
    """
    if gamma == 0.0:
        return torch.ones_like(t_values)
    return torch.exp(-gamma * t_values)


def generate_temporal_windows(schedule, num_windows, tmin, tmax, startup_dt, custom_windows=None):
    """
    Generate temporal window boundaries for curriculum training.
    
    Args:
        schedule: 'linear' or 'custom'
        num_windows: Number of windows (ignored if schedule='custom')
        tmin: Minimum time
        tmax: Maximum time  
        startup_dt: Startup time offset
        custom_windows: List of [t_min, t_max] pairs for custom schedule
    
    Returns:
        List of (t_min, t_max) tuples
    """
    if schedule == "custom":
        if custom_windows is None:
            raise ValueError("custom_windows required for custom schedule")
        return [(float(w[0]), float(w[1])) for w in custom_windows]
    elif schedule == "linear":
        # Linear progression from startup_dt to tmax
        t_start = max(tmin, startup_dt)
        window_times = np.linspace(t_start, tmax, num_windows + 1)
        return [(float(window_times[i]), float(window_times[i+1])) for i in range(num_windows)]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def compute_epsilon_for_window(window_idx, num_windows, epsilon_min, epsilon_max, 
                                epsilon_base, epsilon_floor, use_annealing):
    """
    Compute epsilon value for current window in adaptive mode.
    
    Args:
        window_idx: Current window index (0-based)
        num_windows: Total number of windows
        epsilon_min: Minimum epsilon (for annealing)
        epsilon_max: Maximum epsilon (for annealing)
        epsilon_base: Base epsilon value (for linear decay)
        epsilon_floor: Floor epsilon value (for linear decay)
        use_annealing: Whether to use automatic annealing
    
    Returns:
        epsilon value for this window
    """
    if use_annealing:
        # Automatic interpolation between MIN and MAX (ensures monotonic increase)
        progress = window_idx / max(1, num_windows - 1)
        return epsilon_min + (epsilon_max - epsilon_min) * progress
    else:
        # Linear decay (original method): strong early → moderate late
        return epsilon_floor + (epsilon_base - epsilon_floor) * (1.0 - window_idx / max(1, num_windows - 1))


def compute_gamma_for_window(window_idx, num_windows, gamma_max, gamma_min):
    """
    Compute gamma value for current window in static mode.
    
    Args:
        window_idx: Current window index (0-based)
        num_windows: Total number of windows
        gamma_max: Maximum gamma value (early windows)
        gamma_min: Minimum gamma value (final window)
    
    Returns:
        gamma value for this window
    """
    # Linear decay from gamma_max to gamma_min
    progress = window_idx / max(1, num_windows - 1)
    return gamma_max * (1.0 - progress) + gamma_min * progress


class CausalTrainer:
    """
    Orchestrates causal training with temporal curriculum and adaptive/static weighting.
    """
    
    def __init__(self, model, net, optimizer, optimizerL, mse_cost_function,
                 train_func, config, device):
        """
        Args:
            model: ASTPN model for collocation generation
            net: Neural network
            optimizer: Adam optimizer
            optimizerL: LBFGS optimizer
            mse_cost_function: Loss function
            train_func: Training function (from solver.py)
            config: Dictionary with all causal config parameters
            device: PyTorch device
        """
        self.model = model
        self.net = net
        self.optimizer = optimizer
        self.optimizerL = optimizerL
        self.mse_cost_function = mse_cost_function
        self.train_func = train_func
        self.config = config
        self.device = device
        
        # Initialize residual tracker for adaptive mode
        self.residual_tracker = None
        if config['weighting_mode'] == 'adaptive':
            t_start = max(config['tmin'], config['startup_dt'])
            self.residual_tracker = ResidualTracker(
                t_min=t_start,
                t_max=config['tmax'],
                num_bins=config['num_time_bins'],
                epsilon=config['epsilon'],
                device=device
            )
    
    def train_with_curriculum(self, collocation_IC, **train_kwargs):
        """
        Train with temporal curriculum (progressive time windows).
        
        Args:
            collocation_IC: Initial condition collocation points
            **train_kwargs: Additional kwargs for train function (rho_1, lam, etc.)
        
        Returns:
            Trained network
        """
        cfg = self.config
        use_restarts = cfg['use_restarts']
        
        # Generate temporal windows
        windows = generate_temporal_windows(
            schedule=cfg['window_schedule'],
            num_windows=cfg['num_windows'],
            tmin=cfg['tmin'],
            tmax=cfg['tmax'],
            startup_dt=cfg['startup_dt'],
            custom_windows=cfg['custom_windows']
        )
        
        # Compute iterations per window
        adam_per_window = cfg['adam_per_window'] or (cfg['iteration_adam'] // cfg['num_windows'])
        lbfgs_per_window = cfg['lbfgs_per_window'] or (cfg['iteration_lbfgs'] // cfg['num_windows'])
        
        print(f"Training schedule: {len(windows)} windows, {adam_per_window} Adam + {lbfgs_per_window} LBFGS per window")
        
        # Train on each window
        for window_idx, (t_window_min, t_window_max) in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Window {window_idx + 1}/{len(windows)}: t in [{t_window_min:.3f}, {t_window_max:.3f}]")
            print(f"{'='*60}")
            
            # Compute causal parameters for this window
            if cfg['weighting_mode'] == 'static':
                causal_gamma = compute_gamma_for_window(
                    window_idx, len(windows), cfg['gamma_max'], cfg['gamma_min']
                )
                print(f"  Static weighting: gamma = {causal_gamma:.4f}")
                causal_mode = 'static'
            else:  # adaptive
                causal_gamma = 0.0
                causal_mode = 'adaptive'
                
                # Update epsilon for this window
                if self.residual_tracker is not None:
                    eps_k = compute_epsilon_for_window(
                        window_idx, len(windows),
                        cfg['epsilon_min'], cfg['epsilon_max'],
                        cfg['epsilon'], cfg['epsilon_floor'],
                        cfg['use_epsilon_annealing']
                    )
                    self.residual_tracker.epsilon = eps_k
                    if cfg['use_epsilon_annealing']:
                        print(f"  Adaptive weighting: epsilon = {eps_k:.3f} (annealed)")
                    else:
                        print(f"  Adaptive weighting: epsilon = {eps_k:.3f} (linear decay)")
            
            # Update model's temporal bounds for this window
            if use_restarts and window_idx > 0:
                # Restart marching: train only in current window [t_k, t_{k+1}]
                t_start = t_window_min
            else:
                # Expanding windows: train from t=0 to current t_max
                t_start = max(cfg['tmin'], cfg['startup_dt'])
            
            # Build rmin/rmax based on spatial dimension (2D or 3D)
            dimension = cfg.get('dimension', 2)  # Default to 2D for backward compatibility
            if dimension == 3:
                self.model.rmin = [cfg['xmin'], cfg['ymin'], cfg['zmin'], t_start]
                self.model.rmax = [cfg['xmax'], cfg['ymax'], cfg['zmax'], t_window_max]
            else:  # 2D or 1D
                self.model.rmin = [cfg['xmin'], cfg['ymin'], t_start]
                self.model.rmax = [cfg['xmax'], cfg['ymax'], t_window_max]
            
            # Regenerate domain collocation for this window
            collocation_domain_window = self.model.geo_time_coord(option="Domain")
            
            # Train on this window
            self.train_func(
                net=self.net,
                model=self.model,
                collocation_domain=collocation_domain_window,
                collocation_IC=collocation_IC,
                optimizer=self.optimizer,
                optimizerL=self.optimizerL,
                closure=None,
                mse_cost_function=self.mse_cost_function,
                iteration_adam=adam_per_window,
                iterationL=lbfgs_per_window,
                device=self.device,
                causal_gamma=causal_gamma,
                causal_mode=causal_mode,
                residual_tracker=self.residual_tracker,
                window_idx=window_idx,
                **train_kwargs
            )
        
        # Restore full time/space range for final evaluation/plotting
        dimension = cfg.get('dimension', 2)  # Default to 2D for backward compatibility
        if dimension == 3:
            self.model.rmin = [cfg['xmin'], cfg['ymin'], cfg['zmin'], cfg['tmin']]
            self.model.rmax = [cfg['xmax'], cfg['ymax'], cfg['zmax'], cfg['tmax']]
        else:  # 2D or 1D
            self.model.rmin = [cfg['xmin'], cfg['ymin'], cfg['tmin']]
            self.model.rmax = [cfg['xmax'], cfg['ymax'], cfg['tmax']]
        
        if use_restarts:
            print(f"\nCausal training completed (restart marching). Final time range: [{cfg['tmin']}, {cfg['tmax']}]")
        else:
            print(f"\nCausal training completed (expanding windows). Final time range: [{cfg['tmin']}, {cfg['tmax']}]")
        
        return self.net
    
    def train_without_curriculum(self, collocation_IC, **train_kwargs):
        """
        Train on full domain with adaptive weighting (no temporal windows).
        
        Args:
            collocation_IC: Initial condition collocation points
            **train_kwargs: Additional kwargs for train function
        
        Returns:
            Trained network
        """
        cfg = self.config
        
        print("Training on full temporal domain with adaptive weighting...")
        
        # Generate full domain collocation
        collocation_domain = self.model.geo_time_coord(option="Domain")
        
        # Train with adaptive weighting
        self.train_func(
            net=self.net,
            model=self.model,
            collocation_domain=collocation_domain,
            collocation_IC=collocation_IC,
            optimizer=self.optimizer,
            optimizerL=self.optimizerL,
            closure=None,
            mse_cost_function=self.mse_cost_function,
            iteration_adam=cfg['iteration_adam'],
            iterationL=cfg['iteration_lbfgs'],
            device=self.device,
            causal_gamma=0.0,  # Not used in adaptive mode
            causal_mode=cfg['weighting_mode'],
            residual_tracker=self.residual_tracker,
            window_idx=None,
            **train_kwargs
        )
        
        print(f"\nCausal training completed (full domain)")
        
        return self.net

