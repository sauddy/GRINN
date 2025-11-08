import os
import numpy as np
import torch
import matplotlib.pyplot as plt


class TrainingDiagnostics:
    """Lightweight real-time diagnostics during training."""

    def __init__(self, save_dir='./diagnostics/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Storage for tracking metrics over iterations
        self.history = {
            'iteration': [],
            'total_loss': [],
            'pde_loss': [],
            'ic_loss': [],
            'mean_rho': [],
            'max_rho': [],
            'mean_grad_rho': [],
            'max_grad_rho': [],
            'mean_residual': [],
            'max_residual': []
        }

    def _prepare_inputs(self, geomtime_col):
        """Normalize collocation inputs to the model's expected format."""
        if isinstance(geomtime_col, (list, tuple)):
            return geomtime_col
        # Single tensor [N, D] -> list of [N,1]
        return [geomtime_col[:, i:i+1] for i in range(geomtime_col.shape[1])]

    def log_iteration(self, iteration, model, loss_dict, geomtime_col):
        """Call this every N iterations during training."""
        self.history['iteration'].append(iteration)
        self.history['total_loss'].append(float(loss_dict.get('total', np.nan)))
        self.history['pde_loss'].append(float(loss_dict.get('pde', np.nan)))
        self.history['ic_loss'].append(float(loss_dict.get('ic', np.nan)))

        # Compute diagnostics from current model state
        with torch.no_grad():
            inputs = self._prepare_inputs(geomtime_col)
            pred = model(inputs)
            # Use first channel as density/log-density proxy
            rho = pred[:, 0].detach().cpu().numpy()

            # Density statistics
            self.history['mean_rho'].append(np.nanmean(rho))
            self.history['max_rho'].append(np.nanmax(rho))

            # Gradient proxies from predictions (fast approximate)
            self.history['mean_grad_rho'].append(float(np.nanstd(rho)))
            self.history['max_grad_rho'].append(float(np.nanmax(np.abs(np.diff(rho)))))

            # Residual statistics (optional, if provided as array-like)
            residual = loss_dict.get('residual', None)
            if residual is not None:
                res_np = np.asarray(residual)
                self.history['mean_residual'].append(float(np.nanmean(np.abs(res_np))))
                self.history['max_residual'].append(float(np.nanmax(np.abs(res_np))))

    def plot_diagnostics(self, iteration):
        """Generate diagnostic plots at this iteration."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        iters = self.history['iteration']

        # Plot 1: Loss evolution
        axes[0, 0].semilogy(iters, self.history['total_loss'], 'b-', linewidth=2, label='Total')
        axes[0, 0].semilogy(iters, self.history['pde_loss'], 'r-', linewidth=2, label='PDE')
        axes[0, 0].semilogy(iters, self.history['ic_loss'], 'g-', linewidth=2, label='IC')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss (log scale)')
        axes[0, 0].set_title('Loss Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Loss ratio (PDE/IC)
        ic_arr = np.array(self.history['ic_loss'])
        pde_arr = np.array(self.history['pde_loss'])
        ratio = pde_arr / (ic_arr + 1e-10)
        axes[0, 1].plot(iters, ratio, 'purple', linewidth=2)
        axes[0, 1].axhline(1.0, color='k', linestyle='--', label='Balanced')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('PDE Loss / IC Loss')
        axes[0, 1].set_title('Loss Balance (Should be ~1)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Density statistics
        axes[1, 0].plot(iters, self.history['mean_rho'], 'b-', linewidth=2, label='Mean ρ')
        axes[1, 0].plot(iters, self.history['max_rho'], 'r-', linewidth=2, label='Max ρ')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Density Evolution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Max density growth rate
        if len(self.history['max_rho']) > 10:
            growth_rate = np.diff(self.history['max_rho'])
            axes[1, 1].plot(iters[1:], growth_rate, 'orange', linewidth=2)
            axes[1, 1].axhline(0, color='k', linestyle='--')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Δ(Max ρ)')
            axes[1, 1].set_title('Density Growth Rate (Should be positive)')
            axes[1, 1].grid(True, alpha=0.3)

        # Plot 5: Gradient magnitude
        axes[2, 0].semilogy(iters, self.history['mean_grad_rho'], 'b-', linewidth=2, label='Mean')
        axes[2, 0].semilogy(iters, self.history['max_grad_rho'], 'r-', linewidth=2, label='Max')
        axes[2, 0].set_xlabel('Iteration')
        axes[2, 0].set_ylabel('Gradient Magnitude (log)')
        axes[2, 0].set_title('Density Gradients')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 6: Residual statistics (if available)
        if len(self.history['mean_residual']) > 0:
            axes[2, 1].semilogy(iters, self.history['mean_residual'], 'b-', linewidth=2, label='Mean')
            axes[2, 1].semilogy(iters, self.history['max_residual'], 'r-', linewidth=2, label='Max')
            axes[2, 1].set_xlabel('Iteration')
            axes[2, 1].set_ylabel('Residual Magnitude (log)')
            axes[2, 1].set_title('PDE Residuals')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/diagnostics_iter_{iteration}.png', dpi=120, bbox_inches='tight')
        plt.close()


