"""
Diagnostic tools for analyzing FD solution quality and identifying regions
where PINN-FD comparisons may be invalid due to numerical artifacts.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional


def diagnose_fd_solution(rho_fd: np.ndarray, vx_fd: np.ndarray, vy_fd: np.ndarray, 
                         rho_pinn: np.ndarray, rho_o: float = 1.0,
                         verbose: bool = True) -> Dict:
    """
    Analyze FD solution for numerical artifacts and quality issues.
    
    Args:
        rho_fd: FD density field
        vx_fd, vy_fd: FD velocity components
        rho_pinn: PINN density field for comparison
        rho_o: Background density
        verbose: Print diagnostic messages
    
    Returns:
        Dictionary with diagnostic information and quality masks
    """
    diagnostics = {}
    
    # 1. Check for NaN/Inf values
    nan_mask = np.isnan(rho_fd) | np.isinf(rho_fd)
    diagnostics['has_nan_inf'] = np.any(nan_mask)
    diagnostics['nan_inf_fraction'] = np.mean(nan_mask)
    diagnostics['nan_inf_mask'] = nan_mask
    
    # 2. Check for negative densities (physical violation)
    negative_mask = rho_fd < 0
    diagnostics['has_negative'] = np.any(negative_mask)
    diagnostics['negative_fraction'] = np.mean(negative_mask)
    diagnostics['negative_mask'] = negative_mask
    
    # 3. Check for extreme density ratios (only flag voids, not cores)
    density_ratio = rho_fd / rho_o
    extreme_low_mask = density_ratio < 0.4   # Less than 0.4x background (voids)
    diagnostics['extreme_low_fraction'] = np.mean(extreme_low_mask)
    diagnostics['extreme_low_mask'] = extreme_low_mask
    diagnostics['min_density_ratio'] = np.min(density_ratio)
    diagnostics['max_density_ratio'] = np.max(density_ratio)
    # Note: High density cores are physically valid, not flagged as invalid
    
    # 4. Check for velocity shear (instability indicator)
    # Compute velocity gradients
    dvx_dx = np.gradient(vx_fd, axis=0)
    dvx_dy = np.gradient(vx_fd, axis=1)
    dvy_dx = np.gradient(vy_fd, axis=0)
    dvy_dy = np.gradient(vy_fd, axis=1)
    
    # Velocity gradient magnitude (shear)
    v_grad_mag = np.sqrt(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)
    cs = 1.0  # Sound speed
    # Normalize by sound speed to get dimensionless shear
    normalized_shear = v_grad_mag / cs
    # Large shear indicates velocity discontinuities/instabilities
    # Use adaptive threshold: 75th percentile + 1.5*IQR to catch outliers
    shear_75 = np.percentile(normalized_shear, 75)
    shear_25 = np.percentile(normalized_shear, 25)
    shear_iqr = shear_75 - shear_25
    shear_threshold = max(0.01, shear_75 + 1.5 * shear_iqr)  # Adaptive threshold, min 0.01
    high_shear_mask = normalized_shear > shear_threshold
    diagnostics['shear_threshold'] = shear_threshold
    diagnostics['high_shear_fraction'] = np.mean(high_shear_mask)
    diagnostics['max_shear'] = np.max(normalized_shear)
    diagnostics['high_shear_mask'] = high_shear_mask
    
    # 5. Check for velocity divergence anomalies (inconsistent flow)
    div_v = dvx_dx + dvy_dy  # Velocity divergence
    # Large divergence in regions of moderate density indicates flow inconsistencies
    density_moderate = (density_ratio > 0.6) & (density_ratio < 1.5)  # Not cores, not voids
    div_anomaly_mask = (np.abs(div_v) > 0.2) & density_moderate
    diagnostics['div_anomaly_fraction'] = np.mean(div_anomaly_mask)
    diagnostics['div_anomaly_mask'] = div_anomaly_mask
    
    # 6. Check for velocity-density inconsistency (shear near density gradients)
    # Compute density gradients
    drho_dx = np.gradient(rho_fd, axis=0)
    drho_dy = np.gradient(rho_fd, axis=1)
    rho_grad_mag = np.sqrt(drho_dx**2 + drho_dy**2)
    normalized_rho_grad = rho_grad_mag / (rho_fd + 1e-6)
    
    # Regions with both high velocity shear AND density gradients (shear layers)
    # but NOT in high-density cores (which are physically valid)
    moderate_density = density_ratio < 1.5  # Exclude cores
    # Use adaptive threshold for shear layers too
    rho_grad_75 = np.percentile(normalized_rho_grad, 75)
    rho_grad_25 = np.percentile(normalized_rho_grad, 25)
    rho_grad_iqr = rho_grad_75 - rho_grad_25
    rho_grad_threshold = max(0.05, rho_grad_75 + 1.0 * rho_grad_iqr)  # Adaptive threshold
    shear_layer_mask = (normalized_shear > shear_threshold * 0.5) & (normalized_rho_grad > rho_grad_threshold) & moderate_density
    diagnostics['shear_layer_fraction'] = np.mean(shear_layer_mask)
    diagnostics['shear_layer_mask'] = shear_layer_mask
    
    # 7. Check for supersonic flow (only flag if not in cores)
    v_mag = np.sqrt(vx_fd**2 + vy_fd**2)
    mach = v_mag / cs
    # Only flag supersonic if not in high-density core (cores can have high velocities)
    supersonic_mask = (mach > 1.5) & (density_ratio < 1.5)
    diagnostics['supersonic_fraction'] = np.mean(supersonic_mask)
    diagnostics['max_mach'] = np.max(mach)
    diagnostics['supersonic_mask'] = supersonic_mask
    
    # 6. Compute error metrics
    epsilon = 200.0 * np.abs(rho_pinn - rho_fd) / (rho_pinn + rho_fd + 1e-6)
    diagnostics['mean_epsilon'] = np.mean(epsilon)
    diagnostics['median_epsilon'] = np.median(epsilon)
    diagnostics['max_epsilon'] = np.max(epsilon)
    diagnostics['p95_epsilon'] = np.percentile(epsilon, 95)
    
    # 8. Check for high error regions not explained by high density (instability indicator)
    # High error in moderate density regions suggests FD instability
    epsilon_75 = np.percentile(epsilon, 75)
    epsilon_25 = np.percentile(epsilon, 25)
    epsilon_iqr = epsilon_75 - epsilon_25
    # Use 95th percentile as threshold (more sensitive) for moderate density regions
    epsilon_threshold = np.percentile(epsilon, 95)
    # Flag regions with error > 95th percentile AND moderate density (not cores)
    high_error_moderate_density = (epsilon > epsilon_threshold) & (density_ratio < 1.5) & (density_ratio > 0.5)
    diagnostics['high_error_instability_fraction'] = np.mean(high_error_moderate_density)
    diagnostics['high_error_instability_mask'] = high_error_moderate_density
    diagnostics['epsilon_threshold'] = epsilon_threshold
    
    # 9. Create composite "invalid region" mask (regions where FD is unreliable)
    # Focus on velocity instabilities (shear) rather than high-density cores
    invalid_mask = (nan_mask | negative_mask | extreme_low_mask |
                   high_shear_mask | div_anomaly_mask | shear_layer_mask | supersonic_mask |
                   high_error_moderate_density)
    diagnostics['invalid_mask'] = invalid_mask
    diagnostics['invalid_fraction'] = np.mean(invalid_mask)
    
    # 8. Compute error statistics excluding invalid regions
    valid_epsilon = epsilon[~invalid_mask]
    if len(valid_epsilon) > 0:
        diagnostics['mean_epsilon_valid'] = np.mean(valid_epsilon)
        diagnostics['median_epsilon_valid'] = np.median(valid_epsilon)
        diagnostics['max_epsilon_valid'] = np.max(valid_epsilon)
        diagnostics['p95_epsilon_valid'] = np.percentile(valid_epsilon, 95)
    else:
        diagnostics['mean_epsilon_valid'] = np.nan
        diagnostics['median_epsilon_valid'] = np.nan
        diagnostics['max_epsilon_valid'] = np.nan
        diagnostics['p95_epsilon_valid'] = np.nan
    
    # 9. Overall quality score (0 = bad, 1 = good)
    quality_score = 1.0 - diagnostics['invalid_fraction']
    diagnostics['quality_score'] = quality_score
    
    if verbose:
        print("\n" + "="*60)
        print("FD SOLUTION DIAGNOSTICS")
        print("="*60)
        print(f"Overall Quality Score: {quality_score:.1%}")
        print(f"Invalid regions: {diagnostics['invalid_fraction']:.1%}")
        print()
        print("Issue breakdown:")
        print(f"  - NaN/Inf values: {diagnostics['nan_inf_fraction']:.1%}")
        print(f"  - Negative density: {diagnostics['negative_fraction']:.1%}")
        print(f"  - Extreme low density (<0.4x): {diagnostics['extreme_low_fraction']:.1%} (min ratio: {diagnostics['min_density_ratio']:.2f})")
        print(f"  - High velocity shear (instability): {diagnostics['high_shear_fraction']:.1%} (max shear: {diagnostics['max_shear']:.2f}, threshold: {diagnostics.get('shear_threshold', 0):.3f})")
        print(f"  - Velocity divergence anomalies: {diagnostics['div_anomaly_fraction']:.1%}")
        print(f"  - Shear layers (velocity+density gradients): {diagnostics['shear_layer_fraction']:.1%}")
        print(f"  - High error in moderate density (instability): {diagnostics['high_error_instability_fraction']:.1%} (threshold: {diagnostics.get('epsilon_threshold', 0):.1f}%)")
        print(f"  - Supersonic flow (outside cores): {diagnostics['supersonic_fraction']:.1%} (Max Mach: {diagnostics['max_mach']:.2f})")
        print(f"  - Density range: {diagnostics['min_density_ratio']:.2f} - {diagnostics['max_density_ratio']:.2f}x background")
        print()
        print("Error metrics (all regions):")
        print(f"  - Mean error: {diagnostics['mean_epsilon']:.2f}%")
        print(f"  - Median error: {diagnostics['median_epsilon']:.2f}%")
        print(f"  - 95th percentile: {diagnostics['p95_epsilon']:.2f}%")
        print(f"  - Max error: {diagnostics['max_epsilon']:.2f}%")
        print()
        if not np.isnan(diagnostics['mean_epsilon_valid']):
            print("Error metrics (valid regions only):")
            print(f"  - Mean error: {diagnostics['mean_epsilon_valid']:.2f}%")
            print(f"  - Median error: {diagnostics['median_epsilon_valid']:.2f}%")
            print(f"  - 95th percentile: {diagnostics['p95_epsilon_valid']:.2f}%")
            print(f"  - Max error: {diagnostics['max_epsilon_valid']:.2f}%")
        print()
        
        # Provide recommendations
        if quality_score < 0.90:
            print("[WARNING] FD solution quality is poor!")
            print("Recommendations:")
            if diagnostics['high_shear_fraction'] > 0.05:
                print("  - Velocity instabilities detected (shear layers)")
                print("  - Reduce Courant number (--nu) for better stability")
                print("  - Increase resolution (--N-fd) to resolve velocity gradients")
            if diagnostics['supersonic_fraction'] > 0.05:
                print("  - Supersonic flow detected outside cores")
                print("  - Reduce Courant number (--nu) for better stability")
            if diagnostics['invalid_fraction'] > 0.10:
                print("  - Time may be too late; FD solver reaching limits")
                print("  - Consider comparing at earlier times")
            print("  - Try different random seed if this is power spectrum")
        elif quality_score < 0.98:
            print("[INFO] FD solution has minor quality issues in some regions")
            print("Consider filtering error metrics to exclude invalid regions")
        else:
            print("[OK] FD solution quality is good")
        print("="*60)
    
    return diagnostics


def plot_diagnostic_masks(x: np.ndarray, y: np.ndarray, diagnostics: Dict,
                         rho_fd: np.ndarray, epsilon: np.ndarray,
                         vx_fd: Optional[np.ndarray] = None, vy_fd: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None):
    """
    Create visualization of diagnostic masks to identify problematic regions.
    
    Args:
        x, y: Grid coordinates
        diagnostics: Output from diagnose_fd_solution
        rho_fd: FD density field
        epsilon: Error metric field
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    XX, YY = np.meshgrid(x, y, indexing='ij')
    
    # Row 1: FD solution and individual issue masks
    ax = axes[0, 0]
    im = ax.pcolormesh(XX, YY, rho_fd, shading='auto', cmap='YlOrBr')
    ax.set_title("FD Density")
    plt.colorbar(im, ax=ax, label=r"$\rho$")
    
    ax = axes[0, 1]
    # Combine issue types: 1=Neg, 2=Low density, 3=High shear, 4=Shear layer, 5=High error instability, 6=Supersonic
    combined_mask = (diagnostics['negative_mask'].astype(int) + 
                    diagnostics['extreme_low_mask'].astype(int) * 2 +
                    diagnostics['high_shear_mask'].astype(int) * 3 +
                    diagnostics['shear_layer_mask'].astype(int) * 4 +
                    diagnostics.get('high_error_instability_mask', np.zeros_like(rho_fd)).astype(int) * 5 +
                    diagnostics['supersonic_mask'].astype(int) * 6)
    im = ax.pcolormesh(XX, YY, combined_mask, shading='auto', cmap='tab10', vmin=0, vmax=7)
    ax.set_title("Issue Type\n(0=OK, 1=Neg, 2=Low, 3=Shear, 4=Layer, 5=HighErr, 6=Supersonic)")
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 2]
    # Show velocity shear magnitude
    if vx_fd is not None and vy_fd is not None:
        dvx_dx = np.gradient(vx_fd, axis=0)
        dvx_dy = np.gradient(vx_fd, axis=1)
        dvy_dx = np.gradient(vy_fd, axis=0)
        dvy_dy = np.gradient(vy_fd, axis=1)
        v_shear = np.sqrt(dvx_dx**2 + dvx_dy**2 + dvy_dx**2 + dvy_dy**2)
        im = ax.pcolormesh(XX, YY, v_shear, shading='auto', cmap='RdYlBu_r')
        ax.set_title("Velocity Shear")
        plt.colorbar(im, ax=ax, label="Shear magnitude")
    else:
        # Fallback: show shear mask
        im = ax.pcolormesh(XX, YY, diagnostics['high_shear_mask'].astype(float), 
                          shading='auto', cmap='RdYlBu_r')
        ax.set_title("High Shear Regions")
        plt.colorbar(im, ax=ax, label="Has high shear")
    
    # Row 2: Error analysis
    ax = axes[1, 0]
    im = ax.pcolormesh(XX, YY, epsilon, shading='auto', cmap='coolwarm', vmax=50)
    ax.set_title("Error ε (%)")
    plt.colorbar(im, ax=ax, label="ε (%)")
    
    ax = axes[1, 1]
    im = ax.pcolormesh(XX, YY, diagnostics['invalid_mask'].astype(float),
                      shading='auto', cmap='RdYlGn_r')
    ax.set_title(f"Invalid Regions ({diagnostics['invalid_fraction']:.1%})")
    plt.colorbar(im, ax=ax, label="Is invalid")
    
    ax = axes[1, 2]
    # Masked error: show only valid regions
    epsilon_masked = epsilon.copy()
    epsilon_masked[diagnostics['invalid_mask']] = 0
    im = ax.pcolormesh(XX, YY, epsilon_masked, shading='auto', cmap='coolwarm', vmax=50)
    ax.set_title("Error (valid regions only)")
    plt.colorbar(im, ax=ax, label="ε (%)")
    ax.text(0.02, 0.98, f"Mean: {diagnostics['mean_epsilon_valid']:.2f}%", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostic plot saved to {save_path}")
    
    plt.show()
    return fig

