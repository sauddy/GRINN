import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import torch
import torch.nn as nn

# Example usage for rho as a callable function:
# For 1D: rho_func = lambda x: np.sin(x)  or  lambda x: torch.sin(x)
# For 2D: rho_func = lambda x, y: np.sin(x) * np.cos(y)
# For 3D: def rho_func(x, y, z):
#             X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
#             return 1.0 + 0.1 * np.sin(X) * np.cos(Y) * np.sin(Z)

G = 1

Lx = 2 * np.pi
Ly = 2 * np.pi
Lz = 2 * np.pi
Nx = 300

dimension = 3

neurons = 64
layers = 5
activation = nn.Tanh
iterations = 2000

def fft_solver(rho, Nx, Lx, Ny = None, Ly = None, Nz = None, Lz=None):

    if (Ly is not None and Ny is None) or (Ny is not None and Ly is None):
        raise ValueError("Both Ny and Ly must be provided together for 2D/3D")
    if (Lz is not None and Nz is None) or (Nz is not None and Lz is None):
        raise ValueError("Both Nz and Lz must be provided together for 3D")

    dx = Lx / Nx
    dy = Ly / Ny if Ly is not None else None
    dz = Lz / Nz if Lz is not None else None

    # If rho is callable, create coordinate arrays and call it
    if callable(rho):
        x = np.linspace(0, Lx, Nx)
        if Ly is not None:
            if Lz is not None:
                # 3D case
                y = np.linspace(0, Ly, Ny)
                z = np.linspace(0, Lz, Nz)
                rho = rho(x, y, z)
            else:
                # 2D case
                y = np.linspace(0, Ly, Ny)
                rho = rho(x, y)
        else:
            # 1D case
            rho = rho(x)

    if Ly is not None:
        if Lz is not None:
            rho_hat = np.fft.fftn(rho)
        else:
            rho_hat = np.fft.fft2(rho)
    else:
        rho_hat = np.fft.fft(rho)

    kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, dy) if dy is not None else None
    kz = 2 * np.pi * np.fft.fftfreq(Nz, dz) if dz is not None else None

    if ky is not None:
        if kz is not None:
            kx2, ky2, kz2 = np.meshgrid(kx**2, ky**2, kz**2, indexing='ij')
            laplace = -(kx2 + ky2 + kz2)
            laplace[0, 0, 0] = 1
        else:
            kx2, ky2 = np.meshgrid(kx**2, ky**2, indexing='ij')
            laplace = -(kx2 + ky2)
            laplace[0, 0] = 1
    else:
        laplace = -kx**2
        laplace[0] = 1

    phi_hat = 4 * np.pi * G * rho_hat / laplace

    if ky is not None:
        if kz is not None:
            phi_hat[0, 0, 0] = 0
            phi = np.real(np.fft.ifftn(phi_hat))
        else:
            phi_hat[0, 0] = 0
            phi = np.real(np.fft.ifft2(phi_hat))
    else:
        phi_hat[0] = 0
        phi = np.real(np.fft.ifft(phi_hat))
    
    #g = -np.gradient(phi, dx)  # Negative for gravity
    
    return phi

class PINN(nn.Module):
    def __init__(self, hidden_dim = neurons, num_layers = layers, activation_type = activation, dimension = dimension):
        super(PINN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_type = activation_type
        self.dimension = dimension

        if dimension == 1:
            in_dim = 1
        elif dimension == 2:
            in_dim = 2
        else:
            in_dim = 3
        
        out_dim = 1

        layers = []
        for i in range(num_layers - 1):
            if i == 0:
                layers.append(nn.Linear(in_features = in_dim, out_features = hidden_dim))
                layers.append(activation_type())
            else:
                layers.append(nn.Linear(in_features = hidden_dim, out_features = hidden_dim))
                layers.append(activation_type())

        layers.append(nn.Linear(in_features = hidden_dim, out_features = out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x, y = None, z = None):

        if self.dimension == 1:
            Nx = x.shape[0]

            grid = x.reshape(Nx, 1)
            return self.linear(grid).reshape(Nx)
        
        if self.dimension == 2:
            Nx = x.shape[0]
            Ny = y.shape[0]
            X, Y = torch.meshgrid(x, y, indexing='ij')

            grid = torch.stack([X, Y], dim = -1).reshape(Nx*Ny, 2)
            return self.linear(grid).reshape(Nx, Ny)
        
        if self.dimension == 3:
            Nx = x.shape[0]
            Ny = y.shape[0]
            Nz = z.shape[0]
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')

            grid = torch.stack([X, Y, Z], dim = -1).reshape(Nx*Ny*Nz, 3)
            return self.linear(grid).reshape(Nx, Ny, Nz)
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def pde_loss(x, model, dim, rho, G, y = None, z = None):

    phi = model(x, y, z)

    if dim == 1:
        phi_x = torch.autograd.grad(phi, x, create_graph = True, grad_outputs=torch.ones_like(phi))[0]
        phi_xx = torch.autograd.grad(phi_x, x, create_graph = True, grad_outputs=torch.ones_like(phi_x))[0]
        laplacian = phi_xx
        return ((laplacian - 4*np.pi*G*rho)**2).mean()

    if dim == 2:
        phi_x = torch.autograd.grad(phi, x, create_graph = True, grad_outputs=torch.ones_like(phi))[0]
        phi_xx = torch.autograd.grad(phi_x, x, create_graph = True, grad_outputs=torch.ones_like(phi_x))[0]
        phi_y = torch.autograd.grad(phi, y, create_graph = True, grad_outputs=torch.ones_like(phi))[0]
        phi_yy = torch.autograd.grad(phi_y, y, create_graph = True, grad_outputs=torch.ones_like(phi_y))[0]
        laplacian = phi_xx + phi_yy
        return ((laplacian - 4*np.pi*G*rho)**2).mean()

    if dim == 3:
        phi_x = torch.autograd.grad(phi, x, create_graph = True, grad_outputs=torch.ones_like(phi))[0]
        phi_xx = torch.autograd.grad(phi_x, x, create_graph = True, grad_outputs=torch.ones_like(phi_x))[0]
        phi_y = torch.autograd.grad(phi, y, create_graph = True, grad_outputs=torch.ones_like(phi))[0]
        phi_yy = torch.autograd.grad(phi_y, y, create_graph = True, grad_outputs=torch.ones_like(phi_y))[0]
        phi_z = torch.autograd.grad(phi, z, create_graph = True, grad_outputs=torch.ones_like(phi))[0]
        phi_zz = torch.autograd.grad(phi_z, z, create_graph = True, grad_outputs=torch.ones_like(phi_z))[0]
        laplacian = phi_xx + phi_yy + phi_zz
        return ((laplacian - 4*np.pi*G*rho)**2).mean()
    
def bc_loss(x, model, dim, y = None, z = None):

    if dim == 1:
        boundary_left = x[0].reshape(-1, 1)
        boundary_right = x[-1].reshape(-1, 1)

        phi_left = model.linear(boundary_left)
        phi_right = model.linear(boundary_right)

        return ((phi_right - phi_left)**2).mean()

    if dim == 2:
        boundary_left = torch.stack([torch.full(y.shape, x[0].item(), device=y.device), y], dim = -1)
        boundary_right = torch.stack([torch.full(y.shape, x[-1].item(), device=y.device), y], dim = -1)

        boundary_below = torch.stack([x, torch.full(x.shape, y[0].item(), device=x.device)], dim=-1)
        boundary_above = torch.stack([x, torch.full(x.shape, y[-1].item(), device=x.device)], dim=-1)

        phi_left = model.linear(boundary_left)
        phi_right = model.linear(boundary_right)

        phi_below = model.linear(boundary_below)
        phi_above = model.linear(boundary_above)

        return (((phi_right - phi_left)**2).mean() + ((phi_above - phi_below)**2).mean())
    
    if dim == 3:
        Y_left, Z_left = torch.meshgrid(y, z, indexing='ij')
        boundary_left = torch.stack([torch.full_like(Y_left, x[0].item()), Y_left, Z_left], dim=-1).reshape(-1, 3)
        
        Y_right, Z_right = torch.meshgrid(y, z, indexing='ij')
        boundary_right = torch.stack([torch.full_like(Y_right, x[-1].item()), Y_right, Z_right], dim=-1).reshape(-1, 3)
        
        X_bottom, Z_bottom = torch.meshgrid(x, z, indexing='ij')
        boundary_bottom = torch.stack([X_bottom, torch.full_like(X_bottom, y[0].item()), Z_bottom], dim=-1).reshape(-1, 3)
        
        X_top, Z_top = torch.meshgrid(x, z, indexing='ij')
        boundary_top = torch.stack([X_top, torch.full_like(X_top, y[-1].item()), Z_top], dim=-1).reshape(-1, 3)
        
        X_below, Y_below = torch.meshgrid(x, y, indexing='ij')
        boundary_below = torch.stack([X_below, Y_below, torch.full_like(X_below, z[0].item())], dim=-1).reshape(-1, 3)
        
        X_above, Y_above = torch.meshgrid(x, y, indexing='ij')
        boundary_above = torch.stack([X_above, Y_above, torch.full_like(X_above, z[-1].item())], dim=-1).reshape(-1, 3)

        phi_left = model.linear(boundary_left)
        phi_right = model.linear(boundary_right)

        phi_bottom = model.linear(boundary_bottom)
        phi_top = model.linear(boundary_top)

        phi_below = model.linear(boundary_below)
        phi_above = model.linear(boundary_above)

        return (((phi_right - phi_left)**2).mean() + ((phi_top - phi_bottom)**2).mean() + ((phi_above - phi_below)**2).mean())

def train(dim, Lx, Nx, rho, G, neurons, layers, activation, iterations, Ly = None, Ny = None, Lz = None, Nz = None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dim == 1:
        x = torch.linspace(0, Lx, Nx, requires_grad = True).to(device)
        
        # If rho is callable, evaluate it at the coordinate points
        rho_field = rho(x) if callable(rho) else rho
        rho_field = rho_field.detach()

        model = PINN(neurons, layers, activation, dim).to(device)
        model.apply(init_weights)
        optimizer_adam = torch.optim.Adam(model.parameters(), lr = 1e-4)

        for i in range(iterations):
            optimizer_adam.zero_grad()
            
            loss = pde_loss(x, model, dim, rho_field, G) + bc_loss(x, model, dim)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
            loss.backward()
            
            optimizer_adam.step()

        return model
    
    if dim == 2:
        x = torch.linspace(0, Lx, Nx, requires_grad = True).to(device)
        y = torch.linspace(0, Ly, Ny, requires_grad = True).to(device)
        
        # If rho is callable, evaluate it at the coordinate points
        rho_field = rho(x, y) if callable(rho) else rho
        rho_field = rho_field.detach()

        model = PINN(neurons, layers, activation, dim).to(device)
        model.apply(init_weights)
        optimizer_adam = torch.optim.Adam(model.parameters(), lr = 1e-4)

        for i in range(iterations):
            optimizer_adam.zero_grad()
            
            loss = pde_loss(x, model, dim, rho_field, G, y) + bc_loss(x, model, dim, y)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
            loss.backward()
            
            optimizer_adam.step()

        return model

    if dim == 3:
        x = torch.linspace(0, Lx, Nx, requires_grad = True).to(device)
        y = torch.linspace(0, Ly, Ny, requires_grad = True).to(device)
        z = torch.linspace(0, Lz, Nz, requires_grad = True).to(device)
        
        # If rho is callable, evaluate it at the coordinate points
        rho_field = rho(x, y, z) if callable(rho) else rho
        rho_field = rho_field.detach()

        model = PINN(neurons, layers, activation, dim).to(device)
        model.apply(init_weights)
        optimizer_adam = torch.optim.Adam(model.parameters(), lr = 1e-4)

        for i in range(iterations):
            optimizer_adam.zero_grad()
            
            loss = pde_loss(x, model, dim, rho_field, G, y, z) + bc_loss(x, model, dim, y, z)
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}")
            loss.backward()
            
            optimizer_adam.step()

        return model
    
if __name__ == "__main__":
    # Test parameters - Memory efficiency comparison
    # FFT: High resolution (memory intensive)
    # PINN: Train on low resolution, evaluate at high resolution
    
    dim = 3
    Lx = 2 * np.pi
    Ly = 2 * np.pi
    Lz = 2 * np.pi
    
    # FFT uses high resolution
    Nx_fft = 128
    Ny_fft = 128
    Nz_fft = 128
    
    # PINN trains on low resolution (memory efficient)
    Nx_pinn_train = 24
    Ny_pinn_train = 24
    Nz_pinn_train = 24
    
    # PINN evaluates at high resolution (same as FFT for comparison)
    Nx_pinn_eval = 128
    Ny_pinn_eval = 128
    Nz_pinn_eval = 128
    
    print("="*70)
    print("MEMORY EFFICIENCY TEST: PINN vs FFT")
    print("="*70)
    print(f"FFT Resolution:           {Nx_fft}³ = {Nx_fft**3:,} points")
    print(f"PINN Training Resolution: {Nx_pinn_train}³ = {Nx_pinn_train**3:,} points")
    print(f"PINN Eval Resolution:     {Nx_pinn_eval}³ = {Nx_pinn_eval**3:,} points")
    print(f"Memory Ratio:             {(Nx_pinn_eval/Nx_pinn_train)**3:.1f}x more points with same memory!")
    print("="*70)
    
    # Define 3D density perturbation: sin(2x)*sin(3y)*sin(4z)
    def rho_test(x, y=None, z=None):
        # Check if it's a torch tensor or numpy array and use appropriate function
        if isinstance(x, torch.Tensor):
            if z is not None:  # 3D case
                X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
                return torch.sin(2*X) * torch.sin(3*Y) * torch.sin(4*Z)
            elif y is not None:  # 2D case
                X, Y = torch.meshgrid(x, y, indexing='ij')
                return torch.sin(2*X) * torch.sin(3*Y)
            else:  # 1D case
                return torch.sin(2*x)
        else:  # numpy
            if z is not None:  # 3D case
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                return np.sin(2*X) * np.sin(3*Y) * np.sin(4*Z)
            elif y is not None:  # 2D case
                X, Y = np.meshgrid(x, y, indexing='ij')
                return np.sin(2*X) * np.sin(3*Y)
            else:  # 1D case
                return np.sin(2*x)
    
    # Test FFT solver at HIGH resolution
    print("\n[1/3] Running FFT solver at HIGH resolution...")
    start_fft = time.time()
    phi_fft = fft_solver(rho_test, Nx_fft, Lx, Ny_fft, Ly, Nz_fft, Lz)
    fft_time = time.time() - start_fft
    fft_memory_mb = phi_fft.nbytes / (1024**2)
    print(f"      ✓ FFT solution: {phi_fft.shape}, Memory: {fft_memory_mb:.2f} MB")
    print(f"      ✓ Time: {fft_time:.4f} seconds")
    
    # Test PINN: Train on LOW resolution
    print(f"\n[2/3] Training PINN on LOW resolution ({Nx_pinn_train}³)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Using device: {device}")
    
    start_pinn = time.time()
    model = train(dim, Lx, Nx_pinn_train, rho_test, G, neurons, layers, activation, iterations, 
                  Ly, Ny_pinn_train, Lz, Nz_pinn_train)
    train_time = time.time() - start_pinn
    
    # Calculate model size
    model_params = sum(p.numel() for p in model.parameters())
    model_memory_mb = model_params * 4 / (1024**2)  # 4 bytes per float32
    print(f"      ✓ Model trained: {model_params:,} parameters, Memory: {model_memory_mb:.2f} MB")
    print(f"      ✓ Training time: {train_time:.4f} seconds")
    
    # Evaluate PINN at HIGH resolution (same as FFT)
    print(f"\n[3/3] Evaluating PINN at HIGH resolution ({Nx_pinn_eval}³)...")
    x_eval = torch.linspace(0, Lx, Nx_pinn_eval, requires_grad=False).to(device)
    y_eval = torch.linspace(0, Ly, Ny_pinn_eval, requires_grad=False).to(device)
    z_eval = torch.linspace(0, Lz, Nz_pinn_eval, requires_grad=False).to(device)
    
    start_eval = time.time()
    with torch.no_grad():
        phi_pinn = model(x_eval, y_eval, z_eval).cpu().numpy()
    eval_time = time.time() - start_eval
    pinn_total_time = train_time + eval_time
    
    print(f"      ✓ PINN evaluated: {phi_pinn.shape}")
    print(f"      ✓ Evaluation time: {eval_time:.4f} seconds")
    print(f"      ✓ Total PINN time: {pinn_total_time:.4f} seconds")
    
    # Normalize both solutions (remove constant offset)
    phi_pinn_normalized = phi_pinn - np.mean(phi_pinn)
    phi_fft_normalized = phi_fft - np.mean(phi_fft)
    
    # Calculate errors
    mse = np.mean((phi_pinn_normalized - phi_fft_normalized)**2)
    max_error = np.max(np.abs(phi_pinn_normalized - phi_fft_normalized))
    relative_error = np.linalg.norm(phi_pinn_normalized - phi_fft_normalized) / np.linalg.norm(phi_fft_normalized)
    
    print(f"\n{'='*70}")
    print(f"RESULTS: MEMORY EFFICIENCY vs ACCURACY TRADEOFF")
    print(f"{'='*70}")
    print(f"\nMemory Usage:")
    print(f"  FFT Grid Memory:       {fft_memory_mb:.2f} MB ({Nx_fft}³ points)")
    print(f"  PINN Model Memory:     {model_memory_mb:.2f} MB ({model_params:,} params)")
    print(f"  Memory Savings:        {fft_memory_mb/model_memory_mb:.1f}x less memory!")
    
    print(f"\nTiming:")
    print(f"  FFT Solver:            {fft_time:.4f} seconds")
    print(f"  PINN Training:         {train_time:.4f} seconds")
    print(f"  PINN Evaluation:       {eval_time:.4f} seconds")
    print(f"  PINN Total:            {pinn_total_time:.4f} seconds")
    print(f"  Time Ratio:            {pinn_total_time/fft_time:.2f}x (FFT is faster)")
    
    print(f"\nAccuracy (PINN trained on {Nx_pinn_train}³, evaluated at {Nx_pinn_eval}³):")
    print(f"  Mean Squared Error:    {mse:.6f}")
    print(f"  Max Absolute Error:    {max_error:.6f}")
    print(f"  Relative L2 Error:     {relative_error:.4f} ({relative_error*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"CONCLUSION:")
    print(f"  PINN achieves {relative_error*100:.2f}% error while using {fft_memory_mb/model_memory_mb:.1f}x less memory")
    print(f"  Trained on only {(Nx_pinn_train/Nx_pinn_eval)**3*100:.1f}% of the evaluation points!")
    print(f"{'='*70}")
    
    # Create visualization
    if dim == 1:
        x_plot = np.linspace(0, Lx, Nx_pinn_eval)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'FFT vs PINN Comparison | FFT: {fft_time:.3f}s, PINN: {pinn_total_time:.3f}s | Rel. Error: {relative_error:.4f}', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: FFT solution
        axes[0, 0].plot(x_plot, phi_fft_normalized, 'b-', linewidth=2, label='FFT Solution')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('φ (normalized)')
        axes[0, 0].set_title('FFT Solution')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: PINN solution
        axes[0, 1].plot(x_plot, phi_pinn_normalized, 'r-', linewidth=2, label='PINN Solution')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('φ (normalized)')
        axes[0, 1].set_title('PINN Solution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Comparison (overlay)
        axes[1, 0].plot(x_plot, phi_fft_normalized, 'b-', linewidth=2, label='FFT', alpha=0.7)
        axes[1, 0].plot(x_plot, phi_pinn_normalized, 'r--', linewidth=2, label='PINN', alpha=0.7)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('φ (normalized)')
        axes[1, 0].set_title('FFT vs PINN Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Absolute error
        error_abs = np.abs(phi_pinn_normalized - phi_fft_normalized)
        axes[1, 1].plot(x_plot, error_abs, 'g-', linewidth=2)
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('|φ_PINN - φ_FFT|')
        axes[1, 1].set_title(f'Absolute Error (MSE: {mse:.4f})')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
    elif dim == 3:
        # For 3D, show 2D slices at z = Lz/2
        mid_z = Nx_pinn_eval // 2
        x_plot = np.linspace(0, Lx, Nx_pinn_eval)
        y_plot = np.linspace(0, Ly, Ny_pinn_eval)
        X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        title = f'Memory Efficiency Test: PINN (trained {Nx_pinn_train}³, eval {Nx_pinn_eval}³) vs FFT ({Nx_fft}³)\n'
        title += f'FFT: {fft_time:.3f}s | PINN: {pinn_total_time:.3f}s | Error: {relative_error:.4f} | Memory: {fft_memory_mb/model_memory_mb:.1f}x savings'
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Plot 1: FFT solution (2D slice)
        im1 = axes[0, 0].contourf(X, Y, phi_fft_normalized[:, :, mid_z], levels=20, cmap='viridis')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title(f'FFT Solution (z={Lz/2:.2f})')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot 2: PINN solution (2D slice)
        im2 = axes[0, 1].contourf(X, Y, phi_pinn_normalized[:, :, mid_z], levels=20, cmap='viridis')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_title(f'PINN Solution (z={Lz/2:.2f})')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Plot 3: Difference (2D slice)
        diff_slice = phi_pinn_normalized[:, :, mid_z] - phi_fft_normalized[:, :, mid_z]
        im3 = axes[1, 0].contourf(X, Y, diff_slice, levels=20, cmap='RdBu_r')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title('Difference (PINN - FFT)')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot 4: Absolute error (2D slice)
        error_slice = np.abs(diff_slice)
        im4 = axes[1, 1].contourf(X, Y, error_slice, levels=20, cmap='hot')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title(f'Absolute Error (MSE: {mse:.4f})')
        plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    #plt.savefig('fft_vs_pinn_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as 'fft_vs_pinn_comparison.png'")
    plt.show()