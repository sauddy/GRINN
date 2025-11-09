import numpy as np
import torch
from config import cs, rho_o, const, G

# Device setup
has_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if has_gpu else "cpu")
dtype = torch.float64
print(f"Using device: {device}")

def fft_solver_torch(rho, Lx, nx, Ly, ny):
    """
    PyTorch FFT solver for Poisson equation (gravitational potential).
    """
    dx = Lx / nx
    dy = Ly / ny
    
    # Calculate the Fourier modes of the gas density
    rhohat = torch.fft.fft2(rho)
    
    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
    ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
    
    # Construct the Laplacian operator in Fourier space
    # Match NumPy exactly: default meshgrid uses 'xy' which gives (ny, nx)
    # But we need to transpose to match FFT2 output (nx, ny)
    kx2, ky2 = torch.meshgrid(kx**2, ky**2, indexing='xy')
    # NumPy meshgrid('xy') creates (ny, nx), but FFT2 output is (nx, ny)
    # So we transpose to match the FFT layout
    laplace = -(kx2.T + ky2.T)
    
    # Handle zero mode (k=0) - set to small value to avoid division by zero
    laplace = torch.where(laplace == 0, torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    
    # Solve for the potential in Fourier space
    phihat = rhohat / laplace
    
    # Transform back to real space
    phi = torch.real(torch.fft.ifft2(phihat))
    
    return phi

def generate_velocity_field_power_spectrum_torch(nx, ny, Lx, Ly, power_index=-3.0, amplitude=0.02, random_seed=None):
    """
    PyTorch implementation for generating a 2D velocity field with a power-law spectrum.
    
    This function generates resolution-independent initial conditions by:
    1. Synthesizing at fixed high resolution (1024x1024)
    2. Applying power-law filter with sharp cutoff at target Nyquist frequency
    3. Downsampling to target resolution via interpolation
    
    This ensures that N=300 and N=400 runs start with the same physical velocity field,
    just sampled at different resolutions, making convergence studies meaningful.
    """
    # Use a fixed, high-resolution grid for synthesis to ensure resolution independence
    hires_nx, hires_ny = 1024, 1024
    
    # Use NumPy's random generator for consistency with CPU solver
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    def synthesize_component():
        # 1. Generate random field at high resolution using NumPy (for consistency)
        field_np = rng.standard_normal((hires_nx, hires_ny))
        field = torch.from_numpy(field_np).to(device=device, dtype=dtype)
        F = torch.fft.fft2(field)
        
        # 2. Construct k-space grid at high resolution
        kx = 2 * np.pi * torch.fft.fftfreq(hires_nx, d=Lx / hires_nx).to(device)
        ky = 2 * np.pi * torch.fft.fftfreq(hires_ny, d=Ly / hires_ny).to(device)
        kxg, kyg = torch.meshgrid(kx, ky, indexing='ij')
        
        kk = torch.sqrt(kxg**2 + kyg**2)
        kk[0, 0] = 1.0  # Avoid division by zero
        
        # 3. Apply power-law filter
        filt = kk**(power_index / 2.0)
        filt[kk == 0] = 0.0
        
        # 4. KEY: Apply sharp cutoff at target Nyquist frequency to prevent aliasing
        k_nyquist = np.pi * nx / Lx  # Target grid's Nyquist frequency
        filt[kk > k_nyquist] = 0.0   # Remove unresolvable modes
        
        F_filtered = F * filt
        
        # 5. Transform back to real space at high resolution
        comp_hires = torch.real(torch.fft.ifft2(F_filtered))
        
        # 6. Normalize the high-resolution field
        comp_hires -= torch.mean(comp_hires)
        std = torch.std(comp_hires)
        if std > 0:
            comp_hires = comp_hires * (amplitude / std)
        
        # 7. Downsample to target resolution via interpolation
        # Convert to NumPy for interpolation (scipy doesn't work with torch tensors)
        comp_hires_np = comp_hires.cpu().numpy()
        
        from scipy.interpolate import RegularGridInterpolator
        x_hires = np.linspace(0, Lx, hires_nx, endpoint=False)
        y_hires = np.linspace(0, Ly, hires_ny, endpoint=False)
        x_lores = np.linspace(0, Lx, nx, endpoint=False)
        y_lores = np.linspace(0, Ly, ny, endpoint=False)
        
        interp = RegularGridInterpolator((x_hires, y_hires), comp_hires_np, 
                                        method='linear', bounds_error=False, fill_value=0.0)
        X_lores, Y_lores = np.meshgrid(x_lores, y_lores, indexing='ij')
        comp_np = interp((X_lores, Y_lores))
        
        # Convert back to torch tensor
        comp = torch.from_numpy(comp_np).to(device=device, dtype=dtype)
        
        return comp

    vx0 = synthesize_component()
    vy0 = synthesize_component()
    return vx0, vy0

def lax_solution_torch(time_val, N, nu, lam, num_of_waves, rho_1, gravity=False, use_velocity_ps=False, 
                         ps_index=-3.0, vel_rms=0.02, random_seed=None):
    """
    PyTorch implementation of the LAX method for solving hydrodynamic equations.
    This version is designed to run on a GPU for accelerated computation.
    """
    # Grid and Domain Parameters
    Lx = Ly = lam * num_of_waves
    c_s = cs

    # Grid setup
    Nx = Ny = N
    dx = dy = float(Lx / Nx)

    # Tensors Initialization
    # Use linspace and slice to match NumPy's endpoint=False behavior
    x = torch.linspace(0, Lx, Nx+1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0, Ly, Ny+1, device=device, dtype=dtype)[:-1]
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    rho0 = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    vx0 = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    vy0 = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    
    # Initial Conditions
    if use_velocity_ps:
        rho0 = rho_o * torch.ones((Nx, Ny), device=device, dtype=dtype)
        vx0, vy0 = generate_velocity_field_power_spectrum_torch(Nx, Ny, Lx, Ly, 
                                                              power_index=ps_index, 
                                                              amplitude=vel_rms, 
                                                              random_seed=random_seed)
    else:
        # This part is not being used by the calling script, so it's not converted for now.
        raise NotImplementedError("Only power spectrum initial conditions are supported in the Torch version.")

    rho1 = rho0.clone()
    vx1 = vx0.clone()
    vy1 = vy0.clone()

    Px0 = rho0 * vx0
    Py0 = rho0 * vy0
    
    # Initialize gravitational potential if needed
    phi0 = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    phi1 = torch.zeros((Nx, Ny), device=device, dtype=dtype)
    
    if gravity:
        # Calculate initial potential using FFT solver
        phi0 = fft_solver_torch(const * (rho0 - rho_o), Lx, Nx, Ly, Ny)

    # --- CORRECTED TIME-STEPPING LOOP ---
    t = 0.0
    k = 0
    # Initial dt for the first step
    vmax_initial = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item(), c_s)
    dt = nu * dx / vmax_initial

    while t < time_val:
        # Ensure the last step doesn't overshoot the final time
        if t + dt > time_val:
            dt = time_val - t

        mux = dt / (2 * dx)
        muy = dt / (2 * dy)

        # Evolve one step
        rho1 = (0.25) * (torch.roll(rho0, -1, dims=0) + torch.roll(rho0, 1, dims=0) +
                         torch.roll(rho0, -1, dims=1) + torch.roll(rho0, 1, dims=1)) \
            - (mux * (torch.roll(rho0, -1, dims=0) * torch.roll(vx0, -1, dims=0) - torch.roll(rho0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) \
            - (muy * (torch.roll(rho0, -1, dims=1) * torch.roll(vy0, -1, dims=1) - torch.roll(rho0, 1, dims=1) * torch.roll(vy0, 1, dims=1)))

        if not gravity:
            Px1 = (0.25) * (torch.roll(Px0, -1, dims=0) + torch.roll(Px0, 1, dims=0) +
                            torch.roll(Px0, -1, dims=1) + torch.roll(Px0, 1, dims=1)) \
                - (mux * (torch.roll(Px0, -1, dims=0) * torch.roll(vx0, -1, dims=0) - torch.roll(Px0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) \
                - (muy * (torch.roll(Px0, -1, dims=1) * torch.roll(vy0, -1, dims=1) - torch.roll(Px0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) \
                - ((c_s**2) * mux * (torch.roll(rho0, -1, dims=0) - torch.roll(rho0, 1, dims=0)))

            Py1 = (0.25) * (torch.roll(Py0, -1, dims=0) + torch.roll(Py0, 1, dims=0) +
                            torch.roll(Py0, -1, dims=1) + torch.roll(Py0, 1, dims=1)) \
                - (muy * (torch.roll(Py0, -1, dims=1) * torch.roll(vy0, -1, dims=1) - torch.roll(Py0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) \
                - (mux * (torch.roll(Py0, -1, dims=0) * torch.roll(vx0, -1, dims=0) - torch.roll(Py0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) \
                - ((c_s**2) * muy * (torch.roll(rho0, -1, dims=1) - torch.roll(rho0, 1, dims=1)))
        else:
            # With self-gravity
            Px1 = (0.25) * (torch.roll(Px0, -1, dims=0) + torch.roll(Px0, 1, dims=0) +
                            torch.roll(Px0, -1, dims=1) + torch.roll(Px0, 1, dims=1)) \
                - (mux * (torch.roll(Px0, -1, dims=0) * torch.roll(vx0, -1, dims=0) - torch.roll(Px0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) \
                - (muy * (torch.roll(Px0, -1, dims=1) * torch.roll(vy0, -1, dims=1) - torch.roll(Px0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) \
                - ((c_s**2) * mux * (torch.roll(rho0, -1, dims=0) - torch.roll(rho0, 1, dims=0))) \
                - (mux * rho0 * (torch.roll(phi0, -1, dims=0) - torch.roll(phi0, 1, dims=0)))

            Py1 = (0.25) * (torch.roll(Py0, -1, dims=0) + torch.roll(Py0, 1, dims=0) +
                            torch.roll(Py0, -1, dims=1) + torch.roll(Py0, 1, dims=1)) \
                - (muy * (torch.roll(Py0, -1, dims=1) * torch.roll(vy0, -1, dims=1) - torch.roll(Py0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) \
                - (mux * (torch.roll(Py0, -1, dims=0) * torch.roll(vx0, -1, dims=0) - torch.roll(Py0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) \
                - ((c_s**2) * muy * (torch.roll(rho0, -1, dims=1) - torch.roll(rho0, 1, dims=1))) \
                - (muy * rho0 * (torch.roll(phi0, -1, dims=1) - torch.roll(phi0, 1, dims=1)))
            
            # Update gravitational potential
            phi1 = fft_solver_torch(const * (rho1 - rho_o), Lx, Nx, Ly, Ny)

        vx1 = Px1 / rho1
        vy1 = Py1 / rho1
        
        # Update state for next iteration
        rho0, vx0, vy0 = rho1, vx1, vy1
        Px0, Py0 = Px1, Py1
        if gravity:
            phi0 = phi1
        
        # Increment time and step counter
        t += dt
        k += 1

        # Calculate dt for the *next* step
        vmax = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item())
        dt1 = nu * dx / vmax if vmax > 1e-9 else float('inf')
        dt2 = nu * dx / c_s
        dt = min(dt1, dt2)

    n = k
    rho_max = torch.max(rho0).item()
    
    return x.cpu().numpy(), rho0.cpu().numpy(), vx0.cpu().numpy(), vy0.cpu().numpy(), None, n, rho_max
