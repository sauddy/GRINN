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
    Uses NumPy's RNG for consistency with the CPU solver, then converts to PyTorch tensors.
    """
    # Use NumPy's random generator for consistency
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    def synthesize_component():
        # Generate random field using NumPy
        field_np = rng.standard_normal((nx, ny))
        field = torch.from_numpy(field_np).to(device=device, dtype=dtype)
        F = torch.fft.fft2(field)
        
        kx = 2 * np.pi * torch.fft.fftfreq(nx, d=Lx / nx).to(device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, d=Ly / ny).to(device)
        kxg, kyg = torch.meshgrid(kx, ky, indexing='ij')
        
        kk = torch.sqrt(kxg**2 + kyg**2)
        kk[0, 0] = 1.0  # Avoid division by zero
        
        filt = kk**(power_index / 2.0)
        filt[kk == 0] = 0.0
        
        F_filtered = F * filt
        comp = torch.real(torch.fft.ifft2(F_filtered))
        comp -= torch.mean(comp)
        
        std = torch.std(comp)
        if std > 0:
            comp = comp * (amplitude / std)
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
    dt = nu * dx / c_s
    
    n = int(time_val / dt)
    mux = dt / (2 * dx)
    muy = dt / (2 * dy)

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

    # Main time-stepping loop
    for k in range(1, n):
        rho1 = (0.25) * (torch.roll(rho0, -1, dims=0) + torch.roll(rho0, 1, dims=0) +
                         torch.roll(rho0, -1, dims=1) + torch.roll(rho0, 1, dims=1)) \
            - (mux * (torch.roll(rho0, -1, dims=0) * torch.roll(vx0, -1, dims=0) - torch.roll(rho0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) \
            - (muy * (torch.roll(rho0, -1, dims=1) * torch.roll(vy0, -1, dims=1) - torch.roll(rho0, 1, dims=1) * torch.roll(vy0, 1, dims=1)))

        if gravity == False:
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
        
        # Update time step based on maximum signal speed
        # Match NumPy exactly: np.max([abs(vx1), abs(vy1)]) takes max of two scalars
        vmax = max(torch.max(torch.abs(vx1)).item(), torch.max(torch.abs(vy1)).item())
        dt1 = nu * dx / vmax if vmax != 0 else nu * dx / c_s
        dt2 = nu * dx / c_s
        dt = min(dt1, dt2)
        mux = dt / (2 * dx)
        muy = dt / (2 * dy)
        
        # Update n dynamically (NumPy recalculates n but loop continues with original range)
        n = int(time_val / dt) if dt > 0 else n
        
        rho0, vx0, vy0 = rho1, vx1, vy1
        Px0, Py0 = Px1, Py1
        if gravity:
            phi0 = phi1
        
    rho_max = torch.max(rho1).item()
    
    return x.cpu().numpy(), rho1.cpu().numpy(), vx1.cpu().numpy(), vy1.cpu().numpy(), None, n, rho_max
