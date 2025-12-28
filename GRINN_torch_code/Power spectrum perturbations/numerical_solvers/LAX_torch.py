import numpy as np
import torch
from config import cs, rho_o, const, G, KX, KY, KZ

# Device setup - check at module import
has_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if has_gpu else "cpu")
dtype = torch.float64
if has_gpu:
    print(f"LAX_torch: GPU available, using device: {device}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
else:
    print(f"LAX_torch: No GPU available, using device: {device}")

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
    
    # Construct the discrete Laplacian operator in Fourier space
    # This ensures consistency with finite-difference gradients used in LAX scheme
    kx_mesh, ky_mesh = torch.meshgrid(kx, ky, indexing='xy')
    # Transpose to match FFT2 output layout (nx, ny)
    laplace = (2*(torch.cos(kx_mesh.T*dx)-1)/(dx**2) + 
               2*(torch.cos(ky_mesh.T*dy)-1)/(dy**2))
    
    # Handle zero mode (k=0) - set to small value to avoid division by zero
    laplace = torch.where(laplace == 0, torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    
    # Solve for the potential in Fourier space
    phihat = rhohat / laplace
    
    # Transform back to real space
    phi = torch.real(torch.fft.ifft2(phihat))
    
    return phi

def fft_solver_torch_3d(rho, Lx, nx, Ly, ny, Lz, nz):
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    rhohat = torch.fft.fftn(rho)
    kx = 2 * np.pi * torch.fft.fftfreq(nx, d=dx).to(device)
    ky = 2 * np.pi * torch.fft.fftfreq(ny, d=dy).to(device)
    kz = 2 * np.pi * torch.fft.fftfreq(nz, d=dz).to(device)
    kx_mesh, ky_mesh, kz_mesh = torch.meshgrid(kx, ky, kz, indexing='ij')
    # Use discrete Laplacian for consistency with finite-difference scheme
    laplace = (2*(torch.cos(kx_mesh*dx)-1)/(dx**2) + 
               2*(torch.cos(ky_mesh*dy)-1)/(dy**2) + 
               2*(torch.cos(kz_mesh*dz)-1)/(dz**2))
    laplace = torch.where(laplace == 0, torch.tensor(1e-9, device=device, dtype=dtype), laplace)
    phihat = rhohat / laplace
    phi = torch.real(torch.fft.ifftn(phihat))
    return phi

def generate_velocity_field_power_spectrum_torch(nx, ny, Lx, Ly, power_index=-3.0, amplitude=0.02, random_seed=None):
    """
    PyTorch implementation for generating a 2D velocity field with a power-law spectrum.
    
    This function generates velocity fields directly at the target resolution (nx, ny)
    without any downsampling or cutoff strategies.
    """
    # Use NumPy's random generator for consistency with CPU solver
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    def synthesize_component():
        # 1. Generate random field at target resolution using NumPy (for consistency)
        field_np = rng.standard_normal((nx, ny))
        field = torch.from_numpy(field_np).to(device=device, dtype=dtype)
        F = torch.fft.fft2(field)
        
        # 2. Construct k-space grid at target resolution
        kx = 2 * np.pi * torch.fft.fftfreq(nx, d=Lx / nx).to(device)
        ky = 2 * np.pi * torch.fft.fftfreq(ny, d=Ly / ny).to(device)
        kxg, kyg = torch.meshgrid(kx, ky, indexing='ij')
        
        kk = torch.sqrt(kxg**2 + kyg**2)
        kk[0, 0] = 1.0  # Avoid division by zero
        
        # 3. Apply power-law filter
        filt = kk**(power_index / 2.0)
        filt[kk == 0] = 0.0
        
        F_filtered = F * filt
        
        # 4. Transform back to real space
        comp = torch.real(torch.fft.ifft2(F_filtered))
        
        # 5. Normalize the field
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
    # Verify device is still correct (in case CUDA becomes available after import)
    current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if current_device != device:
        print(f"Warning: Device changed from {device} to {current_device}")
    
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
        # Sinusoidal perturbations: Use 2D wave pattern: cos(KX*x + KY*y)
        KX_tensor = torch.tensor(KX, device=device, dtype=dtype)
        KY_tensor = torch.tensor(KY, device=device, dtype=dtype)
        rho0 = rho_o + rho_1 * torch.cos(KX_tensor * xx + KY_tensor * yy)
    
    # Copy initial conditions to rho1, vx1, vy1 for t=0 case
    rho1 = rho0.clone()
    vx1 = vx0.clone()
    vy1 = vy0.clone()

    # Set velocity initial conditions for sinusoidal perturbations
    if not use_velocity_ps:
        if not gravity:
            # No gravity case
            v_1 = (c_s * rho_1) / rho_o  # velocity perturbation
            k_magnitude = torch.sqrt(KX_tensor**2 + KY_tensor**2)
            if k_magnitude > 0:
                vx0 = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy) * (KX_tensor / k_magnitude)
                vy0 = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy) * (KY_tensor / k_magnitude)
            else:
                vx0 = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy)
                vy0 = torch.zeros_like(xx)
        else:
            # Gravity case: need to check Jeans length
            jeans = torch.sqrt(torch.tensor(4 * np.pi**2 * c_s**2 / (const * G * rho_o), device=device, dtype=dtype))
            
            if lam >= jeans.item():
                # Gravitational instability case
                alpha = torch.sqrt(torch.tensor(const * G * rho_o - c_s**2 * (2 * np.pi / lam)**2, device=device, dtype=dtype))
                v_1 = (rho_1 / rho_o) * (alpha / (2 * np.pi / lam))
                k_magnitude = torch.sqrt(KX_tensor**2 + KY_tensor**2)
                if k_magnitude > 0:
                    vx0 = -v_1 * torch.sin(KX_tensor * xx + KY_tensor * yy) * (KX_tensor / k_magnitude)
                    vy0 = -v_1 * torch.sin(KX_tensor * xx + KY_tensor * yy) * (KY_tensor / k_magnitude)
                else:
                    vx0 = -v_1 * torch.sin(KX_tensor * xx + KY_tensor * yy)
                    vy0 = torch.zeros_like(xx)
            else:
                # Oscillatory regime
                alpha = torch.sqrt(torch.tensor(c_s**2 * (2 * np.pi / lam)**2 - const * G * rho_o, device=device, dtype=dtype))
                v_1 = (rho_1 / rho_o) * (alpha / (2 * np.pi / lam))
                k_magnitude = torch.sqrt(KX_tensor**2 + KY_tensor**2)
                if k_magnitude > 0:
                    vx0 = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy) * (KX_tensor / k_magnitude)
                    vy0 = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy) * (KY_tensor / k_magnitude)
                else:
                    vx0 = v_1 * torch.cos(KX_tensor * xx + KY_tensor * yy)
                    vy0 = torch.zeros_like(xx)
        
        # Update vx1 and vy1 after setting initial velocities
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


def lax_solution_3d_sinusoidal_torch(time_val, N, nu, lam, num_of_waves, rho_1, gravity=True):
    current_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Nx = Ny = Nz = int(N)
    Lx = Ly = Lz = lam * num_of_waves
    dx = float(Lx / Nx)
    dy = float(Ly / Ny)
    dz = float(Lz / Nz)
    c_s = cs

    x = torch.linspace(0, Lx, Nx+1, device=device, dtype=dtype)[:-1]
    y = torch.linspace(0, Ly, Ny+1, device=device, dtype=dtype)[:-1]
    z = torch.linspace(0, Lz, Nz+1, device=device, dtype=dtype)[:-1]
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    rho0 = rho_o + rho_1 * torch.cos(torch.tensor(KX, device=device, dtype=dtype) * xx +
                                     torch.tensor(KY, device=device, dtype=dtype) * yy +
                                     torch.tensor(KZ, device=device, dtype=dtype) * zz)

    Px0 = rho0 * 0
    Py0 = rho0 * 0
    Pz0 = rho0 * 0

    if gravity:
        jeans = torch.sqrt(torch.tensor(4*np.pi**2*cs**2/(const*G*rho_o), device=device, dtype=dtype))
        if lam >= jeans.item():
            alpha = torch.sqrt(torch.tensor(const*G*rho_o-cs**2*(2*np.pi/lam)**2, device=device, dtype=dtype))
            v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam))
            wave_field = -v_1 * torch.sin(torch.tensor(KX, device=device, dtype=dtype) * xx +
                                          torch.tensor(KY, device=device, dtype=dtype) * yy +
                                          torch.tensor(KZ, device=device, dtype=dtype) * zz)
        else:
            alpha = torch.sqrt(torch.tensor(cs**2*(2*np.pi/lam)**2 - const*G*rho_o, device=device, dtype=dtype))
            v_1 = (rho_1/rho_o) * (alpha/(2*np.pi/lam))
            wave_field = v_1 * torch.cos(torch.tensor(KX, device=device, dtype=dtype) * xx +
                                         torch.tensor(KY, device=device, dtype=dtype) * yy +
                                         torch.tensor(KZ, device=device, dtype=dtype) * zz)
    else:
        v_1 = (cs * rho_1) / rho_o
        wave_field = v_1 * torch.cos(torch.tensor(KX, device=device, dtype=dtype) * xx +
                                     torch.tensor(KY, device=device, dtype=dtype) * yy +
                                     torch.tensor(KZ, device=device, dtype=dtype) * zz)

    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    if k_mag > 0:
        vx0 = wave_field * (KX / k_mag)
        vy0 = wave_field * (KY / k_mag)
        vz0 = wave_field * (KZ / k_mag)
    else:
        vx0 = wave_field
        vy0 = torch.zeros_like(vx0)
        vz0 = torch.zeros_like(vx0)

    Px0 = rho0 * vx0
    Py0 = rho0 * vy0
    Pz0 = rho0 * vz0

    phi0 = torch.zeros_like(rho0)
    phi1 = torch.zeros_like(rho0)
    if gravity:
        phi0 = fft_solver_torch_3d(const*(rho0-rho_o), Lx, Nx, Ly, Ny, Lz, Nz)

    t_val = 0.0
    k_iter = 0
    vmax_initial = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item(), torch.max(torch.abs(vz0)).item(), cs)
    dt = nu * dx / vmax_initial

    while t_val < time_val:
        if t_val + dt > time_val:
            dt = time_val - t_val

        mux = dt / (2 * dx)
        muy = dt / (2 * dy)
        muz = dt / (2 * dz)

        rho1 = (1/6)*(torch.roll(rho0,-1,0) + torch.roll(rho0,1,0) +
                      torch.roll(rho0,-1,1) + torch.roll(rho0,1,1) +
                      torch.roll(rho0,-1,2) + torch.roll(rho0,1,2)) \
               - mux*(torch.roll(rho0,-1,0)*torch.roll(vx0,-1,0) - torch.roll(rho0,1,0)*torch.roll(vx0,1,0)) \
               - muy*(torch.roll(rho0,-1,1)*torch.roll(vy0,-1,1) - torch.roll(rho0,1,1)*torch.roll(vy0,1,1)) \
               - muz*(torch.roll(rho0,-1,2)*torch.roll(vz0,-1,2) - torch.roll(rho0,1,2)*torch.roll(vz0,1,2))

        Px1 = (1/6)*(torch.roll(Px0,-1,0) + torch.roll(Px0,1,0) +
                      torch.roll(Px0,-1,1) + torch.roll(Px0,1,1) +
                      torch.roll(Px0,-1,2) + torch.roll(Px0,1,2)) \
              - mux*(torch.roll(Px0,-1,0)*torch.roll(vx0,-1,0) - torch.roll(Px0,1,0)*torch.roll(vx0,1,0)) \
              - muy*(torch.roll(Px0,-1,1)*torch.roll(vy0,-1,1) - torch.roll(Px0,1,1)*torch.roll(vy0,1,1)) \
              - muz*(torch.roll(Px0,-1,2)*torch.roll(vz0,-1,2) - torch.roll(Px0,1,2)*torch.roll(vz0,1,2)) \
              - ((c_s**2)*mux*(torch.roll(rho0,-1,0) - torch.roll(rho0,1,0)))

        Py1 = (1/6)*(torch.roll(Py0,-1,0) + torch.roll(Py0,1,0) +
                      torch.roll(Py0,-1,1) + torch.roll(Py0,1,1) +
                      torch.roll(Py0,-1,2) + torch.roll(Py0,1,2)) \
              - muy*(torch.roll(Py0,-1,1)*torch.roll(vy0,-1,1) - torch.roll(Py0,1,1)*torch.roll(vy0,1,1)) \
              - mux*(torch.roll(Py0,-1,0)*torch.roll(vx0,-1,0) - torch.roll(Py0,1,0)*torch.roll(vx0,1,0)) \
              - muz*(torch.roll(Py0,-1,2)*torch.roll(vz0,-1,2) - torch.roll(Py0,1,2)*torch.roll(vz0,1,2)) \
              - ((c_s**2)*muy*(torch.roll(rho0,-1,1) - torch.roll(rho0,1,1)))

        Pz1 = (1/6)*(torch.roll(Pz0,-1,0) + torch.roll(Pz0,1,0) +
                      torch.roll(Pz0,-1,1) + torch.roll(Pz0,1,1) +
                      torch.roll(Pz0,-1,2) + torch.roll(Pz0,1,2)) \
              - muz*(torch.roll(Pz0,-1,2)*torch.roll(vz0,-1,2) - torch.roll(Pz0,1,2)*torch.roll(vz0,1,2)) \
              - mux*(torch.roll(Pz0,-1,0)*torch.roll(vx0,-1,0) - torch.roll(Pz0,1,0)*torch.roll(vx0,1,0)) \
              - muy*(torch.roll(Pz0,-1,1)*torch.roll(vy0,-1,1) - torch.roll(Pz0,1,1)*torch.roll(vy0,1,1)) \
              - ((c_s**2)*muz*(torch.roll(rho0,-1,2) - torch.roll(rho0,1,2)))

        if gravity:
            Px1 -= mux * rho0 * (torch.roll(phi0,-1,0) - torch.roll(phi0,1,0))
            Py1 -= muy * rho0 * (torch.roll(phi0,-1,1) - torch.roll(phi0,1,1))
            Pz1 -= muz * rho0 * (torch.roll(phi0,-1,2) - torch.roll(phi0,1,2))
            phi1 = fft_solver_torch_3d(const*(rho1 - rho_o), Lx, Nx, Ly, Ny, Lz, Nz)

        vx1 = Px1 / rho1
        vy1 = Py1 / rho1
        vz1 = Pz1 / rho1

        rho0, vx0, vy0, vz0 = rho1, vx1, vy1, vz1
        Px0, Py0, Pz0 = Px1, Py1, Pz1
        if gravity:
            phi0 = phi1

        t_val += dt
        k_iter += 1
        vmax = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item(), torch.max(torch.abs(vz0)).item())
        dt1 = nu * dx / vmax if vmax > 1e-9 else float('inf')
        dt2 = nu * dx / c_s
        dt = min(dt1, dt2)

    rho_max = torch.max(rho0).item()
    return (x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy(),
            rho0.detach().cpu().numpy(), vx0.detach().cpu().numpy(),
            vy0.detach().cpu().numpy(), vz0.detach().cpu().numpy(),
            phi0.detach().cpu().numpy() if gravity else None, k_iter, rho_max)

def lax_solution_warm_start_torch(rho_ic, vx_ic, vy_ic, x_grid, y_grid, 
                                   t_start, t_end, nu=0.5, save_times=None, gravity=True):
    """
    PyTorch implementation: Run FD solver from custom initial conditions (warm-start).
    
    This function allows restarting the FD solver from a PINN state or any custom state,
    enabling efficient generation of FD data for hybrid PINN-FD training.
    
    Args:
        rho_ic: Initial density field (Nx, Ny) - can be numpy array or torch tensor
        vx_ic: Initial x-velocity field (Nx, Ny) - can be numpy array or torch tensor
        vy_ic: Initial y-velocity field (Nx, Ny) - can be numpy array or torch tensor
        x_grid: x coordinates (Nx,) - can be numpy array or torch tensor
        y_grid: y coordinates (Ny,) - can be numpy array or torch tensor
        t_start: Starting time
        t_end: Ending time
        nu: Courant number
        save_times: List of times to save snapshots [default: [t_end]]
        gravity: Whether to include self-gravity (default: True)
    
    Returns:
        Dictionary: {time: (rho, vx, vy, phi, x, y)} for each saved time (all as numpy arrays)
    """
    if save_times is None:
        save_times = [t_end]
    
    # Convert inputs to torch tensors if needed
    if isinstance(rho_ic, np.ndarray):
        rho_ic = torch.from_numpy(rho_ic).to(device=device, dtype=dtype)
    if isinstance(vx_ic, np.ndarray):
        vx_ic = torch.from_numpy(vx_ic).to(device=device, dtype=dtype)
    if isinstance(vy_ic, np.ndarray):
        vy_ic = torch.from_numpy(vy_ic).to(device=device, dtype=dtype)
    if isinstance(x_grid, np.ndarray):
        x_grid = torch.from_numpy(x_grid).to(device=device, dtype=dtype)
    if isinstance(y_grid, np.ndarray):
        y_grid = torch.from_numpy(y_grid).to(device=device, dtype=dtype)
    
    # Domain setup
    Nx, Ny = rho_ic.shape
    Lx = (x_grid[-1] - x_grid[0] + (x_grid[1] - x_grid[0])).item()  # Approximate domain size
    Ly = (y_grid[-1] - y_grid[0] + (y_grid[1] - y_grid[0])).item()
    dx = Lx / Nx
    dy = Ly / Ny
    
    # Physical constants
    c_s = cs
    
    # Initialize from provided ICs
    rho0 = rho_ic.clone()
    vx0 = vx_ic.clone()
    vy0 = vy_ic.clone()
    
    # Calculate initial potential (gravity is always True for collapse problems)
    phi0 = fft_solver_torch(const * (rho0 - rho_o), Lx, Nx, Ly, Ny)
    
    # Initialize flux terms
    Px0 = rho0 * vx0
    Py0 = rho0 * vy0
    
    # Storage for snapshots
    snapshots = {}
    
    # Time-stepping loop
    t = t_start
    k = 0
    
    # Initial dt
    vmax_initial = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item(), c_s)
    dt = nu * dx / vmax_initial
    
    while t < t_end:
        # Check if we should save a snapshot before this step
        for save_t in save_times:
            if t <= save_t < t + dt and save_t not in snapshots:
                # Save current state (convert to numpy for consistency)
                snapshots[save_t] = (
                    rho0.cpu().numpy().copy(),
                    vx0.cpu().numpy().copy(),
                    vy0.cpu().numpy().copy(),
                    phi0.cpu().numpy().copy(),
                    x_grid.cpu().numpy().copy(),
                    y_grid.cpu().numpy().copy()
                )
        
        # Ensure last step doesn't overshoot
        if t + dt > t_end:
            dt = t_end - t
        
        # LAX time-stepping
        mux = dt / (2 * dx)
        muy = dt / (2 * dy)
        
        # Update density
        rho1 = (0.25) * (torch.roll(rho0, -1, dims=0) + torch.roll(rho0, 1, dims=0) +
                        torch.roll(rho0, -1, dims=1) + torch.roll(rho0, 1, dims=1)) - \
               (mux * (torch.roll(rho0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                       torch.roll(rho0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
               (muy * (torch.roll(rho0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                       torch.roll(rho0, 1, dims=1) * torch.roll(vy0, 1, dims=1)))
        
        # Update momentum (with gravity)
        if gravity:
            Px1 = (0.25) * (torch.roll(Px0, -1, dims=0) + torch.roll(Px0, 1, dims=0) +
                            torch.roll(Px0, -1, dims=1) + torch.roll(Px0, 1, dims=1)) - \
                  (mux * (torch.roll(Px0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Px0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  (muy * (torch.roll(Px0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Px0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  ((c_s**2) * mux * (torch.roll(rho0, -1, dims=0) - torch.roll(rho0, 1, dims=0))) - \
                  (mux * rho0 * (torch.roll(phi0, -1, dims=0) - torch.roll(phi0, 1, dims=0)))
            
            Py1 = (0.25) * (torch.roll(Py0, -1, dims=0) + torch.roll(Py0, 1, dims=0) +
                            torch.roll(Py0, -1, dims=1) + torch.roll(Py0, 1, dims=1)) - \
                  (muy * (torch.roll(Py0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Py0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  (mux * (torch.roll(Py0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Py0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  ((c_s**2) * muy * (torch.roll(rho0, -1, dims=1) - torch.roll(rho0, 1, dims=1))) - \
                  (muy * rho0 * (torch.roll(phi0, -1, dims=1) - torch.roll(phi0, 1, dims=1)))
            
            # Update potential
            phi1 = fft_solver_torch(const * (rho1 - rho_o), Lx, Nx, Ly, Ny)
        else:
            # Without gravity (shouldn't happen for collapse problems, but included for completeness)
            Px1 = (0.25) * (torch.roll(Px0, -1, dims=0) + torch.roll(Px0, 1, dims=0) +
                            torch.roll(Px0, -1, dims=1) + torch.roll(Px0, 1, dims=1)) - \
                  (mux * (torch.roll(Px0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Px0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  (muy * (torch.roll(Px0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Px0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  ((c_s**2) * mux * (torch.roll(rho0, -1, dims=0) - torch.roll(rho0, 1, dims=0)))
            
            Py1 = (0.25) * (torch.roll(Py0, -1, dims=0) + torch.roll(Py0, 1, dims=0) +
                            torch.roll(Py0, -1, dims=1) + torch.roll(Py0, 1, dims=1)) - \
                  (muy * (torch.roll(Py0, -1, dims=1) * torch.roll(vy0, -1, dims=1) -
                          torch.roll(Py0, 1, dims=1) * torch.roll(vy0, 1, dims=1))) - \
                  (mux * (torch.roll(Py0, -1, dims=0) * torch.roll(vx0, -1, dims=0) -
                          torch.roll(Py0, 1, dims=0) * torch.roll(vx0, 1, dims=0))) - \
                  ((c_s**2) * muy * (torch.roll(rho0, -1, dims=1) - torch.roll(rho0, 1, dims=1)))
            
            phi1 = torch.zeros_like(rho1)
        
        # Update velocities
        vx1 = Px1 / rho1
        vy1 = Py1 / rho1
        
        # Update state
        rho0 = rho1
        vx0 = vx1
        vy0 = vy1
        Px0 = Px1
        Py0 = Py1
        if gravity:
            phi0 = phi1
        
        t += dt
        k += 1
        
        # Calculate dt for next step
        vmax = max(torch.max(torch.abs(vx0)).item(), torch.max(torch.abs(vy0)).item())
        dt1 = nu * dx / vmax if vmax > 1e-9 else float('inf')
        dt2 = nu * dx / c_s
        dt = min(dt1, dt2)
    
    # Save final snapshot if not already saved
    if t_end not in snapshots:
        snapshots[t_end] = (
            rho0.cpu().numpy().copy(),
            vx0.cpu().numpy().copy(),
            vy0.cpu().numpy().copy(),
            phi0.cpu().numpy().copy(),
            x_grid.cpu().numpy().copy(),
            y_grid.cpu().numpy().copy()
        )
    
    return snapshots