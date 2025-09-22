from numpy.fft  import fft, ifft,fft2, ifft2,fftn, ifftn
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

print("Importing the LAX Module")

def fft_solver(rho,Lx,nx,Ly,ny,Lz,nz,dim = None):
    
    '''
    A FFT solver that uses discrete Fast Fourier Transform to
    solve the Poisson Equation:
    We apply the correction due to the finite difference grid of phi
    
    Input: 1. The source function density in this case
           2. # of grid point Nx and Ny for 2D
           3. Domain Size in each dimension
           4. Dim : will the updated later to work for any dimension
    
    Output: the potential phi and the field g (optional) not returned currently.
    
    '''

    if dim:
        dx, dy ,dz = Lx / nx, Ly / ny , Lz/nz
    else:
        dx = Lx / nx,
    # Calculate the Fourier modes of the gas density
    rhohat = fftn(rho)

    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
    kz = 2 * np.pi * np.fft.fftfreq(nz, dz)

    # Construct the Laplacian operator in Fourier space
    kx2,ky2,kz2 = np.meshgrid(kx**2, ky**2,kz**2,indexing='ij')
    laplace = -(kx2 + ky2 + kz2)

    ## Correction for the dicrete FFT.  Need to check the calculations
    # laplace = 2*(np.cos(kx*dx)-1)/(dx**2) +  2*(np.cos(ky*dx)-1)/(dy**2)

    laplace[laplace == 0] = 1e-9

    # Solve for the electrostatic potential in Fourier space
    phihat = rhohat / laplace

    # Transform back to real space to obtain the solution
    phi = np.real(ifftn(phihat))
#     dphidx = np.gradient(phi, dx)s
#     dphidy = np.gradient(phi, dy)
#     return phi,dphidx, dphidy 
    return phi

def lax_solution_multiple_times(time_array, N, nu, lam, num_of_waves, rho_1, 
                               gravity=False, isplot=None, comparison=None, 
                               dim_plot=None, animation=None):
    """
    Wrapper function to handle multiple time values
    """
    all_results = []
    
    for time_val in time_array:
        # Ensure time_val is a scalar
        if isinstance(time_val, np.ndarray):
            time_val = float(time_val)
        
        print(f"Processing time: {time_val}")
        
        # Call the original function with scalar time
        result = lax_solution(time_val, N, nu, lam, num_of_waves, rho_1, 
                             gravity=gravity, isplot=False, comparison=comparison, 
                             dim_plot=dim_plot, animation=animation)

        #print(f"Result type: {type(result)}")
        #print(f"Result is None: {result is None}")
        
        #if isinstance(result, dict):
        #    print(f"Keys in result: {list(result.keys())}")
        #    print(f"'x' in result: {'x' in result}")
        #else:
        #    print(f"Result content: {result}")
        
        all_results.append(result)

    #print(f"Length of all_results: {len(all_results)}")
    #print(f"Types in all_results: {[type(r) for r in all_results]}")
    
    # Create comprehensive plots if requested
    if isplot:
        create_comprehensive_plots(all_results, time_array, rho_1, gravity, comparison)
    
    return all_results

def lax_solution(time,N,nu,lam,num_of_waves,rho_1,gravity=False,isplot = None,comparison =None,dim_plot =None,animation=None):
    '''
    This function solves the hydrodynamic Eqns in 3D with/without self gravity using LAX methods 
    described above 
    
    
    Input:  Time till the system is integrated :time
            Number of Xgrid points : N
            Courant number : nu
            Wavelength : If lambda> lambdaJ (with gravity--> Instability) else waves propagation 
            Number of waves : The domain size changes with this maintain periodicity
            Density perturbation : rho1 (for linear or non-linear perturbation)
            Gravity:  If True it deploys the FFT routine to estimate the potential 
            isplot(optional): if True plots the output
            Comparison (optional) : If True then the plots are overplotted with LT solutions for comparison
            Animation (optional): Not used at the moment
    
    Output: Density, velocity + (phi and g if gravity is True)
            isplot: True then the plots are generated 
    
    '''
    
    
    # rho_max = []
    lam = lam          # one wavelength
    num_of_waves  = num_of_waves  
    Lx =  lam * num_of_waves            # Maximum length (two wavelength)
    Ly =  lam * num_of_waves
    Lz =  lam * num_of_waves 
    print("at time= ",time)
    ### Declaring the Constants

    c_s = 1.0            # % Sound Speed  
    rho_o = 1.0          # zeroth order density
    nu = nu              # courant number (\nu = 2 in 2d)
    rho_1 = rho_1        # for linear/nonlinear wave propagation
    const =  1           # The actual value is 4*pi
    G = 1.0              # Gravitational Constant

    ### Grid X-Y-Z-T 
    Nx = N                # The grid resolution values2d:N =(10,50,100,500)
    dx = float(Lx/Nx)      # length spacing          
    
    Ny = N              # The grid resolution values2d:N =(10,50,100,500)
    dy = float(Ly/Ny)      # length spacing  
    
    Nz = N              # The grid resolution values2d:N =(10,50,100,500)
    dz = float(Lz/Nz)      # length spacing
    
    dt = nu*dx/c_s       # time grid spacing
 

    ## For simplification
    mux = dt/(2*dx)      # is the coefficient in the central differencing Eqs above 
    muy = dt/(2*dy)      # is the coefficient in the central differencing Eqs above
    muz = dt/(2*dz)      # is the coefficient in the central differencing Eqs above
    n =  int(time/dt)     # grid points in time
    print("For dx = {} and dt = {} and time gridpoints n = {} ".format(dx,dt,n))
    
    ########## Initializing the ARRAY #######################
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)    
    xx,yy,zz  = np.meshgrid(x,y,z,indexing='ij') ## Mesh for the 3D domain

    
    rho0 = np.zeros((Nx,Ny,Nz))
    rho1 = np.zeros((Nx,Ny,Nz))
    vx0 =np.zeros((Nx,Ny,Nz)) 
    vx1 =np.zeros((Nx,Ny,Nz))
    vy0 =np.zeros((Nx,Ny,Nz))
    vy1 =np.zeros((Nx,Ny,Nz))
    vz0 =np.zeros((Nx,Ny,Nz))
    vz1 =np.zeros((Nx,Ny,Nz))
    Px0 =np.zeros((Nx,Ny,Nz)) # The flux term  U in the above equations
    Px1 =np.zeros((Nx,Ny,Nz))
    Py0 =np.zeros((Nx,Ny,Nz)) # The flux term  W in the above equations 
    Py1 =np.zeros((Nx,Ny,Nz))
    Pz0 =np.zeros((Nx,Ny,Nz)) # The flux term  W in the above equations 
    Pz1 =np.zeros((Nx,Ny,Nz)) 

    phi0 = np.zeros((Nx,Ny,Nz))
    phi1 = np.zeros((Nx,Ny,Nz))

    
    ## Calculating the jeans length is gravity is Turned on
    if gravity:
        jeans = np.sqrt(4*np.pi**2*c_s**2/(const*G*rho_o))
        print("Jean's Length",jeans)

    ######################## Initial Conditions ###########################
    
    rho0 = rho_o + rho_1* np.cos(2*np.pi*xx/lam) # defing the density at t = 0 EQ 11
    
    
    if gravity == False:
        print("Propagation of Sound wave") 
        v_1 = (c_s*rho_1)/rho_o # velocity perturbation
        vx0 = v_1 * np.cos(2*np.pi*xx/lam) # the velocity at t =0

        rho_LT = None
        rho_LT_max = None
        vx_LT = None
        vy_LT = None
        vz_LT = None

        ## Linear Theory
        if comparison:
            rho_LT  = rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time)
            rho_LT_max = np.max(rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time))
            vx_LT = v_1* np.cos(2*np.pi * x/lam - 2*np.pi/lam *time) 
            vy_LT = np.zeros(Ny)
            vz_LT = np.zeros(Nz)
    
    else:    ######## When self-gravity is True and see EQN 12
        if lam >= jeans:  
            print("There is gravitational instabilty  lam = {} > l_jean ={}".format(lam,jeans))
            alpha = np.sqrt(const*G*rho_o-c_s**2*(2*np.pi/lam)**2)
            v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam)) ## With gravity     
            vx0 = - v_1 * np.sin(2*np.pi*xx/lam) # the velocity at t =0
            # print("initial vy",vy[n-1,1,:])
            ##### Density values from Linear Theory at t 
            if comparison:
                rho_LT = rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam))
                vx_LT = -v_1*np.exp(alpha * time)*np.sin(2*np.pi*x/lam)
                vy_LT = np.zeros(Nx)
                vz_LT = np.zeros(Nz)

        else:
            print("There is no gravitational instabilty as lam = {} < l_jean ={}".format(lam,jeans))
            alpha = np.sqrt(c_s**2*(2*np.pi/lam)**2 - const*G*rho_o)
            v_1 = (rho_1/rho_o) * (alpha/(2*np.pi/lam)) # velocity perturbation
            vx0 = v_1 * np.cos(2*np.pi*xx/lam) # the velocity at t =0
            if comparison:
                rho_LT = rho_o + rho_1*np.cos(alpha * time - 2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.cos(alpha * time - 2*np.pi*xx/lam))
                vx_LT = v_1*np.cos(alpha * time - 2*np.pi*x/lam)
                vy_LT = np.zeros((Nx))

        # Calculating the potential and the field using FFT    
#         phi[0,:,:],dphidx,dphidy = fft_solver(const*(rho[0,:,:]-rho_o),Lx,Nx,Ly,Ny,dim = 2)
        phi0 = fft_solver(const*(rho0-rho_o),Lx,Nx,Ly,Ny,Lz,Nz,dim = 3)
        print("shape of Phi",phi0.shape)
#         fft_solver(rho,Lx,nx,Ly,ny,dim = 2)



    

    ####### The Flux term #########
    Px0=rho0*vx0
    Py0=rho0*vy0
    Pz0=rho0*vz0
    
    #################################FINITE DIFFERENCE #######################    
    for k in range(1,n): ## Looping over time 
        rho1 = (1/6)*(np.roll(rho0, -1, axis=0)+ np.roll(rho0, 1, axis=0)\
                        +np.roll(rho0, -1, axis=1)+ np.roll(rho0, 1, axis=1)\
                        +np.roll(rho0, -1, axis=2)+ np.roll(rho0, 1, axis=2))\
        -(mux*(np.roll(rho0,-1,axis=0)*np.roll(vx0,-1,axis=0)-np.roll(rho0,1, axis=0)*np.roll(vx0,1,axis=0)))\
        -(muy*(np.roll(rho0,-1,axis=1)*np.roll(vy0,-1,axis=1)-np.roll(rho0,1, axis=1)*np.roll(vy0,1,axis=1))\
        -(muz*(np.roll(rho0,-1,axis=2)*np.roll(vz0,-1,axis=2)-np.roll(rho0,1, axis=2)*np.roll(vz0,1,axis=2))))

        if gravity == False: ## Hydro sound wave when gravity is absent
            
            Px1 = (1/6)*(np.roll(Px0,-1,axis=0)+ np.roll(Px0,1,axis=0) + np.roll(Px0,-1,axis=1)+ np.roll(Px0,1,axis=1)\
                      + np.roll(Px0,-1,axis=2)+ np.roll(Px0,1,axis=2))\
            -(mux*(np.roll(Px0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Px0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muy*(np.roll(Px0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Px0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -(muz*(np.roll(Px0,-1,axis=2)*np.roll(vz0,-1,axis=2)- np.roll(Px0,1,axis=2)*np.roll(vz0,1,axis=2)))\
            -((c_s**2)*mux*(np.roll(rho0,-1,axis=0)- np.roll(rho0,1,axis=0)))
            
            Py1 = (1/6)*(np.roll(Py0,-1,axis=0)+ np.roll(Py0,1,axis=0) + np.roll(Py0,-1,axis=1)+ np.roll(Py0,1,axis=1)\
                           + np.roll(Py0,-1,axis=2)+ np.roll(Py0,1,axis=2))\
            -(muy*(np.roll(Py0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Py0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -(mux*(np.roll(Py0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Py0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muz*(np.roll(Py0,-1,axis=2)*np.roll(vz0,-1,axis=2)- np.roll(Py0,1,axis=2)*np.roll(vz0,1,axis=2)))\
            -((c_s**2)*muy*(np.roll(rho0,-1,axis=1)- np.roll(rho0,1,axis=1)))
            
            Pz1 = (1/6)*(np.roll(Pz0,-1,axis=0)+ np.roll(Pz0,1,axis=0) + np.roll(Pz0,-1,axis=1)+ np.roll(Pz0,1,axis=1)\
                           + np.roll(Pz0,-1,axis=2)+ np.roll(Pz0,1,axis=2))\
            -(muz*(np.roll(Pz0,-1,axis=2)*np.roll(vz0,-1,axis=2)- np.roll(Pz0,1,axis=2)*np.roll(vz0,1,axis=2)))\
            -(mux*(np.roll(Pz0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Pz0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muy*(np.roll(Pz0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Pz0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -((c_s**2)*muz*(np.roll(rho0,-1,axis=2)- np.roll(rho0,1,axis=2)))
            
        else:
     
            Px1 = (1/6)*(np.roll(Px0,-1,axis=0)+ np.roll(Px0,1,axis=0) + np.roll(Px0,-1,axis=1)+ np.roll(Px0,1,axis=1)\
                      + np.roll(Px0,-1,axis=2)+ np.roll(Px0,1,axis=2))\
            -(mux*(np.roll(Px0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Px0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muy*(np.roll(Px0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Px0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -(muz*(np.roll(Px0,-1,axis=2)*np.roll(vz0,-1,axis=2)- np.roll(Px0,1,axis=2)*np.roll(vz0,1,axis=2)))\
            -((c_s**2)*mux*(np.roll(rho0,-1,axis=0)- np.roll(rho0,1,axis=0)))\
            -(mux*rho0*(np.roll(phi0,-1,axis=0)- np.roll(phi0,1,axis=0)))
        
            Py1 = (1/6)*(np.roll(Py0,-1,axis=0)+ np.roll(Py0,1,axis=0) + np.roll(Py0,-1,axis=1)+ np.roll(Py0,1,axis=1)\
                           + np.roll(Py0,-1,axis=2)+ np.roll(Py0,1,axis=2))\
            -(muy*(np.roll(Py0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Py0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -(mux*(np.roll(Py0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Py0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muz*(np.roll(Py0,-1,axis=2)*np.roll(vz0,-1,axis=2)- np.roll(Py0,1,axis=2)*np.roll(vz0,1,axis=2)))\
            -((c_s**2)*muy*(np.roll(rho0,-1,axis=1)- np.roll(rho0,1,axis=1)))\
            -(muy*rho0*(np.roll(phi0,-1,axis=1)- np.roll(phi0,1,axis=1)))
            
            Pz1 = (1/6)*(np.roll(Pz0,-1,axis=0)+ np.roll(Pz0,1,axis=0) + np.roll(Pz0,-1,axis=1)+ np.roll(Pz0,1,axis=1)\
                           + np.roll(Pz0,-1,axis=2)+ np.roll(Pz0,1,axis=2))\
            -(muz*(np.roll(Pz0,-1,axis=2)*np.roll(vz0,-1,axis=2)- np.roll(Pz0,1,axis=2)*np.roll(vz0,1,axis=2)))\
            -(mux*(np.roll(Pz0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Pz0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muy*(np.roll(Pz0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Pz0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -((c_s**2)*muz*(np.roll(rho0,-1,axis=2)- np.roll(rho0,1,axis=2)))\
            -(muz*rho0*(np.roll(phi0,-1,axis=2)- np.roll(phi0,1,axis=2)))

  
            phi1= fft_solver(const*(rho1-rho_o),Lx,Nx,Ly,Ny,Lz,Nz,dim = 3)

        vx1 = Px1/rho1 ## 2-D velocity vx 
        vy1 = Py1/rho1 ## 2-D velocity vy
        vz1 = Pz1/rho1 ## 2-D velocity vy
        
        ## memory tranfer to overwrite "1" in the next time step
        rho0 = rho1
        vx0 = vx1
        vy0 = vy1
        vz0 = vz1
        Px0 = Px1
        Py0 = Py1
        Pz0 = Pz1
        phi0= phi1
        
        ## Updating dt based on the highest signal speed in the code
        dt1 = nu*dx/np.max([abs(vz1),abs(vx1),abs(vy1)])
        dt2 = nu*dx/c_s
        
        
        dt = np.min([dt1,dt2])
        mux = dt/(2*dx)      # is the coefficient in the central differencing Eqs above 
        muy = dt/(2*dy)      # is the coefficient in the central differencing Eqs above
        muz = dt/(2*dz)  
    
        n = int(time/dt)     # grid points in time updated dynamically

                                                                

    rho_max = np.max(rho1)   ## Maximum density from the FD calculation 

    results = {
    'time': time,
    'x': x,
    'rho': rho1,
    'vx': vx1,
    'n': n,
    'vy': vy1,
    'vz': vz1,
    'phi': phi1 if gravity else None,
    'rho_max': np.max(rho1),
    'rho_LT': rho_LT if comparison else None,
    'vx_LT': vx_LT if comparison else None,
    'vy_LT': vy_LT if comparison else None,
    'vz_LT': vz_LT if comparison else None,
    'rho_LT_max': rho_LT_max if comparison else None}
    
    return results

def create_comprehensive_plots(all_results, time_array, rho_1, gravity, comparison):
    
    n_times = len(time_array)
    n_vars = 6 if gravity else 4
    
    fig, axes = plt.subplots(n_vars, n_times, figsize=(5*n_times, 4*n_vars))
    
    # Ensure axes is always 2D
    if n_times == 1:
        axes = axes.reshape(-1, 1)
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    for col, result in enumerate(all_results):
        time = result['time']
        x = result['x']
        
        # Row 0: Density
        axes[0, col].plot(x, result['rho'][:, 1, 1] - 1.0, linewidth=2, 
                         label=f"FD at t={time:.1f}")
        if comparison and result['rho_LT'] is not None:
            axes[0, col].plot(x, result['rho_LT'] - 1.0, '--', linewidth=2, 
                             label="LT")
        axes[0, col].set_title(f"Density at t={time:.1f}")
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel("ρ - ρ₀")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        
        # Row 1: Velocity vx
        axes[1, col].plot(x, result['vx'][:, 1, 1], linewidth=2, 
                         label=f"FD at t={time:.1f}")
        if comparison and result['vx_LT'] is not None:
            axes[1, col].plot(x, result['vx_LT'], '--', linewidth=2, 
                             label="LT")
        axes[1, col].set_title(f"Velocity vx at t={time:.1f}")
        axes[1, col].set_xlabel("x")
        axes[1, col].set_ylabel("vx")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)
        
        # Row 2: Velocity vy
        y = np.linspace(0, result['x'][-1], len(result['vy'][1, :, 1]))
        axes[2, col].plot(y, result['vy'][1, :, 1], linewidth=2, 
                         label=f"FD at t={time:.1f}")
        if comparison and result['vy_LT'] is not None:
            axes[2, col].plot(y, result['vy_LT'], '--', linewidth=2, 
                             label="LT")
        axes[2, col].set_title(f"Velocity vy at t={time:.1f}")
        axes[2, col].set_xlabel("y")
        axes[2, col].set_ylabel("vy")
        axes[2, col].legend()
        axes[2, col].grid(True, alpha=0.3)
        
        # Row 3: Velocity vz
        z = np.linspace(0, result['x'][-1], len(result['vz'][1, 1, :]))
        axes[3, col].plot(z, result['vz'][1, 1, :], linewidth=2, 
                         label=f"FD at t={time:.1f}")
        if comparison and result['vz_LT'] is not None:
            axes[3, col].plot(z, result['vz_LT'], '--', linewidth=2, 
                             label="LT")
        axes[3, col].set_title(f"Velocity vz at t={time:.1f}")
        axes[3, col].set_xlabel("z")
        axes[3, col].set_ylabel("vz")
        axes[3, col].legend()
        axes[3, col].grid(True, alpha=0.3)
        
        if gravity:
            # Row 4: Gravitational potential
            axes[4, col].plot(x, result['phi'][:, 1, 1], linewidth=2, 
                             label=f"φ at t={time:.1f}")
            axes[4, col].set_title(f"Gravitational Potential at t={time:.1f}")
            axes[4, col].set_xlabel("x")
            axes[4, col].set_ylabel("φ")
            axes[4, col].legend()
            axes[4, col].grid(True, alpha=0.3)
            
            # Row 5: Maximum density evolution
            axes[5, col].scatter(time, result['rho_max'], s=100, 
                               label="FD", color='blue')
            if comparison and result['rho_LT_max'] is not None:
                axes[5, col].scatter(time, result['rho_LT_max'], s=100, 
                                   facecolors='none', edgecolors='red', 
                                   label="LT")
            axes[5, col].set_title(f"Max Density at t={time:.1f}")
            axes[5, col].set_xlabel("Time")
            axes[5, col].set_ylabel("log(ρₘₐₓ - ρ₀)")
            axes[5, col].set_yscale('log')
            axes[5, col].legend()
            axes[5, col].grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Reserve space at top for suptitle
    plt.suptitle(f"LAX Solution Evolution of hydrodynamic variables", 
                fontsize=16, y=0.98)  # Position title higher
    plt.show()
    
# #     ################################# PLOTTING #######################
 
    '''if isplot : 
        plt.figure(1,figsize=(6,4))
        plt.plot(x,rho1[:,1,1]-rho_o,linewidth=1,label="FD at t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{x}$")
        # plt.text(.6,.15,r"dt=%f"%(dt),fontsize=12)
        plt.title("At time {} and rho_1 = {}".format(time,rho_1))
        plt.ylabel(r"$\mathbf{\rho - \rho_{0}}$")
        #plt.savefig(output_folder+'/LAX_density'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)
        if comparison : 
            plt.plot(x,rho_LT-rho_o,'--',linewidth=1,label="LT")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            #plt.savefig(output_folder+'/LAX_density'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)

        plt.figure(2,figsize=(6,4))
        plt.plot(x,vx1[:,1,1],'--',markersize=2,label="t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{x}$")
        plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
        plt.ylabel("vx")
        #plt.savefig(output_folder+'/LAX_velocity'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)
        if comparison : 
            plt.plot(x,vx_LT,'--',linewidth=1,label="LT")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True) 
            #plt.savefig(output_folder+'/LAX_velocity'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)
            
        plt.figure(3,figsize=(6,4))
        plt.plot(y,vy1[1,:,1],'--',markersize=2,label="t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{y}$")
        plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
        plt.ylabel("vy")
        if comparison : 
            plt.plot(y,vy_LT,'--',linewidth=1,label="LT")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)

        plt.figure(4,figsize=(6,4))
        plt.plot(y,vz1[1,1,:],'--',markersize=2,label="t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{z}$")
        plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
        plt.ylabel("vz")
        if comparison : 
            plt.plot(z,vz_LT,'--',linewidth=1,label="LT")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            
        if gravity:
             #### Plotting the comparison of the \rho_max for FD and Linear Theory           
            plt.figure(5,figsize=(6,4))              
            plt.scatter(time,rho_max,label="FD")          
            plt.xlabel("t")
            plt.ylabel(r"$\log (\rho_{\rm max} - \rho_{0}) $")
            plt.yscale('log')
            plt.legend(numpoints=1,loc='upper left',fancybox=True,shadow=True)
            #plt.savefig(output_folder+'/LAX-Comp'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)
            if comparison:
                plt.scatter(time,rho_LT_max,facecolors='none', edgecolors='r',label="LT")   
                plt.legend(numpoints=1,loc='upper left',fancybox=True,shadow=True)
                print(time, rho_LT_max)
                #plt.savefig(output_folder+'/LAX-Comp'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)

        if dim_plot == True: ## PLotting in 3D
        
            fig = plt.figure(6)
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(xx, yy, zz, c=rho0)
            plt.colorbar(sc)
            #plt.savefig(output_folder+'/3D-denstiy'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)
#             plt.show()

    else:
        if gravity:
            return x,rho,v,phi,dphidx,n,rho_LT,rho_LT_max,rho_max,v_LT
        else:
            return x,rho,v,rho_LT,rho_LT_max,rho_max,v_LT
#     ## Clearing the memory
    del rho0,phi0,vx0,vy0,vz0, Px0,Py0,Pz0'''

    
