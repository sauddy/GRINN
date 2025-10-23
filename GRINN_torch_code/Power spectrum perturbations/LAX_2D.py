import numpy as np
import os

# Import TensorFlow and NumPy
# import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import scipy


## For the FFT solver

from numpy.fft import fft, ifft,fft2, ifft2
from scipy import signal

from config import RANDOM_SEED

# Import wave vector components and physical constants for 2D sinusoidal perturbations
try:
    from config import KX, KY, cs, rho_o, const, G
except ImportError:
    # Fallback if config not available
    KX = 2*np.pi/5.0  # Default wavelength
    KY = 0.0
    cs = 1.0
    rho_o = 1.0
    const = 1.0
    G = 1.0

np.random.seed(RANDOM_SEED)
#tf.random.set_seed(1234)

def generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index=-3.0, amplitude=0.02, random_seed=None):
    """
    Generate 2D velocity components (vx, vy) with an isotropic power-law spectrum P(k) ~ k^{power_index}.
    The fields are created by filtering white noise in Fourier space and normalized to the requested RMS amplitude.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    def synthesize_component():
        field = rng.standard_normal((nx, ny))
        F = fft2(field)
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
        kxg, kyg = np.meshgrid(kx, ky, indexing='ij')
        kk = np.sqrt(kxg**2 + kyg**2)
        kk[0, 0] = 1.0
        filt = (kk) ** (power_index / 2.0)
        filt[kk == 0] = 0.0
        F_filtered = F * filt
        comp = np.real(ifft2(F_filtered))
        comp -= np.mean(comp)
        std = np.std(comp)
        if std > 0:
            comp = comp * (amplitude / std)
        return comp

    vx0 = synthesize_component()
    vy0 = synthesize_component()
    return vx0, vy0

def generate_shared_velocity_field(nx, ny, Lx, Ly, power_index=-4.0, amplitude=0.01, random_seed=None):
    """
    Generate shared velocity field for both PINN and FD to ensure identical initial conditions.
    This function creates the velocity field once and returns both numpy arrays (for FD) 
    and interpolation functions (for PINN).
    """
    if random_seed is None:
        random_seed = RANDOM_SEED
    # Generate the velocity field using the FD method
    vx_np, vy_np = generate_velocity_field_power_spectrum(nx, ny, Lx, Ly, power_index, amplitude, random_seed)
    
    # Create interpolation functions for PINN
    from scipy.interpolate import RegularGridInterpolator
    
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    
    vx_interp = RegularGridInterpolator((x_coords, y_coords), vx_np, method='linear', bounds_error=False, fill_value=0.0)
    vy_interp = RegularGridInterpolator((x_coords, y_coords), vy_np, method='linear', bounds_error=False, fill_value=0.0)
    
    return vx_np, vy_np, vx_interp, vy_interp

def lax_solution_with_shared_velocity(time, N, nu, lam, num_of_waves, rho_1, vx0_shared, vy0_shared, gravity=False, isplot=None, comparison=None, animation=None):
    """
    Modified LAX solver that uses pre-generated shared velocity fields for consistent initial conditions.
    This ensures PINN and FD use identical velocity fields at t=0.
    """
    # Call the original lax_solution with shared velocity fields
    result = lax_solution(
        time, N, nu, lam, num_of_waves, rho_1, gravity=gravity, isplot=isplot, 
        comparison=comparison, animation=animation, use_velocity_ps=True,
        vx0_shared=vx0_shared, vy0_shared=vy0_shared
    )
    
    return result

def fft_solver(rho,Lx,nx,Ly,ny,dim = None):
    
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
        dx, dy = Lx / nx, Ly / ny
    else:
        dx = Lx / nx,
    # Calculate the Fourier modes of the gas density
    rhohat = fft2(rho)

    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)

    # Construct the Laplacian operator in Fourier space
    kx2, ky2 = np.meshgrid(kx**2, ky**2)
    laplace = -(kx2 + ky2)

    ## Correction for the dicrete FFT.  Need to check the calculations
#     laplace = 2*(np.cos(kx*dx)-1)/(dx**2) +  2*(np.cos(ky*dx)-1)/(dy**2)

    laplace[laplace == 0] = 1e-9

    # Solve for the electrostatic potential in Fourier space
    phihat = rhohat / laplace

    # Transform back to real space to obtain the solution
    phi = np.real(ifft2(phihat))
#     dphidx = np.gradient(phi, dx)
#     dphidy = np.gradient(phi, dy)
#     return phi,dphidx, dphidy 
    return phi

def lax_solution(time,N,nu,lam,num_of_waves,rho_1,gravity=False,isplot = None,comparison =None,animation=None,
                 use_velocity_ps=False, ps_index=-3.0, vel_rms=0.02, random_seed=None, vx0_shared=None, vy0_shared=None):
    '''
    This function solves the hydrodynamic Eqns in 1D with/without self gravity using LAX methods 
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
    Lx = lam * num_of_waves            # Maximum length (two wavelength)
    Ly = lam * num_of_waves 
    # print("at time= ",time)  # Commented out to reduce output noise
    ### Declaring the Constants

    c_s = cs             # Sound Speed (from config)
    # rho_o, nu, const, G are already imported from config, no need to reassign
    rho_1 = rho_1        # for linear/nonlinear wave propagation

    ### Grid X-T 
    Nx = N                # The grid resolution values2d:N =(10,50,100,500)
    dx = float(Lx/Nx)      # length spacing          
    ### Grid X-T 
    Ny = Nx               # The grid resolution values2d:N =(10,50,100,500)
    dy = float(Ly/Ny)      # length spacing       
    dt = nu*dx/c_s       # time grid spacing
 

    ## For simplification
    mux = dt/(2*dx)      # is the coefficient in the central differencing Eqs above 
    muy = dt/(2*dy)      # is the coefficient in the central differencing Eqs above
    n = int(time/dt)     # grid points in time
    # print("For dx = {} and dt = {} and time gridpoints n = {} ".format(dx,dt,n))  # Commented out to reduce output noise
    
    ########### Initializing the ARRAY #######################
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    xx, yy  = np.meshgrid(x, y,indexing='ij') ## Mesh for the 2D domain
    rho0 = np.zeros((Nx,Ny))
    rho1 = np.zeros((Nx,Ny))
    vx0 =np.zeros((Nx,Ny)) 
    vx1 =np.zeros((Nx,Ny))
    vy0 =np.zeros((Nx,Ny))
    vy1 =np.zeros((Nx,Ny))
    
    Px0 =np.zeros((Nx,Ny)) # The flux term  U in the above equations
    Px1 =np.zeros((Nx,Ny))
    Py0 =np.zeros((Nx,Ny)) # The flux term  V in the above equations 
    Py1 =np.zeros((Nx,Ny))
    
    ## gravitational potential
    phi0 = np.zeros((Nx,Ny))
    phi1 = np.zeros((Nx,Ny))
    # print("shape of Phi",phi1.shape)  # Commented out to reduce output noise
        
   ## Calculating the jeans length is gravity is Turned on
    if gravity:
        jeans = np.sqrt(4*np.pi**2*c_s**2/(const*G*rho_o))
        # print("Jean's Length",jeans)  # Commented out to reduce output noise

    ######################## Initial Conditions ###########################
    
    if use_velocity_ps:
        # Uniform density; initialize velocities from power spectrum
        rho0 = rho_o * np.ones((Nx, Ny))
        if vx0_shared is not None and vy0_shared is not None:
            # Use shared velocity fields for consistent initial conditions
            vx0 = vx0_shared.copy()
            vy0 = vy0_shared.copy()
        else:
            vx0, vy0 = generate_velocity_field_power_spectrum(Nx, Ny, Lx, Ly, power_index=ps_index, amplitude=vel_rms, random_seed=random_seed)
    else:
        # Use 2D wave pattern: cos(KX*x + KY*y) instead of cos(2π*x/λ)
        rho0 = rho_o + rho_1 * np.cos(KX * xx + KY * yy)
    
    # Copy initial conditions to rho1 for t=0 case
    rho1 = rho0.copy()
    
    # Copy initial velocity conditions to vx1, vy1 for t=0 case
    vx1 = vx0.copy()
    vy1 = vy0.copy()

    if gravity == False and not use_velocity_ps:
        v_1 = (c_s*rho_1)/rho_o # velocity perturbation
        # Use coupled 2D velocity components derived from the same wave pattern
        k_magnitude = np.sqrt(KX**2 + KY**2)
        if k_magnitude > 0:
            vx0 = v_1 * np.cos(KX * xx + KY * yy) * (KX / k_magnitude)
            vy0 = v_1 * np.cos(KX * xx + KY * yy) * (KY / k_magnitude)
        else:
            vx0 = v_1 * np.cos(KX * xx + KY * yy)
            vy0 = np.zeros_like(xx)

        ## Linear Theory
        if comparison:
            rho_LT  = rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time)
            rho_LT_max = np.max(rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time))
            vx_LT = v_1* np.cos(2*np.pi * x/lam - 2*np.pi/lam *time) 
            vy_LT = np.zeros(Ny)
          
    
    else:    ######## When self-gravity is True and see EQN 12
        if use_velocity_ps:
            # Already initialized vx0, vy0; keep density uniform
            pass
        elif lam >= jeans:  
            #print("There is gravitational instabilty  lam = {} > l_jean ={}".format(lam,jeans))
            alpha = np.sqrt(const*G*rho_o-c_s**2*(2*np.pi/lam)**2)
            v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam)) ## With gravity     
            # Use coupled 2D velocity components derived from the same wave pattern
            k_magnitude = np.sqrt(KX**2 + KY**2)
            if k_magnitude > 0:
                vx0 = -v_1 * np.sin(KX * xx + KY * yy) * (KX / k_magnitude)
                vy0 = -v_1 * np.sin(KX * xx + KY * yy) * (KY / k_magnitude)
            else:
                vx0 = -v_1 * np.sin(KX * xx + KY * yy)
                vy0 = np.zeros_like(xx)
            # print("initial vy",vy[n-1,1,:])
            ##### Density values from Linear Theory at t 
            if comparison:
                rho_LT = rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam))
                vx_LT = -v_1*np.exp(alpha * time)*np.sin(2*np.pi*x/lam)
                vy_LT = np.zeros(Nx)
                
        else:
            #print("There is no gravitational instabilty as lam = {} < l_jean ={}".format(lam,jeans))
            alpha = np.sqrt(c_s**2*(2*np.pi/lam)**2 - const*G*rho_o)
            v_1 = (rho_1/rho_o) * (alpha/(2*np.pi/lam)) # velocity perturbation
            # Use coupled 2D velocity components derived from the same wave pattern
            k_magnitude = np.sqrt(KX**2 + KY**2)
            if k_magnitude > 0:
                vx0 = v_1 * np.cos(KX * xx + KY * yy) * (KX / k_magnitude)
                vy0 = v_1 * np.cos(KX * xx + KY * yy) * (KY / k_magnitude)
            else:
                vx0 = v_1 * np.cos(KX * xx + KY * yy)
                vy0 = np.zeros_like(xx)
            if comparison:
                rho_LT = rho_o + rho_1*np.cos(alpha * time - 2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.cos(alpha * time - 2*np.pi*xx/lam))
                vx_LT = v_1*np.cos(alpha * time - 2*np.pi*x/lam)
                vy_LT = np.zeros((Nx))

        # Calculating the potential and the field using FFT    
#         phi[0,:,:],dphidx,dphidy = fft_solver(const*(rho[0,:,:]-rho_o),Lx,Nx,Ly,Ny,dim = 2)
        phi0 = fft_solver(const*(rho0-rho_o),Lx,Nx,Ly,Ny,dim = 2)
        # print("shape of Phi",phi0.shape)  # Commented out to reduce output noise
#         fft_solver(rho,Lx,nx,Ly,ny,dim = 2)

    

    ####### The Flux term #########
    Px0=rho0*vx0
    Py0=rho0*vy0
    
    #################################FINITE DIFFERENCE #######################
    for k in range(1,n): ## Looping over time

        rho1 =  (1/4)*(np.roll(rho0, -1, axis=0)+ np.roll(rho0, 1, axis=0)\
                        +np.roll(rho0, -1, axis=1)+ np.roll(rho0, 1, axis=1))\
        -(mux*(np.roll(rho0,-1,axis=0)*np.roll(vx0,-1,axis=0)-np.roll(rho0,1, axis=0)*np.roll(vx0,1,axis=0)))\
        -(muy*(np.roll(rho0,-1,axis=1)*np.roll(vy0,-1,axis=1)-np.roll(rho0,1, axis=1)*np.roll(vy0,1,axis=1)))

        if gravity == False: ## Hydro sound wave when gravity is absent  
            
            Px1 = 0.25*(np.roll(Px0,-1,axis=0)+ np.roll(Px0,1,axis=0) + np.roll(Px0,-1,axis=1)+ np.roll(Px0,1,axis=1))\
            -(mux*(np.roll(Px0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Px0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muy*(np.roll(Px0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Px0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -((c_s**2)*mux*(np.roll(rho0,-1,axis=0)- np.roll(rho0,1,axis=0)))

            Py1 = 0.25*(np.roll(Py0,-1,axis=0)+ np.roll(Py0,1,axis=0) + np.roll(Py0,-1,axis=1)+ np.roll(Py0,1,axis=1))\
            -(muy*(np.roll(Py0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Py0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -(mux*(np.roll(Py0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Py0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -((c_s**2)*muy*(np.roll(rho0,-1,axis=1)- np.roll(rho0,1,axis=1)))

             
        else: ## With self-gravity activated 
            Px1 = 0.25*(np.roll(Px0,-1,axis=0)+ np.roll(Px0,1,axis=0) + np.roll(Px0,-1,axis=1)+ np.roll(Px0,1,axis=1))\
            -(mux*(np.roll(Px0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Px0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -(muy*(np.roll(Px0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Px0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -((c_s**2)*mux*(np.roll(rho0,-1,axis=0)- np.roll(rho0,1,axis=0)))\
            -(mux*rho0*(np.roll(phi0,-1,axis=0)- np.roll(phi0,1,axis=0)))

            Py1 = 0.25*(np.roll(Py0,-1,axis=0)+ np.roll(Py0,1,axis=0) + np.roll(Py0,-1,axis=1)+ np.roll(Py0,1,axis=1))\
            -(muy*(np.roll(Py0,-1,axis=1)*np.roll(vy0,-1,axis=1)- np.roll(Py0,1,axis=1)*np.roll(vy0,1,axis=1)))\
            -(mux*(np.roll(Py0,-1,axis=0)*np.roll(vx0,-1,axis=0)- np.roll(Py0,1,axis=0)*np.roll(vx0,1,axis=0)))\
            -((c_s**2)*muy*(np.roll(rho0,-1,axis=1)- np.roll(rho0,1,axis=1)))\
            -(muy*rho0*(np.roll(phi0,-1,axis=1)- np.roll(phi0,1,axis=1)))

  
            phi1= fft_solver(const*(rho1-rho_o),Lx,Nx,Ly,Ny,dim = 2)

        vx1 = Px1/rho1 ## 2-D velocity vx 
        vy1 = Py1/rho1 ## 2-D velocity vy
               ## memory tranfer to overwrite "1" in the next time step
        rho0 = rho1
        vx0 = vx1
        vy0 = vy1
        
        Px0 = Px1
        Py0 = Py1
        
        phi0= phi1
        
        ## Updating dt based on the highest signal speed in the code
        dt1 = nu*dx/np.max([abs(vx1),abs(vy1)])
        dt2 = nu*dx/c_s
        
        
        dt = np.min([dt1,dt2])
        mux = dt/(2*dx)      # is the coefficient in the central differencing Eqs above 
        muy = dt/(2*dy)      # is the coefficient in the central differencing Eqs above
      
    
        n = int(time/dt)     # grid points in time updated dynamically
    rho_max = np.max(rho1)   ## Maximum density from the FD calculation 
    
    # print(ro1)
# #     ################################# PLOTTING #######################
 
    if isplot : 
        plt.figure(1,figsize=(6,4))
        plt.plot(x,rho0[:,1]-rho_o,linewidth=1,label="FD at t={}".format(round(time,2)))
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
        plt.plot(x,vx1[:,1],'--',markersize=2,label="t={}".format(round(time,2)))
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
        plt.plot(y,vy1[1,:],'--',markersize=2,label="t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{y}$")
        plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
        plt.ylabel("vy")
        if comparison : 
            plt.plot(y,vy_LT,'--',linewidth=1,label="LT")
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

#         if dim_plot == True: ## PLotting in 3D
        
#             fig = plt.figure(6)
#             ax = fig.add_subplot(111, projection='3d')
#             sc = ax.scatter(xx, yy, zz, c=rho0)
#             plt.colorbar(sc)
#             plt.savefig(output_folder+'/3D-denstiy'+str(lam)+'_'+str(num_of_waves)+'_'+str(t)+'.png', dpi=300)
#             plt.show()



    else:
        if gravity:
            if comparison:
                return x,rho1,vx1,phi1,n,rho_LT,rho_LT_max,rho_max,vx_LT
            else:
                return x,rho1,vx1,vy1,phi1,n,rho_max
        else:
            if comparison:
                return rho1,vx1,rho_LT,rho_LT_max,rho_max,vx_LT
            else:
                return rho1,vx1,rho_max


def lax_solution1D_sinusoidal(time,N,nu,lam,num_of_waves,rho_1,gravity=False,isplot = None,comparison =None,animation=None):
    '''
    1D LAX solver for sinusoidal initial conditions with optional self-gravity and linear theory outputs.
    Returns density, velocity, potential (if gravity), and optional linear theory references.
    '''
    lam = lam
    L = lam * num_of_waves

    c_s = cs             # Sound Speed (from config)
    rho0_base = rho_o    # Background density (from config)
    # nu, const, G are already imported from config, no need to reassign

    nx = int(N)
    dx = float(L / nx)
    dt = nu * dx / c_s
    mu = dt / (2 * dx)
    n = int(time / dt)

    x = np.linspace(0, L, nx)

    rho0 = np.zeros(nx)
    phi0 = np.zeros(nx)
    v0 = np.zeros(nx)
    P0 = np.zeros(nx)

    rho1 = np.zeros(nx)
    phi1 = np.zeros(nx)
    v1 = np.zeros(nx)
    P1 = np.zeros(nx)

    if gravity:
        jeans = np.sqrt(4*np.pi**2*c_s**2/(const*G*rho0_base))

    # Initial conditions
    rho0 = rho0_base + rho_1 * np.cos(2*np.pi*x/lam)

    if not gravity:
        v_1 = (c_s * rho_1) / rho0_base
        v0 = v_1 * np.cos(2*np.pi*x/lam)
        if comparison:
            rho_LT = rho0_base + rho_1 * np.cos(2*np.pi * x/lam - 2*np.pi/lam * time)
            rho_LT_max = np.max(rho_LT)
            v_LT = v_1 * np.cos(2*np.pi * x/lam - 2*np.pi/lam * time)
    else:
        if lam >= jeans:
            alpha = np.sqrt(const*G*rho0_base - c_s**2 * (2*np.pi/lam)**2)
            v_1 = (rho_1 / rho0_base) * (alpha / (2*np.pi/lam))
            v0 = - v_1 * np.sin(2*np.pi*x/lam)
            if comparison:
                rho_LT = rho0_base + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam)
                rho_LT_max = np.max(rho_LT)
                v_LT = -v_1*np.exp(alpha * time)*np.sin(2*np.pi*x/lam)
        else:
            alpha = np.sqrt(c_s**2*(2*np.pi/lam)**2 - const*G*rho0_base)
            v_1 = (rho_1 / rho0_base) * (alpha / (2*np.pi/lam))
            v0 = v_1 * np.cos(2*np.pi*x/lam)
            if comparison:
                rho_LT = rho0_base + rho_1*np.cos(alpha * time - 2*np.pi*x/lam)
                rho_LT_max = np.max(rho_LT)
                v_LT = v_1*np.cos(alpha * time - 2*np.pi*x/lam)

        # 1D Poisson (periodic) via FFT: phi_k = rho_k / (-k^2), k=0 set to 0
        k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
        rhohat = np.fft.fft(const*(rho0 - rho0_base))
        denom = -(k**2)
        denom[0] = 1.0
        phihat = rhohat / denom
        phihat[0] = 0.0
        phi0 = np.real(np.fft.ifft(phihat))

    P0 = rho0 * v0

    for _ in range(1, n):
        rho1 = 0.5*(np.roll(rho0,-1)+ np.roll(rho0,1)) - mu*(np.roll(rho0,-1)*np.roll(v0,-1) - np.roll(rho0,1)*np.roll(v0,1))

        if not gravity:
            P1 = 0.5*(np.roll(P0,-1)+ np.roll(P0,1)) - mu*(np.roll(P0,-1)*np.roll(v0,-1) - np.roll(P0,1)*np.roll(v0,1)) - (c_s**2)*mu*(np.roll(rho0,-1) - np.roll(rho0,1))
        else:
            P1 = 0.5*(np.roll(P0,-1)+ np.roll(P0,1)) - mu*(np.roll(P0,-1)*np.roll(v0,-1) - np.roll(P0,1)*np.roll(v0,1)) - (c_s**2)*mu*(np.roll(rho0,-1) - np.roll(rho0,1)) - mu*rho0*(np.roll(phi0,-1) - np.roll(phi0,1))
            k = 2*np.pi*np.fft.fftfreq(nx, d=dx)
            rhohat = np.fft.fft(const*(rho1 - rho0_base))
            denom = -(k**2)
            denom[0] = 1.0
            phihat = rhohat / denom
            phihat[0] = 0.0
            phi1 = np.real(np.fft.ifft(phihat))

        v1 = P1 / rho1

        rho0, v0, P0, phi0 = rho1, v1, P1, phi1

        vmax = np.max(np.abs(v1))
        dt1 = nu*dx/(vmax if vmax != 0 else c_s)
        dt2 = nu*dx/c_s
        dt = min(dt1, dt2)
        mu = dt/(2*dx)
        n = int(time/dt)

    rho_max = np.max(rho1)

    if isplot:
        return
    else:
        if gravity:
            if comparison:
                return x, rho1, v1, phi1, n, rho_LT, rho_LT_max, rho_max, v_LT
            else:
                return x, rho1, v1, phi1, n, rho_max
        else:
            if comparison:
                return rho1, v1, rho_LT, rho_LT_max, rho_max, v_LT
            else:
                return rho1, v1, rho_max