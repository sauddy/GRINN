
from numpy.fft  import fft, ifft,fft2, ifft2,fftn, ifftn
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

print("Importing the LAX Module")

def fft_solver(rho,Lx,N, dim = None):
    
    '''
    A FFT solver that uses discrete Fast Fourier Transform to
    solve the poisson Equation:
    We apply the correction due to the finite difference grid of phi
    
    Input: 1. The source function density in this case
           2. # of grid point N
           3. Domain Size in each dimension
    
    Output: the potential phi and the field g 
    
    '''
    nx = N
    Lx = Lx
    
    dx = Lx / nx
    
    
    # Calculate the Fourier modes of the gas density
    rhohat = fft(rho)

    # Calculate the wave numbers in x and y directions
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    #ky = 2 * np.pi * np.fft.fftfreq(ny, dy)

    # Construct the Laplacian operator in Fourier space
    kx2 = np.meshgrid(kx**2)
    
#     laplace = -(kx**2 )
    ## Laplace with the correction refer to the notes
    laplace = 2*(np.cos(kx*dx)-1)
    
    ## Inorder to avoid the inf we replace zero with a small number
    laplace[laplace == 0] = 1e-9
    
    # Solve for the gravitationa potential in Fourier space
    phihat = rhohat / laplace
    
    phihat = rhohat * dx**2/laplace

    phi = np.real(ifft(phihat))
    
    ## The field ,i.e., gravity 
#     dphidx = np.gradient(phi, dx)

    return phi


def lax_solution1D(time,N,nu,lam,num_of_waves,rho_1,gravity=False,isplot = None,comparison =None,animation=None):
    '''
    This function solves the hydrodynamic Eqns in 1D with/without self gravity using LAX methods 
    described above 
    
    
    Input:  Time till the system is integrated :time
            Number of Xgrid points : N
            Courant number : nu
            Wavelength : If lambda> lambdaJ (with gravity--> Instability) else waves propagation 
            Number of waves : The domain size changes and with this maintains periodicity
            Density perturbation : rho_1 (for linear or non-linear perturbation)
            Gravity:  If True it deploys the FFT routine to estimate the potential 
            isplot(optional): if True plots the outputs
            Comparison (optional) : If True then the plots are overplotted with LT solutions for comparison
            Animation (optional): Not used at the moment
    
    Output: Density, velocity + (phi and g if gravity is True)
            isplot: True then the plots are generated 
    
    '''
    
    
    # rho_max = []
    lam = lam          # one wavelength
    num_of_waves  = num_of_waves  
    L = lam * num_of_waves            # Maximum length (two wavelength)
    
    print("at time= ",time)
    ### Declaring the Constants

    c_s = 1.0            # % Sound Speed  
    rho_o = 1.0          # zeroth order density
    nu = nu              # courant number (\nu = 2 in 2d)
    rho_1 = rho_1        # for linear/nonlinear wave propagation
    const =  1           # The actual value is 4*pi
    G = 1.0              # Gravitational Constant

    ### Grid X-T 
    N = N                # The grid resolution values2d:N =(10,50,100,500)
    dx = float(L/N)      # length spacing          
    dt = nu*dx/c_s       # time grid spacing
 

    ## For simplification
    mu = dt/(2*dx)      # is the coefficient in the central differencing Eqs above 
    
    n = int(time/dt)     # grid points in time
    print("For dx = {} and dt = {} and time gridpoints n = {} ".format(dx,dt,n))
    
    ########### Initializing the ARRAY #######################
    x = np.linspace(0, L, N)
    
    rho0 = np.zeros(N)
    phi0 = np.zeros(N)
    v0 =np.zeros(N)  
    P0 =np.zeros(N) # The flux term 
    
    rho1 = np.zeros(N)
    phi1 = np.zeros(N)
    v1 =np.zeros(N)  
    P1 =np.zeros(N)
    
    ## Calculating the jeans length if gravity is Turned on
    if gravity:
        jeans = np.sqrt(4*np.pi**2*c_s**2/(const*G*rho_o))
        print("Jean's Length",jeans)

    
    ######################## Initial Conditions ###########################
    rho0 = rho_o + rho_1* np.cos(2*np.pi*x/lam) # defing the density at t = 0 EQ 11
    
    if gravity == False:
        print("Propagation of Sound wave") 
        v_1 = (c_s*rho_1)/rho_o # velocity perturbation
        v0 = v_1 * np.cos(2*np.pi*x/lam) # the velocity at t =0
        ## Linear Theory
        if comparison:
            rho_LT  = rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time)
            rho_LT_max = np.max(rho_o + rho_1*np.cos(2*np.pi * x/lam - 2*np.pi/lam *time))
            v_LT = v_1* np.cos(2*np.pi * x/lam - 2*np.pi/lam *time) 
    
    else:    ######## When self-gravity is True and see EQN 22
        if lam >= jeans:  
            print("There is gravitational instabilty  lam = {} > l_jean ={}".format(lam,jeans))
            alpha = np.sqrt(const*G*rho_o-c_s**2*(2*np.pi/lam)**2)
            v_1  = (rho_1/rho_o) * (alpha/(2*np.pi/lam)) ## With gravity        
            v0 = - v_1 * np.sin(2*np.pi*x/lam) # the velocity at t =0
            
            ##### Density  and velocity  from Linear Theory at t 
            if comparison:
                rho_LT = rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.exp(alpha * time)*np.cos(2*np.pi*x/lam))
                v_LT = -v_1*np.exp(alpha * time)*np.sin(2*np.pi*x/lam)
#             print("rho_theory_max={} and the max density {} at time {}".format(rho_LT_max ,rho_max, round(time,2)))

        else:
            print("There is no gravitational instabilty as lam = {} < l_jean ={}".format(lam,jeans))
            alpha = np.sqrt(c_s**2*(2*np.pi/lam)**2 - const*G*rho_o)
            v_1 = (rho_1/rho_o) * (alpha/(2*np.pi/lam)) # velocity perturbation
            v0 = v_1 * np.cos(2*np.pi*x/lam) # the velocity at t =0

            ##### Density  and velocity  from Linear Theory at t
            if comparison:
                rho_LT = rho_o + rho_1*np.cos(alpha * time - 2*np.pi*x/lam)
                rho_LT_max = np.max(rho_o + rho_1*np.cos(alpha * time - 2*np.pi*x/lam))
                v_LT = v_1*np.cos(alpha * time - 2*np.pi*x/lam)

        ## Calculating the potential and the field using FFT    
        phi0 = fft_solver(const*(rho0-rho_o),L,N, dim = None)

    ######### The Flux term #########
    P0=rho0*v0

    #################################FINITE DIFFERENCE #######################
    for k in range(1,n): ## Looping over time 

        rho1 = 0.5*(np.roll(rho0,-1)+ np.roll(rho0,1))-(mu*(np.roll(rho0,-1)*np.roll(v0,-1)-np.roll(rho0,1)*np.roll(v0,1)))

        if gravity == False: ## Hydro sound wave when gravity is absent 

            P1 = 0.5*(np.roll(P0,-1)+ np.roll(P0,1))-(mu*(np.roll(P0,-1)*np.roll(v0,-1)- np.roll(P0,1)*np.roll(v0,1)))-((c_s**2)*mu*(np.roll(rho0,-1)- np.roll(rho0,1)))
        else: ## With self-gravity activated 

            P1 = 0.5*(np.roll(P0,-1)+ np.roll(P0,1))-(mu*(np.roll(P0,-1)*np.roll(v0,-1)- np.roll(P0,1)*np.roll(v0,1)))\
            -((c_s**2)*mu*(np.roll(rho0,-1)- np.roll(rho0,1))) -(mu*rho0*(np.roll(phi0,-1)- np.roll(phi0,1))) #  

            phi1= fft_solver(const*(rho1-rho_o),L,N, dim = None) ## Please note we don't use dphidx rather calculate using central differencing


        ## 1-D velocity 
        v1 = P1/rho1
        
        ## updating the intial array
        rho0 = rho1
        v0 = v1
        P0 = P1
        phi0= phi1
        
        ## updating the dt for numerical stability
        dt1 = nu*dx/np.max(abs(v1))
        dt2 = nu*dx/c_s        
        dt = np.min([dt1,dt2])
        mu = dt/(2*dx)     
        n = int(time/dt)     # grid points in time

    
    
    rho_max = np.max(rho1)   ## Maximum density from the FD calculation 
    
    
    ################################# PLOTTING #######################
 
    if isplot : 
        plt.figure(1,figsize=(6,4))
        plt.plot(x,rho1-rho_o,linewidth=1,label="FD at t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{x}$")
        # plt.text(.6,.15,r"dt=%f"%(dt),fontsize=12)
        plt.title("At time {} and rho_1 = {}".format(time,rho_1))
        plt.ylabel(r"$\mathbf{\rho - \rho_{0}}$")
        if comparison : 
            plt.plot(x,rho_LT-rho_o,'--',linewidth=1,label="LT")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)

        plt.figure(2,figsize=(6,4))
        plt.plot(x,v1,'--',markersize=2,label="t={}".format(round(time,2)))
        plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
        plt.xlabel(r"$\mathbf{x}$")
        plt.title(r"Lax Solution Velocity For $\rho_1$ = {}".format(rho_1))
        plt.ylabel("velocity")
        if comparison : 
            plt.plot(x,v_LT,'--',linewidth=1,label="LT")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)    

        if gravity:
             #### Plotting the comparison of the \rho_max for FD and Linear Theory 
            
            plt.figure(3,figsize=(6,4))                                
            plt.scatter(time,rho_max,label="LT")                     
            plt.xlabel("t")
            plt.ylabel(r"$\log (\rho_{\rm max} - \rho_{0}) $")
            plt.yscale('log')            
            if comparison : 
                plt.scatter(time,rho_LT_max,facecolors='none', edgecolors='r',label="LT") 
                plt.legend(numpoints=1,loc='upper left',fancybox=True,shadow=True)
                
            ## Plotting the gravitational potential (\phi) and field (g)
            plt.figure(4,figsize=(6,4))
            plt.plot(x,phi1,'--',markersize=2,label="t=phi at {}".format(round(time,2)))
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            plt.xlabel(r"$\mathbf{x}$")
            plt.title(r"Lax Solution Phi For $\rho_1$ = {}".format(rho_1))
            plt.ylabel(r"$\Phi$")

    else:
        if gravity:
            if comparison:
                return x,rho1,v1,phi1,n,rho_LT,rho_LT_max,rho_max,v_LT
            else:
                return x,rho1,v1,phi1,n,rho_max
        else:
            if comparison:
                return rho1,v1,rho_LT,rho_LT_max,rho_max,v_LT
            else:
                return rho1,v1,rho_max
            
    ## Clearing the memory
    del rho0, phi0, v0, P0
