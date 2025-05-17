from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import scipy

has_gpu = torch.cuda.is_available()
has_mps = torch.backends.mps.is_built()
device = "mps" if torch.backends.mps.is_built() \
    else "cuda:0" if torch.cuda.is_available() else "cpu"


def plot_function(net,time_array,initial_params,velocity = False, isplot =False, animation=False):
    xmin,xmax,ymax,ymin,rho_1, alpha,lam,output_folder,tmax = initial_params  
    rho_o = 1.0          # zeroth order density
    num_of_waves_x = xmax-xmin/lam
    num_of_waves_y = ymax-ymin/lam
    if animation:
        ## Converting the float (time-input) to an numpy array for animation
        ## Ignore this when the function is called in isolation
        time_array = np.array([time_array])
        # print("time",np.asarray(time_array))
    
    rho_max_Pinns = []    
    peak_lst=[]
    pert_xscale=[]
    for t in time_array:
        print("Plotting at t=", t)
        X = np.linspace(xmin,xmax,1000).reshape(1000, 1)
        Y = np.linspace(ymin,ymax,1000).reshape(1000, 1)
        t_ = t*np.ones(1000).reshape(1000, 1)
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        pt_y_collocation = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        
        # X_0 = np.hstack((pt_x_collocation,pt_t_collocation))
        
        output_0 = net(pt_x_collocation,pt_y_collocation,pt_t_collocation)
        
        rho_pred0 = output_0[:, 0:1].data.cpu().numpy()
        v_pred_x0 = output_0[:, 1:2].data.cpu().numpy()
        v_pred_y0 = output_0[:, 2:3].data.cpu().numpy()
        phi_pred0 = output_0[:, 3:4].data.cpu().numpy()
 
        rho_max_PN = np.max(rho_pred0)
        
        ## Theoretical Values
        rho_theory = np.max(rho_o + rho_1*np.exp(alpha * t)*np.cos(2*np.pi*X[:, 0:1]/lam))
        rho_theory0 = np.max(rho_o + rho_1*np.exp(alpha * 0)*np.cos(2*np.pi*X[:, 0:1]/lam)) ## at t =0 
        
        diff=abs(rho_max_PN-rho_theory)/abs(rho_max_PN+rho_theory) * 2  ## since the den is rhomax+rhotheory

        
#         ### Difference between peaks for the PINNs solution
        
#         rho_pred0Flat=rho_pred0.reshape(-1)
#         peaks,_=scipy.signal.find_peaks(rho_pred0Flat)
#         peak_lst.append(peaks)
        
#         growth_pert=(rho_theory-rho_theory0)/rho_theory0*100 ## growth percentage
        
#         peak_diff=(rho_pred0Flat[peaks[1]]-rho_pred0Flat[peaks[0]])/(rho_pred0Flat[peaks[1]]+rho_pred0Flat[peaks[0]])

        #g_pred0=phi_x = dde.grad.jacobian(phi_pred0, X, i=0, j=0)
        if isplot:              
            print("rho_theory_max={} at time {} in x".format(rho_theory,t))
            plt.figure(1)
            plt.plot(X,rho_pred0,label="t={}".format(round(t,2)))
            plt.ylabel(r"$\rho$")
            plt.xlabel("X")
            plt.grid()
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            # plt.title("Pinns Solution for $\lambda = {}$".format(lam))
            plt.title(r"Pinns Solution for $\lambda$ = {} $\lambda_J$ in x direction".format(round(lam/(2*np.pi),2)))
            plt.savefig(output_folder+'/PINNS_density'+str(lam)+'_'+str(num_of_waves_x)+'_'+str(tmax)+'_X'+'.png', dpi=300)

            print("rho_theory_max={} at time {} in y".format(rho_theory,t))
            plt.figure(2)
            plt.plot(X,rho_pred0,label="t={}".format(round(t,2)))
            plt.ylabel(r"$\rho$")
            plt.xlabel("Y")
            plt.grid()
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            # plt.title("Pinns Solution for $\lambda = {}$".format(lam))
            plt.title(r"Pinns Solution for $\lambda$ = {} $\lambda_J$ in y direction".format(round(lam/(2*np.pi),2)))
            plt.savefig(output_folder+'/PINNS_density'+str(lam)+'_'+str(num_of_waves_y)+'_'+str(tmax)+'Y'+'.png', dpi=300)


            if velocity == True:
              plt.figure(3)
              plt.plot(X,v_pred_x0,'--',label="t={}".format(round(t,2)))
              plt.ylabel("$v$")
              plt.xlabel("X")
              plt.title("Pinns Solution for Velocity in x direction")
              plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
              plt.savefig(output_folder+'/PINNS_velocity'+str(lam)+'_'+str(num_of_waves_x)+'_'+str(tmax)+'_X'+'.png', dpi=300)

              plt.figure(4)
              plt.plot(X,v_pred_y0,'--',label="t={}".format(round(t,2)))
              plt.ylabel("$v$")
              plt.xlabel("Y")
              plt.title("Pinns Solution for Velocity in y direction")
              plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
              plt.savefig(output_folder+'/PINNS_velocity'+str(lam)+'_'+str(num_of_waves_y)+'_'+str(tmax)+'_X'+'.png', dpi=300)


            plt.figure(5)
            plt.plot(X,phi_pred0,'--',label="t={}".format(round(t,2)))
            plt.ylabel(r"$\phi$")
            plt.xlabel("X")
            plt.title("Pinns Solution phi in X direction")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            plt.savefig(output_folder+'/phi'+str(lam)+'_'+str(num_of_waves_x)+'_'+str(tmax)+'_Y'+'.png', dpi=300)

            plt.figure(6)
            plt.plot(X,phi_pred0,'--',label="t={}".format(round(t,2)))
            plt.ylabel(r"$\phi$")
            plt.xlabel("Y")
            plt.title("Pinns Solution phi in Y direction")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            plt.savefig(output_folder+'/phi'+str(lam)+'_'+str(num_of_waves_y)+'_'+str(tmax)+'_Y'+'.png', dpi=300)

            '''plt.figure()
            
            plt.scatter(t,rho_max_PN)
            plt.plot(t,rho_theory,marker='^',label="LT")
            plt.legend(numpoints=1,loc='upper left',fancybox=True,shadow=True)
    #         plt.axhline(rho_theory , color = 'r', linestyle = '--')
            plt.xlabel("t")
            plt.ylabel(r"$\rho_{\rm max}$")
            plt.savefig(output_folder+'/rho_max'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)
            
            
#             plt.figure(6)
#             plt.scatter(growth_pert,peak_diff*100)
#             plt.ylabel("peak difference $\%$")
#             plt.xlabel("growth $\%$")
#             #plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
#             # plt.savefig('lambda7_peak_diff_growth.png')
#             # plt.grid(True)
#             plt.title('DIF bet 1st and central peak')
#             plt.savefig(output_folder+'/gwt2'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)
            
            # plt.figure(7)
            # plt.scatter(growth_pert,diff*100)
            # plt.ylabel("rel DIF-FD-PINN $\%$")
            # plt.xlabel("growth $\%$")
            # #plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            # # plt.savefig('lambda7_peak_diff_growth.png')
            # # plt.grid(True)
            # plt.title('relative DIF-FD-PINN')
            # plt.savefig(output_folder+'/gwt1'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)'''
        
        else:  
            if animation:
                return X ,rho_pred0,v_pred_x0,v_pred_y0,phi_pred0,rho_max_PN,rho_theory
            else:
                return X ,rho_pred0,rho_max_PN,rho_theory 


#         plt.show()

