import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal
from torch.autograd import Variable
from config import rho_o
from LAX import fft_solver, lax_solution1D

def plot_function(net,time_array,initial_params, N, velocity = False,isplot =False ,animation=False):
    xmin, xmax, rho_1, alpha, v_1, jeans, lam, tmax, device = initial_params  
    rho_o = 1.0          # zeroth order density
    num_of_waves = xmax-xmin/lam

    res = N
    
    Y_cut = Z_cut = 0.5
    if animation:
        ## Converting the float (time-input) to an numpy array for animation
        ## Ignore this when the function is called in isolation
        time_array = np.array([time_array])
        # print("time",np.asarray(time_array))
    
    rho_max_Pinns = []    
    peak_lst=[]
    pert_xscale=[]
    for t in time_array:
        #print("Plotting at t=", t)
        X = np.linspace(xmin,xmax,res).reshape(res, 1)
        #Y = Y_cut*np.ones(res).reshape(res, 1)
        #Z = Z_cut*np.ones(res).reshape(res, 1)
        t_ = t*np.ones(res).reshape(res, 1)
        pt_x_collocation = Variable(torch.from_numpy(X).float(), requires_grad=True).to(device)
        #pt_y_collocation = Variable(torch.from_numpy(Y).float(), requires_grad=True).to(device)
        #pt_z_collocation = Variable(torch.from_numpy(Z).float(), requires_grad=True).to(device)
        pt_t_collocation = Variable(torch.from_numpy(t_).float(), requires_grad=True).to(device)
        test_coor = []
        test_coor.append(pt_x_collocation)
        #test_coor.append(pt_y_collocation)
        #test_coor.append(pt_z_collocation)
        test_coor.append(pt_t_collocation)
        # X_0 = np.hstack((pt_x_collocation,pt_t_collocation))
        # test_coor = torch.cat([pt_x_collocation,pt_y_collocation, pt_z_collocation,pt_t_collocation],axis=1)
        alpha_coor = torch.tensor([alpha], device=device, dtype=torch.float32).repeat(res, 1)
        test_coor.append(alpha_coor)
        output_0 = net(test_coor)
        
        rho_pred0 = output_0[:,0:1].data.cpu().numpy()
        #print(np.shape(rho_pred0))
        vx_pred0 = output_0[:, 1:2].data.cpu().numpy()
        phi_pred0=output_0[:, 2:3].data.cpu().numpy()
 
        rho_max_PN = np.max(rho_pred0)
        ## Theoretical Values
        rho_theory = rho_o + alpha*np.exp(alpha * t)*np.cos(2*np.pi*X[:, 0:1]/lam)
        vx_theory = -v_1*np.exp(alpha * t)*np.sin(2*np.pi*X[:, 0:1]/lam) if lam > jeans else v_1*np.exp(alpha * t)*np.cos(2*np.pi*X[:, 0:1]/lam) ## velocity theory
        
        ## Theoretical Values
        rho_theorymax = np.max(rho_o + alpha*np.exp(alpha * t)*np.cos(2*np.pi*X[:, 0:1]/lam))
        rho_theory0 = np.max(rho_o + alpha*np.exp(alpha * 0)*np.cos(2*np.pi*X[:, 0:1]/lam)) ## at t =0

        diff=abs(rho_max_PN-rho_theory)/abs(rho_max_PN+rho_theory) * 2  ## since the den is rhomax+rhotheory'''

        if isplot:              
            #print("rho_theory_max={} at time {}".format(rho_theorymax,t))
            plt.figure(1)
            plt.plot(X,rho_pred0,label="t={}".format(round(t,2)))
            #plt.plot(X[:, 0:1],rho_theory,'--',label="LT" )
            plt.ylabel(r"$\rho$")
            plt.xlabel("x")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            # plt.title("Pinns Solution for $\lambda = {}$".format(lam))
            plt.title(r"Pinns Solution for $\lambda$ = {} $\lambda_J$ ".format(round(lam/(2*np.pi),2)))
            #plt.savefig(output_folder+'/PINNS_density'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)


            if velocity == True:
              plt.figure(2)
              plt.plot(X,vx_pred0,label="t={}".format(round(t,2)))
              #plt.plot(X[:, 0:1],vx_theory,'--',label="LT" )
              plt.ylabel("$v$")
              plt.xlabel("x")
              plt.title("Pinns Solution Velocity")
              plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
              #plt.savefig(output_folder+'/PINNS_velocity'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)


            plt.figure(3)
            plt.plot(X,phi_pred0,label="t={}".format(round(t,2)))
            plt.ylabel(r"$\phi$")
            plt.xlabel("x")
            plt.title("Pinns Solution phi")
            plt.legend(numpoints=1,loc='upper right',fancybox=True,shadow=True)
            #plt.savefig(output_folder+'/phi'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)

            '''plt.figure(4)
            
            plt.scatter(t,rho_max_PN)
            plt.plot(t,rho_theorymax,marker='^',label="LT")
            plt.legend(numpoints=1,loc='upper left',fancybox=True,shadow=True)
    #         plt.axhline(rho_theory , color = 'r', linestyle = '--')
            plt.xlabel("t")
            plt.ylabel(r"$\rho_{\rm max}$")
            #plt.savefig(output_folder+'/rho_max'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=300)'''
        
        else:  
            if animation:
                return X,rho_pred0,vx_pred0,phi_pred0,rho_max_PN,rho_theory
            else:
                return X ,rho_pred0,rho_max_PN,rho_theory 

    plt.show()

def rel_misfit(net, time_array, initial_params, N, nu, num_of_waves, rho_1, show = True):

    xmin, xmax, rho_1, alpha, v_1, jeans, lam, tmax, device = initial_params

    misfit_rho, misfit_vel = [], []

    plt.style.use('default')
    plt.rc('grid', linestyle='-', color='black', linewidth=0.05)

    fig, axes = plt.subplots(4, len(time_array), sharex=True,  sharey='row',figsize=(12,6), 
                             gridspec_kw={'width_ratios':[1]*len(time_array), 'height_ratios':[3,1.2,3,1.2]})
    plt.subplots_adjust(wspace=0.12, hspace=0.1)
    #initial_params = xmin,xmax,rho_1, alpha ,lam,tmax ## Params for pl
    for time,j in zip(time_array,range(len(time_array))):
        
        x,rho,v,phi,n,rho_LT,rho_LT_max,rho_max,v_LT= lax_solution1D(time,N,nu,lam,num_of_waves,
                                                                     alpha,gravity=True,isplot = False,comparison = True)
        X,rho_pred0,v_pred0,phi_pred0,rho_max_PN,rho_theory = plot_function(net,time,initial_params, N, velocity=True,
                                                                            isplot = False, animation=True)
        axes[0][j].plot(X,rho_pred0,color='c',linewidth=5,label="PN")
        if alpha < 0.1:
            axes[0][j].plot(x,rho_LT,linestyle='dashed',color ='firebrick',linewidth=3,label="LT")
        axes[0][j].plot(x,rho,linestyle='solid',color = 'black',linewidth=1.0,label="FD")
        axes[0][j].set_xlim(xmin,xmax)
        
        axes[0][j].set_title("Time={}".format(round(time,2)))
        axes[0][0].set_ylabel(r"$\rho$",fontsize = 18)
        # axes[0][j].set_xlabel("x",fontsize = 18)
        axes[0][j].grid("True")
        axes[0][j].minorticks_on()
        axes[0][j].tick_params(labelsize=10)
        axes[0][j].tick_params(axis='both', which='major',length=4, width=2)
        axes[0][j].tick_params(axis='both', which='minor',length=2, width=1)
        if alpha < 0.1:
            limu = 1.3*rho_o
            liml = .7*rho_o
        else:
            limu = 2.8*rho_o
            liml = .1*rho_o
        axes[0][j].set_ylim(liml,limu)
        # Show legend only in last column for 1st row
        if j == len(time_array)-1:
            axes[0][j].legend(loc='upper right',fancybox=False, shadow=False, ncol=3,fontsize = 10)
        axes[0][0].text(0.42, 0.82, r"$\rho_1$ = {}, $\lambda$ = {} $\lambda_J$ ".format(alpha,round(lam/(2*np.pi),2)),fontsize = 12, horizontalalignment='center', verticalalignment='center', transform=axes[0][0].transAxes) 

        misfit_r = (rho_pred0[:,0] - rho)/((rho_pred0[:,0] + rho)/2)*100
        
        axes[1][j].plot(X, misfit_r, color = 'black',linewidth=1,label="FD")
        # axes[1][j].plot(x,(rho_pred0.flatten()-rho_LT.flatten())/((rho_pred0.flatten()+ rho_LT.flatten())/2)*100,color = 'firebrick',linestyle='dashed',linewidth=1,label="LT")
        # axes[1][j].plot(x,(rho_pred0[:,0]- rho[n-1,:])/((rho_pred0[:,0]+ rho[n-1,:])/2)*100,color = 'k',linewidth=1,label="FD")
        if alpha < 0.1:
            axes[1][j].plot(X,(rho_pred0[:,0]-rho_LT)/((rho_pred0[:,0]+ rho_LT)/2)*100,color = 'b',linewidth=1,label="LT")
        axes[1][j].set_xlabel("x",fontsize = 18)
        axes[1][j].grid("True")
        axes[1][j].minorticks_on()
        axes[1][j].tick_params(labelsize=10)
        axes[1][j].tick_params(axis='both', which='major',length=4, width=2)
        axes[1][j].tick_params(axis='both', which='minor',length=2, width=1.)
        #axes[1][2].legend(loc='best',fancybox=False, shadow=False, ncol=3,fontsize = 10)
        # axes[1][0].set_ylabel(r"$\rho_{PN}- \rho_{FD or LT}/(0.5 (\times\rho_{PN}+ \rho_{FD or LT}))$",fontsize = 10)
        axes[1][j].set_ylim(-10.0, 10.0)
        axes[1][j].set_xlim(xmin,xmax)
        axes[1][0].set_ylabel(r"Rel misfit $\%$ ",fontsize = 14)
        
        
        ### VELOCITY PART ######
    
        axes[2][j].plot(X,v_pred0,color='c',linewidth=5,label="PN")
        if alpha < 0.1:
            axes[2][j].plot(X,v_LT,linestyle='dashed',color ='firebrick',linewidth=3,label="LT")
        axes[2][j].plot(x,v,linestyle='solid',color = 'black',linewidth=1.0,label="FD")
        
        
        # axes[3][j].set_title("Time={}".format(round(time,2)))
        axes[2][0].set_ylabel(r"$v$",fontsize = 18)
        # axes[0][j].set_xlabel("x",fontsize = 18)
        axes[2][j].grid("True")
        axes[2][j].minorticks_on()
        axes[2][j].tick_params(labelsize=8)
        axes[2][j].tick_params(axis='both', which='major',length=2, width=1)
        axes[2][j].tick_params(axis='both', which='minor',length=1, width=1)
        axes[2][j].autoscale(enable=True, axis='y')
        axes[2][j].margins(y=0.8)
        '''limu = 1.5*v_1
        liml = .1*v_1
        axes[2][j].set_ylim(liml,limu)'''
        # Show legend only in last column for 3rd row
        if j == len(time_array)-1:
            axes[2][j].legend(loc='upper right',fancybox=False, shadow=False, ncol=3,fontsize = 10)
        
    #     axes[2][j].set_xlim(xmin,xmax)
    #     axes[2][j].set_ylim(-0.6,0.6)
        
        misfit_v = (v_pred0[:,0]+1 - (v+1))/((v_pred0[:,0]+1 + v+1)/2)*100
        
        axes[3][j].plot(x, misfit_v, color = 'black',linewidth=1,label="FD")
        if alpha < 0.1:
            axes[3][j].plot(x,(v_pred0[:,0]+1- (v_LT+1))/((v_pred0[:,0]+1+ (v_LT+1))/2)*100,color = 'b',linewidth=1,label="LT")
        axes[3][j].set_xlabel("x",fontsize = 18)
        axes[3][j].grid("True")
        axes[3][j].minorticks_on()
        axes[3][j].tick_params(labelsize=8)
        axes[3][j].tick_params(axis='both', which='major',length=2, width=1)
        axes[3][j].tick_params(axis='both', which='minor',length=1, width=1.)
        #axes[3][2].legend(loc='best',fancybox=False, shadow=False, ncol=3,fontsize = 10)
        # axes[1][0].set_ylabel(r"$\rho_{PN}- \rho_{FD or LT}/(0.5 (\times\rho_{PN}+ \rho_{FD or LT}))$",fontsize = 10)
        axes[3][0].set_ylabel(r"$\epsilon$ ",fontsize = 18)
        axes[3][j].set_ylim(-10.0, 10.0)
        axes[3][j].set_xlim(xmin,xmax)
        
        misfit_rho.append(misfit_r)
        misfit_vel.append(misfit_v)
        
    #plt.savefig(output_folder+'/complete'+str(lam)+'_'+str(num_of_waves)+'_'+str(tmax)+'.png', dpi=500,bbox_inches = 'tight')

    if show is True:
        plt.show()

    return misfit_rho, misfit_vel
 