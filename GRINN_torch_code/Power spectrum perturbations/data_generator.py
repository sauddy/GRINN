import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

def diff(u,var,order=1): #The derivative of a variable with respect to another.
    
    u.requires_grad_()
    var.requires_grad_()
    ones = torch.ones_like(u)
    der, = torch.autograd.grad(u, var, create_graph=True, grad_outputs=ones, allow_unused=True)
    if der is None:
        return torch.zeros_like(var, requires_grad=True)
    else:
        der.requires_grad_()
    for i in range(1, order):
        ones = torch.ones_like(der)
        der, = torch.autograd.grad(der, var, create_graph=True, grad_outputs=ones, allow_unused=True)
        if der is None:
            return torch.zeros_like(var, requires_grad=True)
        else:
            der.requires_grad_()
    return der


class col_gen(object):

    def __init__(self,rmin=[0,0,0,0],rmax=[1,1,1,1], N_0 = 1000,N_b=1000,N_r = 3000, dimension=1):
        
        '''
        This function accepts the range of the computational domain and returns the collocation points depending on the dimension

        xmin=0, xmax=1,: Default is seto to 0 and 1 for xmin and xmax respestively 
        ymin=0, ymax=0 : Default is set is to 0 when only dimension 1 is used
        zmin=0, zmax=0 : Default is set is to 0 when only dimension 1 & 2 is used
        tmin=0, tmax=1 : Default is set is to 0 and 1 
        N_0: Number of collocation points for the IC Default is set to 1000
        N_B: Number of collocation points for the main Domain: Default is set to 1000
        N_r: Number of collocation points for the main Domain: Default is set to 3000
        dimension : Spacial Dimension
               
        
        '''
    
        self.rmin = rmin
        self.rmax = rmax
        self.N_0 = N_0
        self.N_b = N_b
        self.N_r = N_r
        self.dimension = dimension
        
        
    def geo_time_coord(self,option,coordinate=1):
        '''
        option: Takes arguments: "Domain", "BC" for Boundary conditions, "IC" for initial conditions:
        
        '''
        
        if self.dimension == 1: 
            if option == "Domain":
                coor = []
                x_coor = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[0], self.rmax[0]).requires_grad_()
                coor.append(x_coor)
                t_coor = t_collocation=torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[1], self.rmax[1]).requires_grad_()
                coor.append(t_coor)

                return coor

            if option == "IC": ## Intial conditions collocation points

                coor = []
                x_0 = torch.empty(self.N_0,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[0], self.rmax[0]).requires_grad_()
                coor.append(x_0)

                t_0 = torch.empty(self.N_0,1, device='cuda', dtype=torch.float32).fill_(0).requires_grad_()
                coor.append(t_0)

                return coor

            if option == "BC":
                coor_l = []
                coor_r = []
                x_bc_l = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmin[coordinate-1]).requires_grad_()
                x_bc_r = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmax[coordinate-1]).requires_grad_()
                t_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[self.dimension], self.rmax[self.dimension])

                coor_l.append(x_bc_l)
                coor_l.append(t_bc)

                coor_r.append(x_bc_r)
                coor_r.append(t_bc)

                return  coor_l, coor_r
            
        if self.dimension == 2:
            if option == "Domain":
                
                coor = []
                for d in range(self.dimension+1): ## +1 for the time dimension
                    temp_coor = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[d], self.rmax[d]).requires_grad_()
                    coor.append(temp_coor)

                return coor
            
            if option == "IC": ## Intial conditions collocation points
                coor = []
                for d in range(self.dimension): ## time is zero as it is the initial condition
                    temp_coor = torch.empty(self.N_0,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[d], self.rmax[d]).requires_grad_()
                    coor.append(temp_coor)

                t_0 = torch.empty(self.N_0,1, device='cuda', dtype=torch.float32).fill_(0).requires_grad_()
                coor.append(t_0)

                return coor
            
            if option == "BC":
                t_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[self.dimension], self.rmax[self.dimension])
                t_bc.requires_grad_()
        
                if coordinate == 1: 
                    
                    x_bc_l = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmin[coordinate-1]).requires_grad_()
                    x_bc_r = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmax[coordinate-1]).requires_grad_()                   
                    y_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate], self.rmax[coordinate]).requires_grad_()
                    
                    coor_l = []
                    coor_r = []
                    coor_l.append(x_bc_l)
                    coor_l.append(y_bc)
                    coor_l.append(t_bc)
                    
                    
                    coor_r.append(x_bc_r)
                    coor_r.append(y_bc)
                    coor_r.append(t_bc)
                    
                    return coor_l , coor_r
                   
                 
                if coordinate == 2:
                    y_bc_l = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmin[coordinate-1]).requires_grad_()
                    y_bc_r = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmax[coordinate-1]).requires_grad_()
                    x_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate-2], self.rmax[coordinate-2]).requires_grad_()            
                    
                    coor_l = []
                    coor_r = []
                    coor_l.append(x_bc)
                    coor_l.append(y_bc_l)
                    coor_l.append(t_bc)
                    
                    
                    coor_r.append(x_bc)
                    coor_r.append(y_bc_r)
                    coor_r.append(t_bc)
                    
                    return coor_l, coor_r
    
            
        if self.dimension == 3: 
            if option == "Domain":
                
                coor = []
                for d in range(self.dimension+1): ## +1 for the time dimension
                    temp_coor = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[d], self.rmax[d]).requires_grad_()
                    coor.append(temp_coor)
                
                return coor

            if option == "IC": ## Intial conditions collocation points
                coor = []
                for d in range(self.dimension): ## time is zero as it is the initial condition                  
                    temp_coor = torch.empty(self.N_0,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[d], self.rmax[d]).requires_grad_()
                    coor.append(temp_coor)
                
                t_0 = torch.empty(self.N_0,1, device='cuda', dtype=torch.float32).fill_(0).requires_grad_()
                coor.append(t_0)
                
                return coor
             

            if option == "BC":                
                t_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[self.dimension], self.rmax[self.dimension])
                t_bc.requires_grad_()
        
                if coordinate == 1: 
                    
                    x_bc_l = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmin[coordinate-1]).requires_grad_()
                    x_bc_r = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmax[coordinate-1]).requires_grad_()                   
                    y_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate], self.rmax[coordinate]).requires_grad_()
                    z_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate+1], self.rmax[coordinate+1]).requires_grad_()
                    
                    coor_l = []
                    coor_r = []
                    coor_l.append(x_bc_l)
                    coor_l.append(y_bc)
                    coor_l.append(z_bc)
                    coor_l.append(t_bc)
                    
                    
                    coor_r.append(x_bc_r)
                    coor_r.append(y_bc)
                    coor_r.append(z_bc)
                    coor_r.append(t_bc)
                    
                    return coor_l , coor_r
                   
                 
                if coordinate == 2:
                    y_bc_l = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmin[coordinate-1]).requires_grad_()
                    y_bc_r = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmax[coordinate-1]).requires_grad_()
                    x_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate-2], self.rmax[coordinate-2]).requires_grad_()            
                    z_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate], self.rmax[coordinate]).requires_grad_()
                
                    coor_l = []
                    coor_r = []
                    coor_l.append(x_bc)
                    coor_l.append(y_bc_l)
                    coor_l.append(z_bc)
                    coor_l.append(t_bc)
                    
                    
                    coor_r.append(x_bc)
                    coor_r.append(y_bc_r)
                    coor_r.append(z_bc)
                    coor_r.append(t_bc)
                    
                    return coor_l, coor_r
                    
                   
                if coordinate == 3:
                    
                    z_bc_l = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmin[coordinate-1]).requires_grad_()
                    z_bc_r = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).fill_(self.rmax[coordinate-1]).requires_grad_()
                    x_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate-3], self.rmax[coordinate-3]).requires_grad_()
                    y_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[coordinate-2], self.rmax[coordinate-2]).requires_grad_()
                    
                    coor_l = []
                    coor_r = []
                    coor_l.append(x_bc)
                    coor_l.append(y_bc)
                    coor_l.append(z_bc_l)
                    coor_l.append(t_bc)
                    
                    
                    coor_r.append(x_bc)
                    coor_r.append(y_bc)
                    coor_r.append(z_bc_r)
                    coor_r.append(t_bc)
                    
                    return coor_l , coor_r