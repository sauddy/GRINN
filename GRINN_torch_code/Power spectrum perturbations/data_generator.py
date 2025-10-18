import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from config import STARTUP_DT

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
                # Shift PDE enforcement to start at t = STARTUP_DT
                t_coor = t_collocation=torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(max(self.rmin[1], STARTUP_DT), self.rmax[1]).requires_grad_()
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
                # Skip BC generation if N_b is 0 (hard constraints used instead)
                if self.N_b == 0:
                    return [], []
                    
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
                # x, y sampled as before; t starts from STARTUP_DT
                x_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[0], self.rmax[0]).requires_grad_()
                y_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[1], self.rmax[1]).requires_grad_()
                t_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(max(self.rmin[2], STARTUP_DT), self.rmax[2]).requires_grad_()
                coor.append(x_dom)
                coor.append(y_dom)
                coor.append(t_dom)

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
                # Skip BC generation if N_b is 0 (hard constraints used instead)
                if self.N_b == 0:
                    return [], []
                    
                # BC evaluated for t >= STARTUP_DT
                t_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(max(self.rmin[self.dimension], STARTUP_DT), self.rmax[self.dimension])
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
                x_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[0], self.rmax[0]).requires_grad_()
                y_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[1], self.rmax[1]).requires_grad_()
                z_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(self.rmin[2], self.rmax[2]).requires_grad_()
                t_dom = torch.empty(self.N_r,1, device='cuda', dtype=torch.float32).uniform_(max(self.rmin[3], STARTUP_DT), self.rmax[3]).requires_grad_()
                coor.append(x_dom)
                coor.append(y_dom)
                coor.append(z_dom)
                coor.append(t_dom)
                
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
                # Skip BC generation if N_b is 0 (hard constraints used instead)
                if self.N_b == 0:
                    return [], []
                    
                t_bc   = torch.empty(self.N_b,1, device='cuda', dtype=torch.float32).uniform_(max(self.rmin[self.dimension], STARTUP_DT), self.rmax[self.dimension])
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
    
    def geo_time_coord_subdomain(self, option, subdomain_bounds, device='cuda', coordinate=1):
        """
        Generate collocation points within a subdomain.
        
        Args:
            option: "Domain", "IC", or "BC"
            subdomain_bounds: Tuple (x_min, x_max, y_min, y_max) for 2D
            device: PyTorch device
            coordinate: For BC generation (1=x, 2=y)
        
        Returns:
            Collocation points list constrained to subdomain
        """
        # For 2D (current focus)
        if self.dimension == 2:
            x_min, x_max, y_min, y_max = subdomain_bounds
            
            if option == "Domain":
                coor = []
                x_dom = torch.empty(self.N_r, 1, device=device, dtype=torch.float32).uniform_(x_min, x_max).requires_grad_()
                y_dom = torch.empty(self.N_r, 1, device=device, dtype=torch.float32).uniform_(y_min, y_max).requires_grad_()
                t_dom = torch.empty(self.N_r, 1, device=device, dtype=torch.float32).uniform_(max(self.rmin[2], STARTUP_DT), self.rmax[2]).requires_grad_()
                coor.append(x_dom)
                coor.append(y_dom)
                coor.append(t_dom)
                return coor
            
            elif option == "IC":
                coor = []
                x_ic = torch.empty(self.N_0, 1, device=device, dtype=torch.float32).uniform_(x_min, x_max).requires_grad_()
                y_ic = torch.empty(self.N_0, 1, device=device, dtype=torch.float32).uniform_(y_min, y_max).requires_grad_()
                t_ic = torch.empty(self.N_0, 1, device=device, dtype=torch.float32).fill_(0).requires_grad_()
                coor.append(x_ic)
                coor.append(y_ic)
                coor.append(t_ic)
                return coor
            
            elif option == "BC":
                # Only generate BC if subdomain touches exterior boundary
                # This is handled by get_exterior_boundary_info
                if self.N_b == 0:
                    return [], []
                
                t_bc = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).uniform_(max(self.rmin[2], STARTUP_DT), self.rmax[2])
                t_bc.requires_grad_()
                
                if coordinate == 1:
                    x_bc_l = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).fill_(x_min).requires_grad_()
                    x_bc_r = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).fill_(x_max).requires_grad_()
                    y_bc = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).uniform_(y_min, y_max).requires_grad_()
                    
                    coor_l = [x_bc_l, y_bc, t_bc]
                    coor_r = [x_bc_r, y_bc, t_bc]
                    return coor_l, coor_r
                
                elif coordinate == 2:
                    y_bc_l = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).fill_(y_min).requires_grad_()
                    y_bc_r = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).fill_(y_max).requires_grad_()
                    x_bc = torch.empty(self.N_b, 1, device=device, dtype=torch.float32).uniform_(x_min, x_max).requires_grad_()
                    
                    coor_l = [x_bc, y_bc_l, t_bc]
                    coor_r = [x_bc, y_bc_r, t_bc]
                    return coor_l, coor_r
        
        else:
            raise NotImplementedError(f"Subdomain collocation not yet implemented for dimension {self.dimension}")
    
    def get_exterior_boundary_info(self, subdomain_idx, nx_sub, ny_sub):
        """
        Determine which boundaries of a subdomain are exterior boundaries.
        Uses xpinn_decomposition utilities.
        
        Args:
            subdomain_idx: Linear subdomain index
            nx_sub: Number of subdomain splits in x-direction
            ny_sub: Number of subdomain splits in y-direction
        
        Returns:
            Dict with keys 'left', 'right', 'bottom', 'top' indicating if boundary is exterior
        """
        from xpinn_decomposition import get_exterior_boundary_info
        
        # Extract global domain bounds from self.rmin, self.rmax
        if self.dimension == 2:
            xmin, ymin = self.rmin[0], self.rmin[1]
            xmax, ymax = self.rmax[0], self.rmax[1]
        else:
            raise NotImplementedError(f"Exterior boundary info not yet implemented for dimension {self.dimension}")
        
        return get_exterior_boundary_info(subdomain_idx, nx_sub, ny_sub, xmin, xmax, ymin, ymax)