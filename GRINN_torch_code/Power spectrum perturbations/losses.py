from data_generator import col_gen
from data_generator import diff

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from config import cs, const, G, rho_o

class ASTPN(col_gen):
    
    def __init__(self, rmin=[0,0,0,0], rmax=[1,1,1,1], N_0 = 1000, N_b=1000, N_r=3000, dimension=1):
        super().__init__(rmin,rmax, N_0,0,N_r, dimension)  # N_b set to 0 due to hard constraints
        
       
        self.coord_Lx, self.coord_Rx = self.geo_time_coord(option="BC",coordinate=1)
        
        if dimension == 2:
            self.coord_Ly, self.coord_Ry = self.geo_time_coord(option="BC",coordinate=2)

        if dimension == 3:
            self.coord_Ly, self.coord_Ry = self.geo_time_coord(option="BC",coordinate=2)
            self.coord_Lz, self.coord_Rz = self.geo_time_coord(option="BC",coordinate=3)
    
    
    
    def periodic_BC(self,net,coordinate=1,derivative_order=0,component=0):
        
        '''
           INPUT: geomtime: The collocatin grids
           since it is BC the derivative always wrt to spacial coordinate: coordinate =1 is the first x coordinate
           derivative order: what order derivative
           component: output component's derivative is taken component = 0 is the First output fron the network
           default is set to 0,i.e., the first output
           coordinate : 1: xaxis (default), 2: with y axis 3: with zaxis
         '''

        if coordinate==1:   
            coord_L, coord_R = self.coord_Lx, self.coord_Rx
        if coordinate==2:       
            coord_L, coord_R = self.coord_Ly, self.coord_Ry
        if coordinate==3:       
            coord_L, coord_R = self.coord_Lz, self.coord_Rz
        
        
        # return coord_L, coord_R
          
        variable_l = net(coord_L)[:,component:component+1] 
        variable_r = net(coord_R)[:,component:component+1] 
        
        if derivative_order == 0:

            return torch.mean((variable_l - variable_r)**2)

        elif derivative_order == 1:        
            der_l = diff(variable_l,coord_L[coordinate-1],order=derivative_order)
            der_r = diff(variable_r,coord_R[coordinate-1],order=derivative_order)

            return torch.mean((der_l-der_r)**2)


def pde_residue(colloc, net, dimension = 1):
    
    '''
    This is the main function that returns all the PDE residue
    '''
    net_outputs = net(colloc)
    
    x = colloc[0]
    
    if dimension == 1:
        t = colloc[1]

    elif dimension == 2:
        y = colloc[1]
        t = colloc[2]

    elif dimension == 3:
        y = colloc[1]
        z = colloc[2]
        t = colloc[3]
    
    rho, vx = net_outputs[:,0:1], net_outputs[:,1:2]

    if dimension == 1:

        phi = net_outputs[:,2:3]

        rho_t = diff(rho,t,order=1)  
        rho_x = diff(rho,x,order=1)

        vx_t = diff(vx, t,order=1)
        vx_x = diff(vx, x,order=1)
        
        phi_x = diff(phi,x,order=1)
        phi_x_x = diff(phi,x,order=2)

    elif dimension == 2:

        vy = net_outputs[:,2:3]
        phi = net_outputs[:,3:4]

        rho_t = diff(rho,t,order=1)  
        rho_x = diff(rho,x,order=1)
        rho_y = diff(rho,y,order=1)

        vx_t = diff(vx, t,order=1)
        vy_t = diff(vy, t,order=1)

        vx_x = diff(vx, x,order=1)
        vx_y = diff(vx, y,order=1)
        vy_x = diff(vy, x,order=1)
        vy_y = diff(vy, y,order=1)
        
        phi_x = diff(phi,x,order=1)
        phi_x_x = diff(phi,x,order=2)

        phi_y = diff(phi,y,order=1)
        phi_y_y = diff(phi,y,order=2)

    elif dimension == 3:
        vy = net_outputs[:,2:3]
        vz = net_outputs[:,3:4]
        phi = net_outputs[:,4:5]

        rho_t = diff(rho,t,order=1)  
        rho_x = diff(rho,x,order=1)
        rho_y = diff(rho,y,order=1)
        rho_z = diff(rho,z,order=1)

        vx_t = diff(vx, t,order=1)
        vy_t = diff(vy, t,order=1)
        vz_t = diff(vz, t,order=1)

        vx_x = diff(vx, x,order=1)
        vy_x = diff(vy, x,order=1)
        vz_x = diff(vz, x,order=1)

        vx_y = diff(vx, y,order=1)
        vy_y = diff(vy, y,order=1)
        vz_y = diff(vz, y,order=1)
        
        vx_z = diff(vx, z,order=1)
        vy_z = diff(vy, z,order=1)
        vz_z = diff(vz, z,order=1)
        
        phi_x = diff(phi,x,order=1)
        phi_x_x = diff(phi,x,order=2)

        phi_y = diff(phi,y,order=1)
        phi_y_y = diff(phi,y,order=2)
    
        phi_z = diff(phi,z,order=1)
        phi_z_z = diff(phi,z,order=2)

    
    ## The residues from the equations

    if dimension == 1:
        rho_r = rho_t + vx * rho_x + rho * vx_x
        vx_r = rho*vx_t + rho*(vx*vx_x) + cs*cs*rho_x +rho*phi_x
        phi_r = phi_x_x - const*(rho - rho_o)

        return rho_r, vx_r, phi_r

    elif dimension == 2:
        rho_r = rho_t + vx * rho_x + vy * rho_y + rho * vx_x + rho * vy_y
        vx_r = rho*vx_t + rho*(vx*vx_x + vy*vx_y) + cs*cs*rho_x + rho*phi_x
        vy_r = rho*vy_t + rho*(vy*vy_y + vx*vy_x) + cs*cs*rho_y + rho*phi_y
        phi_r = phi_x_x + phi_y_y - const*(rho - rho_o)

        return rho_r, vx_r, vy_r, phi_r
    
    elif dimension == 3:
        rho_r = rho_t + vx * rho_x + rho * vx_x + vy *rho_y + rho * vy_y + vz *rho_z +rho * vz_z
        vx_r = rho*vx_t + rho*(vx*vx_x + vy*vx_y+vz*vx_z) + cs*cs*rho_x + rho*phi_x
        vy_r = rho*vy_t + rho*(vy*vy_y + vx*vy_x+vz*vy_z) + cs*cs*rho_y + rho*phi_y
        vz_r = rho*vz_t + rho*(vz*vz_z + vx*vz_x+vy*vz_y) + cs*cs*rho_z + rho*phi_z
        phi_r = phi_x_x + phi_y_y +phi_z_z - const*(rho - rho_o)
        
        return rho_r,vx_r,vy_r,vz_r,phi_r
