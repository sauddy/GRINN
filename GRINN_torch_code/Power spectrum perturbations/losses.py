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


class XPINN_Loss:
    """
    XPINN Loss computation for domain decomposition.
    
    Computes:
    - PDE residual loss per subdomain
    - Initial condition loss per subdomain
    - Periodic BC loss (only exterior boundaries)
    - Interface continuity losses (solution + residual)
    """
    
    def __init__(self, rmin, rmax, dimension=2):
        """
        Initialize XPINN loss computer.
        
        Args:
            rmin: List of minimum values [xmin, ymin, tmin] (for 2D)
            rmax: List of maximum values [xmax, ymax, tmax] (for 2D)
            dimension: Spatial dimension (default 2)
        """
        self.rmin = rmin
        self.rmax = rmax
        self.dimension = dimension
        
        # Import config values
        from config import (INTERFACE_SOLUTION_WEIGHT, INTERFACE_RESIDUAL_WEIGHT,
                           INTERFACE_SOLUTION_COMPONENTS)
        self.interface_solution_weight = INTERFACE_SOLUTION_WEIGHT
        self.interface_residual_weight = INTERFACE_RESIDUAL_WEIGHT
        self.interface_components = INTERFACE_SOLUTION_COMPONENTS
        
        # Component name to index mapping
        self.component_map = {'rho': 0, 'vx': 1, 'vy': 2, 'phi': 3}
    
    def compute_pde_loss(self, colloc, net):
        """
        Compute PDE residual loss for a subdomain.
        
        Args:
            colloc: Collocation points [x, y, t]
            net: Neural network for this subdomain
        
        Returns:
            PDE residual loss (scalar)
        """
        residuals = pde_residue(colloc, net, dimension=self.dimension)
        
        # Sum squared residuals
        loss = sum(torch.mean(r**2) for r in residuals)
        return loss
    
    def compute_ic_loss(self, colloc_ic, net, ic_functions):
        """
        Compute initial condition loss for a subdomain.
        
        Args:
            colloc_ic: Initial condition collocation points [x, y, t=0]
            net: Neural network for this subdomain
            ic_functions: Dictionary of initial condition functions
                          {'rho': func, 'vx': func, 'vy': func, 'phi': func}
        
        Returns:
            IC loss (scalar)
        """
        # Get network predictions at t=0
        u_pred = net(colloc_ic)
        
        # Compute initial conditions
        ic_rho = ic_functions['rho'](colloc_ic)
        ic_vx = ic_functions['vx'](colloc_ic)
        ic_vy = ic_functions['vy'](colloc_ic)
        ic_phi = ic_functions['phi'](colloc_ic)
        
        # Compute MSE for each component
        loss_rho = torch.mean((u_pred[:, 0:1] - ic_rho)**2)
        loss_vx = torch.mean((u_pred[:, 1:2] - ic_vx)**2)
        loss_vy = torch.mean((u_pred[:, 2:3] - ic_vy)**2)
        loss_phi = torch.mean((u_pred[:, 3:4] - ic_phi)**2)
        
        total_ic_loss = loss_rho + loss_vx + loss_vy + loss_phi
        return total_ic_loss
    
    def compute_interface_solution_loss(self, colloc_interface, net1, net2):
        """
        Compute solution continuity loss at interface.
        
        Enforces: u_avg = (u1 + u2)/2 for both networks.
        Minimizes: |u1 - u_avg|^2 + |u2 - u_avg|^2
        
        Args:
            colloc_interface: Interface collocation points [x, y, t]
            net1, net2: Neural networks for adjacent subdomains
        
        Returns:
            Solution continuity loss (scalar)
        """
        # Get predictions from both networks at interface
        u1 = net1(colloc_interface)
        u2 = net2(colloc_interface)
        
        # Compute average
        u_avg = (u1 + u2) / 2.0
        
        # Compute loss for specified components only
        loss = 0.0
        for comp_name in self.interface_components:
            if comp_name in self.component_map:
                idx = self.component_map[comp_name]
                if idx < u1.shape[1]:  # Check component exists
                    u1_comp = u1[:, idx:idx+1]
                    u2_comp = u2[:, idx:idx+1]
                    u_avg_comp = u_avg[:, idx:idx+1]
                    
                    # XPINN approach: enforce average value
                    loss += torch.mean((u1_comp - u_avg_comp)**2)
                    loss += torch.mean((u2_comp - u_avg_comp)**2)
        
        return self.interface_solution_weight * loss
    
    def compute_interface_residual_loss(self, colloc_interface, net1, net2):
        """
        Compute residual continuity loss at interface.
        
        Enforces: R1(interface) = R2(interface)
        
        Args:
            colloc_interface: Interface collocation points [x, y, t]
            net1, net2: Neural networks for adjacent subdomains
        
        Returns:
            Residual continuity loss (scalar)
        """
        # Compute PDE residuals from both networks at interface
        residuals1 = pde_residue(colloc_interface, net1, dimension=self.dimension)
        residuals2 = pde_residue(colloc_interface, net2, dimension=self.dimension)
        
        # Enforce residual matching for all PDE components
        loss = 0.0
        for r1, r2 in zip(residuals1, residuals2):
            loss += torch.mean((r1 - r2)**2)
        
        return self.interface_residual_weight * loss
    
    # NOTE: Periodic boundary conditions are enforced via periodic feature encoding
    # (n_harmonics in _periodic_features method of PINN class), NOT via loss term.
    # This is a hard constraint approach where the network architecture guarantees periodicity.
    
    def compute_total_loss(self, nets, subdomain_collocs, interface_collocs, 
                          subdomain_ic_collocs, ic_functions, interfaces, 
                          exterior_boundaries):
        """
        Compute total XPINN loss aggregating all components.
        
        Args:
            nets: List of neural networks (one per subdomain)
            subdomain_collocs: List of subdomain collocation points
            interface_collocs: Dict mapping interface tuple to collocation points
            subdomain_ic_collocs: List of IC collocation points per subdomain
            ic_functions: Initial condition functions
            interfaces: List of interface tuples (subdomain_i, subdomain_j, type, pos)
            exterior_boundaries: Dict mapping subdomain_idx to boundary info
        
        Returns:
            Tuple (total_loss, loss_dict) where loss_dict contains component losses
        """
        loss_dict = {
            'pde': 0.0,
            'ic': 0.0,
            'interface_solution': 0.0,
            'interface_residual': 0.0
        }
        
        # PDE and IC losses for each subdomain
        for i, (net, colloc, colloc_ic) in enumerate(zip(nets, subdomain_collocs, subdomain_ic_collocs)):
            loss_dict['pde'] += self.compute_pde_loss(colloc, net)
            loss_dict['ic'] += self.compute_ic_loss(colloc_ic, net, ic_functions)
        
        # Interface losses
        for interface in interfaces:
            subdomain_i, subdomain_j, _, _ = interface
            interface_key = (subdomain_i, subdomain_j)
            
            if interface_key in interface_collocs:
                colloc_interface = interface_collocs[interface_key]
                net1 = nets[subdomain_i]
                net2 = nets[subdomain_j]
                
                loss_dict['interface_solution'] += self.compute_interface_solution_loss(
                    colloc_interface, net1, net2)
                loss_dict['interface_residual'] += self.compute_interface_residual_loss(
                    colloc_interface, net1, net2)
        
        # NOTE: Periodic BC NOT included here - enforced via periodic feature encoding (harmonics)
        # in the PINN architecture, which is a hard constraint approach.
        
        # Total loss
        total_loss = sum(loss_dict.values())
        
        return total_loss, loss_dict
