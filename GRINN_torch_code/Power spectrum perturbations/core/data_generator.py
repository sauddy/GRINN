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
    """
    Collocation point generator for PINNs.
    
    Generates collocation points for:
    - Domain (PDE enforcement)
    - Initial conditions (IC)
    - Boundary conditions (BC)
    
    Args:
        rmin: List of minimum values [xmin, ymin, zmin, tmin] (depending on dimension)
        rmax: List of maximum values [xmax, ymax, zmax, tmax] (depending on dimension)
        N_0: Number of initial condition collocation points
        N_b: Number of boundary condition points (per boundary)
        N_r: Number of residual/domain collocation points
        dimension: Spatial dimension (1, 2, or 3)
    """

    def __init__(self,rmin=[0,0,0,0],rmax=[1,1,1,1], N_0 = 1000,N_b=1000,N_r = 3000, dimension=1):
        self.rmin = rmin
        self.rmax = rmax
        self.N_0 = N_0
        self.N_b = N_b
        self.N_r = N_r
        self.dimension = dimension
    
    def _generate_uniform_tensor(self, n_points, lower, upper, device='cuda', requires_grad=True):
        """
        Helper function to generate uniformly distributed tensor.
        
        Args:
            n_points: Number of points to generate
            lower: Lower bound
            upper: Upper bound
            device: PyTorch device
            requires_grad: Whether tensor requires gradients
        
        Returns:
            Tensor of shape [n_points, 1]
        """
        tensor = torch.empty(n_points, 1, device=device, dtype=torch.float32).uniform_(lower, upper)
        if requires_grad:
            tensor = tensor.requires_grad_()
        return tensor
    
    def _generate_constant_tensor(self, n_points, value, device='cuda', requires_grad=True):
        """
        Helper function to generate constant-valued tensor.
        
        Args:
            n_points: Number of points
            value: Constant value
            device: PyTorch device
            requires_grad: Whether tensor requires gradients
        
        Returns:
            Tensor of shape [n_points, 1]
        """
        tensor = torch.empty(n_points, 1, device=device, dtype=torch.float32).fill_(value)
        if requires_grad:
            tensor = tensor.requires_grad_()
        return tensor
        
        
    def geo_time_coord(self,option,coordinate=1):
        '''
        option: Takes arguments: "Domain", "BC" for Boundary conditions, "IC" for initial conditions:
        
        '''
        
        if self.dimension == 1: 
            if option == "Domain":
                x_coor = self._generate_uniform_tensor(self.N_r, self.rmin[0], self.rmax[0])
                t_coor = self._generate_uniform_tensor(self.N_r, max(self.rmin[1], STARTUP_DT), self.rmax[1])
                return [x_coor, t_coor]

            if option == "IC":
                x_0 = self._generate_uniform_tensor(self.N_0, self.rmin[0], self.rmax[0])
                t_0 = self._generate_constant_tensor(self.N_0, 0.0)
                return [x_0, t_0]

            if option == "BC":
                if self.N_b == 0:
                    return [], []
                
                x_bc_l = self._generate_constant_tensor(self.N_b, self.rmin[coordinate-1])
                x_bc_r = self._generate_constant_tensor(self.N_b, self.rmax[coordinate-1])
                t_bc = self._generate_uniform_tensor(self.N_b, self.rmin[self.dimension], self.rmax[self.dimension])
                
                return [x_bc_l, t_bc], [x_bc_r, t_bc]
            
        if self.dimension == 2:
            if option == "Domain":
                x_dom = self._generate_uniform_tensor(self.N_r, self.rmin[0], self.rmax[0])
                y_dom = self._generate_uniform_tensor(self.N_r, self.rmin[1], self.rmax[1])
                t_dom = self._generate_uniform_tensor(self.N_r, max(self.rmin[2], STARTUP_DT), self.rmax[2])
                return [x_dom, y_dom, t_dom]
            
            if option == "IC":
                x_0 = self._generate_uniform_tensor(self.N_0, self.rmin[0], self.rmax[0])
                y_0 = self._generate_uniform_tensor(self.N_0, self.rmin[1], self.rmax[1])
                t_0 = self._generate_constant_tensor(self.N_0, 0.0)
                return [x_0, y_0, t_0]
            
            if option == "BC":
                if self.N_b == 0:
                    return [], []
                
                t_bc = self._generate_uniform_tensor(self.N_b, max(self.rmin[self.dimension], STARTUP_DT), self.rmax[self.dimension])
                
                if coordinate == 1:
                    x_bc_l = self._generate_constant_tensor(self.N_b, self.rmin[0])
                    x_bc_r = self._generate_constant_tensor(self.N_b, self.rmax[0])
                    y_bc = self._generate_uniform_tensor(self.N_b, self.rmin[1], self.rmax[1])
                    return [x_bc_l, y_bc, t_bc], [x_bc_r, y_bc, t_bc]
                
                if coordinate == 2:
                    y_bc_l = self._generate_constant_tensor(self.N_b, self.rmin[1])
                    y_bc_r = self._generate_constant_tensor(self.N_b, self.rmax[1])
                    x_bc = self._generate_uniform_tensor(self.N_b, self.rmin[0], self.rmax[0])
                    return [x_bc, y_bc_l, t_bc], [x_bc, y_bc_r, t_bc]
    
            
        if self.dimension == 3: 
            if option == "Domain":
                x_dom = self._generate_uniform_tensor(self.N_r, self.rmin[0], self.rmax[0])
                y_dom = self._generate_uniform_tensor(self.N_r, self.rmin[1], self.rmax[1])
                z_dom = self._generate_uniform_tensor(self.N_r, self.rmin[2], self.rmax[2])
                t_dom = self._generate_uniform_tensor(self.N_r, max(self.rmin[3], STARTUP_DT), self.rmax[3])
                return [x_dom, y_dom, z_dom, t_dom]

            if option == "IC":
                x_0 = self._generate_uniform_tensor(self.N_0, self.rmin[0], self.rmax[0])
                y_0 = self._generate_uniform_tensor(self.N_0, self.rmin[1], self.rmax[1])
                z_0 = self._generate_uniform_tensor(self.N_0, self.rmin[2], self.rmax[2])
                t_0 = self._generate_constant_tensor(self.N_0, 0.0)
                return [x_0, y_0, z_0, t_0]
             

            if option == "BC":
                if self.N_b == 0:
                    return [], []
                
                t_bc = self._generate_uniform_tensor(self.N_b, max(self.rmin[self.dimension], STARTUP_DT), self.rmax[self.dimension])
                
                if coordinate == 1:
                    x_bc_l = self._generate_constant_tensor(self.N_b, self.rmin[0])
                    x_bc_r = self._generate_constant_tensor(self.N_b, self.rmax[0])
                    y_bc = self._generate_uniform_tensor(self.N_b, self.rmin[1], self.rmax[1])
                    z_bc = self._generate_uniform_tensor(self.N_b, self.rmin[2], self.rmax[2])
                    return [x_bc_l, y_bc, z_bc, t_bc], [x_bc_r, y_bc, z_bc, t_bc]
                
                if coordinate == 2:
                    y_bc_l = self._generate_constant_tensor(self.N_b, self.rmin[1])
                    y_bc_r = self._generate_constant_tensor(self.N_b, self.rmax[1])
                    x_bc = self._generate_uniform_tensor(self.N_b, self.rmin[0], self.rmax[0])
                    z_bc = self._generate_uniform_tensor(self.N_b, self.rmin[2], self.rmax[2])
                    return [x_bc, y_bc_l, z_bc, t_bc], [x_bc, y_bc_r, z_bc, t_bc]
                
                if coordinate == 3:
                    z_bc_l = self._generate_constant_tensor(self.N_b, self.rmin[2])
                    z_bc_r = self._generate_constant_tensor(self.N_b, self.rmax[2])
                    x_bc = self._generate_uniform_tensor(self.N_b, self.rmin[0], self.rmax[0])
                    y_bc = self._generate_uniform_tensor(self.N_b, self.rmin[1], self.rmax[1])
                    return [x_bc, y_bc, z_bc_l, t_bc], [x_bc, y_bc, z_bc_r, t_bc]
    
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
        from methods.xpinn_decomposition import get_exterior_boundary_info
        
        # Extract global domain bounds from self.rmin, self.rmax
        if self.dimension == 2:
            xmin, ymin = self.rmin[0], self.rmin[1]
            xmax, ymax = self.rmax[0], self.rmax[1]
        else:
            raise NotImplementedError(f"Exterior boundary info not yet implemented for dimension {self.dimension}")
        
        return get_exterior_boundary_info(subdomain_idx, nx_sub, ny_sub, xmin, xmax, ymin, ymax)


# ==================== Utility Functions ====================

def input_taker(lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r):
    """
    Parse and validate input parameters for simulation.
    
    Args:
        lam: Wavelength
        rho_1: Amplitude of perturbation
        num_of_waves: Number of waves
        tmax: Maximum time
        N_0: Number of initial condition points
        N_b: Number of boundary points (legacy, not used with hard constraints)
        N_r: Number of collocation points
    
    Returns:
        Tuple (lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r) with correct types
    """
    lam = float(lam)
    rho_1 = float(rho_1)
    num_of_waves = int(num_of_waves)
    tmax = float(tmax)
    N_0 = int(N_0)
    N_r = int(N_r)
    
    return lam, rho_1, num_of_waves, tmax, N_0, N_b, N_r


def req_consts_calc(lam, rho_1):
    """
    Calculate required physical constants for Jeans instability.
    
    Args:
        lam: Wavelength
        rho_1: Amplitude of perturbation (not used in calculation but kept for API consistency)
    
    Returns:
        Tuple (jeans_length, alpha) where:
        - jeans_length: Jeans wavelength (critical wavelength for instability)
        - alpha: Growth rate or oscillation frequency depending on lam vs jeans_length
    """
    from config import cs, const, G, rho_o
    
    if rho_o != 0:
        jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*rho_o))
    else:
        jeans = np.sqrt(4*np.pi**2*cs**2/(const*G*(rho_o + 1)))

    if lam > jeans:
        # Unstable regime: exponential growth
        if rho_o != 0:
            alpha = np.sqrt(const*G*rho_o-cs**2*(2*np.pi/lam)**2)
        else:
            alpha = np.sqrt(const*G*(rho_o + 1)-cs**2*(2*np.pi/lam)**2)
    else:
        # Stable regime: oscillations
        if rho_o != 0:
            alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*rho_o)
        else:
            alpha = np.sqrt(cs**2*(2*np.pi/lam)**2 - const*G*(rho_o + 1))

    return jeans, alpha


def distribute_collocation_points(n_total, num_subdomains):
    """
    Distribute collocation points across subdomains for XPINN.
    
    Args:
        n_total: Total number of collocation points
        num_subdomains: Number of subdomains
    
    Returns:
        List of point counts per subdomain
    """
    from config import N_r_PER_SUBDOMAIN, N_0_PER_SUBDOMAIN
    
    # If per-subdomain count is specified, use it
    if n_total == N_r_PER_SUBDOMAIN and N_r_PER_SUBDOMAIN is not None:
        return [N_r_PER_SUBDOMAIN] * num_subdomains
    elif n_total == N_0_PER_SUBDOMAIN and N_0_PER_SUBDOMAIN is not None:
        return [N_0_PER_SUBDOMAIN] * num_subdomains
    
    # Otherwise, auto-distribute evenly
    base_count = n_total // num_subdomains
    remainder = n_total % num_subdomains
    
    counts = [base_count] * num_subdomains
    # Distribute remainder to first few subdomains
    for i in range(remainder):
        counts[i] += 1
    
    return counts