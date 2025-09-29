import numpy as np

import torch
import torch.nn as nn
#from torch.autograd import Variable
from config import rho_o

class Sin(nn.Module):
    def forward(self, input):
        return torch.sin(input)

class PINN(nn.Module):
    def __init__(self, num_neurons=48, n_harmonics=1):
        super(PINN, self).__init__()
        self.num_neurons = num_neurons
        self.n_harmonics = n_harmonics
        
        # Domain extents for periodic embeddings (set via set_domain)
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.zmin = None
        self.zmax = None
    
    # 1D branch (periodic x features + t)
        in_dim_1d = 2*self.n_harmonics + 1
        self.branch_1d = nn.Sequential(
            nn.Linear(in_dim_1d, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, 3))
        
    # 2D branch (periodic x,y features + t)
        in_dim_2d = 4*self.n_harmonics + 1
        self.branch_2d = nn.Sequential(
            nn.Linear(in_dim_2d, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, 4))
        
    # 3D branch (periodic x,y,z features + t)
        in_dim_3d = 6*self.n_harmonics + 1
        self.branch_3d = nn.Sequential(
            nn.Linear(in_dim_3d, num_neurons),
            Sin(),
            nn.Linear(num_neurons, num_neurons),
            Sin(),
            nn.Linear(num_neurons, 5))
        
        # Output layers per branch
        #self.output_layer_1d = nn.Linear(3, 1)
        #self.output_layer_2d = nn.Linear(4, 1)
        #self.output_layer_3d = nn.Linear(5, 1)


    def set_domain(self, rmin, rmax, dimension):
        # rmin/rmax exclude time; follow ASTPN usage
        if dimension >= 1:
            self.xmin, self.xmax = float(rmin[0]), float(rmax[0])
        if dimension >= 2:
            self.ymin, self.ymax = float(rmin[1]), float(rmax[1])
        if dimension >= 3:
            self.zmin, self.zmax = float(rmin[2]), float(rmax[2])

    def _periodic_features(self, u, umin, umax):
        # u is [N,1]
        L = umax - umin
        theta = 2*np.pi*(u - umin)/L
        features = []
        for k in range(1, self.n_harmonics+1):
            features.append(torch.sin(k*theta))
            features.append(torch.cos(k*theta))
        return torch.cat(features, dim=1) if len(features) > 0 else u

    def forward(self,X):
        x, t = X[0],X[-1]
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        t = t.unsqueeze(-1) if t.dim() == 1 else t

        if len(X) == 2:
            if self.xmin is None or self.xmax is None:
                raise RuntimeError("Domain not set: call net.set_domain for dimension=1")
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            inputs = torch.cat([x_feat, t], dim=1)
            outputs = self.branch_1d(inputs)
            # Hard-enforce uniform density at t=0 without in-place ops
            rho_hat = outputs[:,0:1]
            other = outputs[:,1:]
            rho = rho_o + t * rho_hat
            outputs_mod = torch.cat([rho, other], dim=1)
            return outputs_mod
        
        elif len(X) == 3:
            if self.xmin is None or self.xmax is None or self.ymin is None or self.ymax is None:
                raise RuntimeError("Domain not set: call net.set_domain for dimension=2")
            y = X[1]
            y = y.unsqueeze(-1) if y.dim() == 1 else y
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            y_feat = self._periodic_features(y, self.ymin, self.ymax)
            inputs = torch.cat([x_feat, y_feat, t], dim=1)
            outputs = self.branch_2d(inputs)
            rho_hat = outputs[:,0:1]
            other = outputs[:,1:]
            rho = rho_o + t * rho_hat
            outputs_mod = torch.cat([rho, other], dim=1)
            return outputs_mod
        
        elif len(X) == 4:
            if (self.xmin is None or self.xmax is None or
                self.ymin is None or self.ymax is None or
                self.zmin is None or self.zmax is None):
                raise RuntimeError("Domain not set: call net.set_domain for dimension=3")
            y = X[1]
            z = X[2]
            y = y.unsqueeze(-1) if y.dim() == 1 else y
            z = z.unsqueeze(-1) if z.dim() == 1 else z
            x_feat = self._periodic_features(x, self.xmin, self.xmax)
            y_feat = self._periodic_features(y, self.ymin, self.ymax)
            z_feat = self._periodic_features(z, self.zmin, self.zmax)
            inputs = torch.cat([x_feat, y_feat, z_feat, t], dim=1)
            outputs = self.branch_3d(inputs)
            rho_hat = outputs[:,0:1]
            other = outputs[:,1:]
            rho = rho_o + t * rho_hat
            outputs_mod = torch.cat([rho, other], dim=1)
            return outputs_mod
        
        else:
            raise ValueError(f"Expected len(X) in [2, 3, 4] but got {len(X)}")
        
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
